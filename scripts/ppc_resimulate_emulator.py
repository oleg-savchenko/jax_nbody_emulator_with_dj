#!/usr/bin/env python3
"""
Staged posterior-predictive resimulation pipeline for emulator IC posterior samples.

Workflow:
1) Forward stage (GPU env):
   - Load sample_*.npy ICs from a PPC sim folder (.../ppc/<tag>/<sim_id>/).
   - Upsample ICs from coarse grid (typically 64^3) to n_part (typically 128^3)
     via random high-k mode injection using a linear CLASS P(k).
   - Run emulator forward modeling and save:
       emu_delta_{id:04d}.npy
       emu_dis_{id:04d}.npy
2) Halos stage (CPU/MPI env):
   - Run distributed FoF on emu_dis_{id:04d}.npy for each sample.
   - Save:
       fof_catalog_{id:04d}.npz
       mass_function_{id:04d}.npy  (2 cols: [M_center, dn/dM])
3) HOD stage (CPU env):
   - Run HOD galaxy population from fof_catalog_{id:04d}.npz.
   - Save:
       hod_catalog_{id:04d}.npz
4) Optional auto-PPC:
   - Run g2f_emu/ppc.py on rank 0 after halos/all stage.

Designed to mirror scripts/generate_dataset.py + my_generate_dataset_job.sh conventions,
while operating inside a PPC sample folder for one simulation id.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Avoid matplotlib cache write warnings triggered by importing scripts/halos.py.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = f"/tmp/matplotlib-{os.environ.get('USER', 'user')}"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from halos import (
    gather_halos_to_root,
    load_local_positions_from_displacement,
    particle_mass_msun_h,
    run_fof,
)
from utils import add_import_paths, get_pk_class, parse_ndiv

REPO_ROOT = Path(__file__).resolve().parent.parent
add_import_paths(repo_root=REPO_ROOT)

try:
    from core import QUIJOTE_FIDUCIAL, run_lpt_emulator_pipeline
except ImportError as exc:  # pragma: no cover - optional dependency at import time
    QUIJOTE_FIDUCIAL = {"Omega_m": 0.3175, "Omega_b": 0.0490, "h": 0.6711, "n_s": 0.9624, "sigma8": 0.8340}
    run_lpt_emulator_pipeline = None
    _CORE_IMPORT_ERROR = exc
else:
    _CORE_IMPORT_ERROR = None

try:
    from mpi4py import MPI
except Exception as exc:  # pragma: no cover - optional dependency at import time
    MPI = None
    _MPI_IMPORT_ERROR = exc
else:
    _MPI_IMPORT_ERROR = None

try:
    from nbodykit.algorithms.fof import FOF, fof_catalog
    from nbodykit.source.catalog import ArrayCatalog
except Exception as exc:  # pragma: no cover - optional dependency at import time
    FOF = None
    fof_catalog = None
    ArrayCatalog = None
    _NBODYKIT_IMPORT_ERROR = exc
else:
    _NBODYKIT_IMPORT_ERROR = None

try:
    import pyhod.hodfunc as HODF
except Exception as exc:  # pragma: no cover - optional dependency at import time
    HODF = None
    _PYHOD_IMPORT_ERROR = exc
else:
    _PYHOD_IMPORT_ERROR = None

try:
    import pyhod.utils as HOD_UTILS
except Exception:  # pragma: no cover - optional dependency at import time
    HOD_UTILS = None


DEFAULT_G2F_PPC_SCRIPT = Path("/home/osavchenko/GNN_FLI/g2f_yannick/g2f_emu/ppc.py")
DEFAULT_G2F_RUN_ROOT = Path("/home/osavchenko/GNN_FLI/g2f_yannick/g2f_emu/gnpe_plots")

HOD_DEFAULT_LOGMMIN = 13.0
HOD_DEFAULT_SIGMALOGM = 0.2
HOD_DEFAULT_LOGM0 = 13.1
HOD_DEFAULT_ALPHA = 0.75
HOD_DEFAULT_LOGM1 = 14.3
# HOD cosmology must match the simulations (flat Quijote fiducial); it was Planck15
# (Om=0.3089, h=0.6774) before. pyhod only uses Om (satellite NFW r_vir), so older
# catalogs differ by <1% in satellite radii.
HOD_DEFAULT_OMEGA_M = float(QUIJOTE_FIDUCIAL["Omega_m"])
HOD_DEFAULT_OMEGA_L = 1.0 - HOD_DEFAULT_OMEGA_M
HOD_DEFAULT_H = float(QUIJOTE_FIDUCIAL["h"])
HOD_DEFAULT_Z = 0.0
HOD_DEFAULT_M_STAR = 1.670992e13
HOD_DEFAULT_DELTAC = 200
HOD_DEFAULT_ZERR = 0.0


class _SerialComm:
    """Minimal MPI-like communicator for single-process stages."""

    rank = 0
    size = 1

    def Barrier(self) -> None:
        return None

    def gather(self, value: Any, root: int = 0) -> list[Any] | None:
        return [value] if int(root) == 0 else None

    def bcast(self, value: Any, root: int = 0) -> Any:
        return value

    def allreduce(self, value: Any, op: Any = None) -> Any:
        return value


def log_progress(message: str, *, rank: int, root_only: bool = True) -> None:
    """Print a timestamped progress message (root-only by default)."""
    if root_only and rank != 0:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [rank {rank}] {message}", flush=True)


def format_hms(seconds: float) -> str:
    """Format seconds as HHh MMm SSs."""
    total = int(round(float(seconds)))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}h {mm:02d}m {ss:02d}s"


def extract_numeric_id(path: Path, fallback_index: int) -> int:
    """Extract integer id from a filename stem, fallback to provided index."""
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        return int(fallback_index)
    return int(match.group(1))


def sort_paths_by_numeric_stem(paths: list[Path]) -> list[Path]:
    """Sort paths by first numeric token in stem, then lexicographically."""

    def _key(p: Path) -> tuple[int, str]:
        m = re.search(r"(\d+)", p.stem)
        if m is None:
            return (10**9, p.name)
        return (int(m.group(1)), p.name)

    return sorted(paths, key=_key)


def select_paths(
    *,
    paths: list[Path],
    sample_start: int,
    sample_stride: int,
    num_samples: int | None,
) -> list[Path]:
    """Apply start/stride/optional cap selection to an already sorted path list."""
    selected = paths[int(sample_start) :: int(sample_stride)]
    if num_samples is not None:
        selected = selected[: int(num_samples)]
    return selected


def ppc_sample_entries(
    *,
    ppc_sim_dir: Path,
    sample_pattern: str,
    sample_start: int,
    sample_stride: int,
    num_samples: int | None,
    require_input_samples: bool,
    fallback_patterns: list[str],
) -> list[dict[str, Any]]:
    """
    Build selected sample entries.

    Preferred source is sample_*.npy files. For non-forward runs, if no sample files are
    found, fallback to configured existing per-sample artifact patterns.
    """
    sample_files = sort_paths_by_numeric_stem(list(ppc_sim_dir.glob(str(sample_pattern))))
    if sample_files:
        selected = select_paths(
            paths=sample_files,
            sample_start=sample_start,
            sample_stride=sample_stride,
            num_samples=num_samples,
        )
        entries: list[dict[str, Any]] = []
        for i, path in enumerate(selected):
            sid = extract_numeric_id(path, fallback_index=i)
            entries.append({"sample_id": int(sid), "sample_file": str(path)})
        if entries:
            return entries

    if require_input_samples:
        raise FileNotFoundError(
            f"No files matching {sample_pattern!r} found in {ppc_sim_dir}; "
            "forward stage requires sample_*.npy inputs."
        )

    for patt in fallback_patterns:
        fallback_files = sort_paths_by_numeric_stem(list(ppc_sim_dir.glob(str(patt))))
        if not fallback_files:
            continue
        selected = select_paths(
            paths=fallback_files,
            sample_start=sample_start,
            sample_stride=sample_stride,
            num_samples=num_samples,
        )
        entries = []
        for i, path in enumerate(selected):
            sid = extract_numeric_id(path, fallback_index=i)
            entries.append({"sample_id": int(sid), "sample_file": None})
        if entries:
            return entries

    fallback_desc = ", ".join(repr(p) for p in fallback_patterns) if fallback_patterns else "(no fallback patterns)"
    raise FileNotFoundError(
        f"No sample files ({sample_pattern!r}) or fallback files ({fallback_desc}) found in {ppc_sim_dir}."
    )


def sim_file_paths(ppc_sim_dir: Path, sample_id: int) -> dict[str, Path]:
    """Return canonical output paths for one PPC sample id."""
    sid = int(sample_id)
    return {
        "emu_delta": ppc_sim_dir / f"emu_delta_{sid:04d}.npy",
        "emu_dis": ppc_sim_dir / f"emu_dis_{sid:04d}.npy",
        "fof_catalog": ppc_sim_dir / f"fof_catalog_{sid:04d}.npz",
        "mass_function": ppc_sim_dir / f"mass_function_{sid:04d}.npy",
        "hod_catalog": ppc_sim_dir / f"hod_catalog_{sid:04d}.npz",
    }


def _quijote_class_params(fiducial: dict[str, float]) -> dict[str, float]:
    """Build CLASS parameter dictionary from Quijote-style fiducial cosmology."""
    omega_m = float(fiducial["Omega_m"])
    omega_b = float(fiducial["Omega_b"])
    return {
        "Omega_cdm": float(omega_m - omega_b),
        "Omega_b": float(omega_b),
        "h": float(fiducial["h"]),
        "n_s": float(fiducial["n_s"]),
        "sigma8": float(fiducial["sigma8"]),
    }


def build_class_pk_table_once(
    *,
    boxsize: float,
    target_res: int,
    class_n_modes: int,
    class_params: dict[str, float],
    z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a linear CLASS P(k) table once for mode-injection upsampling."""
    k_fund = 2.0 * np.pi / float(boxsize)
    k_nyq = np.pi * int(target_res) / float(boxsize)
    kmin = min(0.5 * k_fund, 1.0e-5)
    kmax = max(2.0 * k_nyq * np.sqrt(3.0), 10.0)
    k, pk = get_pk_class(
        cosmo_params=class_params,
        z=float(z),
        kmin=float(kmin),
        kmax=float(kmax),
        n_modes=int(class_n_modes),
        non_lin=False,
    )
    k = np.asarray(k, dtype=np.float64)
    pk = np.asarray(pk, dtype=np.float64)
    order = np.argsort(k)
    return np.asarray(k[order], dtype=np.float64), np.asarray(pk[order], dtype=np.float64)


def prepare_mode_injection(
    *,
    input_res: int,
    target_res: int,
    boxsize: float,
    k_tab: np.ndarray,
    pk_tab: np.ndarray,
) -> dict[str, Any]:
    """
    Prepare masks and amplitude arrays for random high-k mode injection.

    Adapted from /home/osavchenko/Pinocchio/my_scripts/upsample_field.py.
    """
    n_in = int(input_res)
    n_out = int(target_res)
    if n_out <= n_in:
        raise ValueError(f"target_res must be > input_res, got {n_out} <= {n_in}")
    if (n_out - n_in) % 2 != 0:
        raise ValueError(f"(target_res - input_res) must be even, got {n_out - n_in}")

    lo = (n_out - n_in) // 2
    hi = lo + n_in
    idx = np.arange(n_out) - n_out // 2
    nx, ny, nz = np.meshgrid(idx, idx, idx, indexing="ij")
    n2 = nx * nx + ny * ny + nz * nz

    low_mask = n2 < (n_in / 2.0) ** 2
    high_mask = (n2 >= (n_in / 2.0) ** 2) & (n2 < (n_out / 2.0) ** 2)

    k_fund = 2.0 * np.pi / float(boxsize)
    k_mag = k_fund * np.sqrt(n2.astype(np.float64))
    pk_vals = np.interp(
        k_mag.ravel(),
        np.asarray(k_tab, dtype=np.float64),
        np.asarray(pk_tab, dtype=np.float64),
        left=float(pk_tab[0]),
        right=float(pk_tab[-1]),
    ).reshape(k_mag.shape)
    pk_vals = np.maximum(pk_vals, 0.0)

    volume = float(boxsize) ** 3
    amplitude = (float(n_out) ** 1.5 / np.sqrt(volume)) * np.sqrt(pk_vals)

    return {
        "input_res": int(n_in),
        "target_res": int(n_out),
        "lo": int(lo),
        "hi": int(hi),
        "low_mask": np.asarray(low_mask, dtype=bool),
        "high_mask": np.asarray(high_mask, dtype=bool),
        "amplitude": np.asarray(amplitude, dtype=np.float64),
    }


def upsample_field_mode_inject(
    field: np.ndarray,
    *,
    prep: dict[str, Any],
    seed: int | None,
) -> np.ndarray:
    """
    Upsample a cubic field with preserved low-k modes and random high-k mode injection.

    Adapted core logic from /home/osavchenko/Pinocchio/my_scripts/upsample_field.py.
    """
    delta_in = np.asarray(field, dtype=np.float32)
    if delta_in.ndim != 3 or not (delta_in.shape[0] == delta_in.shape[1] == delta_in.shape[2]):
        raise ValueError(f"Input field must be cubic 3D, got shape={delta_in.shape}")

    n_in = int(prep["input_res"])
    n_out = int(prep["target_res"])
    if int(delta_in.shape[0]) != n_in:
        raise ValueError(f"Input field size mismatch: prep expects {n_in}, field has {delta_in.shape[0]}")

    lo = int(prep["lo"])
    hi = int(prep["hi"])
    low_mask = np.asarray(prep["low_mask"], dtype=bool)
    high_mask = np.asarray(prep["high_mask"], dtype=bool)
    amplitude = np.asarray(prep["amplitude"], dtype=np.float64)

    f_in_shift = np.fft.fftshift(np.fft.fftn(delta_in))
    f_out_shift = np.zeros((n_out, n_out, n_out), dtype=np.complex128)
    f_out_shift[lo:hi, lo:hi, lo:hi] = f_in_shift * (float(n_out) / float(n_in)) ** 3

    # Keep only low-k spherical modes from the embedded coarse cube.
    f_out_shift *= low_mask

    rng = np.random.default_rng(seed)
    white_noise = rng.standard_normal((n_out, n_out, n_out))
    wn_shift = np.fft.fftshift(np.fft.fftn(white_noise))

    f_total = f_out_shift + (wn_shift * amplitude * high_mask)
    delta_out = np.fft.ifftn(np.fft.ifftshift(f_total)).real
    return np.asarray(delta_out, dtype=np.float32)


def compute_mass_function_table(
    *,
    masses: np.ndarray,
    boxsize: float,
    nbins: int,
    mmin: float | None,
    mmax: float | None,
    empty_fallback_mass: float,
) -> np.ndarray:
    """Compute [M_center, dn/dM] table with logarithmic mass bins."""
    masses_arr = np.asarray(masses, dtype=np.float64)
    masses_arr = masses_arr[np.isfinite(masses_arr) & (masses_arr > 0.0)]

    if masses_arr.size == 0:
        mmin_val = float(mmin) if mmin is not None else float(max(empty_fallback_mass, 1.0))
        mmax_val = float(mmax) if mmax is not None else float(max(10.0 * mmin_val, mmin_val * 1.01))
    else:
        mmin_val = float(np.min(masses_arr)) if mmin is None else float(mmin)
        mmax_val = float(np.max(masses_arr)) if mmax is None else float(mmax)

    if not np.isfinite(mmin_val) or mmin_val <= 0.0:
        raise ValueError(f"Invalid HMF lower mass bound: {mmin_val}")
    if not np.isfinite(mmax_val) or mmax_val <= mmin_val:
        mmax_val = mmin_val * 1.01

    edges = np.logspace(np.log10(mmin_val), np.log10(mmax_val), int(nbins) + 1)
    counts, _ = np.histogram(masses_arr, bins=edges)
    dM = edges[1:] - edges[:-1]
    volume = float(boxsize) ** 3
    dndM = counts.astype(np.float64) / (dM * volume)
    m_center = np.sqrt(edges[:-1] * edges[1:])
    return np.column_stack([m_center, dndM]).astype(np.float64)


def true_catalog_metadata(path: Path) -> dict[str, Any]:
    """Load metadata-like fields from a true FoF catalog if present."""
    if not path.is_file():
        return {}

    out: dict[str, Any] = {}
    with np.load(path) as nz:
        for key in ["BoxSize", "NpartPerDim", "LinkingLength", "AbsoluteLinking", "Nmin"]:
            if key in nz:
                val = nz[key]
                arr = np.asarray(val)
                if arr.ndim == 0:
                    out[key] = arr.item()
                else:
                    out[key] = arr.tolist()
    return out


def resolve_layout(
    *,
    ppc_sim_dir: Path,
    dataset_root_override: Path | None,
    sim_id_override: int | None,
    true_sim_dir_override: Path | None,
) -> dict[str, Any]:
    """Infer tag/sim_id/dataset_root/true_sim_dir from PPC folder layout and overrides."""
    ppc_sim_dir = ppc_sim_dir.resolve()
    if not ppc_sim_dir.is_dir():
        raise FileNotFoundError(f"PPC sim directory not found: {ppc_sim_dir}")

    sim_id_str = str(ppc_sim_dir.name) if sim_id_override is None else str(int(sim_id_override))
    try:
        sim_id_int = int(sim_id_str)
    except ValueError as exc:
        raise ValueError(
            f"Could not infer integer sim_id from {ppc_sim_dir.name!r}; pass --sim-id explicitly."
        ) from exc

    tag = str(ppc_sim_dir.parent.name)

    if dataset_root_override is not None:
        dataset_root = Path(dataset_root_override).resolve()
    else:
        parent_ppc = ppc_sim_dir.parent.parent
        if parent_ppc.name != "ppc":
            raise ValueError(
                "Expected --ppc-sim-dir layout .../ppc/<tag>/<sim_id>. "
                f"Got {ppc_sim_dir} (its parent is {parent_ppc})."
            )
        dataset_root = parent_ppc.parent.resolve()

    if true_sim_dir_override is not None:
        true_sim_dir = Path(true_sim_dir_override).resolve()
    else:
        true_sim_dir = (dataset_root / str(sim_id_int)).resolve()

    return {
        "ppc_sim_dir": ppc_sim_dir,
        "tag": str(tag),
        "sim_id": int(sim_id_int),
        "dataset_root": dataset_root,
        "true_sim_dir": true_sim_dir,
    }


def resolve_runtime_params(args: argparse.Namespace, true_meta: dict[str, Any]) -> dict[str, Any]:
    """Resolve stage parameters, inheriting FoF defaults from true catalog when available."""
    n_part_meta = true_meta.get("NpartPerDim")
    box_meta = true_meta.get("BoxSize")
    link_meta = true_meta.get("LinkingLength")
    abs_link_meta = true_meta.get("AbsoluteLinking")
    nmin_meta = true_meta.get("Nmin")

    if isinstance(box_meta, list):
        box_meta_val = float(np.asarray(box_meta, dtype=np.float64).ravel()[0])
    elif box_meta is None:
        box_meta_val = None
    else:
        box_meta_val = float(box_meta)

    n_part = int(args.n_part) if args.n_part is not None else int(n_part_meta) if n_part_meta is not None else 128
    boxsize = float(args.boxsize) if args.boxsize is not None else box_meta_val if box_meta_val is not None else 250.0

    linking_length = (
        float(args.linking_length)
        if args.linking_length is not None
        else float(link_meta)
        if link_meta is not None
        else 0.2
    )
    absolute_linking = (
        bool(args.absolute_linking)
        if args.absolute_linking is not None
        else bool(abs_link_meta)
        if abs_link_meta is not None
        else False
    )
    nmin = int(args.nmin) if args.nmin is not None else int(nmin_meta) if nmin_meta is not None else 20

    omega_m = float(args.omega_m) if args.omega_m is not None else float(QUIJOTE_FIDUCIAL["Omega_m"])

    return {
        "n_part": int(n_part),
        "res": int(args.res),
        "boxsize": float(boxsize),
        "ndiv": tuple(int(x) for x in args.ndiv),
        "emu_precision": str(args.emu_precision),
        "mas_worder": int(args.mas_worder),
        "deconvolve_mas": bool(args.deconvolve_mas),
        "class_n_modes": int(args.class_n_modes),
        "linking_length": float(linking_length),
        "absolute_linking": bool(absolute_linking),
        "nmin": int(nmin),
        "omega_m": float(omega_m),
    }


def jsonable(value: Any) -> Any:
    """Convert common objects to JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Persist run manifest JSON."""
    path.write_text(json.dumps(manifest, indent=2))


def ensure_emulator_dependencies() -> None:
    """Ensure emulator/JAX pipeline is importable."""
    if _CORE_IMPORT_ERROR is not None or run_lpt_emulator_pipeline is None:
        raise ImportError(
            "Forward stage requires core emulator dependencies (JAX, DISCO-DJ, jax_nbody_emulator)."
        ) from _CORE_IMPORT_ERROR


def ensure_fof_dependencies() -> None:
    """Ensure MPI + nbodykit FoF dependencies are available."""
    if _MPI_IMPORT_ERROR is not None or MPI is None:
        raise ImportError("Halos stage requires mpi4py.") from _MPI_IMPORT_ERROR
    if _NBODYKIT_IMPORT_ERROR is not None or FOF is None or fof_catalog is None or ArrayCatalog is None:
        raise ImportError("Halos stage requires nbodykit.") from _NBODYKIT_IMPORT_ERROR


def ensure_hod_dependencies() -> None:
    """Ensure pyhod dependency is available for HOD stage."""
    if _PYHOD_IMPORT_ERROR is not None or HODF is None:
        raise ImportError("HOD stage requires pyhod (pyhod.hodfunc).") from _PYHOD_IMPORT_ERROR


def parse_hod_extra_args(extra_args: str) -> dict[str, Any]:
    """Parse free-form HOD passthrough args provided as key=value tokens."""
    text = str(extra_args).strip()
    if not text:
        return {}

    normalized = text.replace(",", " ")
    tokens = [tok for tok in shlex.split(normalized) if tok]
    out: dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"Invalid --hod-extra-args token {tok!r}; expected key=value.")
        key, raw = tok.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if not key:
            raise ValueError(f"Invalid --hod-extra-args token {tok!r}; key cannot be empty.")
        low = raw.lower()
        if low in {"true", "false"}:
            val: Any = (low == "true")
        else:
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw)
                except ValueError:
                    val = raw
        out[key] = val
    return out


def _hod_seed_for_sample(sample_id: int, args: argparse.Namespace) -> int | None:
    """Resolve deterministic/random HOD seed for one sample id."""
    policy = str(args.hod_seed_policy)
    if policy == "sample_id":
        return int(sample_id)
    if policy == "fixed":
        return int(args.hod_seed)
    if policy == "none":
        return None
    raise ValueError(f"Unknown hod_seed_policy={policy!r}")


def _fof_catalog_to_hod_input(catalog_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-sample FoF catalog.npz into pyhod input array and halo masses."""
    with np.load(catalog_path) as catalog:
        masses = np.asarray(catalog["Mass"], dtype=np.float64).reshape(-1)
        pos = np.asarray(catalog["CMPosition"], dtype=np.float64).reshape(-1, 3)
        npart = np.asarray(catalog["Npart"], dtype=np.float64).reshape(-1)
    if masses.size != pos.shape[0] or masses.size != npart.size:
        raise ValueError(
            f"Inconsistent FoF catalog arrays in {catalog_path}: "
            f"Mass={masses.shape}, CMPosition={pos.shape}, Npart={npart.shape}"
        )
    vel = np.zeros_like(pos, dtype=np.float64)
    halos = np.column_stack([masses, pos, vel, npart]).astype(np.float64, copy=False)
    return halos, masses


def run_hod_stage_for_sample(
    *,
    sample_id: int,
    ppc_sim_dir: Path,
    args: argparse.Namespace,
    runtime: dict[str, Any],
    comm: Any,
) -> dict[str, Any]:
    """Run HOD galaxy population for one sample and save hod_catalog_{id:04d}.npz."""
    rank = int(comm.rank)
    paths = sim_file_paths(ppc_sim_dir=ppc_sim_dir, sample_id=int(sample_id))
    rec: dict[str, Any] = {
        "sample_id": int(sample_id),
        "fof_catalog_path": str(paths["fof_catalog"]),
        "hod_catalog_path": str(paths["hod_catalog"]),
        "status": "completed",
        "reused": False,
        "n_halos_input": None,
        "n_galaxies": None,
        "hod_seed": None,
        "runtime_seconds": 0.0,
    }

    t0 = time.perf_counter()
    try:
        if bool(args.reuse_existing) and paths["hod_catalog"].is_file():
            rec["status"] = "skipped_reuse"
            rec["reused"] = True
            comm.Barrier()
        else:
            has_catalog = bool(paths["fof_catalog"].is_file()) if rank == 0 else None
            has_catalog = bool(comm.bcast(has_catalog, root=0))
            if not has_catalog:
                raise FileNotFoundError(
                    f"Missing FoF catalog for HOD stage: {paths['fof_catalog']}. "
                    "Run stage=halos first (or stage=all)."
                )

            root_error: str | None = None
            root_stats: dict[str, Any] | None = None
            if rank == 0:
                try:
                    halos, halo_masses = _fof_catalog_to_hod_input(paths["fof_catalog"])
                    extra = parse_hod_extra_args(str(args.hod_extra_args))

                    logmmin = float(extra.get("logMmin", args.hod_logMmin))
                    sigmalogm = float(extra.get("sigmalogM", args.hod_sigmalogM))
                    logm0 = float(extra.get("logM0", args.hod_logM0))
                    alpha = float(extra.get("alpha", args.hod_alpha))
                    logm1 = float(extra.get("logM1", args.hod_logM1))

                    hod_seed = _hod_seed_for_sample(sample_id=int(sample_id), args=args)
                    if "seed" in extra:
                        hod_seed = int(extra["seed"])

                    rsd = int(extra.get("rsd", args.hod_rsd))
                    if "iRSD" in extra:
                        rsd = int(extra["iRSD"])

                    params: dict[str, Any] = {
                        "om": float(extra.get("om", args.hod_omega_m)),
                        "ol": float(extra.get("ol", args.hod_omega_l)),
                        "h": float(extra.get("h", args.hod_hubble)),
                        "z": float(extra.get("z", args.hod_redshift)),
                        "box": float(extra.get("box", runtime["boxsize"])),
                        "Deltac": int(extra.get("Deltac", args.hod_deltac)),
                        "M_star": float(extra.get("M_star", args.hod_m_star)),
                        "model": str(extra.get("model", "Zheng")),
                        "method": str(extra.get("method", args.hod_method)),
                        "iRSD": int(rsd),
                        "zerr": float(extra.get("zerr", args.hod_zerr)),
                    }
                    if hod_seed is not None:
                        params["seed"] = int(hod_seed)

                    n_cen, n_sat = HODF.cent_sat(
                        halos,
                        float(logmmin),
                        float(sigmalogm),
                        float(logm0),
                        float(alpha),
                        float(logm1),
                        params=params,
                    )
                    gal_pos, gal_vel, gal_type, halo_idx = HODF.sample_catalog(halos, n_cen, n_sat, params=params)
                    if int(params["iRSD"]) > 0:
                        if HOD_UTILS is None:
                            raise ImportError("HOD RSD requested but pyhod.utils is unavailable.")
                        gal_pos = HOD_UTILS.displaceRSD(gal_pos, gal_vel, params)

                    host_idx = np.asarray(halo_idx, dtype=np.int64).reshape(-1)
                    valid = (host_idx >= 0) & (host_idx < halo_masses.size)
                    host_mass = np.full(host_idx.shape, np.nan, dtype=np.float64)
                    host_mass[valid] = halo_masses[host_idx[valid]]
                    is_central = np.asarray(gal_type, dtype=np.int32).reshape(-1) == 1

                    np.savez_compressed(
                        paths["hod_catalog"],
                        Position=np.asarray(gal_pos, dtype=np.float32),
                        Velocity=np.asarray(gal_vel, dtype=np.float32),
                        IsCentral=np.asarray(is_central, dtype=np.bool_),
                        HostHaloIndex=np.asarray(host_idx, dtype=np.int32),
                        HostMass=np.asarray(host_mass, dtype=np.float64),
                        BoxSize=np.array(
                            [float(runtime["boxsize"]), float(runtime["boxsize"]), float(runtime["boxsize"])],
                            dtype=np.float64,
                        ),
                        HODModel=np.array(str(params["model"])),
                        HODMethod=np.array(str(params["method"])),
                        HODSeed=np.int64(-1 if hod_seed is None else int(hod_seed)),
                        HODSeedPolicy=np.array(str(args.hod_seed_policy)),
                        HODRSD=np.int32(int(params["iRSD"])),
                        logMmin=np.float64(logmmin),
                        sigmalogM=np.float64(sigmalogm),
                        logM0=np.float64(logm0),
                        alpha=np.float64(alpha),
                        logM1=np.float64(logm1),
                        omega_m=np.float64(float(params["om"])),
                        omega_l=np.float64(float(params["ol"])),
                        hubble=np.float64(float(params["h"])),
                        redshift=np.float64(float(params["z"])),
                        M_star=np.float64(float(params["M_star"])),
                        Deltac=np.int32(int(params["Deltac"])),
                        zerr=np.float64(float(params["zerr"])),
                        n_halos_input=np.int32(int(halos.shape[0])),
                        n_galaxies=np.int32(int(np.asarray(gal_pos).shape[0])),
                        HODExtraArgs=np.array(str(args.hod_extra_args)),
                    )
                    root_stats = {
                        "n_halos_input": int(halos.shape[0]),
                        "n_galaxies": int(np.asarray(gal_pos).shape[0]),
                        "hod_seed": None if hod_seed is None else int(hod_seed),
                    }
                except Exception as exc:
                    root_error = repr(exc)

            root_error = comm.bcast(root_error, root=0)
            root_stats = comm.bcast(root_stats, root=0)
            if root_error is not None:
                raise RuntimeError(f"HOD stage failed on root: {root_error}")

            if root_stats is not None:
                rec["n_halos_input"] = int(root_stats["n_halos_input"])
                rec["n_galaxies"] = int(root_stats["n_galaxies"])
                rec["hod_seed"] = root_stats["hod_seed"]

            comm.Barrier()
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = repr(exc)

    elapsed_local = float(time.perf_counter() - t0)
    if MPI is not None:
        rec["runtime_seconds"] = float(comm.allreduce(elapsed_local, op=MPI.MAX))
    else:
        rec["runtime_seconds"] = float(elapsed_local)
    rec["runtime_hms"] = format_hms(float(rec["runtime_seconds"]))
    return rec


def materialize_true_mass_function_if_needed(
    *,
    true_sim_dir: Path,
    runtime: dict[str, Any],
    hmf_nbins: int,
    hmf_mmin: float | None,
    hmf_mmax: float | None,
    do_materialize: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Create <true_sim_dir>/mass_function.npy from true fof_catalog.npz when needed."""
    out = {
        "status": "not_requested",
        "path": str(true_sim_dir / "mass_function.npy"),
    }

    if not do_materialize:
        out["status"] = "disabled"
        return out

    true_mf_path = true_sim_dir / "mass_function.npy"
    true_cat_path = true_sim_dir / "fof_catalog.npz"
    had_existing_mf = bool(true_mf_path.is_file())

    if had_existing_mf and not overwrite:
        out["status"] = "exists"
        return out

    if not true_cat_path.is_file():
        out["status"] = "missing_true_catalog"
        return out

    with np.load(true_cat_path) as nz:
        if "Mass" in nz:
            masses = np.asarray(nz["Mass"], dtype=np.float64)
        elif "Npart" in nz:
            n_h = np.asarray(nz["Npart"], dtype=np.float64)
            m_particle = particle_mass_msun_h(
                omega_m=float(runtime["omega_m"]),
                boxsize=float(runtime["boxsize"]),
                n_part=int(runtime["n_part"]),
            )
            masses = n_h * float(m_particle)
        else:
            raise ValueError(f"True catalog has neither 'Mass' nor 'Npart': {true_cat_path}")

    m_particle = particle_mass_msun_h(
        omega_m=float(runtime["omega_m"]),
        boxsize=float(runtime["boxsize"]),
        n_part=int(runtime["n_part"]),
    )
    mf = compute_mass_function_table(
        masses=np.asarray(masses, dtype=np.float64),
        boxsize=float(runtime["boxsize"]),
        nbins=int(hmf_nbins),
        mmin=hmf_mmin,
        mmax=hmf_mmax,
        empty_fallback_mass=float(m_particle) * max(1, int(runtime["nmin"])),
    )
    np.save(true_mf_path, np.asarray(mf, dtype=np.float64))

    out["status"] = "overwritten" if had_existing_mf else "written"
    out["path"] = str(true_mf_path)
    out["n_bins"] = int(mf.shape[0])
    return out


def run_g2f_ppc(
    *,
    g2f_ppc_script: Path,
    run_dir: Path,
    sim_id: int,
    ppc_root: Path,
    ppc_n_workers: int | None,
) -> dict[str, Any]:
    """Run g2f_emu/ppc.py and return invocation metadata."""
    cmd = [
        sys.executable,
        str(g2f_ppc_script),
        "--run_dir",
        str(run_dir),
        "--sim_id",
        str(int(sim_id)),
        "--ppc_root",
        str(ppc_root),
    ]
    if ppc_n_workers is not None:
        cmd.extend(["--n_workers", str(int(ppc_n_workers))])

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, check=False)
    elapsed = float(time.perf_counter() - t0)
    if int(proc.returncode) != 0:
        raise RuntimeError(f"g2f ppc.py failed with return code {proc.returncode}: {' '.join(cmd)}")

    return {
        "status": "completed",
        "command": cmd,
        "returncode": int(proc.returncode),
        "runtime_seconds": float(elapsed),
        "runtime_hms": format_hms(elapsed),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Resimulate emulator posterior IC samples in staged mode (forward/halos/hod/all), "
            "write PPC-compatible outputs, and optionally auto-run g2f_emu/ppc.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--stage", choices=("forward", "halos", "hod", "all"), default="all", help="Execution stage.")
    parser.add_argument(
        "--ppc-sim-dir",
        type=Path,
        required=True,
        help="PPC sim folder containing sample_*.npy and z_MAP.npy (layout .../ppc/<tag>/<sim_id>/).",
    )
    parser.add_argument("--dataset-root", type=Path, default=None, help="Override inferred dataset root.")
    parser.add_argument("--sim-id", type=int, default=None, help="Override sim id inferred from folder name.")
    parser.add_argument("--true-sim-dir", type=Path, default=None, help="Override true simulation directory path.")

    parser.add_argument("--sample-pattern", type=str, default="sample_*.npy", help="Pattern for IC posterior sample files.")
    parser.add_argument("--num-samples", type=int, default=None, help="Optional cap on number of samples to process.")
    parser.add_argument("--sample-start", type=int, default=0, help="Start index in the sorted sample list.")
    parser.add_argument("--sample-stride", type=int, default=1, help="Stride in the sorted sample list.")

    parser.add_argument(
        "--n-part",
        type=int,
        default=None,
        help="Particle/displacement lattice size. If omitted, uses true-catalog NpartPerDim or 128.",
    )
    parser.add_argument("--res", type=int, default=64, help="Density summary grid size.")
    parser.add_argument(
        "--boxsize",
        type=float,
        default=None,
        help="Box size [Mpc/h]. If omitted, uses true-catalog BoxSize or 250.",
    )
    parser.add_argument("--ndiv", type=parse_ndiv, default=(1, 1, 1), help="Emulator subbox partition.")
    parser.add_argument("--emu-precision", choices=("f16", "f32"), default="f16", help="Emulator precision.")
    parser.add_argument("--mas-worder", type=int, choices=(2, 3, 4), default=2, help="MAS order.")
    parser.add_argument(
        "--deconvolve-mas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deconvolve MAS during density meshing in forward stage.",
    )
    parser.add_argument("--class-n-modes", type=int, default=4000, help="CLASS sampling count for linear P(k).")
    parser.add_argument(
        "--mode-inject-seed-policy",
        choices=("sample_id", "fixed", "none"),
        default="sample_id",
        help="Seed policy for random high-k mode injection during upsampling.",
    )
    parser.add_argument("--mode-inject-fixed-seed", type=int, default=0, help="Seed used when policy=fixed.")
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing per-sample outputs for selected stage.",
    )

    parser.add_argument(
        "--linking-length",
        type=float,
        default=None,
        help="FoF linking length b. If omitted, inherit true catalog metadata or 0.2.",
    )
    parser.add_argument(
        "--absolute-linking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Treat FoF linking length as absolute Mpc/h. If omitted, inherit true metadata or False.",
    )
    parser.add_argument(
        "--nmin",
        type=int,
        default=None,
        help="FoF minimum particles. If omitted, inherit true metadata or 20.",
    )
    parser.add_argument(
        "--omega-m",
        type=float,
        default=None,
        help="Omega_m for particle mass conversion. Defaults to Quijote fiducial.",
    )
    parser.add_argument(
        "--cleanup-temp-displacement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete emu_dis_{id}.npy after successful halo build.",
    )

    parser.add_argument("--hod-logMmin", type=float, default=HOD_DEFAULT_LOGMMIN, help="HOD logMmin.")
    parser.add_argument("--hod-sigmalogM", type=float, default=HOD_DEFAULT_SIGMALOGM, help="HOD sigmalogM.")
    parser.add_argument("--hod-logM0", type=float, default=HOD_DEFAULT_LOGM0, help="HOD logM0.")
    parser.add_argument("--hod-alpha", type=float, default=HOD_DEFAULT_ALPHA, help="HOD alpha.")
    parser.add_argument("--hod-logM1", type=float, default=HOD_DEFAULT_LOGM1, help="HOD logM1.")
    parser.add_argument("--hod-method", choices=("NFW", "NFWv"), default="NFW", help="HOD position sampling method.")
    parser.add_argument("--hod-rsd", type=int, choices=(0, 1, 2, 3), default=0, help="RSD mode for HOD catalog.")
    parser.add_argument(
        "--hod-seed-policy",
        choices=("sample_id", "fixed", "none"),
        default="sample_id",
        help="Seed policy for stochastic HOD sampling.",
    )
    parser.add_argument("--hod-seed", type=int, default=42, help="Base seed used for fixed-seed policy.")
    parser.add_argument("--hod-omega-m", type=float, default=HOD_DEFAULT_OMEGA_M, help="HOD cosmology Omega_m.")
    parser.add_argument("--hod-omega-l", type=float, default=HOD_DEFAULT_OMEGA_L, help="HOD cosmology Omega_l.")
    parser.add_argument("--hod-hubble", type=float, default=HOD_DEFAULT_H, help="HOD cosmology h.")
    parser.add_argument("--hod-redshift", type=float, default=HOD_DEFAULT_Z, help="HOD redshift.")
    parser.add_argument("--hod-m-star", type=float, default=HOD_DEFAULT_M_STAR, help="HOD M_star [Msun/h].")
    parser.add_argument("--hod-deltac", type=int, default=HOD_DEFAULT_DELTAC, help="HOD Deltac.")
    parser.add_argument("--hod-zerr", type=float, default=HOD_DEFAULT_ZERR, help="HOD zerr nuisance parameter.")
    parser.add_argument(
        "--hod-extra-args",
        type=str,
        default="",
        help=(
            "Optional passthrough key=value pairs for advanced HOD params. "
            "Accepted separators: space and/or comma, e.g. 'zerr=0.01 foo=1'."
        ),
    )

    parser.add_argument("--hmf-nbins", type=int, default=25, help="Number of log-mass HMF bins.")
    parser.add_argument("--hmf-mmin", type=float, default=None, help="Lower HMF mass bound [Msun/h].")
    parser.add_argument("--hmf-mmax", type=float, default=None, help="Upper HMF mass bound [Msun/h].")

    parser.add_argument(
        "--materialize-true-mass-function-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create <true_sim_dir>/mass_function.npy from true fof_catalog.npz when missing.",
    )
    parser.add_argument(
        "--force-overwrite-true-mass-function",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow overwriting an existing true mass_function.npy.",
    )

    parser.add_argument(
        "--run-ppc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run g2f_emu/ppc.py automatically on rank 0 after halos/all stage.",
    )
    parser.add_argument(
        "--g2f-run-dir",
        type=Path,
        default=None,
        help="Override g2f run dir. Default: /home/osavchenko/GNN_FLI/.../gnpe_plots/<tag>",
    )
    parser.add_argument(
        "--g2f-ppc-script",
        type=Path,
        default=DEFAULT_G2F_PPC_SCRIPT,
        help="Path to g2f_emu/ppc.py.",
    )
    parser.add_argument(
        "--ppc-n-workers",
        type=int,
        default=None,
        help="Optional --n_workers passed to g2f_emu/ppc.py.",
    )

    parser.add_argument(
        "--manifest-name",
        type=str,
        default="ppc_resimulation_manifest.json",
        help="Manifest filename inside --ppc-sim-dir.",
    )

    args = parser.parse_args()

    if args.sample_start < 0:
        parser.error("--sample-start must be >= 0")
    if args.sample_stride < 1:
        parser.error("--sample-stride must be >= 1")
    if args.num_samples is not None and int(args.num_samples) < 1:
        parser.error("--num-samples must be >= 1 when provided")
    if int(args.res) < 1:
        parser.error("--res must be >= 1")
    if args.n_part is not None and int(args.n_part) < 1:
        parser.error("--n-part must be >= 1")
    if int(args.class_n_modes) < 2:
        parser.error("--class-n-modes must be >= 2")
    if args.nmin is not None and int(args.nmin) < 1:
        parser.error("--nmin must be >= 1")
    if int(args.hod_seed) < 0:
        parser.error("--hod-seed must be >= 0")
    if int(args.hmf_nbins) < 3:
        parser.error("--hmf-nbins must be >= 3")

    return args


def main() -> None:
    """Entry point."""
    args = parse_args()

    layout = resolve_layout(
        ppc_sim_dir=Path(args.ppc_sim_dir),
        dataset_root_override=args.dataset_root,
        sim_id_override=args.sim_id,
        true_sim_dir_override=args.true_sim_dir,
    )
    ppc_sim_dir = Path(layout["ppc_sim_dir"])
    tag = str(layout["tag"])
    sim_id = int(layout["sim_id"])
    dataset_root = Path(layout["dataset_root"])
    true_sim_dir = Path(layout["true_sim_dir"])

    true_cat_path = true_sim_dir / "fof_catalog.npz"
    true_meta = true_catalog_metadata(true_cat_path)
    runtime = resolve_runtime_params(args=args, true_meta=true_meta)

    if str(args.stage) in {"forward", "all"}:
        ensure_emulator_dependencies()
    if str(args.stage) in {"halos", "all"}:
        ensure_fof_dependencies()
    if str(args.stage) in {"hod", "all"}:
        ensure_hod_dependencies()

    if str(args.stage) == "halos":
        comm = MPI.COMM_WORLD
    elif str(args.stage) == "hod":
        if MPI is None:
            comm = _SerialComm()
        else:
            comm = MPI.COMM_WORLD
    else:
        if MPI is None:
            comm = _SerialComm()
        else:
            world_size = int(MPI.COMM_WORLD.size)
            if world_size != 1:
                raise RuntimeError(
                    f"stage={args.stage} must run with --ntasks=1, but MPI world size is {world_size}."
                )
            comm = MPI.COMM_SELF

    rank = int(comm.rank)
    size = int(comm.size)

    if rank == 0:
        log_progress(f"PPC sim dir: {ppc_sim_dir}", rank=rank)
        log_progress(f"Inferred tag={tag}, sim_id={sim_id}", rank=rank)
        log_progress(f"Dataset root: {dataset_root}", rank=rank)
        log_progress(f"True sim dir: {true_sim_dir}", rank=rank)
        if str(args.stage) in {"halos", "hod", "all"}:
            log_progress(
                "Resolved FoF defaults: "
                f"n_part={runtime['n_part']}, boxsize={runtime['boxsize']}, "
                f"linking_length={runtime['linking_length']}, absolute_linking={runtime['absolute_linking']}, "
                f"nmin={runtime['nmin']}",
                rank=rank,
            )
        if str(args.stage) in {"halos", "hod"}:
            log_progress(f"MPI world size: {size}", rank=rank)

    stage_name = str(args.stage)
    if stage_name == "halos":
        fallback_patterns = ["emu_dis_*.npy", "fof_catalog_*.npz"]
    elif stage_name == "hod":
        fallback_patterns = ["fof_catalog_*.npz"]
    else:
        fallback_patterns = []

    entries_local = None
    if rank == 0:
        entries_local = ppc_sample_entries(
            ppc_sim_dir=ppc_sim_dir,
            sample_pattern=str(args.sample_pattern),
            sample_start=int(args.sample_start),
            sample_stride=int(args.sample_stride),
            num_samples=None if args.num_samples is None else int(args.num_samples),
            require_input_samples=str(args.stage) in {"forward", "all"},
            fallback_patterns=fallback_patterns,
        )
        log_progress(f"Selected {len(entries_local)} sample entries.", rank=rank)

    entries = comm.bcast(entries_local, root=0)

    manifest_path = ppc_sim_dir / str(args.manifest_name)
    manifest: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": str(args.stage),
        "layout": {
            "ppc_sim_dir": str(ppc_sim_dir),
            "tag": str(tag),
            "sim_id": int(sim_id),
            "dataset_root": str(dataset_root),
            "true_sim_dir": str(true_sim_dir),
        },
        "config": jsonable(vars(args)),
        "resolved_runtime": jsonable(runtime),
        "selected_sample_ids": [int(e["sample_id"]) for e in entries],
        "forward_records": [],
        "halo_records": [],
        "hod_records": [],
        "true_mass_function": {},
        "ppc_run": {"status": "not_requested"},
    }

    t_total = time.perf_counter()

    # Forward stage
    if str(args.stage) in {"forward", "all"}:
        class_params = _quijote_class_params(QUIJOTE_FIDUCIAL)
        pk_table: tuple[np.ndarray, np.ndarray] | None = None
        prep_cache: dict[tuple[int, int], dict[str, Any]] = {}

        n_entries = len(entries)
        for i, entry in enumerate(entries, start=1):
            sample_id = int(entry["sample_id"])
            sample_file_str = entry.get("sample_file")
            if sample_file_str is None:
                raise RuntimeError(
                    f"Forward stage requires sample files, but sample_id={sample_id} has no sample_file entry."
                )

            sample_path = Path(sample_file_str)
            out_paths = sim_file_paths(ppc_sim_dir=ppc_sim_dir, sample_id=sample_id)
            rec: dict[str, Any] = {
                "sample_id": int(sample_id),
                "sample_file": str(sample_path),
                "emu_delta_path": str(out_paths["emu_delta"]),
                "emu_dis_path": str(out_paths["emu_dis"]),
                "status": "completed",
                "reused": False,
                "upsampled": False,
                "upsample_seed": None,
                "input_shape": None,
                "upsampled_shape": None,
                "runtime_seconds": 0.0,
            }

            if rank == 0:
                log_progress(f"[forward {i}/{n_entries}] sample_id={sample_id} from {sample_path.name}", rank=rank)

            t0 = time.perf_counter()
            try:
                all_complete_for_reuse = (
                    bool(args.reuse_existing)
                    and str(args.stage) == "all"
                    and out_paths["emu_delta"].is_file()
                    and out_paths["fof_catalog"].is_file()
                    and out_paths["mass_function"].is_file()
                    and out_paths["hod_catalog"].is_file()
                )
                if all_complete_for_reuse:
                    rec["status"] = "skipped_reuse_all_complete"
                    rec["reused"] = True
                elif bool(args.reuse_existing) and out_paths["emu_delta"].is_file() and out_paths["emu_dis"].is_file():
                    rec["status"] = "skipped_reuse"
                    rec["reused"] = True
                else:
                    ic = np.asarray(np.load(sample_path), dtype=np.float32)
                    if ic.ndim != 3 or not (ic.shape[0] == ic.shape[1] == ic.shape[2]):
                        raise ValueError(f"IC sample must be cubic 3D, got {sample_path} shape={ic.shape}")

                    in_res = int(ic.shape[0])
                    out_res = int(runtime["n_part"])
                    rec["input_shape"] = [int(x) for x in ic.shape]

                    if in_res == out_res:
                        ic_npart = np.asarray(ic, dtype=np.float32)
                    else:
                        if out_res < in_res:
                            raise ValueError(
                                f"Input IC resolution {in_res} exceeds n_part={out_res}; only upsampling is supported here."
                            )
                        if out_res % in_res != 0:
                            raise ValueError(
                                f"Upsampling requires integer ratio, got in_res={in_res}, n_part={out_res}."
                            )

                        if str(args.mode_inject_seed_policy) == "sample_id":
                            up_seed = int(sample_id)
                        elif str(args.mode_inject_seed_policy) == "fixed":
                            up_seed = int(args.mode_inject_fixed_seed)
                        else:
                            up_seed = None
                        rec["upsample_seed"] = None if up_seed is None else int(up_seed)
                        rec["upsampled"] = True

                        if pk_table is None:
                            if rank == 0:
                                log_progress(
                                    "Building CLASS linear P(k) table once for mode-injection upsampling...",
                                    rank=rank,
                                )
                            pk_table = build_class_pk_table_once(
                                boxsize=float(runtime["boxsize"]),
                                target_res=out_res,
                                class_n_modes=int(runtime["class_n_modes"]),
                                class_params=class_params,
                                z=0.0,
                            )

                        prep_key = (in_res, out_res)
                        if prep_key not in prep_cache:
                            k_tab, pk_tab = pk_table
                            prep_cache[prep_key] = prepare_mode_injection(
                                input_res=int(in_res),
                                target_res=int(out_res),
                                boxsize=float(runtime["boxsize"]),
                                k_tab=np.asarray(k_tab, dtype=np.float64),
                                pk_tab=np.asarray(pk_tab, dtype=np.float64),
                            )

                        ic_npart = upsample_field_mode_inject(
                            field=np.asarray(ic, dtype=np.float32),
                            prep=prep_cache[prep_key],
                            seed=None if up_seed is None else int(up_seed),
                        )

                    rec["upsampled_shape"] = [int(x) for x in ic_npart.shape]

                    pipeline = run_lpt_emulator_pipeline(
                        delta_linear=np.asarray(ic_npart, dtype=np.float32),
                        n_part=int(runtime["n_part"]),
                        density_res=int(runtime["res"]),
                        boxsize=float(runtime["boxsize"]),
                        ndiv=tuple(runtime["ndiv"]),
                        emu_precision=str(runtime["emu_precision"]),
                        num_sims=1,
                        mas_worder=int(runtime["mas_worder"]),
                        deconvolve_mas=bool(runtime["deconvolve_mas"]),
                        z_target=0.0,
                        compute_vel=False,
                        delta_upsample_method="mode_inject",
                        downsample_method="fourier_truncation",
                        downsample_gaussian_sigma_mpc_h=None,
                        downsample_threads=1,
                        class_n_modes=int(runtime["class_n_modes"]),
                        mode_inject_seed=None,
                        quijote_fiducial=QUIJOTE_FIDUCIAL,
                        output_dir=None,
                        save_outputs=False,
                        show_progress=True,
                        progress_desc_prefix=f"PPC sample {sample_id} subboxes",
                    )

                    np.save(out_paths["emu_delta"], np.asarray(pipeline["delta_emu"], dtype=np.float32))
                    np.save(out_paths["emu_dis"], np.asarray(pipeline["psi_emu"], dtype=np.float32))

            except Exception as exc:
                rec["status"] = "failed"
                rec["error"] = repr(exc)

            rec["runtime_seconds"] = float(time.perf_counter() - t0)
            rec["runtime_hms"] = format_hms(float(rec["runtime_seconds"]))

            if rank == 0:
                manifest["forward_records"].append(rec)
                write_manifest(manifest_path, manifest)
                if rec["status"] == "failed":
                    log_progress(
                        f"  forward FAILED for sample_id={sample_id}: {rec.get('error', 'unknown error')}",
                        rank=rank,
                    )
                elif str(rec["status"]).startswith("skipped"):
                    log_progress(f"  forward reused existing outputs for sample_id={sample_id}", rank=rank)
                else:
                    log_progress(
                        f"  forward completed in {rec['runtime_hms']} for sample_id={sample_id}",
                        rank=rank,
                    )

    # Halos stage
    if str(args.stage) in {"halos", "all"}:
        n_entries = len(entries)
        for i, entry in enumerate(entries, start=1):
            sample_id = int(entry["sample_id"])
            paths = sim_file_paths(ppc_sim_dir=ppc_sim_dir, sample_id=sample_id)

            rec: dict[str, Any] = {
                "sample_id": int(sample_id),
                "emu_dis_path": str(paths["emu_dis"]),
                "fof_catalog_path": str(paths["fof_catalog"]),
                "mass_function_path": str(paths["mass_function"]),
                "status": "completed",
                "reused": False,
                "n_halos": None,
                "runtime_seconds": 0.0,
            }

            if rank == 0:
                log_progress(f"[halos {i}/{n_entries}] sample_id={sample_id}", rank=rank)

            t0 = time.perf_counter()
            try:
                if bool(args.reuse_existing) and paths["fof_catalog"].is_file() and paths["mass_function"].is_file():
                    rec["status"] = "skipped_reuse"
                    rec["reused"] = True
                    if rank == 0 and bool(args.cleanup_temp_displacement) and paths["emu_dis"].is_file():
                        paths["emu_dis"].unlink()
                    comm.Barrier()
                else:
                    has_dis = bool(paths["emu_dis"].is_file()) if rank == 0 else None
                    has_dis = bool(comm.bcast(has_dis, root=0))
                    if not has_dis:
                        raise FileNotFoundError(
                            f"Missing displacement for halos stage: {paths['emu_dis']}. Run forward stage first."
                        )

                    local_pos = load_local_positions_from_displacement(
                        displacement_file=paths["emu_dis"],
                        boxsize=float(runtime["boxsize"]),
                        n_part=int(runtime["n_part"]),
                        rank=rank,
                        size=size,
                    )
                    cm_local, length_local = run_fof(
                        local_positions=np.asarray(local_pos, dtype=np.float32),
                        boxsize=float(runtime["boxsize"]),
                        n_part=int(runtime["n_part"]),
                        linking_length=float(runtime["linking_length"]),
                        nmin=int(runtime["nmin"]),
                        absolute_linking=bool(runtime["absolute_linking"]),
                        comm=comm,
                        fof_cls=FOF,
                        fof_catalog_fn=fof_catalog,
                        array_catalog_cls=ArrayCatalog,
                    )
                    cm_pos, length = gather_halos_to_root(
                        cm_pos_local=np.asarray(cm_local, dtype=np.float32),
                        length_local=np.asarray(length_local, dtype=np.int32),
                        comm=comm,
                    )

                    if rank == 0:
                        m_particle = particle_mass_msun_h(
                            omega_m=float(runtime["omega_m"]),
                            boxsize=float(runtime["boxsize"]),
                            n_part=int(runtime["n_part"]),
                        )
                        masses = np.asarray(length, dtype=np.float64) * float(m_particle)
                        np.savez_compressed(
                            paths["fof_catalog"],
                            CMPosition=np.asarray(cm_pos, dtype=np.float32),
                            Npart=np.asarray(length, dtype=np.int32),
                            Mass=np.asarray(masses, dtype=np.float64),
                            BoxSize=np.array(
                                [float(runtime["boxsize"]), float(runtime["boxsize"]), float(runtime["boxsize"])],
                                dtype=np.float64,
                            ),
                            NpartPerDim=np.int32(int(runtime["n_part"])),
                            LinkingLength=np.float64(float(runtime["linking_length"])),
                            AbsoluteLinking=np.bool_(bool(runtime["absolute_linking"])),
                            Nmin=np.int32(int(runtime["nmin"])),
                        )

                        mf = compute_mass_function_table(
                            masses=np.asarray(masses, dtype=np.float64),
                            boxsize=float(runtime["boxsize"]),
                            nbins=int(args.hmf_nbins),
                            mmin=args.hmf_mmin,
                            mmax=args.hmf_mmax,
                            empty_fallback_mass=float(m_particle) * max(1, int(runtime["nmin"])),
                        )
                        np.save(paths["mass_function"], np.asarray(mf, dtype=np.float64))

                        rec["n_halos"] = int(np.asarray(length).size)

                    comm.Barrier()
                    if rank == 0 and bool(args.cleanup_temp_displacement) and paths["emu_dis"].is_file():
                        paths["emu_dis"].unlink()
                    comm.Barrier()

            except Exception as exc:
                rec["status"] = "failed"
                rec["error"] = repr(exc)

            elapsed_local = float(time.perf_counter() - t0)
            if MPI is not None:
                rec["runtime_seconds"] = float(comm.allreduce(elapsed_local, op=MPI.MAX))
            else:
                rec["runtime_seconds"] = float(elapsed_local)
            rec["runtime_hms"] = format_hms(float(rec["runtime_seconds"]))

            if rank == 0:
                manifest["halo_records"].append(rec)
                write_manifest(manifest_path, manifest)
                if rec["status"] == "failed":
                    log_progress(
                        f"  halos FAILED for sample_id={sample_id}: {rec.get('error', 'unknown error')}",
                        rank=rank,
                    )
                elif rec["status"] == "skipped_reuse":
                    log_progress(f"  halos reused existing outputs for sample_id={sample_id}", rank=rank)
                else:
                    log_progress(
                        f"  halos completed in {rec['runtime_hms']} for sample_id={sample_id} "
                        f"(n_halos={rec['n_halos']})",
                        rank=rank,
                    )

    # HOD stage
    if str(args.stage) in {"hod", "all"}:
        n_entries = len(entries)
        for i, entry in enumerate(entries, start=1):
            sample_id = int(entry["sample_id"])
            if rank == 0:
                log_progress(f"[hod {i}/{n_entries}] sample_id={sample_id}", rank=rank)

            rec = run_hod_stage_for_sample(
                sample_id=int(sample_id),
                ppc_sim_dir=ppc_sim_dir,
                args=args,
                runtime=runtime,
                comm=comm,
            )

            if rank == 0:
                manifest["hod_records"].append(rec)
                write_manifest(manifest_path, manifest)
                if rec["status"] == "failed":
                    log_progress(
                        f"  hod FAILED for sample_id={sample_id}: {rec.get('error', 'unknown error')}",
                        rank=rank,
                    )
                elif rec["status"] == "skipped_reuse":
                    log_progress(f"  hod reused existing outputs for sample_id={sample_id}", rank=rank)
                else:
                    log_progress(
                        f"  hod completed in {rec['runtime_hms']} for sample_id={sample_id} "
                        f"(n_halos_input={rec['n_halos_input']}, n_galaxies={rec['n_galaxies']})",
                        rank=rank,
                    )

    if str(args.stage) in {"halos", "all"} and rank == 0:
        true_mf_info = materialize_true_mass_function_if_needed(
            true_sim_dir=true_sim_dir,
            runtime=runtime,
            hmf_nbins=int(args.hmf_nbins),
            hmf_mmin=args.hmf_mmin,
            hmf_mmax=args.hmf_mmax,
            do_materialize=bool(args.materialize_true_mass_function_if_missing),
            overwrite=bool(args.force_overwrite_true_mass_function),
        )
        manifest["true_mass_function"] = true_mf_info
        write_manifest(manifest_path, manifest)
        log_progress(
            f"True mass-function materialization status: {true_mf_info.get('status', 'unknown')}",
            rank=rank,
        )

        if bool(args.run_ppc):
            run_dir = (
                Path(args.g2f_run_dir).resolve()
                if args.g2f_run_dir is not None
                else (DEFAULT_G2F_RUN_ROOT / str(tag)).resolve()
            )
            ppc_info = run_g2f_ppc(
                g2f_ppc_script=Path(args.g2f_ppc_script).resolve(),
                run_dir=run_dir,
                sim_id=int(sim_id),
                ppc_root=dataset_root,
                ppc_n_workers=args.ppc_n_workers,
            )
            manifest["ppc_run"] = ppc_info
            write_manifest(manifest_path, manifest)
            log_progress(
                f"g2f PPC finished in {ppc_info['runtime_hms']} (run_dir={run_dir})",
                rank=rank,
            )

    if rank == 0:
        total = float(time.perf_counter() - t_total)
        n_forward_failed = int(sum(1 for r in manifest["forward_records"] if r.get("status") == "failed"))
        n_halo_failed = int(sum(1 for r in manifest["halo_records"] if r.get("status") == "failed"))
        n_hod_failed = int(sum(1 for r in manifest["hod_records"] if r.get("status") == "failed"))
        log_progress(f"Done in {format_hms(total)}", rank=rank)
        log_progress(f"Manifest: {manifest_path}", rank=rank)
        if manifest["forward_records"]:
            log_progress(
                f"Forward summary: {len(manifest['forward_records']) - n_forward_failed}/"
                f"{len(manifest['forward_records'])} successful",
                rank=rank,
            )
        if manifest["halo_records"]:
            log_progress(
                f"Halos summary: {len(manifest['halo_records']) - n_halo_failed}/"
                f"{len(manifest['halo_records'])} successful",
                rank=rank,
            )
        if manifest["hod_records"]:
            log_progress(
                f"HOD summary: {len(manifest['hod_records']) - n_hod_failed}/"
                f"{len(manifest['hod_records'])} successful",
                rank=rank,
            )


if __name__ == "__main__":
    main()
