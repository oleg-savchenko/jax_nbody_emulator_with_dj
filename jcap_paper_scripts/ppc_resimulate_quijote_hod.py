#!/usr/bin/env python3
"""
PPC resimulation pipeline for Quijote-HOD IC posterior samples.

Stages:
1) forward (GPU):
   - Load sample_*.npy ICs from ic_samples/sim_*/
   - Include delta_ic.npy as an extra forward input (when present)
   - Upsample 64^3 -> 512^3 via random high-k mode injection
   - Run emulator forward model
   - Save per-sample displacement + final DM field

2) halos (CPU/MPI):
   - Build FoF halos with nbodykit
   - Save FoF catalog + halo mass function + halo field (mass-weighted PCS+deconv)
   - Remove emu_dis_NNNN.npy after successful halo processing

3) hod (GPU JAX):
   - Build HOD field from FoF catalog with field_level_sbi.jax_hod.JaxHodForwardModel
   - Save raw and sigma_h-normalized HOD fields

4) plot analysis (default after hod/all):
   - Use observed field from delta_gal.npy when present (fallback: delta_obs.npy)
   - Compare sample resimulated HOD fields against resimulated delta_ic truth field
   - Run field_level_sbi.plot_utils.plot_samples_analysis
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = f"/tmp/matplotlib-{os.environ.get('USER', 'user')}"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from halos import (
    gather_halos_to_root,
    load_local_positions_from_displacement,
    particle_mass_msun_h,
    run_fof,
)
from utils import add_import_paths, get_pk_class, parse_ndiv

add_import_paths(repo_root=REPO_ROOT)

try:
    from core import QUIJOTE_FIDUCIAL, run_lpt_emulator_pipeline
except ImportError as exc:  # pragma: no cover
    QUIJOTE_FIDUCIAL = {"Omega_m": 0.3175, "Omega_b": 0.0490, "h": 0.6711, "n_s": 0.9624, "sigma8": 0.8340}
    run_lpt_emulator_pipeline = None
    _CORE_IMPORT_ERROR = exc
else:
    _CORE_IMPORT_ERROR = None

try:
    from mpi4py import MPI
except Exception as exc:  # pragma: no cover
    MPI = None
    _MPI_IMPORT_ERROR = exc
else:
    _MPI_IMPORT_ERROR = None

try:
    from nbodykit.algorithms.fof import FOF, fof_catalog
    from nbodykit.source.catalog import ArrayCatalog
except Exception as exc:  # pragma: no cover
    FOF = None
    fof_catalog = None
    ArrayCatalog = None
    _NBODYKIT_IMPORT_ERROR = exc
else:
    _NBODYKIT_IMPORT_ERROR = None

try:
    from discodj.core.scatter_and_gather import compensate_mak, scatter
except Exception as exc:  # pragma: no cover
    compensate_mak = None
    scatter = None
    _DISCODJ_IMPORT_ERROR = exc
else:
    _DISCODJ_IMPORT_ERROR = None


DEFAULT_IC_SAMPLES_ROOT = Path(
    "/gpfs/scratch1/shared/osavchenko/jcap_paper/"
    "260531_152133_hod_onthefly_64_1ksims_60ep_logMmin13.0/ic_samples"
)
SIM_INDEX_OFFSET = 2047
SIM_INDEX_MIN = 1
SIM_INDEX_MAX = 16
DEFAULT_SIM_INDEX = 1
DEFAULT_RUN_CONFIG = Path(
    "/home/osavchenko/field_level_sbi/runs/"
    "260531_152133_hod_onthefly_64_1ksims_60ep_logMmin13.0/config.json"
)
DEFAULT_PLOT_ROOT = Path(
    "/home/osavchenko/field_level_sbi/paper_scripts/"
    "260531_152133_hod_onthefly_64_1ksims_60ep_logMmin13.0"
)


class _SerialComm:
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
    if root_only and rank != 0:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [rank {rank}] {message}", flush=True)


def format_hms(seconds: float) -> str:
    total = int(round(float(seconds)))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}h {mm:02d}m {ss:02d}s"


def extract_numeric_id(name: str, fallback: int) -> int:
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else int(fallback)


def sort_paths_by_numeric_stem(paths: list[Path]) -> list[Path]:
    def _key(p: Path) -> tuple[int, str]:
        m = re.search(r"(\d+)", p.stem)
        if m is None:
            return (10**9, p.name)
        return (int(m.group(1)), p.name)

    return sorted(paths, key=_key)


def select_paths(paths: list[Path], sample_start: int, sample_stride: int, num_samples: int | None) -> list[Path]:
    selected = paths[int(sample_start) :: int(sample_stride)]
    if num_samples is not None:
        selected = selected[: int(num_samples)]
    return selected


def _quijote_class_params(fiducial: dict[str, float]) -> dict[str, float]:
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


def upsample_field_mode_inject(field: np.ndarray, *, prep: dict[str, Any], seed: int | None) -> np.ndarray:
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


def mesh_delta_from_positions(
    *,
    positions: np.ndarray,
    boxsize: float,
    res: int,
    worder: int,
    weights: np.ndarray | None,
) -> np.ndarray:
    if scatter is None or compensate_mak is None:
        raise ImportError("DISCO-DJ scatter/compensate are required for field meshing.") from _DISCODJ_IMPORT_ERROR

    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    mesh = np.zeros((int(res), int(res), int(res)), dtype=np.float32)
    if pos.shape[0] == 0:
        return mesh

    pos = np.mod(pos, np.float32(boxsize))
    if weights is None:
        weight_arr = None
    else:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if w.shape[0] != pos.shape[0]:
            raise ValueError(f"weights length {w.shape[0]} != positions length {pos.shape[0]}")
        weight_arr = w[:, None]

    field = scatter(
        mesh,
        pos,
        n_part_tot=int(pos.shape[0]),
        res=int(res),
        boxsize=float(boxsize),
        dtype_num=32,
        weight=weight_arr,
        worder=int(worder),
        with_jax=False,
    )
    field = np.asarray(field, dtype=np.float32)
    mean = float(np.mean(field, dtype=np.float64))
    if not np.isfinite(mean) or mean <= 0.0:
        return np.zeros_like(field, dtype=np.float32)

    delta = field / mean - np.float32(1.0)
    delta = compensate_mak(delta, double_exponent=False, worder=int(worder), with_jax=False)
    return np.asarray(delta, dtype=np.float32)


def ensure_emulator_dependencies() -> None:
    if _CORE_IMPORT_ERROR is not None or run_lpt_emulator_pipeline is None:
        raise ImportError("Forward stage requires emulator/JAX dependencies.") from _CORE_IMPORT_ERROR


def ensure_fof_dependencies() -> None:
    if _MPI_IMPORT_ERROR is not None or MPI is None:
        raise ImportError("Halos stage requires mpi4py.") from _MPI_IMPORT_ERROR
    if _NBODYKIT_IMPORT_ERROR is not None or FOF is None or fof_catalog is None or ArrayCatalog is None:
        raise ImportError("Halos stage requires nbodykit.") from _NBODYKIT_IMPORT_ERROR


def has_discodj_meshing() -> bool:
    return (scatter is not None) and (compensate_mak is not None)


def import_field_level_modules(field_level_sbi_root: Path) -> dict[str, Any]:
    root = Path(field_level_sbi_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"field_level_sbi root not found: {root}")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from field_level_sbi.jax_hod import JaxHodForwardModel
    from field_level_sbi import plot_utils
    from field_level_sbi import utils as fsbi_utils

    return {
        "JaxHodForwardModel": JaxHodForwardModel,
        "plot_utils": plot_utils,
        "fsbi_utils": fsbi_utils,
    }


def load_run_config(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Run config not found: {p}")
    return json.loads(p.read_text())


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    return value


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2))


def parse_sim_ids(value: str) -> set[int]:
    raw = str(value).replace(",", " ").split()
    out: set[int] = set()
    for tok in raw:
        out.add(int(tok))
    return out


def parse_ndiv_fallbacks(value: str) -> list[tuple[int, int, int]]:
    """Parse fallback ndiv list from 'a,b,c;d,e,f' or 'a,b,c d,e,f'."""
    text = str(value).strip()
    if not text:
        return []
    parts = [p.strip() for p in text.replace(";", " ").split() if p.strip()]
    out: list[tuple[int, int, int]] = []
    for part in parts:
        out.append(tuple(int(x) for x in parse_ndiv(part)))
    return out


def _is_oom_error(exc: Exception) -> bool:
    msg = repr(exc)
    needles = (
        "RESOURCE_EXHAUSTED",
        "Out of memory",
        "out of memory",
        "CUDA_ERROR_OUT_OF_MEMORY",
        "ran out of memory",
    )
    return any(n in msg for n in needles)


def discover_sim_dirs(ic_samples_root: Path, sim_pattern: str, sim_ids: set[int] | None) -> list[Path]:
    sims = [p for p in ic_samples_root.glob(sim_pattern) if p.is_dir()]

    def _sim_key(p: Path) -> tuple[int, str]:
        return (extract_numeric_id(p.name, fallback=10**9), p.name)

    sims = sorted(sims, key=_sim_key)
    if sim_ids is None:
        return sims

    selected: list[Path] = []
    for p in sims:
        sid = extract_numeric_id(p.name, fallback=-1)
        if sid in sim_ids:
            selected.append(p)
    return selected


def entry_output_paths(out_sim_dir: Path, entry_stem: str) -> dict[str, Path]:
    stem = str(entry_stem)
    return {
        "dis": out_sim_dir / f"emu_dis_{stem}.npy",
        "dm_field": out_sim_dir / f"dm_field_{stem}.npy",
        "fof": out_sim_dir / f"fof_catalog_{stem}.npz",
        "hmf": out_sim_dir / f"mass_function_{stem}.npy",
        "halo_field": out_sim_dir / f"halo_field_mass_{stem}.npy",
        "hod_field_raw": out_sim_dir / f"hod_field_raw_{stem}.npy",
        "hod_field_obsnorm": out_sim_dir / f"hod_field_obsnorm_{stem}.npy",
    }


def _entry_sort_key(entry: dict[str, Any]) -> tuple[int, int, str]:
    kind = str(entry.get("kind", "other"))
    if kind == "sample":
        sid = int(entry.get("sample_id", 10**9))
        return (0, sid, str(entry.get("entry_stem", "")))
    if kind == "delta_ic":
        return (1, 0, str(entry.get("entry_stem", "")))
    return (2, 0, str(entry.get("entry_stem", "")))


def _make_entry(
    *,
    kind: str,
    entry_stem: str,
    sample_file: str | None,
    sample_id: int | None = None,
) -> dict[str, Any]:
    return {
        "kind": str(kind),
        "entry_stem": str(entry_stem),
        "entry_label": str(entry_stem),
        "sample_id": None if sample_id is None else int(sample_id),
        "sample_file": sample_file,
    }


def _entry_seed(entry: dict[str, Any]) -> int:
    sid = entry.get("sample_id")
    if sid is not None:
        return int(sid)
    return int(zlib.crc32(str(entry["entry_stem"]).encode("utf-8")) & 0x7FFFFFFF)


def _finite_std(x: np.ndarray) -> float | None:
    arr = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None
    std = float(np.std(arr[finite]))
    if not np.isfinite(std):
        return None
    return std


def build_hod_input_from_fof(fof_path: Path, h_max: int) -> tuple[np.ndarray, int, int]:
    with np.load(fof_path) as nz:
        masses = np.asarray(nz["Mass"], dtype=np.float32).reshape(-1)
        pos = np.asarray(nz["CMPosition"], dtype=np.float32).reshape(-1, 3)
        npart = np.asarray(nz["Npart"], dtype=np.float32).reshape(-1)

    n_halos_input = int(masses.shape[0])
    if not (pos.shape[0] == n_halos_input == npart.shape[0]):
        raise ValueError("FoF catalog arrays have inconsistent lengths")

    order = np.argsort(masses)[::-1]
    masses = masses[order]
    pos = pos[order]
    npart = npart[order]

    take = min(int(h_max), int(n_halos_input))
    cat = np.zeros((int(h_max), 5), dtype=np.float32)
    if take > 0:
        cat[:take, 0] = masses[:take]
        cat[:take, 1:4] = pos[:take]
        cat[:take, 4] = npart[:take]
    return cat, int(n_halos_input), int(take)


def ensure_finite_field(arr: np.ndarray, *, label: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(out)
    if not np.all(finite):
        frac = float(np.mean(finite))
        raise ValueError(f"Non-finite values in {label} (finite_fraction={frac:.6f})")
    return out


def maybe_rescale_sample_ic(
    *,
    ic: np.ndarray,
    entry_kind: str,
    sample_space: str,
    rescaling_factor: float | None,
    delta_ic_std_ref: float | None,
    auto_ratio_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Convert sampled ICs to physical units expected by emulator forward stage.

    sample_space:
      - physical: no conversion
      - rescaled: force ic -> ic * rescaling_factor
      - auto: infer from amplitude mismatch against delta_ic reference std
    """
    info: dict[str, Any] = {
        "mode_requested": str(sample_space),
        "mode_applied": "none",
        "scale_factor": 1.0,
        "std_raw": _finite_std(ic),
        "std_used": None,
        "delta_ic_std_ref": None if delta_ic_std_ref is None else float(delta_ic_std_ref),
        "reason": "no_change",
    }

    kind = str(entry_kind)
    mode = str(sample_space)
    rf = None if rescaling_factor is None else float(rescaling_factor)
    raw_std = info["std_raw"]

    if kind != "sample":
        info["reason"] = "entry_is_not_sample"
        info["std_used"] = raw_std
        return np.asarray(ic, dtype=np.float32), info

    if mode == "physical":
        info["reason"] = "forced_physical"
        info["std_used"] = raw_std
        return np.asarray(ic, dtype=np.float32), info

    if rf is None or (not np.isfinite(rf)) or rf <= 0.0:
        info["reason"] = "missing_or_invalid_rescaling_factor"
        info["std_used"] = raw_std
        return np.asarray(ic, dtype=np.float32), info

    if mode == "rescaled":
        out = np.asarray(ic, dtype=np.float32) * np.float32(rf)
        info["mode_applied"] = "mul_rf"
        info["scale_factor"] = float(rf)
        info["reason"] = "forced_rescaled_to_physical"
        info["std_used"] = _finite_std(out)
        return out, info

    # auto mode
    if raw_std is None or raw_std <= 0.0:
        info["reason"] = "nonfinite_or_zero_raw_std"
        info["std_used"] = raw_std
        return np.asarray(ic, dtype=np.float32), info

    # Strong prior: if a delta_ic reference is available, prefer conversion that
    # makes sample amplitude consistent with that reference.
    if delta_ic_std_ref is not None and np.isfinite(delta_ic_std_ref) and delta_ic_std_ref > 0.0:
        std_keep = abs(raw_std - float(delta_ic_std_ref))
        std_mul = abs(raw_std * rf - float(delta_ic_std_ref))
        amp_ratio = raw_std / float(delta_ic_std_ref)
        if amp_ratio >= float(auto_ratio_threshold) and std_mul < std_keep:
            out = np.asarray(ic, dtype=np.float32) * np.float32(rf)
            info["mode_applied"] = "mul_rf"
            info["scale_factor"] = float(rf)
            info["reason"] = "auto_mismatch_vs_delta_ic"
            info["std_used"] = _finite_std(out)
            return out, info
        info["reason"] = "auto_kept_physical_by_delta_ic_check"
        info["std_used"] = raw_std
        return np.asarray(ic, dtype=np.float32), info

    # Fallback heuristic when delta_ic reference is unavailable.
    # Typical physical z=0 IC std is O(1), while GNPE-rescaled samples are often O(1/rf) >> 1.
    if raw_std > float(auto_ratio_threshold) and (raw_std * rf) < raw_std:
        out = np.asarray(ic, dtype=np.float32) * np.float32(rf)
        info["mode_applied"] = "mul_rf"
        info["scale_factor"] = float(rf)
        info["reason"] = "auto_high_amplitude_fallback"
        info["std_used"] = _finite_std(out)
        return out, info

    info["reason"] = "auto_no_conversion_triggered"
    info["std_used"] = raw_std
    return np.asarray(ic, dtype=np.float32), info


def sample_entries_for_sim(
    *,
    sim_dir: Path,
    out_sim_dir: Path,
    sample_pattern: str,
    sample_start: int,
    sample_stride: int,
    num_samples: int | None,
    stage: str,
    include_delta_ic: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    samples = sort_paths_by_numeric_stem(list(sim_dir.glob(sample_pattern)))
    if samples:
        selected = select_paths(samples, sample_start, sample_stride, num_samples)
        for idx, p in enumerate(selected):
            sid = extract_numeric_id(p.stem, fallback=idx)
            out.append(
                _make_entry(
                    kind="sample",
                    entry_stem=f"{int(sid):04d}",
                    sample_file=str(p),
                    sample_id=int(sid),
                )
            )

    if include_delta_ic:
        delta_ic_path = sim_dir / "delta_ic.npy"
        if delta_ic_path.is_file():
            out.append(
                _make_entry(kind="delta_ic", entry_stem="delta_ic", sample_file=str(delta_ic_path), sample_id=None)
            )

    if out:
        return sorted(out, key=_entry_sort_key)

    if stage in {"forward", "all"}:
        raise FileNotFoundError(
            f"No forward IC inputs found in {sim_dir}. "
            f"Checked {sample_pattern!r} plus optional delta_ic.npy."
        )

    if stage == "halos":
        prefix = "emu_dis_"
        ext = ".npy"
        fallback = sort_paths_by_numeric_stem(list(out_sim_dir.glob("emu_dis_*.npy")))
    else:
        prefix = "fof_catalog_"
        ext = ".npz"
        fallback = sort_paths_by_numeric_stem(list(out_sim_dir.glob("fof_catalog_*.npz")))

    fallback = select_paths(fallback, sample_start, sample_stride, num_samples)
    if not fallback:
        raise FileNotFoundError(
            f"No sample files and no stage fallback files found for stage={stage} in {sim_dir} / {out_sim_dir}"
        )

    for idx, p in enumerate(fallback):
        name = p.name
        if not (name.startswith(prefix) and name.endswith(ext)):
            sid = extract_numeric_id(p.stem, fallback=idx)
            out.append(
                _make_entry(
                    kind="sample",
                    entry_stem=f"{int(sid):04d}",
                    sample_file=None,
                    sample_id=int(sid),
                )
            )
            continue

        stem = name[len(prefix) : -len(ext)]
        if stem == "z_map":
            # Legacy outputs may contain z_map artifacts from older runs.
            # Intentionally skip them: z_map must not be resimulated.
            continue
        elif stem == "delta_ic":
            out.append(_make_entry(kind="delta_ic", entry_stem="delta_ic", sample_file=None, sample_id=None))
        elif stem.isdigit():
            out.append(
                _make_entry(
                    kind="sample",
                    entry_stem=f"{int(stem):04d}",
                    sample_file=None,
                    sample_id=int(stem),
                )
            )
        else:
            out.append(_make_entry(kind="other", entry_stem=stem, sample_file=None, sample_id=None))
    return sorted(out, key=_entry_sort_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPC resimulation for Quijote-HOD IC samples: forward + FoF + HOD + plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--stage", choices=("forward", "halos", "hod", "all"), default="all")

    parser.add_argument(
        "--ic-samples-root",
        type=Path,
        default=DEFAULT_IC_SAMPLES_ROOT,
        help=(
            "Path to IC samples collection root (.../ic_samples) or one explicit sim dir "
            "(.../ic_samples/sim_<id>)."
        ),
    )
    parser.add_argument(
        "--sim-index",
        type=int,
        default=DEFAULT_SIM_INDEX,
        help=(
            f"Select sim id via sim_id = {SIM_INDEX_OFFSET} + sim_index "
            f"(allowed range: {SIM_INDEX_MIN}..{SIM_INDEX_MAX})."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=None, help="Default: <run_root>/ppc")
    parser.add_argument("--run-config", type=Path, default=None, help="Default: /home/osavchenko/field_level_sbi/runs/<tag>/config.json")
    parser.add_argument("--field-level-sbi-root", type=Path, default=Path("/home/osavchenko/field_level_sbi"))

    parser.add_argument("--sim-pattern", type=str, default="sim_*")
    parser.add_argument("--sim-ids", type=str, default="", help="Optional ids, e.g. '2048 2050' or '2048,2050'")

    parser.add_argument("--sample-pattern", type=str, default="sample_*.npy")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument(
        "--include-delta-ic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include delta_ic.npy in resimulation stages when present.",
    )

    parser.add_argument("--n-part", type=int, default=512)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--boxsize", type=float, default=1000.0)
    parser.add_argument("--ndiv", type=parse_ndiv, default=(4, 2, 2))
    parser.add_argument(
        "--ndiv-fallbacks",
        type=str,
        default="8,4,4;8,8,8",
        help=(
            "Fallback ndiv list for OOM retries in forward stage. "
            "Format: '8,4,4;8,8,8' or empty string to disable."
        ),
    )
    parser.add_argument("--emu-precision", choices=("f16", "f32"), default="f16")
    parser.add_argument(
        "--emu-num-sims",
        type=int,
        default=2,
        help="Number of emulator forward passes per IC call (matches optimized 512^3 workflow).",
    )
    parser.add_argument("--class-n-modes", type=int, default=5000)
    parser.add_argument("--mode-inject-seed-policy", choices=("sample_id", "fixed", "none"), default="sample_id")
    parser.add_argument("--mode-inject-fixed-seed", type=int, default=0)
    parser.add_argument(
        "--ic-sample-space",
        choices=("auto", "physical", "rescaled"),
        default="auto",
        help=(
            "Interpretation of sample_*.npy amplitude before emulator forward pass. "
            "'physical': use as-is; 'rescaled': convert via sample * rescaling_factor; "
            "'auto': detect mismatch versus delta_ic amplitude and convert when needed."
        ),
    )
    parser.add_argument(
        "--ic-rescaling-factor",
        type=float,
        default=None,
        help="Optional override for run config rescaling_factor used by --ic-sample-space.",
    )
    parser.add_argument(
        "--ic-auto-scale-threshold",
        type=float,
        default=5.0,
        help=(
            "Auto-conversion trigger ratio for sample std mismatch vs delta_ic std "
            "(only used when --ic-sample-space=auto)."
        ),
    )

    parser.add_argument("--linking-length", type=float, default=0.2)
    parser.add_argument("--absolute-linking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--nmin", type=int, default=20)
    parser.add_argument("--omega-m", type=float, default=0.3175)
    parser.add_argument("--cleanup-temp-displacement", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--hmf-nbins", type=int, default=25)
    parser.add_argument("--hmf-mmin", type=float, default=None)
    parser.add_argument("--hmf-mmax", type=float, default=None)

    parser.add_argument("--mas-worder", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--deconvolve-mas", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--sigma-h", type=float, default=None, help="Override sigma_h from run config")
    parser.add_argument("--h-max", type=int, default=None, help="Override H_max from run config")
    parser.add_argument("--per-host-cap", type=int, default=None, help="Override per_host_cap from run config")
    parser.add_argument("--n-max-sat", type=int, default=None, help="Override N_max_sat from run config")
    parser.add_argument("--hod-batch-size", type=int, default=8, help="Micro-batch size for GPU JAX HOD inference.")

    parser.add_argument("--run-plot-analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-output-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--plot-n-workers", type=int, default=None)

    parser.add_argument("--manifest-name", type=str, default="ppc_manifest.json")

    args = parser.parse_args()

    if args.sample_start < 0:
        parser.error("--sample-start must be >= 0")
    if args.sample_stride < 1:
        parser.error("--sample-stride must be >= 1")
    if args.sim_index < SIM_INDEX_MIN or args.sim_index > SIM_INDEX_MAX:
        parser.error(f"--sim-index must be in [{SIM_INDEX_MIN}, {SIM_INDEX_MAX}]")
    if args.num_samples is not None and args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.n_part < 1 or args.res < 1:
        parser.error("--n-part and --res must be >= 1")
    try:
        _ = parse_ndiv_fallbacks(str(args.ndiv_fallbacks))
    except Exception as exc:
        parser.error(f"--ndiv-fallbacks parse error: {exc}")
    if args.emu_num_sims < 1:
        parser.error("--emu-num-sims must be >= 1")
    if args.hmf_nbins < 3:
        parser.error("--hmf-nbins must be >= 3")
    if args.ic_auto_scale_threshold <= 1.0:
        parser.error("--ic-auto-scale-threshold must be > 1")
    if args.hod_batch_size < 1:
        parser.error("--hod-batch-size must be >= 1")

    return args


def main() -> None:
    args = parse_args()

    ic_samples_root = Path(args.ic_samples_root).resolve()
    if not ic_samples_root.is_dir():
        raise FileNotFoundError(f"IC samples root not found: {ic_samples_root}")

    # Detect explicit single-sim input like .../ic_samples/sim_2048
    is_single_sim_root = bool(re.fullmatch(r"sim_\d+", ic_samples_root.name))
    if is_single_sim_root:
        run_root = ic_samples_root.parent.parent
        ic_samples_collection_root = ic_samples_root.parent
    else:
        run_root = ic_samples_root.parent
        ic_samples_collection_root = ic_samples_root

    run_tag = run_root.name
    output_root = Path(args.output_root).resolve() if args.output_root is not None else (run_root / "ppc").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.run_config is not None:
        run_config_path = Path(args.run_config).resolve()
    else:
        run_config_path = Path(f"/home/osavchenko/field_level_sbi/runs/{run_tag}/config.json").resolve()
        if not run_config_path.is_file():
            run_config_path = DEFAULT_RUN_CONFIG.resolve()

    if str(args.stage) in {"forward", "all"}:
        ensure_emulator_dependencies()
    if str(args.stage) in {"halos", "all"}:
        ensure_fof_dependencies()
    if str(args.stage) in {"hod", "all"}:
        _ = import_field_level_modules(Path(args.field_level_sbi_root))

    if str(args.stage) == "halos":
        comm = MPI.COMM_WORLD
    elif str(args.stage) == "hod":
        comm = _SerialComm() if MPI is None else MPI.COMM_WORLD
    else:
        if MPI is None:
            comm = _SerialComm()
        else:
            if int(MPI.COMM_WORLD.size) != 1:
                raise RuntimeError(f"stage={args.stage} must run with one rank (got {MPI.COMM_WORLD.size})")
            comm = MPI.COMM_SELF

    rank = int(comm.rank)
    size = int(comm.size)

    if str(args.stage) == "hod" and MPI is None:
        step_ntasks = int(
            os.environ.get(
                "SLURM_STEP_NUM_TASKS",
                os.environ.get("PMI_SIZE", os.environ.get("SLURM_NTASKS", "1")),
            )
        )
        if step_ntasks > 1:
            raise RuntimeError(
                "stage=hod detected multi-task execution but mpi4py is unavailable. "
                "This would spawn duplicate serial writers. Re-run with one task or an env with mpi4py."
            )

    run_cfg_local = load_run_config(run_config_path) if rank == 0 else None
    run_cfg = comm.bcast(run_cfg_local, root=0)

    box_params = dict(run_cfg.get("box_params", {}))
    hod_params = dict(run_cfg.get("hod_params", {}))
    cosmo_params = dict(run_cfg.get("cosmo_params", {}))
    ic_rescaling_factor = (
        float(args.ic_rescaling_factor)
        if args.ic_rescaling_factor is not None
        else (float(run_cfg.get("rescaling_factor")) if run_cfg.get("rescaling_factor") is not None else None)
    )
    sigma_h = float(args.sigma_h) if args.sigma_h is not None else float(run_cfg.get("sigma_h"))
    h_max = int(args.h_max) if args.h_max is not None else int(run_cfg.get("H_max"))
    per_host_cap = int(args.per_host_cap) if args.per_host_cap is not None else int(run_cfg.get("per_host_cap", 16))
    n_max_sat = int(args.n_max_sat) if args.n_max_sat is not None else int(run_cfg.get("N_max_sat", 48))

    if rank == 0:
        log_progress(f"IC samples root: {ic_samples_root}", rank=rank)
        log_progress(
            f"sim-index={args.sim_index} => target sim_{SIM_INDEX_OFFSET + int(args.sim_index)}",
            rank=rank,
        )
        log_progress(f"Output root: {output_root}", rank=rank)
        log_progress(f"Run config: {run_config_path}", rank=rank)
        log_progress(
            f"IC sample-space={args.ic_sample_space}, rescaling_factor={ic_rescaling_factor}",
            rank=rank,
        )
        log_progress(
            f"FoF params: b={args.linking_length}, absolute={args.absolute_linking}, nmin={args.nmin}, "
            f"n_part={args.n_part}, box={args.boxsize}",
            rank=rank,
        )
        if str(args.stage) in {"halos", "hod"}:
            log_progress(f"MPI world size: {size}", rank=rank)

    sim_id_set = parse_sim_ids(args.sim_ids) if str(args.sim_ids).strip() else None
    sim_dirs_local: list[Path] | None = None
    if rank == 0:
        if is_single_sim_root:
            sim_dirs_local = [ic_samples_root]
        elif sim_id_set is not None:
            sim_dirs_local = discover_sim_dirs(ic_samples_collection_root, args.sim_pattern, sim_id_set)
        else:
            target_sim_id = SIM_INDEX_OFFSET + int(args.sim_index)
            target_sim_dir = ic_samples_collection_root / f"sim_{target_sim_id}"
            if not target_sim_dir.is_dir():
                raise FileNotFoundError(
                    f"Target sim directory from --sim-index was not found: {target_sim_dir}"
                )
            sim_dirs_local = [target_sim_dir]
    sim_dirs = comm.bcast(sim_dirs_local, root=0)

    if rank == 0:
        log_progress(f"Discovered {len(sim_dirs)} sim directories.", rank=rank)

    class_params = _quijote_class_params(QUIJOTE_FIDUCIAL)
    pk_table: tuple[np.ndarray, np.ndarray] | None = None
    prep_cache: dict[tuple[int, int], dict[str, Any]] = {}
    halo_field_enabled = bool(has_discodj_meshing())

    if rank == 0 and str(args.stage) in {"halos", "all"}:
        if halo_field_enabled:
            log_progress("Halo-field meshing: enabled (DISCO-DJ available).", rank=rank)
        else:
            log_progress(
                "Halo-field meshing: disabled (DISCO-DJ not available in current environment). "
                "FoF and HMF outputs will still be produced.",
                rank=rank,
            )

    t_total = time.perf_counter()

    for sim_idx, sim_dir in enumerate(sim_dirs, start=1):
        sim_dir = Path(sim_dir)
        sim_name = sim_dir.name
        sim_id = extract_numeric_id(sim_name, fallback=sim_idx)
        out_sim_dir = output_root / sim_name
        if rank == 0:
            out_sim_dir.mkdir(parents=True, exist_ok=True)

        entries_local = None
        if rank == 0:
            entries_local = sample_entries_for_sim(
                sim_dir=sim_dir,
                out_sim_dir=out_sim_dir,
                sample_pattern=args.sample_pattern,
                sample_start=int(args.sample_start),
                sample_stride=int(args.sample_stride),
                num_samples=None if args.num_samples is None else int(args.num_samples),
                stage=str(args.stage),
                include_delta_ic=bool(args.include_delta_ic),
            )
            log_progress(f"[{sim_name}] selected {len(entries_local)} sample entries", rank=rank)
        entries = comm.bcast(entries_local, root=0)

        delta_ic_std_ref_local: float | None = None
        if rank == 0:
            delta_ic_ref_path = sim_dir / "delta_ic.npy"
            if delta_ic_ref_path.is_file():
                try:
                    delta_ic_ref = np.asarray(np.load(delta_ic_ref_path), dtype=np.float32)
                    delta_ic_std_ref_local = _finite_std(delta_ic_ref)
                except Exception:
                    delta_ic_std_ref_local = None
            log_progress(
                f"[{sim_name}] delta_ic std reference for auto scaling: {delta_ic_std_ref_local}",
                rank=rank,
            )
        delta_ic_std_ref = comm.bcast(delta_ic_std_ref_local, root=0)

        manifest_path = out_sim_dir / str(args.manifest_name)
        manifest: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sim_name": sim_name,
            "sim_id": int(sim_id),
            "ic_sim_dir": str(sim_dir),
            "output_sim_dir": str(out_sim_dir),
            "stage": str(args.stage),
            "config": jsonable(vars(args)),
            "run_config_path": str(run_config_path),
            "selected_sample_ids": [int(e["sample_id"]) for e in entries if e.get("sample_id") is not None],
            "selected_entries": [
                {
                    "kind": str(e.get("kind")),
                    "entry_stem": str(e.get("entry_stem")),
                    "sample_file": e.get("sample_file"),
                    "sample_id": e.get("sample_id"),
                }
                for e in entries
            ],
            "forward_records": [],
            "halo_records": [],
            "hod_records": [],
            "hod_batch": {
                "configured_batch_size": int(args.hod_batch_size),
                "num_batch_calls": 0,
                "num_batch_fallback_calls": 0,
            },
            "plot_analysis": {"status": "not_requested"},
        }

        # Forward stage
        if str(args.stage) in {"forward", "all"}:
            for i, entry in enumerate(entries, start=1):
                entry_kind = str(entry.get("kind", "sample"))
                entry_stem = str(entry.get("entry_stem"))
                entry_label = str(entry.get("entry_label", entry_stem))
                sample_id = entry.get("sample_id")
                sample_file = entry.get("sample_file")
                if sample_file is None:
                    raise RuntimeError(f"forward stage needs sample file for entry={entry_label}")
                sample_path = Path(sample_file)
                paths = entry_output_paths(out_sim_dir, entry_stem)

                rec: dict[str, Any] = {
                    "entry_kind": entry_kind,
                    "entry_stem": entry_stem,
                    "entry_label": entry_label,
                    "sample_id": None if sample_id is None else int(sample_id),
                    "sample_file": str(sample_path),
                    "dm_field_path": str(paths["dm_field"]),
                    "displacement_path": str(paths["dis"]),
                    "status": "completed",
                    "reused": False,
                    "upsampled": False,
                    "upsample_seed": None,
                    "ic_sample_space_mode_requested": str(args.ic_sample_space),
                    "ic_sample_space_mode_applied": "none",
                    "ic_sample_scale_factor": 1.0,
                    "ic_std_raw": None,
                    "ic_std_used": None,
                    "ic_scale_reason": None,
                    "emu_attempts": [],
                    "emu_ndiv_used": None,
                    "emu_num_sims_used": None,
                    "runtime_seconds": 0.0,
                }

                t0 = time.perf_counter()
                if rank == 0:
                    log_progress(f"[{sim_name}] [forward {i}/{len(entries)}] entry={entry_label}", rank=rank)

                try:
                    all_complete = (
                        bool(args.reuse_existing)
                        and str(args.stage) == "all"
                        and paths["dm_field"].is_file()
                        and paths["fof"].is_file()
                        and paths["hmf"].is_file()
                        and paths["halo_field"].is_file()
                        and paths["hod_field_obsnorm"].is_file()
                        and paths["hod_field_raw"].is_file()
                    )
                    if all_complete:
                        rec["status"] = "skipped_reuse_all_complete"
                        rec["reused"] = True
                    elif bool(args.reuse_existing) and paths["dm_field"].is_file() and paths["dis"].is_file():
                        rec["status"] = "skipped_reuse"
                        rec["reused"] = True
                    else:
                        ic = np.asarray(np.load(sample_path), dtype=np.float32)
                        if ic.ndim != 3 or not (ic.shape[0] == ic.shape[1] == ic.shape[2]):
                            raise ValueError(f"Sample must be cubic 3D, got {sample_path} shape={ic.shape}")

                        ic, ic_scale_info = maybe_rescale_sample_ic(
                            ic=np.asarray(ic, dtype=np.float32),
                            entry_kind=entry_kind,
                            sample_space=str(args.ic_sample_space),
                            rescaling_factor=ic_rescaling_factor,
                            delta_ic_std_ref=delta_ic_std_ref,
                            auto_ratio_threshold=float(args.ic_auto_scale_threshold),
                        )
                        rec["ic_sample_space_mode_applied"] = str(ic_scale_info.get("mode_applied", "none"))
                        rec["ic_sample_scale_factor"] = float(ic_scale_info.get("scale_factor", 1.0))
                        rec["ic_std_raw"] = ic_scale_info.get("std_raw")
                        rec["ic_std_used"] = ic_scale_info.get("std_used")
                        rec["ic_scale_reason"] = str(ic_scale_info.get("reason", "unknown"))

                        if rank == 0 and rec["ic_sample_space_mode_applied"] != "none":
                            log_progress(
                                f"[{sim_name}] [{entry_label}] converted sample IC: "
                                f"{rec['ic_sample_space_mode_applied']} (factor={rec['ic_sample_scale_factor']}) "
                                f"std_raw={rec['ic_std_raw']} -> std_used={rec['ic_std_used']} "
                                f"reason={rec['ic_scale_reason']}",
                                rank=rank,
                            )

                        in_res = int(ic.shape[0])
                        out_res = int(args.n_part)
                        if in_res == out_res:
                            ic_npart = np.asarray(ic, dtype=np.float32)
                        else:
                            if out_res < in_res:
                                raise ValueError(f"input resolution {in_res} > n_part {out_res}")
                            if out_res % in_res != 0:
                                raise ValueError(f"n_part/out_res ratio must be integer, got {in_res}->{out_res}")

                            if str(args.mode_inject_seed_policy) == "sample_id":
                                up_seed = int(_entry_seed(entry))
                            elif str(args.mode_inject_seed_policy) == "fixed":
                                up_seed = int(args.mode_inject_fixed_seed)
                            else:
                                up_seed = None
                            rec["upsample_seed"] = None if up_seed is None else int(up_seed)
                            rec["upsampled"] = True

                            if pk_table is None:
                                if rank == 0:
                                    log_progress("Building CLASS P(k) table for mode-injection upsampling...", rank=rank)
                                pk_table = build_class_pk_table_once(
                                    boxsize=float(args.boxsize),
                                    target_res=out_res,
                                    class_n_modes=int(args.class_n_modes),
                                    class_params=class_params,
                                    z=0.0,
                                )

                            key = (in_res, out_res)
                            if key not in prep_cache:
                                k_tab, pk_tab = pk_table
                                prep_cache[key] = prepare_mode_injection(
                                    input_res=in_res,
                                    target_res=out_res,
                                    boxsize=float(args.boxsize),
                                    k_tab=np.asarray(k_tab, dtype=np.float64),
                                    pk_tab=np.asarray(pk_tab, dtype=np.float64),
                                )

                            ic_npart = upsample_field_mode_inject(
                                np.asarray(ic, dtype=np.float32), prep=prep_cache[key], seed=up_seed
                            )

                        preferred_ndiv = tuple(int(x) for x in args.ndiv)
                        fallback_ndivs = parse_ndiv_fallbacks(str(args.ndiv_fallbacks))
                        ndiv_candidates: list[tuple[int, int, int]] = []
                        for cand in [preferred_ndiv, *fallback_ndivs]:
                            cand_t = tuple(int(x) for x in cand)
                            if cand_t not in ndiv_candidates:
                                ndiv_candidates.append(cand_t)

                        num_sims_candidates = [int(args.emu_num_sims)]
                        if int(args.emu_num_sims) != 1:
                            num_sims_candidates.append(1)

                        attempts: list[dict[str, Any]] = []
                        pipe = None
                        last_exc: Exception | None = None
                        for ndiv_try in ndiv_candidates:
                            for num_sims_try in num_sims_candidates:
                                attempts.append({"ndiv": [int(x) for x in ndiv_try], "num_sims": int(num_sims_try)})
                                if rank == 0:
                                    log_progress(
                                        f"  forward attempt with ndiv={ndiv_try}, num_sims={num_sims_try}",
                                        rank=rank,
                                    )
                                try:
                                    pipe = run_lpt_emulator_pipeline(
                                        delta_linear=np.asarray(ic_npart, dtype=np.float32),
                                        n_part=int(args.n_part),
                                        density_res=int(args.res),
                                        boxsize=float(args.boxsize),
                                        ndiv=tuple(int(x) for x in ndiv_try),
                                        emu_precision=str(args.emu_precision),
                                        num_sims=int(num_sims_try),
                                        mas_worder=int(args.mas_worder),
                                        deconvolve_mas=bool(args.deconvolve_mas),
                                        z_target=0.0,
                                        compute_vel=False,
                                        delta_upsample_method="mode_inject",
                                        downsample_method="fourier_truncation",
                                        downsample_gaussian_sigma_mpc_h=None,
                                        downsample_threads=1,
                                        class_n_modes=int(args.class_n_modes),
                                        mode_inject_seed=None,
                                        quijote_fiducial=QUIJOTE_FIDUCIAL,
                                        output_dir=None,
                                        save_outputs=False,
                                        show_progress=True,
                                        progress_desc_prefix=f"{sim_name} {entry_label} subboxes",
                                    )
                                    rec["emu_ndiv_used"] = [int(x) for x in ndiv_try]
                                    rec["emu_num_sims_used"] = int(num_sims_try)
                                    last_exc = None
                                    break
                                except Exception as exc:
                                    last_exc = exc
                                    if not _is_oom_error(exc):
                                        raise
                                    if rank == 0:
                                        log_progress(
                                            f"  OOM detected for ndiv={ndiv_try}, num_sims={num_sims_try}; retrying...",
                                            rank=rank,
                                        )
                            if pipe is not None:
                                break

                        rec["emu_attempts"] = attempts
                        if pipe is None:
                            if last_exc is None:
                                raise RuntimeError("Forward stage failed without an explicit exception.")
                            raise last_exc

                        delta_emu = np.asarray(pipe["delta_emu"], dtype=np.float32)
                        psi_emu = np.asarray(pipe["psi_emu"], dtype=np.float32)
                        if not np.all(np.isfinite(delta_emu)):
                            frac = float(np.isfinite(delta_emu).mean())
                            raise RuntimeError(
                                f"Non-finite values in emulator dm_field output for entry={entry_label} "
                                f"(finite_fraction={frac:.6f})"
                            )
                        if not np.all(np.isfinite(psi_emu)):
                            frac = float(np.isfinite(psi_emu).mean())
                            raise RuntimeError(
                                f"Non-finite values in emulator displacement output for entry={entry_label} "
                                f"(finite_fraction={frac:.6f})"
                            )

                        np.save(paths["dm_field"], delta_emu)
                        np.save(paths["dis"], psi_emu)

                except Exception as exc:
                    rec["status"] = "failed"
                    rec["error"] = repr(exc)

                rec["runtime_seconds"] = float(time.perf_counter() - t0)
                rec["runtime_hms"] = format_hms(float(rec["runtime_seconds"]))

                if rank == 0:
                    manifest["forward_records"].append(rec)
                    write_manifest(manifest_path, manifest)

        # Halos stage
        if str(args.stage) in {"halos", "all"}:
            for i, entry in enumerate(entries, start=1):
                entry_kind = str(entry.get("kind", "sample"))
                entry_stem = str(entry.get("entry_stem"))
                entry_label = str(entry.get("entry_label", entry_stem))
                sample_id = entry.get("sample_id")
                paths = entry_output_paths(out_sim_dir, entry_stem)

                rec: dict[str, Any] = {
                    "entry_kind": entry_kind,
                    "entry_stem": entry_stem,
                    "entry_label": entry_label,
                    "sample_id": None if sample_id is None else int(sample_id),
                    "status": "completed",
                    "reused": False,
                    "n_halos": None,
                    "runtime_seconds": 0.0,
                }

                t0 = time.perf_counter()
                if rank == 0:
                    log_progress(f"[{sim_name}] [halos {i}/{len(entries)}] entry={entry_label}", rank=rank)

                try:
                    halo_output_ready = (not halo_field_enabled) or paths["halo_field"].is_file()
                    if bool(args.reuse_existing) and paths["fof"].is_file() and paths["hmf"].is_file() and halo_output_ready:
                        rec["status"] = "skipped_reuse"
                        rec["reused"] = True
                        if rank == 0 and bool(args.cleanup_temp_displacement) and paths["dis"].is_file():
                            paths["dis"].unlink()
                        comm.Barrier()
                    else:
                        has_dis = bool(paths["dis"].is_file()) if rank == 0 else None
                        has_dis = bool(comm.bcast(has_dis, root=0))
                        if not has_dis:
                            raise FileNotFoundError(
                                f"Missing displacement for halos stage: {paths['dis']}. Run forward first."
                            )

                        local_pos = load_local_positions_from_displacement(
                            displacement_file=paths["dis"],
                            boxsize=float(args.boxsize),
                            n_part=int(args.n_part),
                            rank=rank,
                            size=size,
                        )

                        cm_local, length_local = run_fof(
                            local_positions=np.asarray(local_pos, dtype=np.float32),
                            boxsize=float(args.boxsize),
                            n_part=int(args.n_part),
                            linking_length=float(args.linking_length),
                            nmin=int(args.nmin),
                            absolute_linking=bool(args.absolute_linking),
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

                        root_err = None
                        root_n_halos = None
                        if rank == 0:
                            try:
                                m_particle = particle_mass_msun_h(
                                    omega_m=float(args.omega_m),
                                    boxsize=float(args.boxsize),
                                    n_part=int(args.n_part),
                                )
                                masses = np.asarray(length, dtype=np.float64) * float(m_particle)
                                np.savez_compressed(
                                    paths["fof"],
                                    CMPosition=np.asarray(cm_pos, dtype=np.float32),
                                    Npart=np.asarray(length, dtype=np.int32),
                                    Mass=np.asarray(masses, dtype=np.float64),
                                    BoxSize=np.array([float(args.boxsize)] * 3, dtype=np.float64),
                                    NpartPerDim=np.int32(int(args.n_part)),
                                    LinkingLength=np.float64(float(args.linking_length)),
                                    AbsoluteLinking=np.bool_(bool(args.absolute_linking)),
                                    Nmin=np.int32(int(args.nmin)),
                                )

                                mf = compute_mass_function_table(
                                    masses=np.asarray(masses, dtype=np.float64),
                                    boxsize=float(args.boxsize),
                                    nbins=int(args.hmf_nbins),
                                    mmin=args.hmf_mmin,
                                    mmax=args.hmf_mmax,
                                    empty_fallback_mass=float(m_particle) * max(1, int(args.nmin)),
                                )
                                np.save(paths["hmf"], np.asarray(mf, dtype=np.float64))

                                if halo_field_enabled:
                                    halo_delta = mesh_delta_from_positions(
                                        positions=np.asarray(cm_pos, dtype=np.float32),
                                        boxsize=float(args.boxsize),
                                        res=int(args.res),
                                        worder=int(args.mas_worder),
                                        weights=np.asarray(masses, dtype=np.float32),
                                    )
                                    np.save(paths["halo_field"], np.asarray(halo_delta, dtype=np.float32))

                                root_n_halos = int(np.asarray(length).size)
                            except Exception as exc:
                                root_err = repr(exc)

                        root_err = comm.bcast(root_err, root=0)
                        root_n_halos = comm.bcast(root_n_halos, root=0)
                        if root_err is not None:
                            raise RuntimeError(f"Halos stage failed on root for entry={entry_label}: {root_err}")
                        if root_n_halos is not None:
                            rec["n_halos"] = int(root_n_halos)

                        comm.Barrier()

                        if rank == 0 and bool(args.cleanup_temp_displacement) and paths["dis"].is_file():
                            paths["dis"].unlink()
                        comm.Barrier()

                except Exception as exc:
                    rec["status"] = "failed"
                    rec["error"] = repr(exc)

                elapsed_local = float(time.perf_counter() - t0)
                rec["runtime_seconds"] = (
                    float(comm.allreduce(elapsed_local, op=MPI.MAX)) if MPI is not None else float(elapsed_local)
                )
                rec["runtime_hms"] = format_hms(float(rec["runtime_seconds"]))

                if rank == 0:
                    manifest["halo_records"].append(rec)
                    write_manifest(manifest_path, manifest)

        # HOD stage
        if str(args.stage) in {"hod", "all"}:
            mods = import_field_level_modules(Path(args.field_level_sbi_root))
            JaxHodForwardModel = mods["JaxHodForwardModel"]
            plot_utils = mods["plot_utils"]
            fsbi_utils = mods["fsbi_utils"]

            hod_model = None
            box_for_plots = None
            if rank == 0:
                hod_model = JaxHodForwardModel(
                    hod_params=hod_params,
                    box_size=float(args.boxsize),
                    grid_res=int(args.res),
                    H_max=int(h_max),
                    per_host_cap=int(per_host_cap),
                    N_max_sat=int(n_max_sat),
                    verbose=False,
                )
                box_for_plots = fsbi_utils.Power_Spectrum_Sampler(
                    {"box_size": float(args.boxsize), "grid_res": int(args.res), "h": float(box_params.get("h", 0.6711)), "dim": 3},
                    device="cpu",
                )

            if rank == 0:
                def _append_hod_record(rec: dict[str, Any], t0: float) -> None:
                    rec["runtime_seconds"] = float(time.perf_counter() - t0)
                    rec["runtime_hms"] = format_hms(float(rec["runtime_seconds"]))
                    manifest["hod_records"].append(rec)
                    write_manifest(manifest_path, manifest)

                pending: list[dict[str, Any]] = []
                batch_size = int(args.hod_batch_size)
                manifest["hod_batch"]["configured_batch_size"] = int(batch_size)

                for i, entry in enumerate(entries, start=1):
                    entry_kind = str(entry.get("kind", "sample"))
                    entry_stem = str(entry.get("entry_stem"))
                    entry_label = str(entry.get("entry_label", entry_stem))
                    sample_id = entry.get("sample_id")
                    paths = entry_output_paths(out_sim_dir, entry_stem)

                    rec: dict[str, Any] = {
                        "entry_kind": entry_kind,
                        "entry_stem": entry_stem,
                        "entry_label": entry_label,
                        "sample_id": None if sample_id is None else int(sample_id),
                        "status": "completed",
                        "reused": False,
                        "n_halos_input": None,
                        "n_halos_used": None,
                        "hod_batch_size_used": None,
                        "runtime_seconds": 0.0,
                    }
                    t0 = time.perf_counter()
                    log_progress(f"[{sim_name}] [hod {i}/{len(entries)}] entry={entry_label}", rank=rank)

                    try:
                        if bool(args.reuse_existing) and paths["hod_field_raw"].is_file() and paths["hod_field_obsnorm"].is_file():
                            rec["status"] = "skipped_reuse"
                            rec["reused"] = True
                            rec["hod_batch_size_used"] = 0
                            _append_hod_record(rec, t0)
                            continue

                        if not paths["fof"].is_file():
                            raise FileNotFoundError(f"Missing FoF catalog for HOD stage: {paths['fof']}")

                        cat, n_halos_input, n_halos_used = build_hod_input_from_fof(paths["fof"], int(h_max))
                        rec["n_halos_input"] = int(n_halos_input)
                        rec["n_halos_used"] = int(n_halos_used)
                        pending.append({"rec": rec, "paths": paths, "cat": cat, "t0": t0})
                    except Exception as exc:
                        rec["status"] = "failed"
                        rec["error"] = repr(exc)
                        _append_hod_record(rec, t0)

                for bstart in range(0, len(pending), batch_size):
                    batch_items = pending[bstart : bstart + batch_size]
                    if not batch_items:
                        continue
                    cats = np.stack([item["cat"] for item in batch_items], axis=0).astype(np.float32)
                    manifest["hod_batch"]["num_batch_calls"] = int(manifest["hod_batch"]["num_batch_calls"]) + 1
                    try:
                        delta_batch = np.asarray(hod_model(cats), dtype=np.float32)
                        if delta_batch.shape[0] != len(batch_items):
                            raise RuntimeError(
                                f"HOD batch output size mismatch: got {delta_batch.shape[0]} for {len(batch_items)} inputs"
                            )

                        for j, item in enumerate(batch_items):
                            rec = item["rec"]
                            paths = item["paths"]
                            t0 = float(item["t0"])
                            try:
                                delta_hod = ensure_finite_field(
                                    delta_batch[j], label=f"hod_field_raw entry={rec['entry_label']}"
                                )
                                delta_hod_obs = ensure_finite_field(
                                    delta_hod / np.float32(sigma_h),
                                    label=f"hod_field_obsnorm entry={rec['entry_label']}",
                                )
                                np.save(paths["hod_field_raw"], np.asarray(delta_hod, dtype=np.float32))
                                np.save(paths["hod_field_obsnorm"], np.asarray(delta_hod_obs, dtype=np.float32))
                                rec["hod_batch_size_used"] = int(len(batch_items))
                            except Exception as exc:
                                rec["status"] = "failed"
                                rec["error"] = repr(exc)
                            _append_hod_record(rec, t0)
                    except Exception as batch_exc:
                        log_progress(
                            f"[{sim_name}] HOD micro-batch failed (size={len(batch_items)}), retrying entry-by-entry: {batch_exc!r}",
                            rank=rank,
                        )
                        for item in batch_items:
                            rec = item["rec"]
                            paths = item["paths"]
                            t0 = float(item["t0"])
                            manifest["hod_batch"]["num_batch_fallback_calls"] = (
                                int(manifest["hod_batch"]["num_batch_fallback_calls"]) + 1
                            )
                            try:
                                delta_hod = np.asarray(hod_model(item["cat"][np.newaxis])[0], dtype=np.float32)
                                delta_hod = ensure_finite_field(
                                    delta_hod, label=f"hod_field_raw entry={rec['entry_label']} fallback"
                                )
                                delta_hod_obs = ensure_finite_field(
                                    delta_hod / np.float32(sigma_h),
                                    label=f"hod_field_obsnorm entry={rec['entry_label']} fallback",
                                )
                                np.save(paths["hod_field_raw"], np.asarray(delta_hod, dtype=np.float32))
                                np.save(paths["hod_field_obsnorm"], np.asarray(delta_hod_obs, dtype=np.float32))
                                rec["hod_batch_size_used"] = 1
                            except Exception as exc:
                                rec["status"] = "failed"
                                rec["error"] = f"{repr(exc)} | batch_error={repr(batch_exc)}"
                            _append_hod_record(rec, t0)
            comm.Barrier()

            if rank == 0 and bool(args.run_plot_analysis):
                t_plot = time.perf_counter()
                try:
                    obs_path = sim_dir / "delta_gal.npy"
                    if not obs_path.is_file():
                        obs_path = sim_dir / "delta_obs.npy"
                    if not obs_path.is_file():
                        raise FileNotFoundError(f"Observed field not found (checked delta_gal.npy and delta_obs.npy) in {sim_dir}")

                    sample_entries = [e for e in entries if str(e.get("kind")) == "sample"]
                    if not sample_entries:
                        raise RuntimeError("No sample_* entries available for sample-vs-truth PPC comparison.")

                    sample_fields: list[np.ndarray] = []
                    for e in sample_entries:
                        stem = str(e["entry_stem"])
                        p = entry_output_paths(out_sim_dir, stem)["hod_field_raw"]
                        if p.is_file():
                            sample_fields.append(np.asarray(np.load(p), dtype=np.float32))

                    if not sample_fields:
                        raise RuntimeError("No sample HOD raw fields available for plot_samples_analysis")

                    truth_entries = [e for e in entries if str(e.get("kind")) == "delta_ic"]
                    if not truth_entries:
                        manifest["plot_analysis"] = {
                            "status": "skipped",
                            "reason": "delta_ic entry not selected (set INCLUDE_DELTA_IC=1 for apples-to-apples truth comparison)",
                            "runtime_seconds": float(time.perf_counter() - t_plot),
                            "runtime_hms": format_hms(float(time.perf_counter() - t_plot)),
                        }
                        write_manifest(manifest_path, manifest)
                        continue
                    truth_stem = str(truth_entries[0]["entry_stem"])
                    truth_path = entry_output_paths(out_sim_dir, truth_stem)["hod_field_raw"]
                    if not truth_path.is_file():
                        manifest["plot_analysis"] = {
                            "status": "skipped",
                            "reason": f"truth resimulated HOD field missing: {truth_path}",
                            "runtime_seconds": float(time.perf_counter() - t_plot),
                            "runtime_hms": format_hms(float(time.perf_counter() - t_plot)),
                        }
                        write_manifest(manifest_path, manifest)
                        continue

                    samples = np.stack(sample_fields, axis=0).astype(np.float32)
                    z_map_raw = np.asarray(samples.mean(axis=0), dtype=np.float32)
                    delta_truth = np.asarray(np.load(truth_path), dtype=np.float32)
                    delta_obs = np.asarray(np.load(obs_path), dtype=np.float32)

                    plot_root = Path(args.plot_output_root).resolve() / sim_name
                    plot_root.mkdir(parents=True, exist_ok=True)

                    plot_utils.plot_samples_analysis(
                        delta_truth,
                        delta_obs,
                        samples,
                        z_map_raw,
                        box_for_plots,
                        cosmo_params=cosmo_params,
                        MAS=None,
                        save_dir=str(plot_root),
                        run_name=f"ppc_{sim_name}",
                        n_workers=args.plot_n_workers,
                    )
                    manifest["plot_analysis"] = {
                        "status": "completed",
                        "runtime_seconds": float(time.perf_counter() - t_plot),
                        "runtime_hms": format_hms(float(time.perf_counter() - t_plot)),
                        "plot_dir": str(plot_root),
                        "n_samples": int(samples.shape[0]),
                        "observed_field_path": str(obs_path),
                        "truth_field_path": str(truth_path),
                        "z_map_source": "sample_mean_proxy",
                    }
                except Exception as exc:
                    manifest["plot_analysis"] = {
                        "status": "failed",
                        "error": repr(exc),
                        "runtime_seconds": float(time.perf_counter() - t_plot),
                        "runtime_hms": format_hms(float(time.perf_counter() - t_plot)),
                    }
                write_manifest(manifest_path, manifest)

        if rank == 0:
            n_f = sum(1 for r in manifest["forward_records"] if r.get("status") == "failed")
            n_h = sum(1 for r in manifest["halo_records"] if r.get("status") == "failed")
            n_g = sum(1 for r in manifest["hod_records"] if r.get("status") == "failed")
            log_progress(
                f"[{sim_name}] manifest saved: {manifest_path} | failures forward/halos/hod = {n_f}/{n_h}/{n_g}",
                rank=rank,
            )

    if rank == 0:
        total = float(time.perf_counter() - t_total)
        log_progress(f"All done in {format_hms(total)}", rank=rank)


if __name__ == "__main__":
    main()
