#!/usr/bin/env python3
"""
Generate emulator+halo+HOD datasets in numbered simulation folders.

This script supports four execution stages:
1) `emulator`: run IC->emulator forward only and save
   - delta_linear_z0.npy
   - emu_delta_z0.npy
   - emu_dis_z0.npy (temporary intermediate for halo generation)
2) `halos`: build FoF catalogs from precomputed emu_dis_z0.npy and save
   - fof_catalog.npz
   Then optionally remove emu_dis_z0.npy (default: enabled).
3) `hod`: populate FoF halos with galaxies and save
   - hod_catalog.npz
4) `all`: run emulator then halos then HOD in one process.

Each simulation is stored in a numbered folder:
  <dataset_dir>/<sim_index>/
with seeds equal to simulation indices.

Examples:
  # Stage 1 (GPU env with emulator deps):
  python scripts/generate_dataset.py --stage emulator

  # Stage 2 (nbodykit env):
  python scripts/generate_dataset.py --stage halos --dataset-name box250_npart128_res64_nsims100

  # Stage 3 (HOD env with pyhod dependency):
  python scripts/generate_dataset.py --stage hod --dataset-name box250_npart128_res64_nsims100

  # Single-stage run (requires environment with emulator + nbodykit + pyhod deps):
  python scripts/generate_dataset.py --stage all --n-sims 10
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import time
from pathlib import Path
from typing import Any

import numpy as np

from halos import (
    gather_halos_to_root,
    load_local_positions_from_displacement,
    particle_mass_msun_h,
    run_fof,
)
from utils import add_import_paths, format_hms, parse_ndiv

REPO_ROOT = Path(__file__).resolve().parent.parent
add_import_paths(repo_root=REPO_ROOT)

try:
    from core import QUIJOTE_FIDUCIAL, run_lpt_emulator_pipeline
except ImportError as exc:  # pragma: no cover - optional dependency at import time
    QUIJOTE_FIDUCIAL = {"Omega_m": 0.3175}
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


class _SerialComm:
    """Minimal MPI-like communicator for non-MPI single-process stages."""

    rank = 0
    size = 1

    def Barrier(self) -> None:
        """No-op barrier for serial execution."""
        return None

    def gather(self, value: Any, root: int = 0) -> list[Any] | None:
        """Return single-item gather result on root."""
        return [value] if int(root) == 0 else None

    def bcast(self, value: Any, root: int = 0) -> Any:
        """Return broadcast value unchanged in serial execution."""
        return value

    def allreduce(self, value: Any, op: Any = None) -> Any:
        """Return value unchanged in serial execution."""
        return value


HOD_DEFAULT_LOGMMIN = 13.0
HOD_DEFAULT_SIGMALOGM = 0.2
HOD_DEFAULT_LOGM0 = 13.1
HOD_DEFAULT_ALPHA = 0.75
HOD_DEFAULT_LOGM1 = 14.3
HOD_DEFAULT_OMEGA_M = 0.3089
HOD_DEFAULT_OMEGA_L = 0.6911
HOD_DEFAULT_H = 0.6774
HOD_DEFAULT_Z = 0.0
HOD_DEFAULT_M_STAR = 1.670992e13
HOD_DEFAULT_DELTAC = 200
HOD_DEFAULT_ZERR = 0.0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for staged dataset generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate emulator dataset folders with IC/final density fields, FoF halo catalogs, and HOD galaxies. "
            "Supports staged execution for split environments."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=("emulator", "halos", "hod", "all"),
        default="all",
        help="Execution stage.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/gpfs/scratch1/shared/osavchenko/emulator_datasets"),
        help="Root directory containing dataset folders.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            "Dataset folder name under --dataset-root. If omitted, auto-generates "
            "'box{BOX}_npart{NPART}_res{RES}_nsims{NSIMS}' and appends _vN on collisions."
        ),
    )
    parser.add_argument("--n-sims", type=int, default=100, help="Number of simulations to process.")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed/index.")
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip already completed simulations for the selected stage.",
    )
    parser.add_argument(
        "--cleanup-temp-displacement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove temporary emu_dis_z0.npy after successful halo generation.",
    )

    # Emulator defaults chosen by user.
    parser.add_argument("--n-part", type=int, default=128, help="Particle/displacement lattice resolution.")
    parser.add_argument("--res", type=int, default=64, help="Output density grid resolution.")
    parser.add_argument("--boxsize", type=float, default=250.0, help="Box size in Mpc/h.")
    parser.add_argument(
        "--ndiv",
        type=parse_ndiv,
        default=(1, 1, 1),
        help="Emulator subbox partition (one int or tuple).",
    )
    parser.add_argument("--emu-precision", choices=("f16", "f32"), default="f16", help="Emulator precision.")
    parser.add_argument(
        "--num-sims-internal",
        type=int,
        default=1,
        help="Internal repeat count inside one emulator call (last result kept).",
    )
    parser.add_argument("--mas-worder", type=int, choices=(2, 3, 4), default=2, help="MAS order.")
    parser.add_argument(
        "--deconvolve-mas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deconvolve MAS in generated density fields.",
    )
    parser.add_argument("--class-n-modes", type=int, default=4000, help="CLASS table sampling count.")
    parser.add_argument(
        "--delta-upsample-method",
        choices=("mode_inject", "fourier", "linear"),
        default="mode_inject",
        help="Method used if IC resizing to n_part is needed.",
    )
    parser.add_argument(
        "--downsample-method",
        choices=("fourier_truncation", "block_average", "gaussian"),
        default="fourier_truncation",
        help="Method used if IC downsampling is needed.",
    )
    parser.add_argument(
        "--downsample-gaussian-sigma-mpc-h",
        type=float,
        default=None,
        help="Gaussian smoothing scale for downsample_method=gaussian.",
    )
    parser.add_argument("--downsample-threads", type=int, default=1, help="Threads for Gaussian downsampling.")

    # FoF defaults.
    parser.add_argument("--omega-m", type=float, default=float(QUIJOTE_FIDUCIAL["Omega_m"]), help="Omega_m for mass.")
    parser.add_argument(
        "--linking-length",
        type=float,
        default=0.2,
        help="FoF linking length in units of mean inter-particle spacing.",
    )
    parser.add_argument(
        "--absolute-linking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat linking length as absolute Mpc/h.",
    )
    parser.add_argument("--nmin", type=int, default=20, help="Minimum halo particle count.")
    parser.add_argument("--catalog-file", type=str, default="fof_catalog.npz", help="Halo catalog filename.")

    # HOD defaults adapted from Pinocchio run_hod.py (Zheng+2007).
    parser.add_argument("--hod-catalog-file", type=str, default="hod_catalog.npz", help="HOD catalog filename.")
    parser.add_argument("--hod-logMmin", type=float, default=HOD_DEFAULT_LOGMMIN, help="HOD logMmin.")
    parser.add_argument("--hod-sigmalogM", type=float, default=HOD_DEFAULT_SIGMALOGM, help="HOD sigmalogM.")
    parser.add_argument("--hod-logM0", type=float, default=HOD_DEFAULT_LOGM0, help="HOD logM0.")
    parser.add_argument("--hod-alpha", type=float, default=HOD_DEFAULT_ALPHA, help="HOD alpha.")
    parser.add_argument("--hod-logM1", type=float, default=HOD_DEFAULT_LOGM1, help="HOD logM1.")
    parser.add_argument("--hod-method", choices=("NFW", "NFWv"), default="NFW", help="HOD position sampling method.")
    parser.add_argument("--hod-rsd", type=int, choices=(0, 1, 2, 3), default=0, help="RSD mode for HOD catalog.")
    parser.add_argument(
        "--hod-seed-policy",
        choices=("sim_index", "fixed", "none"),
        default="sim_index",
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

    args = parser.parse_args()
    if args.n_sims < 1:
        parser.error("--n-sims must be >= 1")
    if args.seed_start < 0:
        parser.error("--seed-start must be >= 0")
    if args.n_part < 1 or args.res < 1:
        parser.error("--n-part and --res must be >= 1")
    if args.num_sims_internal < 1:
        parser.error("--num-sims-internal must be >= 1")
    if args.hod_seed < 0:
        parser.error("--hod-seed must be >= 0")
    return args


def format_box_for_name(boxsize: float) -> str:
    """Format box size token for dataset name."""
    box = float(boxsize)
    if box.is_integer():
        return str(int(box))
    return f"{box:g}".replace(".", "p")


def default_dataset_base_name(*, boxsize: float, n_part: int, res: int, n_sims: int) -> str:
    """Build default dataset base name."""
    return f"box{format_box_for_name(boxsize)}_npart{int(n_part)}_res{int(res)}_nsims{int(n_sims)}"


def find_latest_existing_dataset_dir(dataset_root: Path, base: str) -> Path | None:
    """Find latest existing dataset dir among base and base_vN variants."""
    candidates: list[tuple[int, Path]] = []
    base_dir = dataset_root / base
    if base_dir.is_dir():
        candidates.append((1, base_dir))

    patt = re.compile(rf"^{re.escape(base)}_v(\d+)$")
    for p in dataset_root.iterdir():
        if not p.is_dir():
            continue
        m = patt.match(p.name)
        if m is not None:
            candidates.append((int(m.group(1)), p))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def allocate_unique_dataset_dir(dataset_root: Path, base: str) -> Path:
    """Allocate a unique dataset directory path using base, base_v2, base_v3, ..."""
    candidate = dataset_root / base
    if not candidate.exists():
        return candidate
    version = 2
    while True:
        candidate = dataset_root / f"{base}_v{version}"
        if not candidate.exists():
            return candidate
        version += 1


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    """Resolve dataset directory path from explicit or auto naming rules."""
    dataset_root = Path(args.dataset_root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    if args.dataset_name is not None:
        return dataset_root / str(args.dataset_name)

    base = default_dataset_base_name(
        boxsize=float(args.boxsize),
        n_part=int(args.n_part),
        res=int(args.res),
        n_sims=int(args.n_sims),
    )
    if str(args.stage) in {"halos", "hod"}:
        latest = find_latest_existing_dataset_dir(dataset_root=dataset_root, base=base)
        if latest is None:
            raise FileNotFoundError(
                "No existing dataset folder found for halos/hod stage with auto name. "
                f"Expected one of: {dataset_root / base} or {dataset_root / (base + '_vN')}. "
                "Pass --dataset-name explicitly, or run emulator stage first."
            )
        return latest
    return allocate_unique_dataset_dir(dataset_root=dataset_root, base=base)


def jsonable(value: Any) -> Any:
    """Convert common runtime objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def sim_paths(sim_dir: Path, catalog_file: str, hod_catalog_file: str) -> dict[str, Path]:
    """Build canonical file paths for one simulation folder."""
    return {
        "ic": sim_dir / "delta_linear_z0.npy",
        "final": sim_dir / "emu_delta_z0.npy",
        "displacement_tmp": sim_dir / "emu_dis_z0.npy",
        "catalog": sim_dir / str(catalog_file),
        "hod_catalog": sim_dir / str(hod_catalog_file),
    }


def run_emulator_stage_for_sim(sim_index: int, sim_dir: Path, args: argparse.Namespace) -> tuple[str, float]:
    """Run emulator stage for one simulation index and save required fields."""
    paths = sim_paths(sim_dir=sim_dir, catalog_file=args.catalog_file, hod_catalog_file=args.hod_catalog_file)
    if bool(args.reuse_existing) and paths["ic"].is_file() and paths["final"].is_file() and paths["displacement_tmp"].is_file():
        return "skipped_reuse", 0.0

    sim_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    pipeline = run_lpt_emulator_pipeline(
        seed=int(sim_index),
        n_part=int(args.n_part),
        density_res=int(args.res),
        boxsize=float(args.boxsize),
        ndiv=tuple(args.ndiv),
        emu_precision=str(args.emu_precision),
        num_sims=int(args.num_sims_internal),
        mas_worder=int(args.mas_worder),
        deconvolve_mas=bool(args.deconvolve_mas),
        z_target=0.0,
        compute_vel=False,
        delta_upsample_method=str(args.delta_upsample_method),
        downsample_method=str(args.downsample_method),
        downsample_gaussian_sigma_mpc_h=args.downsample_gaussian_sigma_mpc_h,
        downsample_threads=int(args.downsample_threads),
        class_n_modes=int(args.class_n_modes),
        quijote_fiducial=QUIJOTE_FIDUCIAL,
        output_dir=None,
        save_outputs=False,
        show_progress=True,
        progress_desc_prefix=f"Dataset sim {sim_index} subboxes",
    )

    np.save(paths["ic"], np.asarray(pipeline["ics_grf"], dtype=np.float32))
    np.save(paths["final"], np.asarray(pipeline["delta_emu"], dtype=np.float32))
    np.save(paths["displacement_tmp"], np.asarray(pipeline["psi_emu"], dtype=np.float32))
    return "completed", float(time.perf_counter() - t0)


def ensure_fof_dependencies() -> None:
    """Validate availability of MPI + nbodykit dependencies for FoF stage."""
    if _MPI_IMPORT_ERROR is not None or MPI is None:
        raise ImportError(
            "FoF stage requires mpi4py. Activate .venv_nbodykit or equivalent environment."
        ) from _MPI_IMPORT_ERROR
    if _NBODYKIT_IMPORT_ERROR is not None or FOF is None or fof_catalog is None or ArrayCatalog is None:
        raise ImportError(
            "FoF stage requires nbodykit. Activate .venv_nbodykit or equivalent environment."
        ) from _NBODYKIT_IMPORT_ERROR


def ensure_emulator_dependencies() -> None:
    """Validate availability of emulator pipeline dependencies."""
    if _CORE_IMPORT_ERROR is not None or run_lpt_emulator_pipeline is None:
        raise ImportError(
            "Emulator stage requires core emulator dependencies (JAX, DISCO-DJ, jax_nbody_emulator). "
            "Activate the emulator environment (e.g. .venv)."
        ) from _CORE_IMPORT_ERROR


def ensure_hod_dependencies() -> None:
    """Validate availability of pyhod dependency for HOD stage."""
    if _PYHOD_IMPORT_ERROR is not None or HODF is None:
        raise ImportError(
            "HOD stage requires pyhod (pyhod.hodfunc). Activate an environment with pyhod installed."
        ) from _PYHOD_IMPORT_ERROR


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


def _hod_seed_for_sim(sim_index: int, args: argparse.Namespace) -> int | None:
    """Resolve deterministic/random HOD seed for one simulation index."""
    policy = str(args.hod_seed_policy)
    if policy == "sim_index":
        return int(sim_index)
    if policy == "fixed":
        return int(args.hod_seed)
    if policy == "none":
        return None
    raise ValueError(f"Unknown hod_seed_policy={policy!r}")


def _fof_catalog_to_hod_input(catalog_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Convert emulator FoF catalog.npz into pyhod input array and halo masses."""
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


def run_hod_stage_for_sim(
    sim_index: int,
    sim_dir: Path,
    args: argparse.Namespace,
    comm: Any,
) -> tuple[str, float]:
    """Run HOD galaxy population for one simulation and save `hod_catalog.npz` on root."""
    paths = sim_paths(sim_dir=sim_dir, catalog_file=args.catalog_file, hod_catalog_file=args.hod_catalog_file)
    rank = int(comm.rank)
    t0 = time.perf_counter()

    if bool(args.reuse_existing) and paths["hod_catalog"].is_file():
        comm.Barrier()
        return "skipped_reuse", 0.0

    has_catalog = bool(paths["catalog"].is_file()) if rank == 0 else None
    has_catalog = bool(comm.bcast(has_catalog, root=0))
    if not has_catalog:
        raise FileNotFoundError(
            f"Missing FoF catalog for HOD stage: {paths['catalog']}. "
            "Run stage=halos first (or stage=all)."
        )

    root_error: str | None = None
    if rank == 0:
        try:
            halos, halo_masses = _fof_catalog_to_hod_input(paths["catalog"])
            extra = parse_hod_extra_args(str(args.hod_extra_args))

            logmmin = float(extra.get("logMmin", args.hod_logMmin))
            sigmalogm = float(extra.get("sigmalogM", args.hod_sigmalogM))
            logm0 = float(extra.get("logM0", args.hod_logM0))
            alpha = float(extra.get("alpha", args.hod_alpha))
            logm1 = float(extra.get("logM1", args.hod_logM1))

            hod_seed = _hod_seed_for_sim(sim_index=sim_index, args=args)
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
                "box": float(extra.get("box", args.boxsize)),
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
                BoxSize=np.array([float(args.boxsize), float(args.boxsize), float(args.boxsize)], dtype=np.float64),
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
        except Exception as exc:
            root_error = repr(exc)

    root_error = comm.bcast(root_error, root=0)
    if root_error is not None:
        raise RuntimeError(f"HOD stage failed on root: {root_error}")

    comm.Barrier()
    elapsed_local = float(time.perf_counter() - t0)
    if MPI is not None:
        elapsed = float(comm.allreduce(elapsed_local, op=MPI.MAX))
    else:
        elapsed = elapsed_local
    return "completed", elapsed


def run_halos_stage_for_sim(
    sim_index: int,
    sim_dir: Path,
    args: argparse.Namespace,
    comm: Any,
) -> tuple[str, float]:
    """Run distributed FoF stage for one simulation index and write halo catalog on root."""
    paths = sim_paths(sim_dir=sim_dir, catalog_file=args.catalog_file, hod_catalog_file=args.hod_catalog_file)
    rank = int(comm.rank)
    size = int(comm.size)

    if bool(args.reuse_existing) and paths["catalog"].is_file():
        if rank == 0 and bool(args.cleanup_temp_displacement) and paths["displacement_tmp"].is_file():
            paths["displacement_tmp"].unlink()
        comm.Barrier()
        return "skipped_reuse", 0.0

    has_displacement = bool(paths["displacement_tmp"].is_file())
    has_displacement = bool(comm.bcast(has_displacement if rank == 0 else None, root=0))
    if not has_displacement:
        raise FileNotFoundError(
            f"Missing temporary displacement for halos stage: {paths['displacement_tmp']}. "
            "Run stage=emulator first (or stage=all), or disable cleanup in previous halos runs."
        )

    t0 = time.perf_counter()
    local_pos = load_local_positions_from_displacement(
        displacement_file=paths["displacement_tmp"],
        boxsize=float(args.boxsize),
        n_part=int(args.n_part),
        rank=rank,
        size=size,
    )
    cm_pos_local, length_local = run_fof(
        local_positions=local_pos,
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
        cm_pos_local=cm_pos_local,
        length_local=length_local,
        comm=comm,
    )
    if rank == 0:
        m_particle = particle_mass_msun_h(
            omega_m=float(args.omega_m),
            boxsize=float(args.boxsize),
            n_part=int(args.n_part),
        )
        masses = np.asarray(length, dtype=np.float64) * float(m_particle)
        np.savez_compressed(
            paths["catalog"],
            CMPosition=np.asarray(cm_pos, dtype=np.float32),
            Npart=np.asarray(length, dtype=np.int32),
            Mass=np.asarray(masses, dtype=np.float64),
            BoxSize=np.array([float(args.boxsize), float(args.boxsize), float(args.boxsize)], dtype=np.float64),
            NpartPerDim=np.int32(int(args.n_part)),
            LinkingLength=np.float64(float(args.linking_length)),
            AbsoluteLinking=np.bool_(bool(args.absolute_linking)),
            Nmin=np.int32(int(args.nmin)),
        )
    comm.Barrier()
    if rank == 0 and bool(args.cleanup_temp_displacement) and paths["displacement_tmp"].is_file():
        paths["displacement_tmp"].unlink()
    comm.Barrier()
    elapsed_local = float(time.perf_counter() - t0)
    elapsed = float(comm.allreduce(elapsed_local, op=MPI.MAX))
    return "completed", elapsed


def write_manifest(
    dataset_dir: Path,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> None:
    """Write dataset-level manifest with runtime status for all simulation indices."""
    completed = [int(r["index"]) for r in records if str(r["status"]) != "failed"]
    failed = [
        {"index": int(r["index"]), "seed": int(r["seed"]), "error": str(r.get("error", ""))}
        for r in records
        if str(r["status"]) == "failed"
    ]
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": str(dataset_dir),
        "stage": str(args.stage),
        "n_sims_requested": int(args.n_sims),
        "seed_start": int(args.seed_start),
        "indices": [int(args.seed_start + i) for i in range(int(args.n_sims))],
        "config": jsonable(vars(args)),
        "completed_indices": completed,
        "failed": failed,
        "records": records,
    }
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    """Run staged dataset generation and persist required outputs/manifest."""
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if str(args.stage) in {"emulator", "all"}:
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

    indices = [int(args.seed_start + i) for i in range(int(args.n_sims))]
    records: list[dict[str, Any]] = []
    t_total = time.perf_counter()

    if rank == 0:
        print(f"Dataset directory: {dataset_dir}")
        print(f"Stage: {args.stage}")
        if str(args.stage) in {"halos", "hod"}:
            print(f"MPI world size: {size}")
        print(f"Simulations requested: {len(indices)} (indices {indices[0]}..{indices[-1]})")

    for i, sim_index in enumerate(indices, start=1):
        sim_dir = dataset_dir / str(sim_index)
        paths = sim_paths(sim_dir=sim_dir, catalog_file=args.catalog_file, hod_catalog_file=args.hod_catalog_file)
        if rank == 0:
            print(f"[{i}/{len(indices)}] sim_index={sim_index} -> {sim_dir}")
            rec: dict[str, Any] = {
                "index": int(sim_index),
                "seed": int(sim_index),
                "sim_dir": str(sim_dir),
                "status": "completed",
                "emulator_stage_status": "not_requested",
                "halos_stage_status": "not_requested",
                "hod_stage_status": "not_requested",
                "emulator_runtime_seconds": 0.0,
                "halos_runtime_seconds": 0.0,
                "hod_runtime_seconds": 0.0,
                "total_runtime_seconds": 0.0,
                "files": {k: str(v) for k, v in paths.items()},
            }
        t_sim = time.perf_counter()

        try:
            if (
                str(args.stage) == "all"
                and bool(args.reuse_existing)
                and paths["ic"].is_file()
                and paths["final"].is_file()
                and paths["catalog"].is_file()
                and paths["hod_catalog"].is_file()
            ):
                if rank == 0:
                    rec["emulator_stage_status"] = "skipped_reuse"
                    rec["halos_stage_status"] = "skipped_reuse"
                    rec["hod_stage_status"] = "skipped_reuse"
                if bool(args.cleanup_temp_displacement) and paths["displacement_tmp"].is_file():
                    paths["displacement_tmp"].unlink()
                if rank == 0:
                    print("  all: skipped (reuse-existing complete simulation)")
            else:
                if str(args.stage) in {"emulator", "all"}:
                    e_status, e_time = run_emulator_stage_for_sim(sim_index=sim_index, sim_dir=sim_dir, args=args)
                    if rank == 0:
                        rec["emulator_stage_status"] = str(e_status)
                        rec["emulator_runtime_seconds"] = float(e_time)
                        if e_status == "completed":
                            print(f"  emulator: completed in {format_hms(float(e_time))} ({e_time:.2f} s)")
                        else:
                            print("  emulator: skipped (reuse-existing)")

                if str(args.stage) in {"halos", "all"}:
                    h_status, h_time = run_halos_stage_for_sim(
                        sim_index=sim_index,
                        sim_dir=sim_dir,
                        args=args,
                        comm=comm,
                    )
                    if rank == 0:
                        rec["halos_stage_status"] = str(h_status)
                        rec["halos_runtime_seconds"] = float(h_time)
                        if h_status == "completed":
                            print(f"  halos: completed in {format_hms(float(h_time))} ({h_time:.2f} s)")
                        else:
                            print("  halos: skipped (reuse-existing)")

                if str(args.stage) in {"hod", "all"}:
                    g_status, g_time = run_hod_stage_for_sim(
                        sim_index=sim_index,
                        sim_dir=sim_dir,
                        args=args,
                        comm=comm,
                    )
                    if rank == 0:
                        rec["hod_stage_status"] = str(g_status)
                        rec["hod_runtime_seconds"] = float(g_time)
                        if g_status == "completed":
                            print(f"  hod: completed in {format_hms(float(g_time))} ({g_time:.2f} s)")
                        else:
                            print("  hod: skipped (reuse-existing)")

        except Exception as exc:
            if rank == 0:
                rec["status"] = "failed"
                rec["error"] = repr(exc)
                print(f"  FAILED: {exc!r}")

        if rank == 0:
            rec["total_runtime_seconds"] = float(time.perf_counter() - t_sim)
            records.append(rec)
            write_manifest(dataset_dir=dataset_dir, args=args, records=records)
        if str(args.stage) in {"halos", "hod"}:
            comm.Barrier()

    if rank == 0:
        failed = [r for r in records if str(r["status"]) == "failed"]
        print("Dataset generation finished.")
        print(f"Total wall time: {format_hms(time.perf_counter() - t_total)}")
        print(f"Manifest: {dataset_dir / 'dataset_manifest.json'}")
        print(f"Completed: {len(records) - len(failed)} / {len(records)}")
        if failed:
            print(f"Failed: {len(failed)}")
            for r in failed:
                print(f"  - index={r['index']}: {r.get('error', 'unknown error')}")


if __name__ == "__main__":
    main()
