#!/usr/bin/env python3
"""
Apples-to-apples HMF comparison:
true Quijote IC -> emulator -> FoF vs posterior-resimulation FoF ensemble.

This script:
1. Loads true IC from a Quijote target .pt file (default key: delta_z127).
2. Runs the same forward emulator pipeline used in PPC to produce emulated final displacement.
3. Runs FoF halo finding on that emulated true displacement with the same FoF settings.
4. Compares this truth-emulated HMF to posterior-resimulation HMF samples from
   an existing aggregated `*_hmf_arrays.npz` file.

Intended to run under MPI (e.g. `srun`) in the nbodykit environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Allow this script to run under .venv_nbodykit while still seeing emulator deps.
_pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
_EXTRA_PATHS = [
    REPO_ROOT / "src",
    REPO_ROOT / ".venv" / "lib" / _pyver / "site-packages",
    Path.home() / ".local" / "lib" / _pyver / "site-packages",
]
for _p in _EXTRA_PATHS:
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime dependency
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from core import QUIJOTE_FIDUCIAL, run_lpt_emulator_pipeline
except ImportError as exc:  # pragma: no cover - runtime dependency
    QUIJOTE_FIDUCIAL = None
    run_lpt_emulator_pipeline = None
    _CORE_IMPORT_ERROR = exc
else:
    _CORE_IMPORT_ERROR = None

try:
    from utils import growth_D_approx, parse_ndiv
except ImportError as exc:  # pragma: no cover - runtime dependency
    growth_D_approx = None
    parse_ndiv = None
    _UTILS_IMPORT_ERROR = exc
else:
    _UTILS_IMPORT_ERROR = None

try:
    from halos import gather_halos_to_root, load_local_positions_from_displacement, particle_mass_msun_h, run_fof
except ImportError as exc:  # pragma: no cover - runtime dependency
    gather_halos_to_root = None
    load_local_positions_from_displacement = None
    particle_mass_msun_h = None
    run_fof = None
    _HALOS_IMPORT_ERROR = exc
else:
    _HALOS_IMPORT_ERROR = None

try:
    from mpi4py import MPI
    from nbodykit.algorithms.fof import FOF, fof_catalog
    from nbodykit.source.catalog import ArrayCatalog
except ImportError as exc:  # pragma: no cover - runtime dependency
    MPI = None
    FOF = None
    fof_catalog = None
    ArrayCatalog = None
    _NBodyKit_IMPORT_ERROR = exc
else:
    _NBodyKit_IMPORT_ERROR = None


def _z_file_tag(z_value: float) -> str:
    """Format redshift for stable file naming."""
    return f"z{z_value:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    """Parse CLI for apples-to-apples true-IC HMF comparison."""
    parser = argparse.ArgumentParser(
        description="Compare true-IC emulated FoF HMF against posterior-resimulation HMF ensemble.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--target-path",
        type=Path,
        default=Path("/home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt"),
        help="Path to Quijote .pt target file containing true IC field.",
    )
    parser.add_argument("--ic-key", type=str, default="delta_z127", help="IC key in --target-path.")
    parser.add_argument("--z-ic", type=float, default=127.0, help="Redshift of IC field.")
    parser.add_argument("--z-target", type=float, default=0.0, help="Target redshift for emulator + FoF.")
    parser.add_argument(
        "--ic-rescale-mode",
        choices=("growth", "none"),
        default="growth",
        help="How to rescale true IC amplitude before forward run.",
    )

    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("/gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD"),
        help="Posterior resimulation directory (used to derive default posterior arrays path).",
    )
    parser.add_argument(
        "--posterior-arrays-path",
        type=Path,
        default=None,
        help="Path to posterior `*_hmf_arrays.npz`. Default is derived from --samples-dir.",
    )
    parser.add_argument(
        "--posterior-prefix",
        type=str,
        default="posterior_predictive",
        help="Prefix used to derive default posterior arrays path.",
    )

    parser.add_argument("--n-part", type=int, default=512, help="Emulator/FoF particle grid resolution.")
    parser.add_argument("--res", type=int, default=128, help="Forward summary density resolution.")
    parser.add_argument("--boxsize", type=float, default=1000.0, help="Box size in Mpc/h.")
    parser.add_argument(
        "--ndiv",
        type=parse_ndiv if callable(parse_ndiv) else str,
        default=(4, 2, 2),
        help="Emulator subbox partition, e.g. 4,2,2.",
    )
    parser.add_argument("--emu-precision", choices=("f16", "f32"), default="f16", help="Emulator precision.")
    parser.add_argument("--num-sims", type=int, default=1, help="Internal emulator repeats (last run used).")
    parser.add_argument("--mas-worder", type=int, choices=(2, 3, 4), default=4, help="MAS order for meshing.")
    parser.add_argument(
        "--deconvolve-mas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deconvolve MAS kernel for pipeline fields.",
    )
    parser.add_argument(
        "--delta-upsample-method",
        choices=("mode_inject", "fourier", "linear"),
        default="mode_inject",
        help="IC resizing method when IC grid != n_part.",
    )
    parser.add_argument(
        "--downsample-method",
        choices=("fourier_truncation", "block_average", "gaussian"),
        default="fourier_truncation",
        help="Downsampling method when IC grid > n_part.",
    )
    parser.add_argument(
        "--downsample-gaussian-sigma-mpc-h",
        type=float,
        default=None,
        help="Gaussian sigma [Mpc/h] for downsample_method=gaussian.",
    )
    parser.add_argument("--downsample-threads", type=int, default=1, help="Threads for gaussian downsampling.")
    parser.add_argument("--class-n-modes", type=int, default=4000, help="CLASS table sampling for mode_inject.")

    parser.add_argument("--omega-m", type=float, default=0.3175, help="Omega_m for particle-mass conversion.")
    parser.add_argument("--fof-linking-length", type=float, default=0.2, help="FoF linking length parameter.")
    parser.add_argument(
        "--fof-absolute-linking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat linking length as absolute Mpc/h instead of b relative to mean spacing.",
    )
    parser.add_argument("--fof-nmin", type=int, default=20, help="FoF minimum particles per halo.")

    parser.add_argument("--title", type=str, default="Friend-of-Friends mass", help="Top panel title.")
    parser.add_argument("--ratio-ymin", type=float, default=0.6, help="Lower y-limit for ratio panel.")
    parser.add_argument("--ratio-ymax", type=float, default=1.4, help="Upper y-limit for ratio panel.")
    parser.add_argument("--dpi", type=int, default=180, help="Output DPI.")
    parser.add_argument(
        "--show-tinker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show Tinker curve from posterior arrays when available.",
    )
    parser.add_argument(
        "--show-original-truth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show original posterior-arrays truth curve (if present) as reference.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/osavchenko/gaussian_npe/paper_plots_scripts/260303_224627_net_IsotropicD"),
        help="Output directory for comparison products.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="posterior_predictive_hmf_apples_to_apples_true_ic_comparison",
        help="Prefix for saved comparison files.",
    )
    parser.add_argument(
        "--truth-forward-dir",
        type=Path,
        default=None,
        help="Directory to store true-IC forward outputs. Default: <output-dir>/true_ic_forward_apples_to_apples.",
    )
    parser.add_argument(
        "--reuse-existing-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing true-IC emulated displacement if present.",
    )

    args = parser.parse_args()

    if args.n_part < 1 or args.res < 1:
        parser.error("--n-part and --res must be >= 1")
    if args.fof_nmin < 1:
        parser.error("--fof-nmin must be >= 1")
    if args.num_sims < 1:
        parser.error("--num-sims must be >= 1")
    if args.class_n_modes < 2:
        parser.error("--class-n-modes must be >= 2")
    if args.ratio_ymax <= args.ratio_ymin:
        parser.error("--ratio-ymax must be greater than --ratio-ymin")

    return args


def default_posterior_arrays_path(samples_dir: Path, prefix: str) -> Path:
    """Infer default posterior HMF arrays path from samples directory name."""
    run_name = Path(samples_dir).resolve().name
    return (
        Path("/home/osavchenko/gaussian_npe/paper_plots_scripts")
        / run_name
        / "posterior_resimulation"
        / f"{prefix}_hmf_arrays.npz"
    )


def _optional_curve(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    """Return optional 1D array or None when key missing/empty."""
    if key not in data.files:
        return None
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.size == 0:
        return None
    return arr


def load_true_ic_field(target_path: Path, key: str) -> np.ndarray:
    """Load true IC field from Quijote target .pt dictionary."""
    if torch is None:
        raise ImportError("torch is required to read target .pt files") from _TORCH_IMPORT_ERROR
    p = Path(target_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Target file not found: {p}")
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {p}, got {type(payload)}")
    if key not in payload:
        raise KeyError(f"Missing key {key!r} in {p}")
    arr = np.asarray(payload[key], dtype=np.float32)
    if arr.ndim != 3 or len(set(arr.shape)) != 1:
        raise ValueError(f"True IC field must be cubic 3D, got shape={arr.shape}")
    return arr


def ic_growth_scale(z_ic: float, z_target: float) -> float:
    """Compute D(z_target)/D(z_ic) with Quijote fiducial cosmology."""
    omega_m = float(QUIJOTE_FIDUCIAL["Omega_m"])
    omega_b = float(QUIJOTE_FIDUCIAL["Omega_b"])
    class_params = {
        "Omega_cdm": omega_m - omega_b,
        "Omega_b": omega_b,
        "h": float(QUIJOTE_FIDUCIAL["h"]),
        "n_s": float(QUIJOTE_FIDUCIAL["n_s"]),
        "sigma8": float(QUIJOTE_FIDUCIAL["sigma8"]),
    }
    d_ic = float(growth_D_approx(class_params, float(z_ic)))
    d_target = float(growth_D_approx(class_params, float(z_target)))
    return float(d_target / d_ic)


def run_distributed_fof_lengths(
    displacement_path: Path,
    *,
    boxsize: float,
    n_part: int,
    linking_length: float,
    nmin: int,
    absolute_linking: bool,
) -> np.ndarray | None:
    """Run MPI-distributed FoF and return root-collected halo lengths."""
    if _NBodyKit_IMPORT_ERROR is not None:
        raise ImportError("nbodykit + mpi4py are required for FoF stage") from _NBodyKit_IMPORT_ERROR
    if _HALOS_IMPORT_ERROR is not None:
        raise ImportError("Failed to import FoF helpers from scripts/halos.py") from _HALOS_IMPORT_ERROR

    comm = MPI.COMM_WORLD
    rank = int(comm.rank)
    size = int(comm.size)

    local_positions = load_local_positions_from_displacement(
        displacement_file=Path(displacement_path),
        boxsize=float(boxsize),
        n_part=int(n_part),
        rank=rank,
        size=size,
    )
    cm_local, length_local = run_fof(
        local_positions=np.asarray(local_positions, dtype=np.float32),
        boxsize=float(boxsize),
        n_part=int(n_part),
        linking_length=float(linking_length),
        nmin=int(nmin),
        absolute_linking=bool(absolute_linking),
        comm=comm,
        fof_cls=FOF,
        fof_catalog_fn=fof_catalog,
        array_catalog_cls=ArrayCatalog,
    )
    _, length = gather_halos_to_root(
        cm_pos_local=np.asarray(cm_local, dtype=np.float32),
        length_local=np.asarray(length_local, dtype=np.int32),
        comm=comm,
    )
    if rank != 0:
        return None
    return np.asarray(length, dtype=np.int32)


def compute_hmf_curve(masses: np.ndarray, log_edges: np.ndarray, boxsize: float) -> np.ndarray:
    """Compute dn/dlog10M for fixed log10-mass binning."""
    masses = np.asarray(masses, dtype=np.float64)
    masses = masses[np.isfinite(masses) & (masses > 0.0)]
    counts, _ = np.histogram(np.log10(masses), bins=np.asarray(log_edges, dtype=np.float64))
    dlogm = float(log_edges[1] - log_edges[0])
    volume = float(boxsize) ** 3
    return counts.astype(np.float64) / (volume * dlogm)


def plot_apples_to_apples_hmf(
    *,
    m_centers: np.ndarray,
    posterior_hmf_samples: np.ndarray,
    truth_emulated_hmf: np.ndarray,
    mass_resolution_limit: float | None,
    out_path: Path,
    title: str,
    ratio_ymin: float,
    ratio_ymax: float,
    dpi: int,
    tinker_curve: np.ndarray | None,
    original_truth_curve: np.ndarray | None,
) -> None:
    """Plot posterior HMF ensemble vs true-IC-emulated HMF."""
    m_centers = np.asarray(m_centers, dtype=np.float64)
    hmf_samples = np.asarray(posterior_hmf_samples, dtype=np.float64)
    truth_curve = np.asarray(truth_emulated_hmf, dtype=np.float64)

    hmf_samples = np.where(hmf_samples > 0.0, hmf_samples, np.nan)
    truth_curve = np.where(truth_curve > 0.0, truth_curve, np.nan)

    mu = np.nanmean(hmf_samples, axis=0)
    sigma = np.nanstd(hmf_samples, axis=0)
    sigma1_lo = np.where((mu - sigma) > 0.0, mu - sigma, np.nan)
    sigma1_hi = np.where((mu + sigma) > 0.0, mu + sigma, np.nan)
    sigma2_lo = np.where((mu - 2.0 * sigma) > 0.0, mu - 2.0 * sigma, np.nan)
    sigma2_hi = np.where((mu + 2.0 * sigma) > 0.0, mu + 2.0 * sigma, np.nan)

    tinker = None if tinker_curve is None else np.asarray(tinker_curve, dtype=np.float64)
    original_truth = None if original_truth_curve is None else np.asarray(original_truth_curve, dtype=np.float64)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.4, 7.4),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )

    ax = axes[0]
    if mass_resolution_limit is not None and np.isfinite(float(mass_resolution_limit)) and float(mass_resolution_limit) > 0.0:
        ax.axvspan(
            float(np.min(m_centers)),
            float(mass_resolution_limit),
            color="#efe6e6",
            alpha=0.85,
            label="Mass resolution limit",
            zorder=0,
        )

    ax.fill_between(m_centers, sigma2_lo, sigma2_hi, color="0.7", alpha=0.45, lw=0.0, label=r"$\pm 2\sigma$", zorder=1)
    ax.fill_between(m_centers, sigma1_lo, sigma1_hi, color="0.55", alpha=0.55, lw=0.0, label=r"$\pm 1\sigma$", zorder=2)
    ax.plot(m_centers, mu, color="#5aa2c7", lw=2.0, label="Posterior Resimulations", zorder=3)
    if tinker is not None:
        ax.plot(m_centers, tinker, color="0.5", lw=1.8, ls=":", label="Tinker", zorder=4)
    if original_truth is not None:
        ax.plot(m_centers, original_truth, color="0.35", lw=1.4, ls="-.", label="Original truth reference", zorder=5)

    ax.plot(
        m_centers,
        truth_curve,
        color="k",
        ls="--",
        lw=1.8,
        marker="x",
        ms=6,
        mew=1.6,
        label="True IC -> Emulator -> FoF",
        zorder=6,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$dn/d\log_{10}M\ [(h/{\rm Mpc})^3]$")
    ax.set_title(str(title), fontsize=16)
    ax.grid(alpha=0.25, which="both", ls=":")
    ax.legend(framealpha=0.95, fontsize=8, loc="best")

    ax = axes[1]
    denom = np.where(truth_curve > 0.0, truth_curve, np.nan)
    ratio_mu = mu / denom
    ratio_sigma1_lo = sigma1_lo / denom
    ratio_sigma1_hi = sigma1_hi / denom
    ratio_sigma2_lo = sigma2_lo / denom
    ratio_sigma2_hi = sigma2_hi / denom

    if mass_resolution_limit is not None and np.isfinite(float(mass_resolution_limit)) and float(mass_resolution_limit) > 0.0:
        ax.axvspan(float(np.min(m_centers)), float(mass_resolution_limit), color="#efe6e6", alpha=0.85, zorder=0)
    ax.fill_between(m_centers, ratio_sigma2_lo, ratio_sigma2_hi, color="#7eb6d6", alpha=0.45, lw=0.0, zorder=1)
    ax.fill_between(m_centers, ratio_sigma1_lo, ratio_sigma1_hi, color="#5aa2c7", alpha=0.55, lw=0.0, zorder=2)
    ax.plot(m_centers, ratio_mu, color="#2d7ea8", lw=1.8, label="Posterior / true-emulated", zorder=3)
    ax.axhline(1.0, color="k", ls="--", lw=1.3, zorder=4)

    if tinker is not None:
        ax.plot(m_centers, tinker / denom, color="0.5", lw=1.4, ls=":", zorder=5)
    if original_truth is not None:
        ax.plot(m_centers, original_truth / denom, color="0.35", lw=1.2, ls="-.", zorder=6)

    ax.set_xscale("log")
    ax.set_ylim(float(ratio_ymin), float(ratio_ymax))
    ax.set_ylabel("Ratio")
    ax.set_xlabel(r"Mass [$M_\odot h^{-1}$]")
    ax.grid(alpha=0.25, which="both", ls=":")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def main() -> None:
    """Run apples-to-apples true-IC HMF comparison workflow."""
    args = parse_args()

    if _UTILS_IMPORT_ERROR is not None:
        raise ImportError("Failed to import utilities from scripts/utils.py") from _UTILS_IMPORT_ERROR
    if _CORE_IMPORT_ERROR is not None:
        raise ImportError("Failed to import emulator core from scripts/core.py") from _CORE_IMPORT_ERROR

    if _NBodyKit_IMPORT_ERROR is not None:
        raise ImportError("nbodykit + mpi4py are required for FoF stage") from _NBodyKit_IMPORT_ERROR
    if _HALOS_IMPORT_ERROR is not None:
        raise ImportError("Failed to import FoF helpers from scripts/halos.py") from _HALOS_IMPORT_ERROR

    comm = MPI.COMM_WORLD
    rank = int(comm.rank)
    size = int(comm.size)

    samples_dir = Path(args.samples_dir).resolve()
    posterior_arrays_path = (
        Path(args.posterior_arrays_path).resolve()
        if args.posterior_arrays_path is not None
        else default_posterior_arrays_path(samples_dir=samples_dir, prefix=str(args.posterior_prefix)).resolve()
    )

    if rank == 0 and not posterior_arrays_path.is_file():
        raise FileNotFoundError(f"Posterior HMF arrays not found: {posterior_arrays_path}")

    bundle = None
    if rank == 0:
        data = np.load(posterior_arrays_path)
        required = ("hmf_mass_centers", "hmf_log_edges", "hmf_samples", "mass_resolution_limit")
        missing = [k for k in required if k not in data.files]
        if missing:
            raise KeyError(f"Missing required keys in {posterior_arrays_path}: {missing}")

        bundle = {
            "m_centers": np.asarray(data["hmf_mass_centers"], dtype=np.float64),
            "log_edges": np.asarray(data["hmf_log_edges"], dtype=np.float64),
            "hmf_samples": np.asarray(data["hmf_samples"], dtype=np.float64),
            "mass_resolution_limit": float(np.asarray(data["mass_resolution_limit"], dtype=np.float64)),
            "tinker": _optional_curve(data, "hmf_tinker") if bool(args.show_tinker) else None,
            "original_truth": _optional_curve(data, "hmf_truth") if bool(args.show_original_truth) else None,
        }

    bundle = comm.bcast(bundle, root=0)
    m_centers = np.asarray(bundle["m_centers"], dtype=np.float64)
    log_edges = np.asarray(bundle["log_edges"], dtype=np.float64)
    posterior_hmf_samples = np.asarray(bundle["hmf_samples"], dtype=np.float64)
    mass_resolution_limit = float(bundle["mass_resolution_limit"])
    tinker_curve = None if bundle["tinker"] is None else np.asarray(bundle["tinker"], dtype=np.float64)
    original_truth_curve = None if bundle["original_truth"] is None else np.asarray(bundle["original_truth"], dtype=np.float64)

    if rank == 0:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        truth_forward_dir = (
            Path(args.truth_forward_dir).resolve()
            if args.truth_forward_dir is not None
            else (output_dir / "true_ic_forward_apples_to_apples")
        )
        truth_forward_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
        truth_forward_dir = None

    output_dir = comm.bcast(output_dir, root=0)
    truth_forward_dir = comm.bcast(truth_forward_dir, root=0)

    z_tag = _z_file_tag(float(args.z_target))
    truth_displacement_path = Path(truth_forward_dir) / f"emu_dis_{z_tag}.npy"

    forward_runtime_seconds = np.nan
    if rank == 0:
        if bool(args.reuse_existing_forward) and truth_displacement_path.is_file():
            print(f"Reusing existing true-IC forward displacement: {truth_displacement_path}", flush=True)
            metadata_path = Path(truth_forward_dir) / "metadata.json"
            if metadata_path.is_file():
                try:
                    meta = json.loads(metadata_path.read_text())
                    forward_runtime_seconds = float(meta.get("emulator_runtime_seconds", np.nan))
                except Exception:
                    forward_runtime_seconds = np.nan
        else:
            print("Running true IC forward pipeline...", flush=True)
            true_ic = load_true_ic_field(target_path=Path(args.target_path), key=str(args.ic_key))
            scale = 1.0
            if str(args.ic_rescale_mode) == "growth":
                scale = ic_growth_scale(z_ic=float(args.z_ic), z_target=float(args.z_target))
            print(f"True IC rescale mode: {args.ic_rescale_mode}, scale={scale:.8f}", flush=True)

            pipeline = run_lpt_emulator_pipeline(
                delta_linear=np.asarray(true_ic * np.float32(scale), dtype=np.float32),
                n_part=int(args.n_part),
                density_res=int(args.res),
                boxsize=float(args.boxsize),
                ndiv=tuple(args.ndiv),
                emu_precision=str(args.emu_precision),
                num_sims=int(args.num_sims),
                mas_worder=int(args.mas_worder),
                deconvolve_mas=bool(args.deconvolve_mas),
                z_target=float(args.z_target),
                compute_vel=False,
                delta_upsample_method=str(args.delta_upsample_method),
                downsample_method=str(args.downsample_method),
                downsample_gaussian_sigma_mpc_h=args.downsample_gaussian_sigma_mpc_h,
                downsample_threads=int(args.downsample_threads),
                class_n_modes=int(args.class_n_modes),
                quijote_fiducial=QUIJOTE_FIDUCIAL,
                output_dir=Path(truth_forward_dir),
                save_outputs=True,
                show_progress=True,
                progress_desc_prefix="A2A true-IC subboxes",
            )
            saved_paths = dict(pipeline.get("saved_paths", {}))
            saved_psi_emu = saved_paths.get("psi_emu")
            if saved_psi_emu is not None:
                truth_displacement_path = Path(saved_psi_emu).resolve()
            if not truth_displacement_path.is_file():
                raise FileNotFoundError(
                    f"Expected emulated truth displacement not found: {truth_displacement_path}"
                )
            forward_runtime_seconds = float(
                dict(pipeline.get("metadata", {})).get("emulator_runtime_seconds", np.nan)
            )

    comm.Barrier()

    truth_lengths = run_distributed_fof_lengths(
        displacement_path=Path(truth_displacement_path),
        boxsize=float(args.boxsize),
        n_part=int(args.n_part),
        linking_length=float(args.fof_linking_length),
        nmin=int(args.fof_nmin),
        absolute_linking=bool(args.fof_absolute_linking),
    )

    if rank != 0:
        return

    if truth_lengths is None:
        raise RuntimeError("Internal error: rank 0 did not receive truth FoF lengths")
    if truth_lengths.size == 0:
        raise ValueError("No FoF groups found for true-IC emulated displacement")

    m_particle = particle_mass_msun_h(
        omega_m=float(args.omega_m),
        boxsize=float(args.boxsize),
        n_part=int(args.n_part),
    )
    truth_masses = np.asarray(truth_lengths, dtype=np.float64) * float(m_particle)
    truth_emulated_hmf = compute_hmf_curve(
        masses=np.asarray(truth_masses, dtype=np.float64),
        log_edges=np.asarray(log_edges, dtype=np.float64),
        boxsize=float(args.boxsize),
    )

    prefix = str(args.output_prefix)
    comp_arrays_path = Path(output_dir) / f"{prefix}_arrays.npz"
    comp_plot_path = Path(output_dir) / f"{prefix}.png"
    comp_meta_path = Path(output_dir) / f"{prefix}_metadata.json"

    np.savez_compressed(
        comp_arrays_path,
        posterior_hmf_mass_centers=np.asarray(m_centers, dtype=np.float64),
        posterior_hmf_log_edges=np.asarray(log_edges, dtype=np.float64),
        posterior_hmf_samples=np.asarray(posterior_hmf_samples, dtype=np.float64),
        truth_emulated_hmf=np.asarray(truth_emulated_hmf, dtype=np.float64),
        posterior_hmf_tinker=np.asarray([] if tinker_curve is None else tinker_curve, dtype=np.float64),
        posterior_hmf_original_truth=np.asarray([] if original_truth_curve is None else original_truth_curve, dtype=np.float64),
        mass_resolution_limit=np.asarray(mass_resolution_limit, dtype=np.float64),
    )

    plot_apples_to_apples_hmf(
        m_centers=np.asarray(m_centers, dtype=np.float64),
        posterior_hmf_samples=np.asarray(posterior_hmf_samples, dtype=np.float64),
        truth_emulated_hmf=np.asarray(truth_emulated_hmf, dtype=np.float64),
        mass_resolution_limit=float(mass_resolution_limit),
        out_path=Path(comp_plot_path),
        title=str(args.title),
        ratio_ymin=float(args.ratio_ymin),
        ratio_ymax=float(args.ratio_ymax),
        dpi=int(args.dpi),
        tinker_curve=None if tinker_curve is None else np.asarray(tinker_curve, dtype=np.float64),
        original_truth_curve=None if original_truth_curve is None else np.asarray(original_truth_curve, dtype=np.float64),
    )

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_path": str(Path(args.target_path).resolve()),
        "ic_key": str(args.ic_key),
        "z_ic": float(args.z_ic),
        "z_target": float(args.z_target),
        "ic_rescale_mode": str(args.ic_rescale_mode),
        "samples_dir": str(samples_dir),
        "posterior_arrays_path": str(posterior_arrays_path),
        "truth_forward_dir": str(Path(truth_forward_dir).resolve()),
        "truth_displacement_path": str(Path(truth_displacement_path).resolve()),
        "forward_runtime_seconds": None if not np.isfinite(forward_runtime_seconds) else float(forward_runtime_seconds),
        "mpi_world_size": int(size),
        "n_part": int(args.n_part),
        "res": int(args.res),
        "boxsize": float(args.boxsize),
        "omega_m": float(args.omega_m),
        "fof_linking_length": float(args.fof_linking_length),
        "fof_absolute_linking": bool(args.fof_absolute_linking),
        "fof_nmin": int(args.fof_nmin),
        "mass_resolution_limit_msun_h": float(mass_resolution_limit),
        "truth_num_halos": int(truth_lengths.size),
        "comparison_arrays": str(comp_arrays_path),
        "comparison_plot": str(comp_plot_path),
        "output_prefix": str(prefix),
    }
    Path(comp_meta_path).write_text(json.dumps(metadata, indent=2))

    print("Apples-to-apples HMF comparison complete.")
    print(f"Posterior arrays: {posterior_arrays_path}")
    print(f"Truth displacement: {truth_displacement_path}")
    print(f"Comparison arrays: {comp_arrays_path}")
    print(f"Comparison plot:   {comp_plot_path}")
    print(f"Comparison metadata: {comp_meta_path}")


if __name__ == "__main__":
    main()
