#!/usr/bin/env python3
"""
Posterior predictive check for IC posterior samples using the emulator pipeline.

Workflow:
1. Load posterior IC samples `sample_XXXX.npy` from `--samples-dir`.
2. For each sample:
   - Rescale IC from `z_ic` to `z_target` linear amplitude.
   - Run forward emulator pipeline (or reuse existing per-sample outputs).
   - Save emulator outputs in `--samples-dir/<sample_folder_template>`.
   - Compute summaries of final emulated field:
       P(k), reduced bispectrum Q(theta), cross-correlation with truth field,
       Minkowski functionals, and FoF-based HMF.
3. Aggregate summaries across samples.
4. Save aggregated arrays and produce mean ±1sigma/±2sigma PPC plots.
5. Overlay truth summaries from `target_path[<target_key>]` where defined.

Example:
  python paper_scripts/posterior_predictive_check.py \
    --samples-dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD \
    --target-path /home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt \
    --num-samples 100 \
    --n-part 128 \
    --res 128 \
    --boxsize 1000 \
    --ndiv 4,2,2 \
    --emu-precision f16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from core import QUIJOTE_FIDUCIAL, run_lpt_emulator_pipeline
except ImportError as exc:  # pragma: no cover - runtime dependency
    QUIJOTE_FIDUCIAL = None
    run_lpt_emulator_pipeline = None
    _CORE_IMPORT_ERROR = exc
else:
    _CORE_IMPORT_ERROR = None

try:
    from halos import particle_mass_msun_h, run_fof
except ImportError as exc:  # pragma: no cover - runtime dependency
    particle_mass_msun_h = None
    run_fof = None
    _HALOS_IMPORT_ERROR = exc
else:
    _HALOS_IMPORT_ERROR = None

try:
    from utils import (
        compute_minkowski_functionals,
        growth_D_approx,
        mas_from_worder,
        parse_ndiv,
        resize_density_grid,
    )
except ImportError as exc:  # pragma: no cover - runtime dependency
    compute_minkowski_functionals = None
    growth_D_approx = None
    mas_from_worder = None
    parse_ndiv = None
    resize_density_grid = None
    _UTILS_IMPORT_ERROR = exc
else:
    _UTILS_IMPORT_ERROR = None

try:
    import Pk_library as PKL
except ImportError as exc:  # pragma: no cover - runtime dependency
    PKL = None
    _PKL_IMPORT_ERROR = exc
else:
    _PKL_IMPORT_ERROR = None

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime dependency
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from mpi4py import MPI
    from nbodykit.algorithms.fof import FOF, fof_catalog
    from nbodykit.source.catalog import ArrayCatalog
except ImportError as exc:  # pragma: no cover - optional for HMF mode
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
    """Parse command-line options for posterior predictive checks."""
    parser = argparse.ArgumentParser(
        description=(
            "Run emulator posterior predictive checks for IC sample files "
            "sample_XXXX.npy and aggregate summary statistics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("/gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD"),
        help="Directory containing posterior IC samples sample_XXXX.npy.",
    )
    parser.add_argument(
        "--sample-pattern",
        type=str,
        default="sample_*.npy",
        help="Glob pattern for IC sample files inside --samples-dir.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of IC sample files to process (after sorting).",
    )
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="Starting sample index in the sorted sample-file list.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Stride used when selecting samples from the sorted list.",
    )
    parser.add_argument(
        "--sample-folder-template",
        type=str,
        default="sample_{sample_id:04d}",
        help=(
            "Template for per-sample emulator output folder names inside --samples-dir. "
            "Available fields: sample_id, sample_stem."
        ),
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        default=Path("/home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt"),
        help="Path to truth .pt file containing final field key (e.g. delta_z0).",
    )
    parser.add_argument(
        "--target-key",
        type=str,
        default="delta_z0",
        help="Key in --target-path used as truth final field.",
    )
    parser.add_argument(
        "--target-deconvolved",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether target field is already MAS-deconvolved (e.g. *_wout_MAK.pt).",
    )
    parser.add_argument("--z-ic", type=float, default=127.0, help="Redshift of IC posterior samples.")
    parser.add_argument("--z-target", type=float, default=0.0, help="Target redshift of emulator output.")
    parser.add_argument(
        "--ic-rescale-mode",
        choices=("growth", "none"),
        default="growth",
        help="How to scale IC samples before forward run.",
    )

    parser.add_argument("--n-part", type=int, default=128, help="Emulator particle lattice resolution.")
    parser.add_argument("--res", type=int, default=128, help="Summary density-grid resolution.")
    parser.add_argument("--boxsize", type=float, default=1000.0, help="Simulation box size in Mpc/h.")
    parser.add_argument(
        "--ndiv",
        type=parse_ndiv if callable(parse_ndiv) else str,
        default=(4, 2, 2),
        help="Subbox partition for emulator.",
    )
    parser.add_argument(
        "--emu-precision",
        choices=("f16", "f32"),
        default="f16",
        help="Emulator compute precision.",
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=1,
        help="Number of emulator repeats inside each sample run (last one is used).",
    )
    parser.add_argument("--mas-worder", type=int, choices=(2, 3, 4), default=4, help="MAS order for meshing.")
    parser.add_argument(
        "--deconvolve-mas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MAS-deconvolve generated emulator/LPT density fields.",
    )
    parser.add_argument(
        "--delta-upsample-method",
        choices=("mode_inject", "fourier", "linear"),
        default="mode_inject",
        help="Interpolation used when IC field must be upsampled.",
    )
    parser.add_argument(
        "--downsample-method",
        choices=("fourier_truncation", "block_average", "gaussian"),
        default="fourier_truncation",
        help="Downsampling method when IC field must be downsampled.",
    )
    parser.add_argument(
        "--downsample-gaussian-sigma-mpc-h",
        type=float,
        default=None,
        help="Gaussian smoothing scale for downsample_method=gaussian.",
    )
    parser.add_argument(
        "--downsample-threads",
        type=int,
        default=1,
        help="Thread count for gaussian downsampling helper.",
    )
    parser.add_argument("--class-n-modes", type=int, default=4000, help="CLASS k-table sample count.")
    parser.add_argument(
        "--compute-vel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also compute emulator velocity head.",
    )
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse per-sample forward outputs if found in sample folder.",
    )

    parser.add_argument("--pk-threads", type=int, default=1, help="Thread count for Pk_library calls.")
    parser.add_argument("--bispec-k1", type=float, default=0.1, help="First bispectrum config k1 [h/Mpc].")
    parser.add_argument("--bispec-k2", type=float, default=0.1, help="First bispectrum config k2 [h/Mpc].")
    parser.add_argument("--bispec-k1b", type=float, default=0.05, help="Second bispectrum config k1 [h/Mpc].")
    parser.add_argument("--bispec-k2b", type=float, default=0.1, help="Second bispectrum config k2 [h/Mpc].")
    parser.add_argument("--bispec-n-theta", type=int, default=25, help="Number of theta points in bispectrum.")

    parser.add_argument(
        "--minkowski-nu-min",
        type=float,
        default=-3.0,
        help="Minimum threshold for Minkowski curves.",
    )
    parser.add_argument(
        "--minkowski-nu-max",
        type=float,
        default=3.0,
        help="Maximum threshold for Minkowski curves.",
    )
    parser.add_argument(
        "--minkowski-n-thresholds",
        type=int,
        default=41,
        help="Number of threshold bins for Minkowski curves.",
    )

    parser.add_argument(
        "--compute-hmf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute FoF halos and HMF for each emulator sample.",
    )
    parser.add_argument("--fof-linking-length", type=float, default=0.2, help="FoF linking length parameter.")
    parser.add_argument(
        "--fof-absolute-linking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat linking length as absolute Mpc/h.",
    )
    parser.add_argument("--fof-nmin", type=int, default=20, help="FoF minimum particles per halo.")
    parser.add_argument("--hmf-nbins", type=int, default=25, help="Number of logarithmic HMF bins.")
    parser.add_argument("--hmf-mmin", type=float, default=None, help="Lower HMF bound [Msun/h].")
    parser.add_argument("--hmf-mmax", type=float, default=None, help="Upper HMF bound [Msun/h].")
    parser.add_argument(
        "--truth-displacement-path",
        type=Path,
        default=None,
        help="Optional truth displacement file for truth HMF overlay (if unavailable, HMF has no truth line).",
    )

    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=None,
        help="Directory for aggregated arrays/plots (default: <samples-dir>/ppc_results).",
    )
    parser.add_argument(
        "--aggregate-prefix",
        type=str,
        default="posterior_predictive",
        help="Filename prefix for aggregated outputs.",
    )

    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.sample_start < 0:
        parser.error("--sample-start must be >= 0")
    if args.sample_stride < 1:
        parser.error("--sample-stride must be >= 1")
    if args.res < 1 or args.n_part < 1:
        parser.error("--res and --n-part must be >= 1")
    if args.minkowski_n_thresholds < 2:
        parser.error("--minkowski-n-thresholds must be >= 2")
    if args.bispec_n_theta < 3:
        parser.error("--bispec-n-theta must be >= 3")
    if args.hmf_nbins < 3:
        parser.error("--hmf-nbins must be >= 3")
    return args


def load_target_field(target_path: Path, target_key: str) -> np.ndarray:
    """Load truth field from a torch .pt file and return float32 numpy array."""
    if torch is None:
        raise ImportError("torch is required to read target .pt files.") from _TORCH_IMPORT_ERROR
    if not target_path.is_file():
        raise FileNotFoundError(f"Target file not found: {target_path}")
    data = torch.load(target_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {target_path}, got {type(data)}")
    if target_key not in data:
        raise KeyError(f"Missing target key {target_key!r} in {target_path}")
    field = np.asarray(data[target_key], dtype=np.float32)
    if field.ndim != 3 or len(set(field.shape)) != 1:
        raise ValueError(f"Target field must be cubic 3D, got shape={field.shape}")
    return field


def sorted_sample_files(samples_dir: Path, pattern: str) -> list[Path]:
    """Return sample file paths sorted by numeric ID in sample_XXXX filenames."""
    files = list(samples_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching pattern {pattern!r} in {samples_dir}")

    def key_fn(path: Path) -> tuple[int, str]:
        match = re.search(r"(\d+)", path.stem)
        if match is None:
            return (10**9, path.name)
        return (int(match.group(1)), path.name)

    return sorted(files, key=key_fn)


def extract_sample_id(path: Path, fallback_index: int) -> int:
    """Extract numeric sample ID from filename stem."""
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        return int(fallback_index)
    return int(match.group(1))


def ic_growth_scale(*, z_ic: float, z_target: float) -> float:
    """Compute growth-factor scaling D(z_target)/D(z_ic) for Quijote fiducial cosmology."""
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


def field_to_channels_first_displacement(psi: np.ndarray) -> np.ndarray:
    """Normalize displacement array layout to (3, N, N, N)."""
    arr = np.asarray(psi, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Displacement must be 4D, got shape={arr.shape}")
    if arr.shape[0] == 3:
        return arr
    if arr.shape[-1] == 3:
        return np.moveaxis(arr, -1, 0)
    raise ValueError(f"Displacement shape must be (3,N,N,N) or (N,N,N,3), got {arr.shape}")


def positions_from_displacement(psi: np.ndarray, boxsize: float) -> np.ndarray:
    """Construct Eulerian particle positions from displacement field."""
    dis = field_to_channels_first_displacement(psi)
    n_part = int(dis.shape[1])
    if dis.shape[2] != n_part or dis.shape[3] != n_part:
        raise ValueError(f"Displacement must be cubic, got shape={dis.shape}")

    dx = float(boxsize) / float(n_part)
    x = (np.arange(n_part, dtype=np.float32) * dx)[:, None, None]
    y = (np.arange(n_part, dtype=np.float32) * dx)[None, :, None]
    z = (np.arange(n_part, dtype=np.float32) * dx)[None, None, :]

    n_local = n_part * n_part * n_part
    pos = np.empty((n_local, 3), dtype=np.float32)
    pos[:, 0] = np.mod(x + dis[0], boxsize).reshape(-1)
    pos[:, 1] = np.mod(y + dis[1], boxsize).reshape(-1)
    pos[:, 2] = np.mod(z + dis[2], boxsize).reshape(-1)
    return pos


def run_single_rank_fof_masses(
    psi: np.ndarray,
    *,
    boxsize: float,
    n_part: int,
    omega_m: float,
    linking_length: float,
    nmin: int,
    absolute_linking: bool,
) -> np.ndarray:
    """Run FoF on one rank and return halo masses in Msun/h."""
    if _NBodyKit_IMPORT_ERROR is not None:
        raise ImportError(
            "nbodykit + mpi4py are required for HMF/FoF computation."
        ) from _NBodyKit_IMPORT_ERROR
    comm = MPI.COMM_WORLD
    if int(comm.size) != 1:
        raise RuntimeError(
            "This PPC script currently expects a single-rank run for FoF. "
            f"Got MPI size={comm.size}."
        )
    local_positions = positions_from_displacement(psi=psi, boxsize=float(boxsize))
    _, length = run_fof(
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
    m_particle = particle_mass_msun_h(omega_m=float(omega_m), boxsize=float(boxsize), n_part=int(n_part))
    return np.asarray(length, dtype=np.float64) * float(m_particle)


def compute_hmf_curve(masses: np.ndarray, log_edges: np.ndarray, boxsize: float) -> np.ndarray:
    """Compute dn/dlog10M on fixed logarithmic mass-bin edges."""
    masses = np.asarray(masses, dtype=np.float64)
    masses = masses[np.isfinite(masses) & (masses > 0.0)]
    counts, _ = np.histogram(np.log10(masses), bins=log_edges)
    dlogm = float(log_edges[1] - log_edges[0])
    volume = float(boxsize) ** 3
    return counts.astype(np.float64) / (volume * dlogm)


def ensure_target_resolution(
    delta_target: np.ndarray,
    *,
    target_res: int,
    boxsize: float,
    upsample_method: str,
) -> np.ndarray:
    """Resize truth target field to target resolution when needed."""
    delta_target = np.asarray(delta_target, dtype=np.float32)
    if int(delta_target.shape[0]) == int(target_res):
        return delta_target
    return resize_density_grid(
        delta_in=delta_target,
        target_res=int(target_res),
        boxsize=float(boxsize),
        upsample_method=str(upsample_method),
    )


def mean_sigma(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sample mean and standard deviation along axis=0."""
    arr = np.asarray(samples, dtype=np.float64)
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def save_summary_npz(path: Path, payload: dict[str, np.ndarray | list | float | int | bool | str]) -> None:
    """Persist summary payload to a compressed npz archive."""
    to_save: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            to_save[key] = value
        elif isinstance(value, (list, tuple)):
            to_save[key] = np.asarray(value)
        elif value is None:
            to_save[key] = np.asarray([])
        else:
            to_save[key] = np.asarray(value)
    np.savez_compressed(path, **to_save)


def plot_pk_summary(
    *,
    k: np.ndarray,
    pk_true: np.ndarray,
    pk_samples: np.ndarray,
    boxsize: float,
    res: int,
    out_path: Path,
) -> None:
    """Plot posterior predictive P(k) bands and ratio to truth."""
    k = np.asarray(k, dtype=np.float64)
    pk_true = np.asarray(pk_true, dtype=np.float64)
    pk_samples = np.asarray(pk_samples, dtype=np.float64)
    mu, sigma = mean_sigma(pk_samples)
    k_nyq = np.pi * int(res) / float(boxsize)

    ratio = np.divide(pk_samples, pk_true[None, :], out=np.full_like(pk_samples, np.nan), where=pk_true[None, :] > 0)
    ratio_mu, ratio_sigma = mean_sigma(ratio)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.2),
        sharex=True,
        constrained_layout=True,
        height_ratios=(2.0, 1.0),
    )

    ax = axes[0]
    ax.plot(k, pk_true, color="black", lw=1.4, label="Truth final field")
    ax.plot(k, mu, color="tab:blue", lw=1.2, label="Posterior predictive mean")
    ax.fill_between(k, mu - sigma, mu + sigma, color="tab:blue", alpha=0.35, label=r"$\pm 1\sigma$")
    ax.fill_between(k, mu - 2.0 * sigma, mu + 2.0 * sigma, color="tab:blue", alpha=0.16, label=r"$\pm 2\sigma$")
    ax.axvline(k_nyq, color="red", ls="--", lw=0.8, alpha=0.6, label=r"$k_{\rm Nyq}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$P(k)$")
    ax.grid(alpha=0.15, which="both")
    ax.legend(framealpha=0.9, fontsize=8, loc="best")

    ax = axes[1]
    ax.plot(k, ratio_mu, color="tab:blue", lw=1.2, label="Predictive / truth")
    ax.fill_between(k, ratio_mu - ratio_sigma, ratio_mu + ratio_sigma, color="tab:blue", alpha=0.35)
    ax.fill_between(k, ratio_mu - 2.0 * ratio_sigma, ratio_mu + 2.0 * ratio_sigma, color="tab:blue", alpha=0.16)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.axvline(k_nyq, color="red", ls="--", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [$h/{\rm Mpc}$]")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0.0, 2.0)
    ax.grid(alpha=0.15, which="both")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_crosscorr_summary(*, k: np.ndarray, xcorr_samples: np.ndarray, out_path: Path) -> None:
    """Plot posterior predictive cross-correlation bands with the truth field."""
    k = np.asarray(k, dtype=np.float64)
    xcorr_samples = np.asarray(xcorr_samples, dtype=np.float64)
    mu, sigma = mean_sigma(xcorr_samples)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(k, mu, color="tab:green", lw=1.2, label="Predictive mean C(k)")
    ax.fill_between(k, mu - sigma, mu + sigma, color="tab:green", alpha=0.35, label=r"$\pm 1\sigma$")
    ax.fill_between(k, mu - 2.0 * sigma, mu + 2.0 * sigma, color="tab:green", alpha=0.16, label=r"$\pm 2\sigma$")
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Truth self-correlation")
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel(r"$k$ [$h/{\rm Mpc}$]")
    ax.set_ylabel(r"$C(k)$")
    ax.grid(alpha=0.15, which="both")
    ax.legend(framealpha=0.9, fontsize=8, loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bispectrum_summary(
    *,
    theta_deg: np.ndarray,
    q_true_cfg1: np.ndarray,
    q_true_cfg2: np.ndarray,
    q_samples_cfg1: np.ndarray,
    q_samples_cfg2: np.ndarray,
    cfg1_label: str,
    cfg2_label: str,
    out_path: Path,
) -> None:
    """Plot posterior predictive reduced bispectrum curves for two triangle configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True, constrained_layout=True)

    for ax, q_true, q_samples, title in [
        (axes[0], q_true_cfg1, q_samples_cfg1, cfg1_label),
        (axes[1], q_true_cfg2, q_samples_cfg2, cfg2_label),
    ]:
        q_true = np.asarray(q_true, dtype=np.float64)
        q_samples = np.asarray(q_samples, dtype=np.float64)
        mu, sigma = mean_sigma(q_samples)
        ax.plot(theta_deg, q_true, color="black", lw=1.4, label="Truth final field")
        ax.plot(theta_deg, mu, color="tab:purple", lw=1.2, label="Predictive mean")
        ax.fill_between(theta_deg, mu - sigma, mu + sigma, color="tab:purple", alpha=0.35, label=r"$\pm 1\sigma$")
        ax.fill_between(theta_deg, mu - 2.0 * sigma, mu + 2.0 * sigma, color="tab:purple", alpha=0.16, label=r"$\pm 2\sigma$")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$\theta$ [deg]")
        ax.grid(alpha=0.15)

    axes[0].set_ylabel(r"$Q(\theta)$")
    axes[0].legend(framealpha=0.9, fontsize=8, loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_minkowski_summary(
    *,
    nu: np.ndarray,
    truth: dict[str, np.ndarray],
    samples: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Plot posterior predictive Minkowski bands V0..V3 against truth curves."""
    keys = ["v0", "v1", "v2", "v3"]
    ylabels = [
        r"$V_0$",
        r"$V_1$ [$h/{\rm Mpc}$]",
        r"$V_2$ [$h^2/{\rm Mpc}^2$]",
        r"$V_3$ [$(\mathrm{Mpc}/h)^{-3}$]",
    ]
    titles = ["Volume", "Surface", "Curvature", "Euler characteristic"]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), sharex=True, constrained_layout=True)
    axes_flat = axes.ravel()

    for idx, key in enumerate(keys):
        ax = axes_flat[idx]
        arr = np.asarray(samples[key], dtype=np.float64)
        mu, sigma = mean_sigma(arr)
        ax.plot(nu, np.asarray(truth[key], dtype=np.float64), color="black", lw=1.4, label="Truth final field")
        ax.plot(nu, mu, color="tab:blue", lw=1.2, label="Predictive mean")
        ax.fill_between(nu, mu - sigma, mu + sigma, color="tab:blue", alpha=0.35, label=r"$\pm 1\sigma$")
        ax.fill_between(nu, mu - 2.0 * sigma, mu + 2.0 * sigma, color="tab:blue", alpha=0.16, label=r"$\pm 2\sigma$")
        ax.set_title(titles[idx], fontsize=10)
        ax.set_ylabel(ylabels[idx])
        ax.grid(alpha=0.15)
        if idx >= 2:
            ax.set_xlabel(r"$\nu=(\delta-\mu)/\sigma$")

    axes_flat[0].legend(framealpha=0.9, fontsize=8, loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_hmf_summary(
    *,
    m_centers: np.ndarray,
    hmf_samples: np.ndarray,
    hmf_truth: np.ndarray | None,
    out_path: Path,
) -> None:
    """Plot posterior predictive HMF bands and optional truth HMF."""
    m_centers = np.asarray(m_centers, dtype=np.float64)
    hmf_samples = np.asarray(hmf_samples, dtype=np.float64)
    mu, sigma = mean_sigma(hmf_samples)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    if hmf_truth is not None:
        ax.plot(m_centers, np.asarray(hmf_truth, dtype=np.float64), color="black", lw=1.4, label="Truth HMF")
    ax.plot(m_centers, mu, color="tab:orange", lw=1.2, label="Predictive mean HMF")
    ax.fill_between(m_centers, mu - sigma, mu + sigma, color="tab:orange", alpha=0.35, label=r"$\pm 1\sigma$")
    ax.fill_between(m_centers, mu - 2.0 * sigma, mu + 2.0 * sigma, color="tab:orange", alpha=0.16, label=r"$\pm 2\sigma$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_{\rm halo}\ [M_\odot/h]$")
    ax.set_ylabel(r"$dn/d\log_{10}M\ [(h/{\rm Mpc})^3]$")
    ax.grid(alpha=0.15, which="both")
    ax.legend(framealpha=0.9, fontsize=8, loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Run posterior predictive forward simulation and summary aggregation."""
    args = parse_args()

    if _UTILS_IMPORT_ERROR is not None:
        raise ImportError(
            "Failed to import utilities from scripts/utils.py. "
            "Activate the environment used for the emulator pipeline."
        ) from _UTILS_IMPORT_ERROR
    if _CORE_IMPORT_ERROR is not None:
        raise ImportError(
            "Failed to import core emulator pipeline from scripts/core.py. "
            "Ensure DISCO-DJ/JAX dependencies are available."
        ) from _CORE_IMPORT_ERROR
    if PKL is None:
        raise ImportError("Pk_library is required for PPC summaries.") from _PKL_IMPORT_ERROR

    if bool(args.compute_hmf) and _HALOS_IMPORT_ERROR is not None:
        raise ImportError(
            "Failed to import FoF helpers from scripts/halos.py. "
            "Disable HMF mode with --no-compute-hmf or fix the environment."
        ) from _HALOS_IMPORT_ERROR
    samples_dir = args.samples_dir.resolve()
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    aggregate_dir = args.aggregate_dir.resolve() if args.aggregate_dir is not None else (samples_dir / "ppc_results")
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading target field from {args.target_path} [{args.target_key}]...")
    delta_true = load_target_field(target_path=args.target_path, target_key=args.target_key)
    delta_true = ensure_target_resolution(
        delta_target=delta_true,
        target_res=int(args.res),
        boxsize=float(args.boxsize),
        upsample_method=str(args.delta_upsample_method),
    )

    scale = 1.0
    if args.ic_rescale_mode == "growth":
        scale = ic_growth_scale(z_ic=float(args.z_ic), z_target=float(args.z_target))
    print(f"IC scaling mode: {args.ic_rescale_mode}, scale factor={scale:.8f}")

    mas_name = mas_from_worder(int(args.mas_worder))
    mas_model = "None" if bool(args.deconvolve_mas) else mas_name
    mas_target = "None" if bool(args.target_deconvolved) else mas_name

    print("Computing truth summaries...")
    pk_true_obj = PKL.Pk(delta_true, float(args.boxsize), axis=0, MAS=mas_target, threads=int(args.pk_threads), verbose=False)
    k_pk = np.asarray(pk_true_obj.k3D, dtype=np.float64)
    pk_true = np.asarray(pk_true_obj.Pk[:, 0], dtype=np.float64)
    valid_k = np.isfinite(k_pk) & (k_pk > 0) & np.isfinite(pk_true) & (pk_true > 0)
    k_pk = k_pk[valid_k]
    pk_true = pk_true[valid_k]

    theta = np.linspace(0.0, np.pi, int(args.bispec_n_theta), dtype=np.float64)
    theta_deg = np.degrees(theta)
    q_true_cfg1 = np.asarray(
        PKL.Bk(delta_true, float(args.boxsize), float(args.bispec_k1), float(args.bispec_k2), theta, mas_target, threads=int(args.pk_threads)).Q,
        dtype=np.float64,
    )
    q_true_cfg2 = np.asarray(
        PKL.Bk(delta_true, float(args.boxsize), float(args.bispec_k1b), float(args.bispec_k2b), theta, mas_target, threads=int(args.pk_threads)).Q,
        dtype=np.float64,
    )

    nu_grid = np.linspace(float(args.minkowski_nu_min), float(args.minkowski_nu_max), int(args.minkowski_n_thresholds), dtype=np.float32)
    mink_truth_full = compute_minkowski_functionals(
        field=delta_true,
        boxsize=float(args.boxsize),
        thresholds=nu_grid,
        standardize=True,
    )
    mink_truth = {k: np.asarray(mink_truth_full[k], dtype=np.float64) for k in ("v0", "v1", "v2", "v3")}

    files_all = sorted_sample_files(samples_dir=samples_dir, pattern=str(args.sample_pattern))
    files_sel = files_all[int(args.sample_start) :: int(args.sample_stride)]
    files_sel = files_sel[: int(args.num_samples)]
    if len(files_sel) == 0:
        raise ValueError("No sample files selected after applying start/stride/num-samples.")
    print(f"Selected {len(files_sel)} sample files for PPC.")

    sample_ids: list[int] = []
    sample_input_paths: list[str] = []
    sample_output_dirs: list[str] = []
    runtime_s: list[float] = []

    pk_samples: list[np.ndarray] = []
    xcorr_samples: list[np.ndarray] = []
    q_samples_cfg1: list[np.ndarray] = []
    q_samples_cfg2: list[np.ndarray] = []
    mink_samples = {"v0": [], "v1": [], "v2": [], "v3": []}

    hmf_samples: list[np.ndarray] = []
    hmf_log_edges: np.ndarray | None = None
    hmf_centers: np.ndarray | None = None
    hmf_truth: np.ndarray | None = None

    omega_m = float(QUIJOTE_FIDUCIAL["Omega_m"])
    z_tag = _z_file_tag(float(args.z_target))
    for local_idx, sample_path in enumerate(tqdm(files_sel, desc="PPC samples", ncols=100)):
        sample_id = extract_sample_id(sample_path, fallback_index=local_idx)
        sample_ids.append(int(sample_id))
        sample_input_paths.append(str(sample_path))

        sample_folder_name = args.sample_folder_template.format(sample_id=int(sample_id), sample_stem=sample_path.stem)
        out_dir = samples_dir / sample_folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_output_dirs.append(str(out_dir))

        emu_delta_path = out_dir / f"emu_delta_{z_tag}.npy"
        emu_dis_path = out_dir / f"emu_dis_{z_tag}.npy"
        metadata_path = out_dir / "metadata.json"

        if bool(args.reuse_existing) and emu_delta_path.is_file() and emu_dis_path.is_file():
            delta_emu = np.asarray(np.load(emu_delta_path), dtype=np.float32)
            psi_emu = np.asarray(np.load(emu_dis_path), dtype=np.float32)
            if metadata_path.is_file():
                metadata_local = json.loads(metadata_path.read_text())
                runtime_s.append(float(metadata_local.get("emulator_runtime_seconds", np.nan)))
            else:
                runtime_s.append(np.nan)
        else:
            ic_sample = np.asarray(np.load(sample_path), dtype=np.float32)
            if ic_sample.ndim != 3 or len(set(ic_sample.shape)) != 1:
                raise ValueError(f"IC sample must be cubic 3D, got {sample_path} with shape={ic_sample.shape}")
            ic_scaled = np.asarray(ic_sample * np.float32(scale), dtype=np.float32)

            pipeline = run_lpt_emulator_pipeline(
                delta_linear=ic_scaled,
                n_part=int(args.n_part),
                density_res=int(args.res),
                boxsize=float(args.boxsize),
                ndiv=tuple(args.ndiv),
                emu_precision=str(args.emu_precision),
                num_sims=int(args.num_sims),
                mas_worder=int(args.mas_worder),
                deconvolve_mas=bool(args.deconvolve_mas),
                z_target=float(args.z_target),
                compute_vel=bool(args.compute_vel),
                delta_upsample_method=str(args.delta_upsample_method),
                downsample_method=str(args.downsample_method),
                downsample_gaussian_sigma_mpc_h=args.downsample_gaussian_sigma_mpc_h,
                downsample_threads=int(args.downsample_threads),
                class_n_modes=int(args.class_n_modes),
                quijote_fiducial=QUIJOTE_FIDUCIAL,
                output_dir=out_dir,
                save_outputs=True,
                show_progress=False,
                progress_desc_prefix="PPC subboxes",
            )
            delta_emu = np.asarray(pipeline["delta_emu"], dtype=np.float32)
            psi_emu = np.asarray(pipeline["psi_emu"], dtype=np.float32)
            runtime_s.append(float(pipeline["metadata"]["emulator_runtime_seconds"]))

        if int(delta_emu.shape[0]) != int(args.res):
            raise ValueError(f"Unexpected emulator density shape {delta_emu.shape}; expected ({args.res}, {args.res}, {args.res}).")

        xpk = PKL.XPk(
            [np.asarray(delta_emu, dtype=np.float32), np.asarray(delta_true, dtype=np.float32)],
            float(args.boxsize),
            axis=0,
            MAS=[mas_model, mas_target],
            threads=int(args.pk_threads),
        )
        k_local = np.asarray(xpk.k3D, dtype=np.float64)
        pk_local = np.asarray(xpk.Pk[:, 0, 0], dtype=np.float64)
        xcorr_local = np.asarray(xpk.XPk[:, 0, 0], dtype=np.float64) / np.sqrt(
            np.asarray(xpk.Pk[:, 0, 0], dtype=np.float64) * np.asarray(xpk.Pk[:, 0, 1], dtype=np.float64)
        )
        valid_local = (
            np.isfinite(k_local)
            & (k_local > 0)
            & np.isfinite(pk_local)
            & (pk_local > 0)
            & np.isfinite(xcorr_local)
        )
        k_local = k_local[valid_local]
        pk_local = pk_local[valid_local]
        xcorr_local = xcorr_local[valid_local]
        if k_local.shape != k_pk.shape or not np.allclose(k_local, k_pk, rtol=1e-7, atol=1e-10):
            pk_local = np.interp(k_pk, k_local, pk_local)
            xcorr_local = np.interp(k_pk, k_local, xcorr_local)

        pk_samples.append(np.asarray(pk_local, dtype=np.float64))
        xcorr_samples.append(np.asarray(xcorr_local, dtype=np.float64))

        q1 = np.asarray(
            PKL.Bk(
                np.asarray(delta_emu, dtype=np.float32),
                float(args.boxsize),
                float(args.bispec_k1),
                float(args.bispec_k2),
                theta,
                mas_model,
                threads=int(args.pk_threads),
            ).Q,
            dtype=np.float64,
        )
        q2 = np.asarray(
            PKL.Bk(
                np.asarray(delta_emu, dtype=np.float32),
                float(args.boxsize),
                float(args.bispec_k1b),
                float(args.bispec_k2b),
                theta,
                mas_model,
                threads=int(args.pk_threads),
            ).Q,
            dtype=np.float64,
        )
        q_samples_cfg1.append(q1)
        q_samples_cfg2.append(q2)

        mink = compute_minkowski_functionals(
            field=np.asarray(delta_emu, dtype=np.float32),
            boxsize=float(args.boxsize),
            thresholds=nu_grid,
            standardize=True,
        )
        for key in ("v0", "v1", "v2", "v3"):
            mink_samples[key].append(np.asarray(mink[key], dtype=np.float64))

        if bool(args.compute_hmf):
            masses = run_single_rank_fof_masses(
                psi=np.asarray(psi_emu, dtype=np.float32),
                boxsize=float(args.boxsize),
                n_part=int(args.n_part),
                omega_m=float(omega_m),
                linking_length=float(args.fof_linking_length),
                nmin=int(args.fof_nmin),
                absolute_linking=bool(args.fof_absolute_linking),
            )
            masses = masses[np.isfinite(masses) & (masses > 0.0)]
            if masses.size == 0:
                raise ValueError(f"No halos found for sample {sample_path.name}.")

            if hmf_log_edges is None:
                mmin = float(args.hmf_mmin) if args.hmf_mmin is not None else float(np.min(masses))
                mmax = float(args.hmf_mmax) if args.hmf_mmax is not None else float(np.max(masses))
                if not (mmax > mmin > 0.0):
                    raise ValueError(f"Invalid HMF bounds from first sample: mmin={mmin}, mmax={mmax}.")
                hmf_log_edges = np.linspace(np.log10(mmin), np.log10(mmax), int(args.hmf_nbins) + 1)
                hmf_centers = 10.0 ** (0.5 * (hmf_log_edges[1:] + hmf_log_edges[:-1]))

                if args.truth_displacement_path is not None:
                    truth_dis = np.asarray(np.load(args.truth_displacement_path), dtype=np.float32)
                    truth_masses = run_single_rank_fof_masses(
                        psi=truth_dis,
                        boxsize=float(args.boxsize),
                        n_part=int(args.n_part),
                        omega_m=float(omega_m),
                        linking_length=float(args.fof_linking_length),
                        nmin=int(args.fof_nmin),
                        absolute_linking=bool(args.fof_absolute_linking),
                    )
                    hmf_truth = compute_hmf_curve(
                        masses=np.asarray(truth_masses, dtype=np.float64),
                        log_edges=np.asarray(hmf_log_edges, dtype=np.float64),
                        boxsize=float(args.boxsize),
                    )

            hmf_curve = compute_hmf_curve(
                masses=np.asarray(masses, dtype=np.float64),
                log_edges=np.asarray(hmf_log_edges, dtype=np.float64),
                boxsize=float(args.boxsize),
            )
            hmf_samples.append(np.asarray(hmf_curve, dtype=np.float64))

            np.savez_compressed(
                out_dir / "ppc_sample_summary.npz",
                sample_id=np.int32(sample_id),
                k=np.asarray(k_pk, dtype=np.float64),
                pk=np.asarray(pk_local, dtype=np.float64),
                crosscorr=np.asarray(xcorr_local, dtype=np.float64),
                theta_deg=np.asarray(theta_deg, dtype=np.float64),
                bispectrum_q_cfg1=np.asarray(q1, dtype=np.float64),
                bispectrum_q_cfg2=np.asarray(q2, dtype=np.float64),
                nu=np.asarray(nu_grid, dtype=np.float64),
                minkowski_v0=np.asarray(mink["v0"], dtype=np.float64),
                minkowski_v1=np.asarray(mink["v1"], dtype=np.float64),
                minkowski_v2=np.asarray(mink["v2"], dtype=np.float64),
                minkowski_v3=np.asarray(mink["v3"], dtype=np.float64),
                hmf_mass_centers=np.asarray(hmf_centers if hmf_centers is not None else [], dtype=np.float64),
                hmf_dn_dlog10m=np.asarray(hmf_curve, dtype=np.float64),
            )
        else:
            np.savez_compressed(
                out_dir / "ppc_sample_summary.npz",
                sample_id=np.int32(sample_id),
                k=np.asarray(k_pk, dtype=np.float64),
                pk=np.asarray(pk_local, dtype=np.float64),
                crosscorr=np.asarray(xcorr_local, dtype=np.float64),
                theta_deg=np.asarray(theta_deg, dtype=np.float64),
                bispectrum_q_cfg1=np.asarray(q1, dtype=np.float64),
                bispectrum_q_cfg2=np.asarray(q2, dtype=np.float64),
                nu=np.asarray(nu_grid, dtype=np.float64),
                minkowski_v0=np.asarray(mink["v0"], dtype=np.float64),
                minkowski_v1=np.asarray(mink["v1"], dtype=np.float64),
                minkowski_v2=np.asarray(mink["v2"], dtype=np.float64),
                minkowski_v3=np.asarray(mink["v3"], dtype=np.float64),
            )

    pk_samples_arr = np.asarray(pk_samples, dtype=np.float64)
    xcorr_samples_arr = np.asarray(xcorr_samples, dtype=np.float64)
    q_samples_cfg1_arr = np.asarray(q_samples_cfg1, dtype=np.float64)
    q_samples_cfg2_arr = np.asarray(q_samples_cfg2, dtype=np.float64)
    mink_samples_arr = {k: np.asarray(v, dtype=np.float64) for k, v in mink_samples.items()}

    payload = {
        "sample_ids": np.asarray(sample_ids, dtype=np.int32),
        "sample_input_paths": np.asarray(sample_input_paths),
        "sample_output_dirs": np.asarray(sample_output_dirs),
        "emulator_runtime_seconds": np.asarray(runtime_s, dtype=np.float64),
        "k_pk_h_per_mpc": np.asarray(k_pk, dtype=np.float64),
        "pk_true": np.asarray(pk_true, dtype=np.float64),
        "pk_samples": np.asarray(pk_samples_arr, dtype=np.float64),
        "crosscorr_samples": np.asarray(xcorr_samples_arr, dtype=np.float64),
        "theta_deg": np.asarray(theta_deg, dtype=np.float64),
        "bispectrum_q_true_cfg1": np.asarray(q_true_cfg1, dtype=np.float64),
        "bispectrum_q_true_cfg2": np.asarray(q_true_cfg2, dtype=np.float64),
        "bispectrum_q_samples_cfg1": np.asarray(q_samples_cfg1_arr, dtype=np.float64),
        "bispectrum_q_samples_cfg2": np.asarray(q_samples_cfg2_arr, dtype=np.float64),
        "minkowski_nu": np.asarray(nu_grid, dtype=np.float64),
        "minkowski_true_v0": np.asarray(mink_truth["v0"], dtype=np.float64),
        "minkowski_true_v1": np.asarray(mink_truth["v1"], dtype=np.float64),
        "minkowski_true_v2": np.asarray(mink_truth["v2"], dtype=np.float64),
        "minkowski_true_v3": np.asarray(mink_truth["v3"], dtype=np.float64),
        "minkowski_samples_v0": np.asarray(mink_samples_arr["v0"], dtype=np.float64),
        "minkowski_samples_v1": np.asarray(mink_samples_arr["v1"], dtype=np.float64),
        "minkowski_samples_v2": np.asarray(mink_samples_arr["v2"], dtype=np.float64),
        "minkowski_samples_v3": np.asarray(mink_samples_arr["v3"], dtype=np.float64),
        "hmf_mass_centers": np.asarray([] if hmf_centers is None else hmf_centers, dtype=np.float64),
        "hmf_samples": np.asarray(hmf_samples, dtype=np.float64),
        "hmf_truth": np.asarray([] if hmf_truth is None else hmf_truth, dtype=np.float64),
    }

    prefix = str(args.aggregate_prefix)
    arrays_path = aggregate_dir / f"{prefix}_summary_arrays.npz"
    save_summary_npz(arrays_path, payload)

    plot_pk_path = aggregate_dir / f"{prefix}_pk.png"
    plot_xcorr_path = aggregate_dir / f"{prefix}_crosscorr.png"
    plot_bispec_path = aggregate_dir / f"{prefix}_bispectrum.png"
    plot_mink_path = aggregate_dir / f"{prefix}_minkowski.png"
    plot_hmf_path = aggregate_dir / f"{prefix}_hmf.png"

    print("Saving aggregate PPC plots...")
    plot_pk_summary(
        k=np.asarray(k_pk, dtype=np.float64),
        pk_true=np.asarray(pk_true, dtype=np.float64),
        pk_samples=np.asarray(pk_samples_arr, dtype=np.float64),
        boxsize=float(args.boxsize),
        res=int(args.res),
        out_path=plot_pk_path,
    )
    plot_crosscorr_summary(
        k=np.asarray(k_pk, dtype=np.float64),
        xcorr_samples=np.asarray(xcorr_samples_arr, dtype=np.float64),
        out_path=plot_xcorr_path,
    )
    cfg1_lbl = rf"$k_1={args.bispec_k1},\ k_2={args.bispec_k2}\ h/\mathrm{{Mpc}}$"
    cfg2_lbl = rf"$k_1={args.bispec_k1b},\ k_2={args.bispec_k2b}\ h/\mathrm{{Mpc}}$"
    plot_bispectrum_summary(
        theta_deg=np.asarray(theta_deg, dtype=np.float64),
        q_true_cfg1=np.asarray(q_true_cfg1, dtype=np.float64),
        q_true_cfg2=np.asarray(q_true_cfg2, dtype=np.float64),
        q_samples_cfg1=np.asarray(q_samples_cfg1_arr, dtype=np.float64),
        q_samples_cfg2=np.asarray(q_samples_cfg2_arr, dtype=np.float64),
        cfg1_label=cfg1_lbl,
        cfg2_label=cfg2_lbl,
        out_path=plot_bispec_path,
    )
    plot_minkowski_summary(
        nu=np.asarray(nu_grid, dtype=np.float64),
        truth=mink_truth,
        samples={k: np.asarray(v, dtype=np.float64) for k, v in mink_samples_arr.items()},
        out_path=plot_mink_path,
    )
    if bool(args.compute_hmf) and len(hmf_samples) > 0 and hmf_centers is not None:
        plot_hmf_summary(
            m_centers=np.asarray(hmf_centers, dtype=np.float64),
            hmf_samples=np.asarray(hmf_samples, dtype=np.float64),
            hmf_truth=None if hmf_truth is None else np.asarray(hmf_truth, dtype=np.float64),
            out_path=plot_hmf_path,
        )

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples_dir": str(samples_dir),
        "target_path": str(args.target_path),
        "target_key": str(args.target_key),
        "num_samples_requested": int(args.num_samples),
        "num_samples_processed": int(len(sample_ids)),
        "sample_ids": [int(x) for x in sample_ids],
        "sample_output_dirs": [str(x) for x in sample_output_dirs],
        "n_part": int(args.n_part),
        "res": int(args.res),
        "boxsize": float(args.boxsize),
        "z_ic": float(args.z_ic),
        "z_target": float(args.z_target),
        "ic_rescale_mode": str(args.ic_rescale_mode),
        "ic_rescale_factor": float(scale),
        "mas_worder": int(args.mas_worder),
        "deconvolve_mas": bool(args.deconvolve_mas),
        "target_deconvolved": bool(args.target_deconvolved),
        "mas_model_for_summaries": str(mas_model),
        "mas_target_for_summaries": str(mas_target),
        "ndiv": [int(x) for x in args.ndiv],
        "emu_precision": str(args.emu_precision),
        "num_sims_per_sample": int(args.num_sims),
        "compute_hmf": bool(args.compute_hmf),
        "truth_hmf_available": bool(hmf_truth is not None),
        "hmf_truth_note": (
            "truth_displacement_path was provided"
            if hmf_truth is not None
            else "No truth displacement available; HMF truth overlay omitted."
        ),
        "aggregate_arrays_file": str(arrays_path),
        "aggregate_pk_plot": str(plot_pk_path),
        "aggregate_crosscorr_plot": str(plot_xcorr_path),
        "aggregate_bispectrum_plot": str(plot_bispec_path),
        "aggregate_minkowski_plot": str(plot_mink_path),
        "aggregate_hmf_plot": str(plot_hmf_path) if (bool(args.compute_hmf) and len(hmf_samples) > 0 and hmf_centers is not None) else None,
    }
    metadata_path = aggregate_dir / f"{prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("Posterior predictive check finished.")
    print(f"Processed samples: {len(sample_ids)}")
    print(f"Aggregate arrays: {arrays_path}")
    print(f"Aggregate metadata: {metadata_path}")
    print("Aggregate plots:")
    print(f"  - {plot_pk_path}")
    print(f"  - {plot_xcorr_path}")
    print(f"  - {plot_bispec_path}")
    print(f"  - {plot_mink_path}")
    if bool(args.compute_hmf) and len(hmf_samples) > 0 and hmf_centers is not None:
        print(f"  - {plot_hmf_path}")
    else:
        print("  - HMF plot not produced (compute_hmf disabled or unavailable).")


if __name__ == "__main__":
    main()
