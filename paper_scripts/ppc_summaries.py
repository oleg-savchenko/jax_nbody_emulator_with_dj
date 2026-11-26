#!/usr/bin/env python3
"""
CPU summary stage for posterior predictive emulator outputs.

This script expects per-sample forward outputs already present in
`--samples-dir/<sample_folder_template>/emu_delta_<ztag>.npy` and computes:
- P(k)
- reduced bispectrum Q(theta) for two triangle configurations
- cross-correlation with truth final field
- Minkowski functionals V0..V3

Then it aggregates over samples and writes summary arrays + plots.

Example:
  python paper_scripts/ppc_summaries.py \
    --samples-dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD \
    --target-path /home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt \
    --num-samples 100 \
    --res 128 \
    --boxsize 1000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ppc_common import (
    extract_sample_id,
    mean_sigma,
    sample_output_dir,
    save_summary_npz,
    select_sample_files,
    z_file_tag,
)

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
    from utils import compute_minkowski_functionals, mas_from_worder, resize_density_grid
except ImportError as exc:  # pragma: no cover - runtime dependency
    compute_minkowski_functionals = None
    mas_from_worder = None
    resize_density_grid = None
    _UTILS_IMPORT_ERROR = exc
else:
    _UTILS_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for PPC summary aggregation."""
    parser = argparse.ArgumentParser(
        description="Compute PPC summaries from precomputed emulator outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("/gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD"),
        help="Directory containing IC samples and per-sample emulator folders.",
    )
    parser.add_argument("--sample-pattern", type=str, default="sample_*.npy", help="IC sample glob pattern.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of selected samples to summarize.")
    parser.add_argument("--sample-start", type=int, default=0, help="Starting index in sorted sample list.")
    parser.add_argument("--sample-stride", type=int, default=1, help="Stride in sorted sample list.")
    parser.add_argument(
        "--sample-folder-template",
        type=str,
        default="sample_{sample_id:04d}",
        help="Per-sample output folder template under --samples-dir.",
    )

    parser.add_argument(
        "--target-path",
        type=Path,
        default=Path("/home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt"),
        help="Path to truth .pt file.",
    )
    parser.add_argument("--target-key", type=str, default="delta_z0", help="Truth field key inside target .pt file.")
    parser.add_argument(
        "--target-deconvolved",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether target field is already MAS-deconvolved.",
    )

    parser.add_argument("--z-target", type=float, default=0.0, help="Target redshift used for emu_delta_<ztag>.npy lookup.")
    parser.add_argument("--res", type=int, default=128, help="Summary density-grid resolution.")
    parser.add_argument("--boxsize", type=float, default=1000.0, help="Simulation box size in Mpc/h.")
    parser.add_argument("--mas-worder", type=int, choices=(2, 3, 4), default=4, help="MAS order used for emulator fields.")
    parser.add_argument(
        "--deconvolve-mas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether emulator/LPT density fields are MAS-deconvolved.",
    )
    parser.add_argument(
        "--delta-upsample-method",
        choices=("mode_inject", "fourier", "linear"),
        default="mode_inject",
        help="Method for resizing truth field if its resolution differs from --res.",
    )

    parser.add_argument("--pk-threads", type=int, default=1, help="Thread count for Pk_library calls.")
    parser.add_argument("--bispec-k1", type=float, default=0.1, help="First bispectrum config k1 [h/Mpc].")
    parser.add_argument("--bispec-k2", type=float, default=0.1, help="First bispectrum config k2 [h/Mpc].")
    parser.add_argument("--bispec-k1b", type=float, default=0.05, help="Second bispectrum config k1 [h/Mpc].")
    parser.add_argument("--bispec-k2b", type=float, default=0.1, help="Second bispectrum config k2 [h/Mpc].")
    parser.add_argument("--bispec-n-theta", type=int, default=25, help="Number of theta points in bispectrum.")

    parser.add_argument("--minkowski-nu-min", type=float, default=-3.0, help="Minimum Minkowski threshold.")
    parser.add_argument("--minkowski-nu-max", type=float, default=3.0, help="Maximum Minkowski threshold.")
    parser.add_argument("--minkowski-n-thresholds", type=int, default=41, help="Number of Minkowski thresholds.")

    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=None,
        help=(
            "Aggregate output directory. "
            "Default: /home/osavchenko/gaussian_npe/paper_plots_scripts/<RUN_NAME>/posterior_resimulation, "
            "where RUN_NAME is basename(--samples-dir)."
        ),
    )
    parser.add_argument("--aggregate-prefix", type=str, default="posterior_predictive", help="Output filename prefix.")
    parser.add_argument(
        "--save-per-sample-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-sample summary npz inside each sample output folder.",
    )

    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.sample_start < 0:
        parser.error("--sample-start must be >= 0")
    if args.sample_stride < 1:
        parser.error("--sample-stride must be >= 1")
    if args.res < 1:
        parser.error("--res must be >= 1")
    if args.minkowski_n_thresholds < 2:
        parser.error("--minkowski-n-thresholds must be >= 2")
    if args.bispec_n_theta < 3:
        parser.error("--bispec-n-theta must be >= 3")
    return args


def load_target_field(target_path: Path, target_key: str) -> np.ndarray:
    """Load truth field from torch .pt and return a float32 cubic grid."""
    if torch is None:
        raise ImportError("torch is required to read target .pt files") from _TORCH_IMPORT_ERROR
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


def ensure_target_resolution(
    delta_target: np.ndarray,
    *,
    target_res: int,
    boxsize: float,
    upsample_method: str,
) -> np.ndarray:
    """Resize truth target field to summary resolution when required."""
    arr = np.asarray(delta_target, dtype=np.float32)
    if int(arr.shape[0]) == int(target_res):
        return arr
    return resize_density_grid(
        delta_in=arr,
        target_res=int(target_res),
        boxsize=float(boxsize),
        upsample_method=str(upsample_method),
    )


def plot_pk_summary(
    *,
    k: np.ndarray,
    pk_true: np.ndarray,
    pk_samples: np.ndarray,
    xcorr_samples: np.ndarray | None,
    boxsize: float,
    res: int,
    out_path: Path,
) -> None:
    """Plot posterior predictive P(k), ratio, and optional C(k) bands in one figure."""
    k = np.asarray(k, dtype=np.float64)
    pk_true = np.asarray(pk_true, dtype=np.float64)
    pk_samples = np.asarray(pk_samples, dtype=np.float64)
    mu, sigma = mean_sigma(pk_samples)
    k_nyq = np.pi * int(res) / float(boxsize)

    ratio = np.divide(pk_samples, pk_true[None, :], out=np.full_like(pk_samples, np.nan), where=pk_true[None, :] > 0)
    ratio_mu, ratio_sigma = mean_sigma(ratio)

    has_xcorr = xcorr_samples is not None
    if has_xcorr:
        xcorr_arr = np.asarray(xcorr_samples, dtype=np.float64)
        xcorr_mu, xcorr_sigma = mean_sigma(xcorr_arr)
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(7.2, 9.0),
            sharex=True,
            constrained_layout=True,
            height_ratios=(2.0, 1.0, 1.0),
        )
    else:
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

    if has_xcorr:
        ax = axes[2]
        ax.plot(k, xcorr_mu, color="tab:green", lw=1.2, label="Predictive mean C(k)")
        ax.fill_between(k, xcorr_mu - xcorr_sigma, xcorr_mu + xcorr_sigma, color="tab:green", alpha=0.35, label=r"$\pm 1\sigma$")
        ax.fill_between(
            k,
            xcorr_mu - 2.0 * xcorr_sigma,
            xcorr_mu + 2.0 * xcorr_sigma,
            color="tab:green",
            alpha=0.16,
            label=r"$\pm 2\sigma$",
        )
        ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Truth self-correlation")
        ax.axvline(k_nyq, color="red", ls="--", lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel(r"$k$ [$h/{\rm Mpc}$]")
        ax.set_ylabel(r"$C(k)$")
        ax.grid(alpha=0.15, which="both")
        ax.legend(framealpha=0.9, fontsize=8, loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_crosscorr_summary(*, k: np.ndarray, xcorr_samples: np.ndarray, out_path: Path) -> None:
    """Plot posterior predictive cross-correlation C(k) bands."""
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
    """Plot reduced bispectrum posterior predictive bands for two configs."""
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
    """Plot posterior predictive Minkowski bands V0..V3 against truth."""
    keys = ["v0", "v1", "v2", "v3"]
    ylabels = [r"$V_0$", r"$V_1$ [$h/{\rm Mpc}$]", r"$V_2$ [$h^2/{\rm Mpc}^2$]", r"$V_3$ [$(\mathrm{Mpc}/h)^{-3}$]"]
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


def main() -> None:
    """Compute and aggregate CPU summary statistics from existing emulator outputs."""
    args = parse_args()

    if _UTILS_IMPORT_ERROR is not None:
        raise ImportError("Failed to import utilities from scripts/utils.py") from _UTILS_IMPORT_ERROR
    if PKL is None:
        raise ImportError("Pk_library is required for PPC summaries") from _PKL_IMPORT_ERROR

    samples_dir = args.samples_dir.resolve()
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    if args.aggregate_dir is None:
        run_name = samples_dir.name
        aggregate_dir = (
            Path("/home/osavchenko/gaussian_npe/paper_plots_scripts")
            / run_name
            / "posterior_resimulation"
        )
    else:
        aggregate_dir = args.aggregate_dir.resolve()
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    files_sel = select_sample_files(
        samples_dir=samples_dir,
        pattern=str(args.sample_pattern),
        sample_start=int(args.sample_start),
        sample_stride=int(args.sample_stride),
        num_samples=int(args.num_samples),
    )
    print(f"Selected {len(files_sel)} sample files for summary computation.")

    print(f"Loading target field from {args.target_path} [{args.target_key}]...")
    delta_true = load_target_field(target_path=args.target_path, target_key=args.target_key)
    delta_true = ensure_target_resolution(
        delta_target=delta_true,
        target_res=int(args.res),
        boxsize=float(args.boxsize),
        upsample_method=str(args.delta_upsample_method),
    )

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
    mink_truth = {key: np.asarray(mink_truth_full[key], dtype=np.float64) for key in ("v0", "v1", "v2", "v3")}

    sample_ids: list[int] = []
    sample_output_dirs: list[str] = []
    pk_samples: list[np.ndarray] = []
    xcorr_samples: list[np.ndarray] = []
    q_samples_cfg1: list[np.ndarray] = []
    q_samples_cfg2: list[np.ndarray] = []
    mink_samples = {"v0": [], "v1": [], "v2": [], "v3": []}

    z_tag = z_file_tag(float(args.z_target))
    for idx, sample_path in enumerate(files_sel):
        sample_id = extract_sample_id(sample_path, fallback_index=idx)
        out_dir = sample_output_dir(
            samples_dir=samples_dir,
            template=str(args.sample_folder_template),
            sample_id=int(sample_id),
            sample_path=sample_path,
        )
        emu_delta_path = out_dir / f"emu_delta_{z_tag}.npy"
        if not emu_delta_path.is_file():
            raise FileNotFoundError(
                f"Missing forward output: {emu_delta_path}. "
                "Run paper_scripts/ppc_forward.py first or adjust --z-target/--sample-folder-template."
            )

        delta_emu = np.asarray(np.load(emu_delta_path), dtype=np.float32)
        if delta_emu.shape != (int(args.res), int(args.res), int(args.res)):
            raise ValueError(
                f"Unexpected emulator density shape {delta_emu.shape}; expected ({args.res}, {args.res}, {args.res})."
            )

        xpk = PKL.XPk(
            [delta_emu, np.asarray(delta_true, dtype=np.float32)],
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

        valid_local = np.isfinite(k_local) & (k_local > 0) & np.isfinite(pk_local) & (pk_local > 0) & np.isfinite(xcorr_local)
        k_local = k_local[valid_local]
        pk_local = pk_local[valid_local]
        xcorr_local = xcorr_local[valid_local]

        if k_local.shape != k_pk.shape or not np.allclose(k_local, k_pk, rtol=1e-7, atol=1e-10):
            pk_local = np.interp(k_pk, k_local, pk_local)
            xcorr_local = np.interp(k_pk, k_local, xcorr_local)

        q1 = np.asarray(
            PKL.Bk(
                delta_emu,
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
                delta_emu,
                float(args.boxsize),
                float(args.bispec_k1b),
                float(args.bispec_k2b),
                theta,
                mas_model,
                threads=int(args.pk_threads),
            ).Q,
            dtype=np.float64,
        )

        mink = compute_minkowski_functionals(
            field=delta_emu,
            boxsize=float(args.boxsize),
            thresholds=nu_grid,
            standardize=True,
        )

        sample_ids.append(int(sample_id))
        sample_output_dirs.append(str(out_dir))
        pk_samples.append(np.asarray(pk_local, dtype=np.float64))
        xcorr_samples.append(np.asarray(xcorr_local, dtype=np.float64))
        q_samples_cfg1.append(np.asarray(q1, dtype=np.float64))
        q_samples_cfg2.append(np.asarray(q2, dtype=np.float64))
        for key in ("v0", "v1", "v2", "v3"):
            mink_samples[key].append(np.asarray(mink[key], dtype=np.float64))

        if bool(args.save_per_sample_summary):
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
    mink_samples_arr = {key: np.asarray(values, dtype=np.float64) for key, values in mink_samples.items()}

    payload = {
        "sample_ids": np.asarray(sample_ids, dtype=np.int32),
        "sample_output_dirs": np.asarray(sample_output_dirs),
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
    }

    prefix = str(args.aggregate_prefix)
    arrays_path = aggregate_dir / f"{prefix}_summary_arrays.npz"
    save_summary_npz(arrays_path, payload)

    plot_pk_path = aggregate_dir / f"{prefix}_pk.png"
    plot_xcorr_path = aggregate_dir / f"{prefix}_crosscorr.png"
    plot_bispec_path = aggregate_dir / f"{prefix}_bispectrum.png"
    plot_mink_path = aggregate_dir / f"{prefix}_minkowski.png"

    plot_pk_summary(
        k=np.asarray(k_pk, dtype=np.float64),
        pk_true=np.asarray(pk_true, dtype=np.float64),
        pk_samples=np.asarray(pk_samples_arr, dtype=np.float64),
        xcorr_samples=np.asarray(xcorr_samples_arr, dtype=np.float64),
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
        samples={key: np.asarray(vals, dtype=np.float64) for key, vals in mink_samples_arr.items()},
        out_path=plot_mink_path,
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
        "res": int(args.res),
        "boxsize": float(args.boxsize),
        "z_target": float(args.z_target),
        "mas_worder": int(args.mas_worder),
        "deconvolve_mas": bool(args.deconvolve_mas),
        "target_deconvolved": bool(args.target_deconvolved),
        "mas_model_for_summaries": str(mas_model),
        "mas_target_for_summaries": str(mas_target),
        "aggregate_arrays_file": str(arrays_path),
        "aggregate_pk_plot": str(plot_pk_path),
        "aggregate_crosscorr_plot": str(plot_xcorr_path),
        "aggregate_bispectrum_plot": str(plot_bispec_path),
        "aggregate_minkowski_plot": str(plot_mink_path),
    }
    metadata_path = aggregate_dir / f"{prefix}_summary_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("Summary stage complete.")
    print(f"Samples processed: {len(sample_ids)}")
    print(f"Summary arrays: {arrays_path}")
    print(f"Summary metadata: {metadata_path}")


if __name__ == "__main__":
    main()
