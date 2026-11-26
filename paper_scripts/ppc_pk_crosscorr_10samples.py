#!/usr/bin/env python3
"""
Plot posterior IC and posterior-resimulation P(k), ratio to truth, and cross-correlation C(k).

This script reads:
- posterior IC samples: --samples-dir/<sample-pattern>
- posterior resimulations: --samples-dir/<sample_folder_template>/emu_delta_<ztag>.npy

It then creates a 3x2 diagnostic figure:
- left column: IC comparison (posterior ICs vs true IC)
- right column: final-field comparison (posterior resimulations vs true final)

Rows are identical for both columns:
1) P(k) with mean and +-1/2 sigma bands
2) Ratio to truth with mean and +-1/2 sigma bands
3) C(k) cross-correlation with truth, mean and +-1/2 sigma bands

Outputs:
- <prefix>_pk_ratio_crosscorr.png
- <prefix>_pk_ratio_crosscorr_arrays.npz
- <prefix>_pk_ratio_crosscorr_metadata.json
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
    from utils import mas_from_worder, resize_density_grid
except ImportError as exc:  # pragma: no cover - runtime dependency
    mas_from_worder = None
    resize_density_grid = None
    _UTILS_IMPORT_ERROR = exc
else:
    _UTILS_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute and plot IC + final P(k), ratio, and C(k) for posterior samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("/gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD"),
        help="Directory containing sample_*.npy and per-sample output folders.",
    )
    parser.add_argument("--sample-pattern", type=str, default="sample_*.npy", help="IC sample glob pattern.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of selected samples to include.")
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
    parser.add_argument("--ic-key", type=str, default="delta_z127", help="Truth IC field key inside target .pt file.")
    parser.add_argument("--target-key", type=str, default="delta_z0", help="Truth final field key inside target .pt file.")

    parser.add_argument(
        "--ic-target-deconvolved",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether target IC field is already MAS-deconvolved.",
    )
    parser.add_argument(
        "--ic-samples-deconvolved",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether posterior IC samples are already MAS-deconvolved.",
    )
    parser.add_argument(
        "--target-deconvolved",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether target final field is already MAS-deconvolved.",
    )

    parser.add_argument("--z-target", type=float, default=0.0, help="Target redshift for emu_delta_<ztag>.npy.")
    parser.add_argument("--res", type=int, default=128, help="Density-grid resolution.")
    parser.add_argument("--boxsize", type=float, default=1000.0, help="Simulation box size in Mpc/h.")
    parser.add_argument("--mas-worder", type=int, choices=(2, 3, 4), default=4, help="MAS order for emulator outputs.")
    parser.add_argument(
        "--deconvolve-mas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether emulator fields are already MAS-deconvolved.",
    )
    parser.add_argument(
        "--delta-upsample-method",
        choices=("mode_inject", "fourier", "linear"),
        default="mode_inject",
        help="Method for resizing truth/IC fields if their resolution differs from --res.",
    )
    parser.add_argument("--pk-threads", type=int, default=1, help="Thread count for Pk_library calls.")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. "
            "Default: /home/osavchenko/gaussian_npe/paper_plots_scripts/<RUN_NAME>/posterior_resimulation"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="posterior_predictive_10samples",
        help="Output filename prefix.",
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


def ensure_resolution(
    field: np.ndarray,
    *,
    target_res: int,
    boxsize: float,
    upsample_method: str,
) -> np.ndarray:
    """Resize a cubic field to target_res only when needed."""
    arr = np.asarray(field, dtype=np.float32)
    if int(arr.shape[0]) == int(target_res):
        return arr
    return resize_density_grid(
        delta_in=arr,
        target_res=int(target_res),
        boxsize=float(boxsize),
        upsample_method=str(upsample_method),
    )


def compute_pk_truth(
    *,
    delta_true: np.ndarray,
    boxsize: float,
    mas_target: str,
    pk_threads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute truth P(k) on positive k bins."""
    pk_true_obj = PKL.Pk(
        np.asarray(delta_true, dtype=np.float32),
        float(boxsize),
        axis=0,
        MAS=str(mas_target),
        threads=int(pk_threads),
        verbose=False,
    )
    k_pk = np.asarray(pk_true_obj.k3D, dtype=np.float64)
    pk_true = np.asarray(pk_true_obj.Pk[:, 0], dtype=np.float64)
    valid = np.isfinite(k_pk) & (k_pk > 0) & np.isfinite(pk_true) & (pk_true > 0)
    return k_pk[valid], pk_true[valid]


def compute_pk_xcorr_sample(
    *,
    field_sample: np.ndarray,
    field_true: np.ndarray,
    boxsize: float,
    mas_model: str,
    mas_target: str,
    pk_threads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute P(k) and C(k) for one sample field against its truth field."""
    xpk = PKL.XPk(
        [np.asarray(field_sample, dtype=np.float32), np.asarray(field_true, dtype=np.float32)],
        float(boxsize),
        axis=0,
        MAS=[str(mas_model), str(mas_target)],
        threads=int(pk_threads),
    )

    k = np.asarray(xpk.k3D, dtype=np.float64)
    pk_sample = np.asarray(xpk.Pk[:, 0, 0], dtype=np.float64)
    pk_true_local = np.asarray(xpk.Pk[:, 0, 1], dtype=np.float64)
    xpk_sample_true = np.asarray(xpk.XPk[:, 0, 0], dtype=np.float64)
    denom = np.sqrt(pk_sample * pk_true_local)
    xcorr = np.divide(xpk_sample_true, denom, out=np.full_like(xpk_sample_true, np.nan), where=denom > 0)

    valid = np.isfinite(k) & (k > 0) & np.isfinite(pk_sample) & (pk_sample > 0) & np.isfinite(xcorr)
    return k[valid], pk_sample[valid], xcorr[valid]


def _draw_pk_ratio_xcorr_column(
    *,
    axes_col: np.ndarray,
    k: np.ndarray,
    pk_true: np.ndarray,
    pk_samples: np.ndarray,
    xcorr_samples: np.ndarray,
    column_title: str,
    truth_label: str,
    sample_label: str,
    ratio_label: str,
    xcorr_label: str,
    k_nyq: float,
    sample_color: str,
) -> None:
    """Draw one 3-row column: P(k), ratio, and C(k)."""
    k = np.asarray(k, dtype=np.float64)
    pk_true = np.asarray(pk_true, dtype=np.float64)
    pk_samples = np.asarray(pk_samples, dtype=np.float64)
    xcorr_samples = np.asarray(xcorr_samples, dtype=np.float64)

    pk_mu, pk_sigma = mean_sigma(pk_samples)
    ratio_samples = np.divide(
        pk_samples,
        pk_true[None, :],
        out=np.full_like(pk_samples, np.nan),
        where=pk_true[None, :] > 0,
    )
    ratio_mu, ratio_sigma = mean_sigma(ratio_samples)
    xcorr_mu, xcorr_sigma = mean_sigma(xcorr_samples)

    ax = axes_col[0]
    ax.plot(k, pk_true, color="black", lw=1.4, label=truth_label)
    ax.plot(k, pk_mu, color=sample_color, lw=1.2, label=sample_label)
    ax.fill_between(k, pk_mu - pk_sigma, pk_mu + pk_sigma, color=sample_color, alpha=0.35, label=r"$\pm 1\sigma$")
    ax.fill_between(k, pk_mu - 2.0 * pk_sigma, pk_mu + 2.0 * pk_sigma, color=sample_color, alpha=0.16, label=r"$\pm 2\sigma$")
    ax.axvline(k_nyq, color="red", ls="--", lw=0.8, alpha=0.6, label=r"$k_{\rm Nyq}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$P(k)$")
    ax.set_title(column_title)
    ax.grid(alpha=0.15, which="both")
    ax.legend(framealpha=0.9, fontsize=8, loc="best")

    ax = axes_col[1]
    ax.plot(k, ratio_mu, color=sample_color, lw=1.2, label=ratio_label)
    ax.fill_between(k, ratio_mu - ratio_sigma, ratio_mu + ratio_sigma, color=sample_color, alpha=0.35)
    ax.fill_between(k, ratio_mu - 2.0 * ratio_sigma, ratio_mu + 2.0 * ratio_sigma, color=sample_color, alpha=0.16)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.axvline(k_nyq, color="red", ls="--", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0.0, 2.0)
    ax.grid(alpha=0.15, which="both")

    ax = axes_col[2]
    ax.plot(k, xcorr_mu, color="tab:green", lw=1.2, label=xcorr_label)
    ax.fill_between(k, xcorr_mu - xcorr_sigma, xcorr_mu + xcorr_sigma, color="tab:green", alpha=0.35, label=r"$\pm 1\sigma$")
    ax.fill_between(k, xcorr_mu - 2.0 * xcorr_sigma, xcorr_mu + 2.0 * xcorr_sigma, color="tab:green", alpha=0.16, label=r"$\pm 2\sigma$")
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Truth self-correlation")
    ax.axvline(k_nyq, color="red", ls="--", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel(r"$k$ [$h/{\rm Mpc}$]")
    ax.set_ylabel(r"$C(k)$")
    ax.grid(alpha=0.15, which="both")
    ax.legend(framealpha=0.9, fontsize=8, loc="best")


def plot_pk_ratio_crosscorr_two_column(
    *,
    k_ic: np.ndarray,
    pk_true_ic: np.ndarray,
    pk_samples_ic: np.ndarray,
    xcorr_samples_ic: np.ndarray,
    k_final: np.ndarray,
    pk_true_final: np.ndarray,
    pk_samples_final: np.ndarray,
    xcorr_samples_final: np.ndarray,
    boxsize: float,
    res: int,
    out_path: Path,
) -> None:
    """Plot 3x2 diagnostics: IC column on left, final-resim column on right."""
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13.4, 9.0),
        constrained_layout=True,
        sharex="col",
        height_ratios=(2.0, 1.0, 1.0),
    )

    k_nyq = np.pi * int(res) / float(boxsize)

    _draw_pk_ratio_xcorr_column(
        axes_col=axes[:, 0],
        k=np.asarray(k_ic, dtype=np.float64),
        pk_true=np.asarray(pk_true_ic, dtype=np.float64),
        pk_samples=np.asarray(pk_samples_ic, dtype=np.float64),
        xcorr_samples=np.asarray(xcorr_samples_ic, dtype=np.float64),
        column_title="ICs: posterior samples vs true IC",
        truth_label="True IC field",
        sample_label="Posterior IC mean",
        ratio_label="Posterior IC / true IC",
        xcorr_label="Posterior IC mean C(k)",
        k_nyq=float(k_nyq),
        sample_color="tab:orange",
    )

    _draw_pk_ratio_xcorr_column(
        axes_col=axes[:, 1],
        k=np.asarray(k_final, dtype=np.float64),
        pk_true=np.asarray(pk_true_final, dtype=np.float64),
        pk_samples=np.asarray(pk_samples_final, dtype=np.float64),
        xcorr_samples=np.asarray(xcorr_samples_final, dtype=np.float64),
        column_title="Resimulations: emulator outputs vs true final",
        truth_label="True final field",
        sample_label="Posterior resim mean",
        ratio_label="Resim / true final",
        xcorr_label="Resim mean C(k)",
        k_nyq=float(k_nyq),
        sample_color="tab:blue",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Entry point."""
    args = parse_args()

    if _UTILS_IMPORT_ERROR is not None:
        raise ImportError("Failed to import utilities from scripts/utils.py") from _UTILS_IMPORT_ERROR
    if PKL is None:
        raise ImportError("Pk_library is required for this script") from _PKL_IMPORT_ERROR

    samples_dir = args.samples_dir.resolve()
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    if args.output_dir is None:
        run_name = samples_dir.name
        output_dir = (
            Path("/home/osavchenko/gaussian_npe/paper_plots_scripts")
            / run_name
            / "posterior_resimulation"
        )
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files_sel = select_sample_files(
        samples_dir=samples_dir,
        pattern=str(args.sample_pattern),
        sample_start=int(args.sample_start),
        sample_stride=int(args.sample_stride),
        num_samples=int(args.num_samples),
    )
    print(f"Selected {len(files_sel)} sample files.")

    print(f"Loading truth IC field from {args.target_path} [{args.ic_key}]...")
    delta_true_ic = load_target_field(target_path=args.target_path, target_key=args.ic_key)
    delta_true_ic = ensure_resolution(
        field=delta_true_ic,
        target_res=int(args.res),
        boxsize=float(args.boxsize),
        upsample_method=str(args.delta_upsample_method),
    )

    print(f"Loading truth final field from {args.target_path} [{args.target_key}]...")
    delta_true_final = load_target_field(target_path=args.target_path, target_key=args.target_key)
    delta_true_final = ensure_resolution(
        field=delta_true_final,
        target_res=int(args.res),
        boxsize=float(args.boxsize),
        upsample_method=str(args.delta_upsample_method),
    )

    mas_name = mas_from_worder(int(args.mas_worder))

    mas_model_ic = "None" if bool(args.ic_samples_deconvolved) else mas_name
    mas_target_ic = "None" if bool(args.ic_target_deconvolved) else mas_name

    mas_model_final = "None" if bool(args.deconvolve_mas) else mas_name
    mas_target_final = "None" if bool(args.target_deconvolved) else mas_name

    k_ref_ic, pk_true_ic = compute_pk_truth(
        delta_true=np.asarray(delta_true_ic, dtype=np.float32),
        boxsize=float(args.boxsize),
        mas_target=str(mas_target_ic),
        pk_threads=int(args.pk_threads),
    )
    k_ref_final, pk_true_final = compute_pk_truth(
        delta_true=np.asarray(delta_true_final, dtype=np.float32),
        boxsize=float(args.boxsize),
        mas_target=str(mas_target_final),
        pk_threads=int(args.pk_threads),
    )

    z_tag = z_file_tag(float(args.z_target))
    sample_ids: list[int] = []
    sample_output_dirs: list[str] = []

    pk_samples_ic: list[np.ndarray] = []
    xcorr_samples_ic: list[np.ndarray] = []

    pk_samples_final: list[np.ndarray] = []
    xcorr_samples_final: list[np.ndarray] = []

    for idx, sample_path in enumerate(files_sel):
        sample_id = extract_sample_id(sample_path, fallback_index=idx)
        out_dir = sample_output_dir(
            samples_dir=samples_dir,
            template=str(args.sample_folder_template),
            sample_id=int(sample_id),
            sample_path=sample_path,
        )

        # IC sample (left column)
        ic_sample = np.asarray(np.load(sample_path), dtype=np.float32)
        if ic_sample.ndim != 3 or len(set(ic_sample.shape)) != 1:
            raise ValueError(f"IC sample must be cubic 3D, got {sample_path} with shape={ic_sample.shape}")
        ic_sample = ensure_resolution(
            field=ic_sample,
            target_res=int(args.res),
            boxsize=float(args.boxsize),
            upsample_method=str(args.delta_upsample_method),
        )

        k_local_ic, pk_local_ic, xcorr_local_ic = compute_pk_xcorr_sample(
            field_sample=ic_sample,
            field_true=np.asarray(delta_true_ic, dtype=np.float32),
            boxsize=float(args.boxsize),
            mas_model=str(mas_model_ic),
            mas_target=str(mas_target_ic),
            pk_threads=int(args.pk_threads),
        )
        if k_local_ic.shape != k_ref_ic.shape or not np.allclose(k_local_ic, k_ref_ic, rtol=1e-7, atol=1e-10):
            pk_local_ic = np.interp(k_ref_ic, k_local_ic, pk_local_ic)
            xcorr_local_ic = np.interp(k_ref_ic, k_local_ic, xcorr_local_ic)

        # Resimulation sample (right column)
        emu_delta_path = out_dir / f"emu_delta_{z_tag}.npy"
        if not emu_delta_path.is_file():
            raise FileNotFoundError(
                f"Missing forward output: {emu_delta_path}. "
                "Run paper_scripts/ppc_forward.py first or adjust --z-target/--sample-folder-template."
            )
        delta_emu = np.asarray(np.load(emu_delta_path), dtype=np.float32)
        expected_shape = (int(args.res), int(args.res), int(args.res))
        if delta_emu.shape != expected_shape:
            raise ValueError(f"Unexpected emulator density shape {delta_emu.shape}; expected {expected_shape}.")

        k_local_fin, pk_local_fin, xcorr_local_fin = compute_pk_xcorr_sample(
            field_sample=delta_emu,
            field_true=np.asarray(delta_true_final, dtype=np.float32),
            boxsize=float(args.boxsize),
            mas_model=str(mas_model_final),
            mas_target=str(mas_target_final),
            pk_threads=int(args.pk_threads),
        )
        if k_local_fin.shape != k_ref_final.shape or not np.allclose(k_local_fin, k_ref_final, rtol=1e-7, atol=1e-10):
            pk_local_fin = np.interp(k_ref_final, k_local_fin, pk_local_fin)
            xcorr_local_fin = np.interp(k_ref_final, k_local_fin, xcorr_local_fin)

        sample_ids.append(int(sample_id))
        sample_output_dirs.append(str(out_dir))

        pk_samples_ic.append(np.asarray(pk_local_ic, dtype=np.float64))
        xcorr_samples_ic.append(np.asarray(xcorr_local_ic, dtype=np.float64))

        pk_samples_final.append(np.asarray(pk_local_fin, dtype=np.float64))
        xcorr_samples_final.append(np.asarray(xcorr_local_fin, dtype=np.float64))

        print(f"Processed sample_id={sample_id} ({idx + 1}/{len(files_sel)})")

    pk_samples_ic_arr = np.asarray(pk_samples_ic, dtype=np.float64)
    xcorr_samples_ic_arr = np.asarray(xcorr_samples_ic, dtype=np.float64)
    ratio_samples_ic_arr = np.divide(
        pk_samples_ic_arr,
        pk_true_ic[None, :],
        out=np.full_like(pk_samples_ic_arr, np.nan),
        where=pk_true_ic[None, :] > 0,
    )

    pk_samples_final_arr = np.asarray(pk_samples_final, dtype=np.float64)
    xcorr_samples_final_arr = np.asarray(xcorr_samples_final, dtype=np.float64)
    ratio_samples_final_arr = np.divide(
        pk_samples_final_arr,
        pk_true_final[None, :],
        out=np.full_like(pk_samples_final_arr, np.nan),
        where=pk_true_final[None, :] > 0,
    )

    prefix = str(args.output_prefix)
    plot_path = output_dir / f"{prefix}_pk_ratio_crosscorr.png"
    arrays_path = output_dir / f"{prefix}_pk_ratio_crosscorr_arrays.npz"
    metadata_path = output_dir / f"{prefix}_pk_ratio_crosscorr_metadata.json"

    plot_pk_ratio_crosscorr_two_column(
        k_ic=np.asarray(k_ref_ic, dtype=np.float64),
        pk_true_ic=np.asarray(pk_true_ic, dtype=np.float64),
        pk_samples_ic=np.asarray(pk_samples_ic_arr, dtype=np.float64),
        xcorr_samples_ic=np.asarray(xcorr_samples_ic_arr, dtype=np.float64),
        k_final=np.asarray(k_ref_final, dtype=np.float64),
        pk_true_final=np.asarray(pk_true_final, dtype=np.float64),
        pk_samples_final=np.asarray(pk_samples_final_arr, dtype=np.float64),
        xcorr_samples_final=np.asarray(xcorr_samples_final_arr, dtype=np.float64),
        boxsize=float(args.boxsize),
        res=int(args.res),
        out_path=plot_path,
    )

    np.savez_compressed(
        arrays_path,
        sample_ids=np.asarray(sample_ids, dtype=np.int32),
        sample_output_dirs=np.asarray(sample_output_dirs),
        # IC diagnostics
        k_ic_h_per_mpc=np.asarray(k_ref_ic, dtype=np.float64),
        pk_true_ic=np.asarray(pk_true_ic, dtype=np.float64),
        pk_samples_ic=np.asarray(pk_samples_ic_arr, dtype=np.float64),
        ratio_samples_ic=np.asarray(ratio_samples_ic_arr, dtype=np.float64),
        crosscorr_samples_ic=np.asarray(xcorr_samples_ic_arr, dtype=np.float64),
        # Final diagnostics
        k_final_h_per_mpc=np.asarray(k_ref_final, dtype=np.float64),
        pk_true_final=np.asarray(pk_true_final, dtype=np.float64),
        pk_samples_final=np.asarray(pk_samples_final_arr, dtype=np.float64),
        ratio_samples_final=np.asarray(ratio_samples_final_arr, dtype=np.float64),
        crosscorr_samples_final=np.asarray(xcorr_samples_final_arr, dtype=np.float64),
        # Backward-compatible aliases pointing to final diagnostics
        k_pk_h_per_mpc=np.asarray(k_ref_final, dtype=np.float64),
        pk_true=np.asarray(pk_true_final, dtype=np.float64),
        pk_samples=np.asarray(pk_samples_final_arr, dtype=np.float64),
        ratio_samples=np.asarray(ratio_samples_final_arr, dtype=np.float64),
        crosscorr_samples=np.asarray(xcorr_samples_final_arr, dtype=np.float64),
    )

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples_dir": str(samples_dir),
        "sample_pattern": str(args.sample_pattern),
        "sample_folder_template": str(args.sample_folder_template),
        "num_samples_requested": int(args.num_samples),
        "num_samples_processed": int(len(sample_ids)),
        "sample_ids": [int(x) for x in sample_ids],
        "sample_output_dirs": [str(x) for x in sample_output_dirs],
        "target_path": str(args.target_path),
        "ic_key": str(args.ic_key),
        "target_key": str(args.target_key),
        "ic_target_deconvolved": bool(args.ic_target_deconvolved),
        "ic_samples_deconvolved": bool(args.ic_samples_deconvolved),
        "target_deconvolved": bool(args.target_deconvolved),
        "z_target": float(args.z_target),
        "res": int(args.res),
        "boxsize": float(args.boxsize),
        "mas_worder": int(args.mas_worder),
        "deconvolve_mas": bool(args.deconvolve_mas),
        "mas_model_ic": str(mas_model_ic),
        "mas_target_ic": str(mas_target_ic),
        "mas_model_final": str(mas_model_final),
        "mas_target_final": str(mas_target_final),
        "plot_layout": "3x2: left=IC diagnostics, right=resimulation diagnostics",
        "output_plot": str(plot_path),
        "output_arrays": str(arrays_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("Done.")
    print(f"Plot: {plot_path}")
    print(f"Arrays: {arrays_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
