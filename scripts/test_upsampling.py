#!/usr/bin/env python3
"""
Test and visualize power-spectrum damping caused by GRF upsampling.

This script:
1. Loads a low-resolution linear GRF density field from a .npy file or .pt file.
2. Rescales the input field amplitude from `--input-z` to `--z` using Quijote's growth approximation.
3. Upsamples it to a target resolution using `resize_density_grid` from `utils.py`.
4. Compares both against CLASS linear P(k).
5. Downsamples the upsampled field back to input resolution with both Gaussian and block-average methods.
6. Computes cross-correlation C(k) for both downsampling methods on the common input grid.

Example:
JAX_PLATFORMS=cpu python scripts/test_upsampling.py \
  --input-delta-path /home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt \
  --input-delta-key delta_z127 \
  --input-z 127 \
  --target-res 512 \
  --boxsize 1000 \
  --upsample-method mode_inject \
  --z 0.0 \
  --output-dir outputs/upsampling_damping_test

Run on GPU from terminal (Slurm allocation + execution):
srun -p gpu_a100 --gpus=1 --cpus-per-task=18 --time=00:02:00 \
  --export=ALL,JAX_PLATFORMS=cuda \
  python scripts/test_upsampling.py \
  --upsample-method mode_inject --target-res 512

If you are already inside a GPU Slurm job:
JAX_PLATFORMS=cuda python scripts/test_upsampling.py \
  --upsample-method mode_inject --target-res 512
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import Pk_library as PKL
import torch

from utils import (
    QUIJOTE_FIDUCIAL_CLASS,
    add_import_paths,
    get_pk_class,
    growth_D_approx,
    resize_density_grid,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
add_import_paths(repo_root=REPO_ROOT)
PLOT_KMIN = 5.0e-3


def parse_args() -> argparse.Namespace:
    """Define and validate CLI options for the GRF upsampling damping test."""
    parser = argparse.ArgumentParser(
        description="Load a GRF field, upsample it, and compare P(k) to CLASS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-delta-path",
        type=Path,
        default=Path("/home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt"),
        help="Path to low-resolution linear GRF density field (.npy) or Quijote .pt file.",
    )
    parser.add_argument(
        "--input-delta-key",
        type=str,
        default="delta_z127",
        help="Dictionary key used when loading --input-delta-path from a .pt file.",
    )
    parser.add_argument(
        "--target-res",
        type=int,
        default=512,
        help="Target resolution for upsampled field.",
    )
    parser.add_argument("--boxsize", type=float, default=1000.0, help="Box size in Mpc/h.")
    parser.add_argument(
        "--upsample-method",
        choices=("mode_inject", "fourier", "linear"),
        default="mode_inject",
        help="Interpolation method passed to `resize_density_grid`.",
    )
    parser.add_argument("--z", type=float, default=0.0, help="Redshift for CLASS linear P(k).")
    parser.add_argument(
        "--input-z",
        type=float,
        default=127.0,
        help="Redshift of the provided input density field before growth rescaling.",
    )
    parser.add_argument(
        "--class-n-samples",
        type=int,
        default=512,
        help="Number of CLASS k-samples for smooth theory curve.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/upsampling_damping_test"),
        help="Output directory for plot and metadata.",
    )
    parser.add_argument(
        "--save-upsampled-field",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save the upsampled field as .npy (can be large).",
    )

    args = parser.parse_args()
    if args.target_res < 1:
        parser.error("--target-res must be >= 1")
    if args.class_n_samples < 16:
        parser.error("--class-n-samples must be >= 16")
    return args


def load_input_delta(input_path: Path, input_key: str) -> np.ndarray:
    """Load a cubic 3D density field from .npy or .pt (dictionary key required for .pt)."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input field not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".npy":
        delta = np.asarray(np.load(input_path), dtype=np.float32)
    elif suffix == ".pt":
        sample = torch.load(input_path, map_location="cpu", weights_only=False)
        if not isinstance(sample, dict):
            raise TypeError(f"Expected a dict in {input_path}, got {type(sample)}")
        if input_key not in sample:
            raise KeyError(f"Missing key {input_key!r} in {input_path}")
        delta = np.asarray(sample[input_key], dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported input format {input_path.suffix!r}. Expected .npy or .pt."
        )

    if delta.ndim != 3 or len(set(delta.shape)) != 1:
        raise ValueError(f"Input field must be cubic 3D. Got shape={delta.shape}")
    return delta


def compute_pk(delta: np.ndarray, boxsize: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute P(k) for a gridded density field without MAS deconvolution."""
    obj = PKL.Pk(np.asarray(delta, dtype=np.float32), float(boxsize), axis=0, MAS="None", threads=1, verbose=False)
    k = np.asarray(obj.k3D, dtype=np.float64)
    pk = np.asarray(obj.Pk[:, 0], dtype=np.float64)
    valid = (k > 0) & np.isfinite(pk) & (pk > 0)
    return k[valid], pk[valid]


def compute_crosscorr(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    boxsize: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cross-correlation C(k)=Pab/sqrt(Paa*Pbb) for two same-grid fields."""
    a = np.asarray(delta_a, dtype=np.float32)
    b = np.asarray(delta_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Cross-correlation requires same shapes, got {a.shape} and {b.shape}")

    pk_a = PKL.Pk(a, float(boxsize), axis=0, MAS="None", threads=1, verbose=False)
    pk_b = PKL.Pk(b, float(boxsize), axis=0, MAS="None", threads=1, verbose=False)
    xpk = PKL.XPk([a, b], float(boxsize), axis=0, MAS=["None", "None"], threads=1)

    k = np.asarray(pk_a.k3D, dtype=np.float64)
    p_aa = np.asarray(pk_a.Pk[:, 0], dtype=np.float64)
    p_bb = np.asarray(pk_b.Pk[:, 0], dtype=np.float64)
    p_ab = np.asarray(xpk.XPk[:, 0, 0], dtype=np.float64)

    denom = np.sqrt(np.clip(p_aa * p_bb, a_min=0.0, a_max=None))
    valid = (k > 0) & np.isfinite(p_ab) & np.isfinite(denom) & (denom > 0.0)
    c = np.zeros_like(k, dtype=np.float64)
    c[valid] = p_ab[valid] / denom[valid]
    return k[valid], c[valid]


def get_class_cosmo() -> dict[str, float]:
    """Build CLASS cosmology dictionary from fiducial Quijote parameters."""
    return dict(QUIJOTE_FIDUCIAL_CLASS)


def make_plot(
    *,
    k_low: np.ndarray,
    pk_low: np.ndarray,
    k_up: np.ndarray,
    pk_up: np.ndarray,
    k_cross_gaussian: np.ndarray,
    c_cross_gaussian: np.ndarray,
    k_cross_block: np.ndarray,
    c_cross_block: np.ndarray,
    k_class: np.ndarray,
    pk_class: np.ndarray,
    boxsize: float,
    res_low: int,
    res_up: int,
    kmin_plot: float,
    out_path: Path,
) -> None:
    """Plot P(k), ratio-to-CLASS, and cross-correlation diagnostics."""
    pk_class_low = np.interp(k_low, k_class, pk_class)
    pk_class_up = np.interp(k_up, k_class, pk_class)

    k_nyq_low = np.pi * int(res_low) / float(boxsize)
    k_nyq_up = np.pi * int(res_up) / float(boxsize)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 9.2),
        sharex=True,
        constrained_layout=True,
        height_ratios=(2.0, 1.0, 1.0),
    )

    ax = axes[0]
    ax.plot(k_low, pk_low, lw=1.3, label=f"Input GRF ({res_low}^3)")
    ax.plot(k_up, pk_up, lw=1.3, label=f"Upsampled GRF ({res_up}^3)")
    ax.plot(k_class, pk_class, "k--", lw=1.2, alpha=0.85, label="CLASS linear")
    ax.axvline(k_nyq_low, color="tab:green", ls="--", lw=0.9, alpha=0.8, label=r"$k_{\rm Nyq, in}$")
    ax.axvline(k_nyq_up, color="tab:red", ls=":", lw=0.9, alpha=0.8, label=r"$k_{\rm Nyq, up}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=float(kmin_plot))
    ax.set_ylabel(r"$P(k)$")
    ax.set_title("GRF upsampling test: small-scale damping near target Nyquist")
    ax.grid(which="both", alpha=0.15)
    ax.legend(framealpha=0.9)

    ax = axes[1]
    ax.plot(k_low, pk_low / pk_class_low, lw=1.3, label="Input / CLASS")
    ax.plot(k_up, pk_up / pk_class_up, lw=1.3, label="Upsampled / CLASS")
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.axvline(k_nyq_low, color="tab:green", ls="--", lw=0.9, alpha=0.8)
    ax.axvline(k_nyq_up, color="tab:red", ls=":", lw=0.9, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlim(left=float(kmin_plot))
    ax.set_ylim(0.0, 2.0)
    ax.set_xlabel(r"$k$ [$h/{\rm Mpc}$]")
    ax.set_ylabel("Ratio")
    ax.grid(which="both", alpha=0.15)
    ax.legend(framealpha=0.9)

    ax = axes[2]
    ax.plot(
        k_cross_gaussian,
        c_cross_gaussian,
        lw=1.3,
        label=r"$C(k)$ input vs downsample(upsampled, Gaussian)",
    )
    ax.plot(
        k_cross_block,
        c_cross_block,
        lw=1.3,
        label=r"$C(k)$ input vs downsample(upsampled, Block)",
    )
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.axvline(k_nyq_low, color="tab:green", ls="--", lw=0.9, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlim(left=float(kmin_plot))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$k$ [$h/{\rm Mpc}$]")
    ax.set_ylabel(r"$C(k)$")
    ax.grid(which="both", alpha=0.15)
    ax.legend(framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_slice_comparison_plot(
    *,
    delta_low: np.ndarray,
    delta_up: np.ndarray,
    delta_up_back_low_gaussian: np.ndarray,
    delta_up_back_low_block: np.ndarray,
    gaussian_sigma_mpc_h: float,
    out_path: Path,
) -> tuple[int, int, int, int]:
    """Plot middle slices of input, upsampled, and both downsampled-back fields."""
    delta_low = np.asarray(delta_low, dtype=np.float32)
    delta_up = np.asarray(delta_up, dtype=np.float32)
    delta_up_back_low_gaussian = np.asarray(delta_up_back_low_gaussian, dtype=np.float32)
    delta_up_back_low_block = np.asarray(delta_up_back_low_block, dtype=np.float32)

    slice_low = int(delta_low.shape[0] // 2)
    slice_up = int(delta_up.shape[0] // 2)
    slice_back_gaussian = int(delta_up_back_low_gaussian.shape[0] // 2)
    slice_back_block = int(delta_up_back_low_block.shape[0] // 2)
    img_low = delta_low[slice_low, :, :]
    img_up = delta_up[slice_up, :, :]
    img_back_gaussian = delta_up_back_low_gaussian[slice_back_gaussian, :, :]
    img_back_block = delta_up_back_low_block[slice_back_block, :, :]

    vals = np.concatenate(
        [
            img_low.ravel(),
            img_up.ravel(),
            img_back_gaussian.ravel(),
            img_back_block.ravel(),
        ]
    )
    vmin = float(np.percentile(vals, 1.0))
    vmax = float(np.percentile(vals, 99.0))

    fig, axes = plt.subplots(1, 4, figsize=(16.6, 4.2), constrained_layout=True)
    im0 = axes[0].imshow(img_low, origin="lower", cmap="seismic", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Input GRF\nslice x={slice_low}")
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("y")

    im1 = axes[1].imshow(img_up, origin="lower", cmap="seismic", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Upsampled GRF\nslice x={slice_up}")
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("y")

    im2 = axes[2].imshow(img_back_gaussian, origin="lower", cmap="seismic", vmin=vmin, vmax=vmax)
    axes[2].set_title(
        "Downsampled-back GRF (Gaussian, "
        f"$\\sigma$={float(gaussian_sigma_mpc_h):.3g} Mpc/h)\n"
        f"slice x={slice_back_gaussian}"
    )
    axes[2].set_xlabel("z")
    axes[2].set_ylabel("y")

    im3 = axes[3].imshow(img_back_block, origin="lower", cmap="seismic", vmin=vmin, vmax=vmax)
    axes[3].set_title(f"Downsampled-back GRF (Block)\nslice x={slice_back_block}")
    axes[3].set_xlabel("z")
    axes[3].set_ylabel("y")

    fig.colorbar(im3, ax=list(axes), shrink=0.95, label=r"$\delta$")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return slice_low, slice_up, slice_back_gaussian, slice_back_block


def main() -> None:
    """Run the upsampling damping test and save plot plus metadata."""
    t0_total = time.perf_counter()
    args = parse_args()
    add_import_paths(repo_root=REPO_ROOT)

    t0 = time.perf_counter()
    delta_low = load_input_delta(args.input_delta_path, args.input_delta_key)
    cosmo = get_class_cosmo()
    d_input = growth_D_approx(cosmo, float(args.input_z))
    d_target = growth_D_approx(cosmo, float(args.z))
    growth_rescale = d_target / d_input
    delta_low = np.asarray(delta_low, dtype=np.float32) * np.float32(growth_rescale)
    t_load = time.perf_counter() - t0

    res_low = int(delta_low.shape[0])
    res_up = int(args.target_res)
    input_voxel_size_mpc_h = float(args.boxsize) / float(res_low)
    gaussian_sigma_mpc_h = 0.5 * float(input_voxel_size_mpc_h)

    t0 = time.perf_counter()
    delta_up = resize_density_grid(
        delta_in=delta_low,
        target_res=res_up,
        boxsize=float(args.boxsize),
        upsample_method=str(args.upsample_method),
        class_cosmo_params=cosmo,
        class_z=float(args.z),
        class_n_modes=int(args.class_n_samples),
    )
    t_upsample = time.perf_counter() - t0

    t0 = time.perf_counter()
    delta_up_back_low_gaussian = resize_density_grid(
        delta_in=delta_up,
        target_res=res_low,
        boxsize=float(args.boxsize),
        upsample_method=str(args.upsample_method),
        downsample_method="gaussian",
        downsample_gaussian_sigma_mpc_h=float(gaussian_sigma_mpc_h),
    )
    t_downsample_gaussian = time.perf_counter() - t0

    t0 = time.perf_counter()
    delta_up_back_low_block = resize_density_grid(
        delta_in=delta_up,
        target_res=res_low,
        boxsize=float(args.boxsize),
        upsample_method=str(args.upsample_method),
        downsample_method="block_average",
    )
    t_downsample_block = time.perf_counter() - t0

    t0 = time.perf_counter()
    k_low, pk_low = compute_pk(delta_low, boxsize=float(args.boxsize))
    k_up, pk_up = compute_pk(delta_up, boxsize=float(args.boxsize))
    k_cross_gaussian, c_cross_gaussian = compute_crosscorr(
        delta_low, delta_up_back_low_gaussian, boxsize=float(args.boxsize)
    )
    k_cross_block, c_cross_block = compute_crosscorr(
        delta_low, delta_up_back_low_block, boxsize=float(args.boxsize)
    )
    t_pk = time.perf_counter() - t0

    t0 = time.perf_counter()
    kmin = min(float(k_low.min()), float(k_up.min()), 1.0e-4)
    kmax = max(float(k_low.max()), float(k_up.max()), 1.0)
    k_class, pk_class = get_pk_class(
        cosmo_params=cosmo,
        z=float(args.z),
        kmin=max(float(PLOT_KMIN), float(kmin)),
        kmax=kmax,
        n_modes=int(args.class_n_samples),
        non_lin=False,
    )
    t_class = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "upsampling_pk.png"
    make_plot(
        k_low=k_low,
        pk_low=pk_low,
        k_up=k_up,
        pk_up=pk_up,
        k_cross_gaussian=k_cross_gaussian,
        c_cross_gaussian=c_cross_gaussian,
        k_cross_block=k_cross_block,
        c_cross_block=c_cross_block,
        k_class=np.asarray(k_class, dtype=np.float64),
        pk_class=np.asarray(pk_class, dtype=np.float64),
        boxsize=float(args.boxsize),
        res_low=res_low,
        res_up=res_up,
        kmin_plot=float(PLOT_KMIN),
        out_path=plot_path,
    )
    t_plot_pk = time.perf_counter() - t0

    t0 = time.perf_counter()
    slice_plot_path = out_dir / "upsampling_slices.png"
    slice_low, slice_up, slice_back_gaussian, slice_back_block = make_slice_comparison_plot(
        delta_low=delta_low,
        delta_up=delta_up,
        delta_up_back_low_gaussian=delta_up_back_low_gaussian,
        delta_up_back_low_block=delta_up_back_low_block,
        gaussian_sigma_mpc_h=float(gaussian_sigma_mpc_h),
        out_path=slice_plot_path,
    )
    t_plot_slices = time.perf_counter() - t0
    t_plot = t_plot_pk + t_plot_slices

    if args.save_upsampled_field:
        t0 = time.perf_counter()
        np.save(out_dir / f"delta_upsampled_{res_up}.npy", np.asarray(delta_up, dtype=np.float32))
        t_save_up = time.perf_counter() - t0
    else:
        t_save_up = 0.0

    t0 = time.perf_counter()
    k_nyq_low = np.pi * res_low / float(args.boxsize)
    k_nyq_up = np.pi * res_up / float(args.boxsize)
    metadata = {
        "input_delta_path": str(args.input_delta_path),
        "input_delta_key": str(args.input_delta_key),
        "input_res": res_low,
        "target_res": res_up,
        "boxsize_mpc_over_h": float(args.boxsize),
        "input_z": float(args.input_z),
        "z": float(args.z),
        "growth_D_input": float(d_input),
        "growth_D_target": float(d_target),
        "growth_rescale_input_to_target": float(growth_rescale),
        "upsample_method": str(args.upsample_method),
        "downsample_method": "gaussian_and_block_average",
        "downsample_gaussian_sigma_mpc_h": float(gaussian_sigma_mpc_h),
        "input_voxel_size_mpc_h": float(input_voxel_size_mpc_h),
        "plot_kmin_h_per_mpc": float(PLOT_KMIN),
        "k_nyq_input_h_per_mpc": float(k_nyq_low),
        "k_nyq_target_h_per_mpc": float(k_nyq_up),
        "plot_path": str(plot_path),
        "slice_plot_path": str(slice_plot_path),
        "slice_index_input": int(slice_low),
        "slice_index_upsampled": int(slice_up),
        "slice_index_downsampled_gaussian": int(slice_back_gaussian),
        "slice_index_downsampled_block": int(slice_back_block),
        "crosscorr_definition": {
            "gaussian": "C(k) between input field and gaussian-downsampled(upsampled field) on input grid",
            "block_average": "C(k) between input field and block-downsampled(upsampled field) on input grid",
        },
        "save_upsampled_field": bool(args.save_upsampled_field),
        "timing_seconds": {
            "load_input": float(t_load),
            "upsample": float(t_upsample),
            "downsample_back_gaussian": float(t_downsample_gaussian),
            "downsample_back_block": float(t_downsample_block),
            "power_spectra_and_crosscorr": float(t_pk),
            "class_pk": float(t_class),
            "plot": float(t_plot),
            "plot_pk": float(t_plot_pk),
            "plot_slices": float(t_plot_slices),
            "save_upsampled_field": float(t_save_up),
            "metadata_write": None,
            "total": None,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    t_meta = time.perf_counter() - t0
    t_total = time.perf_counter() - t0_total

    metadata["timing_seconds"]["metadata_write"] = float(t_meta)
    metadata["timing_seconds"]["total"] = float(t_total)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Upsampling damping test finished.")
    print(f"Input field: {args.input_delta_path}")
    print(f"Input res -> target res: {res_low} -> {res_up}")
    print("Stage timings:")
    print(f"  - load input: {t_load:.2f} s")
    print(f"  - upsample ({args.upsample_method}): {t_upsample:.2f} s")
    print(
        "  - downsample back (gaussian): "
        f"{t_downsample_gaussian:.2f} s (sigma={gaussian_sigma_mpc_h:.6g} Mpc/h)"
    )
    print(f"  - downsample back (block): {t_downsample_block:.2f} s")
    print(f"  - P(k)+C(k): {t_pk:.2f} s")
    print(f"  - CLASS P(k): {t_class:.2f} s")
    print(f"  - plot (total): {t_plot:.2f} s")
    print(f"    * P(k) figure: {t_plot_pk:.2f} s")
    print(f"    * slice figure: {t_plot_slices:.2f} s")
    if args.save_upsampled_field:
        print(f"  - save upsampled field: {t_save_up:.2f} s")
    print(f"  - total: {t_total:.2f} s")
    print(f"Saved plot: {plot_path}")
    print(f"Saved slice plot: {slice_plot_path}")
    print(f"Saved metadata: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
