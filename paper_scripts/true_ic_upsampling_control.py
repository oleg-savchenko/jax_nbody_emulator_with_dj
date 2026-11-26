#!/usr/bin/env python3
"""
Control test: run emulator from true ICs with the same upsampling path as PPC.

This script is designed to answer a specific diagnostic question:
"If we take the true 128^3 IC field, upsample it to n_part (e.g. 512^3) with the
same method used in PPC, then run the emulator, do we still see the high-k bump
near Nyquist in final P(k)?"

Workflow:
1. Load true IC and true final field from a Quijote target .pt file.
2. Optionally rescale IC amplitude from z_ic to z_target using linear growth.
3. Run the same forward core pipeline as PPC (`run_lpt_emulator_pipeline`).
4. Compare:
   - IC input field vs pipeline IC summary field (`ics_grf`)
   - Emulated final field vs true final field
5. Save diagnostic plots and metadata.

Example:
  python paper_scripts/true_ic_upsampling_control.py \
    --target-path /home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt \
    --ic-key delta_z127 \
    --final-key delta_z0 \
    --n-part 512 \
    --res 128 \
    --boxsize 1000 \
    --ndiv 4,2,2 \
    --emu-precision f16 \
    --delta-upsample-method mode_inject
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    from utils import growth_D_approx, mas_from_worder, parse_ndiv, resize_density_grid
except ImportError as exc:  # pragma: no cover - runtime dependency
    growth_D_approx = None
    mas_from_worder = None
    parse_ndiv = None
    resize_density_grid = None
    _UTILS_IMPORT_ERROR = exc
else:
    _UTILS_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for true-IC upsampling control test."""
    parser = argparse.ArgumentParser(
        description="Run a true-IC upsampling control test and plot P(k) diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        default=Path("/home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt"),
        help="Path to target .pt file with IC and final density fields.",
    )
    parser.add_argument("--ic-key", type=str, default="delta_z127", help="Key for true IC field in --target-path.")
    parser.add_argument("--final-key", type=str, default="delta_z0", help="Key for true final field in --target-path.")
    parser.add_argument("--z-ic", type=float, default=127.0, help="Redshift of IC field.")
    parser.add_argument("--z-target", type=float, default=0.0, help="Target redshift for forward run.")
    parser.add_argument(
        "--ic-rescale-mode",
        choices=("growth", "none"),
        default="growth",
        help="How to scale IC amplitude before forward run.",
    )

    parser.add_argument("--n-part", type=int, default=512, help="Particle grid for emulator forward run.")
    parser.add_argument("--res", type=int, default=128, help="Summary density-grid resolution.")
    parser.add_argument("--boxsize", type=float, default=1000.0, help="Box size in Mpc/h.")
    parser.add_argument(
        "--ndiv",
        type=parse_ndiv if callable(parse_ndiv) else str,
        default=(4, 2, 2),
        help="Subbox partition for emulator.",
    )
    parser.add_argument("--emu-precision", choices=("f16", "f32"), default="f16", help="Emulator precision.")
    parser.add_argument("--num-sims", type=int, default=1, help="Internal emulator repeats (last is used).")
    parser.add_argument("--mas-worder", type=int, choices=(2, 3, 4), default=4, help="MAS order for meshing.")
    parser.add_argument(
        "--deconvolve-mas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MAS deconvolution for pipeline-produced fields.",
    )
    parser.add_argument(
        "--target-deconvolved",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether target final field is already MAS-deconvolved.",
    )
    parser.add_argument(
        "--delta-upsample-method",
        choices=("mode_inject", "fourier", "linear"),
        default="mode_inject",
        help="Upsampling method used by core pipeline.",
    )
    parser.add_argument(
        "--downsample-method",
        choices=("fourier_truncation", "block_average", "gaussian"),
        default="fourier_truncation",
        help="Downsampling method when a resize to lower resolution is required.",
    )
    parser.add_argument(
        "--downsample-gaussian-sigma-mpc-h",
        type=float,
        default=None,
        help="Gaussian smoothing scale for downsample_method=gaussian.",
    )
    parser.add_argument("--downsample-threads", type=int, default=1, help="Threads for gaussian downsampling helper.")
    parser.add_argument("--class-n-modes", type=int, default=4000, help="CLASS table sampling for mode_inject.")
    parser.add_argument("--pk-threads", type=int, default=1, help="Thread count for Pk_library calls.")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "true_ic_upsampling_control",
        help="Directory for saved emulator outputs and diagnostic plots.",
    )
    parser.add_argument(
        "--reuse-existing-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse saved forward outputs if present in --output-dir.",
    )
    args = parser.parse_args()

    if args.n_part < 1 or args.res < 1:
        parser.error("--n-part and --res must be >= 1")
    if args.num_sims < 1:
        parser.error("--num-sims must be >= 1")
    return args


def load_field_from_target(path: Path, key: str) -> np.ndarray:
    """Load a cubic 3D float32 field from a torch .pt dictionary."""
    if torch is None:
        raise ImportError("torch is required to read target .pt files.") from _TORCH_IMPORT_ERROR
    if not path.is_file():
        raise FileNotFoundError(f"Target file not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict in {path}, got {type(payload)}")
    if key not in payload:
        raise KeyError(f"Missing key {key!r} in {path}")
    arr = np.asarray(payload[key], dtype=np.float32)
    if arr.ndim != 3 or len(set(arr.shape)) != 1:
        raise ValueError(f"Field {key!r} must be cubic 3D, got shape={arr.shape}")
    return arr


def ic_growth_scale(z_ic: float, z_target: float) -> float:
    """Compute D(z_target)/D(z_ic) for Quijote fiducial cosmology."""
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


def compute_pk(delta: np.ndarray, boxsize: float, mas: str, threads: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute isotropic P(k) from a density grid using Pylians."""
    obj = PKL.Pk(np.asarray(delta, dtype=np.float32), float(boxsize), axis=0, MAS=str(mas), threads=int(threads), verbose=False)
    k = np.asarray(obj.k3D, dtype=np.float64)
    pk = np.asarray(obj.Pk[:, 0], dtype=np.float64)
    valid = np.isfinite(k) & (k > 0.0) & np.isfinite(pk) & (pk > 0.0)
    return k[valid], pk[valid]


def align_to_reference_k(k_ref: np.ndarray, k: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpolate a spectrum y(k) onto a reference k-grid."""
    return np.interp(np.asarray(k_ref, dtype=np.float64), np.asarray(k, dtype=np.float64), np.asarray(y, dtype=np.float64))


def plot_pk_ratio(
    *,
    k: np.ndarray,
    y_ref: np.ndarray,
    y_test: np.ndarray,
    label_ref: str,
    label_test: str,
    k_nyq: float,
    title: str,
    out_path: Path,
) -> None:
    """Plot spectrum comparison and test/reference ratio."""
    k = np.asarray(k, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    ratio = np.divide(y_test, y_ref, out=np.full_like(y_test, np.nan), where=y_ref > 0.0)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.2),
        sharex=True,
        constrained_layout=True,
        height_ratios=(2.0, 1.0),
    )

    ax = axes[0]
    ax.plot(k, y_ref, color="black", lw=1.4, label=label_ref)
    ax.plot(k, y_test, color="tab:blue", lw=1.2, label=label_test)
    ax.axvline(float(k_nyq), color="red", ls="--", lw=0.8, alpha=0.6, label=r"$k_{\rm Nyq}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$P(k)$")
    ax.set_title(title)
    ax.grid(alpha=0.15, which="both")
    ax.legend(framealpha=0.9, fontsize=8, loc="best")

    ax = axes[1]
    ax.plot(k, ratio, color="tab:blue", lw=1.2, label=f"{label_test}/{label_ref}")
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.axvline(float(k_nyq), color="red", ls="--", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [$h/{\rm Mpc}$]")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0.0, 2.5)
    ax.grid(alpha=0.15, which="both")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def nearest_ratio(k: np.ndarray, ratio: np.ndarray, kval: float) -> tuple[float, float]:
    """Return (k_nearest, ratio_nearest) around requested k."""
    idx = int(np.argmin(np.abs(np.asarray(k, dtype=np.float64) - float(kval))))
    return float(k[idx]), float(ratio[idx])


def main() -> None:
    """Run the true-IC upsampling control test and save diagnostics."""
    args = parse_args()

    if _CORE_IMPORT_ERROR is not None:
        raise ImportError("Failed to import core emulator pipeline from scripts/core.py") from _CORE_IMPORT_ERROR
    if _UTILS_IMPORT_ERROR is not None:
        raise ImportError("Failed to import utilities from scripts/utils.py") from _UTILS_IMPORT_ERROR
    if PKL is None:
        raise ImportError("Pk_library is required for this control test.") from _PKL_IMPORT_ERROR

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading target fields from {args.target_path} ...", flush=True)
    delta_ic_raw = load_field_from_target(path=args.target_path, key=str(args.ic_key))
    delta_final_true_raw = load_field_from_target(path=args.target_path, key=str(args.final_key))

    scale = 1.0
    if str(args.ic_rescale_mode) == "growth":
        scale = ic_growth_scale(z_ic=float(args.z_ic), z_target=float(args.z_target))
    print(f"IC rescaling mode: {args.ic_rescale_mode}, scale={scale:.8f}", flush=True)

    delta_ic_input = np.asarray(delta_ic_raw * np.float32(scale), dtype=np.float32)
    delta_final_true = resize_density_grid(
        delta_in=np.asarray(delta_final_true_raw, dtype=np.float32),
        target_res=int(args.res),
        boxsize=float(args.boxsize),
        upsample_method=str(args.delta_upsample_method),
        downsample_method=str(args.downsample_method),
        downsample_gaussian_sigma_mpc_h=args.downsample_gaussian_sigma_mpc_h,
        downsample_threads=int(args.downsample_threads),
    )
    delta_ic_ref_res = resize_density_grid(
        delta_in=np.asarray(delta_ic_input, dtype=np.float32),
        target_res=int(args.res),
        boxsize=float(args.boxsize),
        upsample_method=str(args.delta_upsample_method),
        downsample_method=str(args.downsample_method),
        downsample_gaussian_sigma_mpc_h=args.downsample_gaussian_sigma_mpc_h,
        downsample_threads=int(args.downsample_threads),
    )

    z_tag = f"z{args.z_target:g}".replace(".", "p")
    emu_delta_path = out_dir / f"emu_delta_{z_tag}.npy"
    ics_summary_path = out_dir / "delta_linear_z0.npy"
    metadata_path = out_dir / "metadata.json"

    if bool(args.reuse_existing_forward) and emu_delta_path.is_file() and ics_summary_path.is_file() and metadata_path.is_file():
        print("Reusing existing forward outputs from output directory.", flush=True)
        delta_emu = np.asarray(np.load(emu_delta_path), dtype=np.float32)
        delta_ic_pipeline = np.asarray(np.load(ics_summary_path), dtype=np.float32)
        pipeline_meta = json.loads(metadata_path.read_text())
    else:
        print("Running forward pipeline from true IC field...", flush=True)
        pipeline = run_lpt_emulator_pipeline(
            delta_linear=np.asarray(delta_ic_input, dtype=np.float32),
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
            output_dir=out_dir,
            save_outputs=True,
            show_progress=True,
            progress_desc_prefix="Control subboxes",
        )
        delta_emu = np.asarray(pipeline["delta_emu"], dtype=np.float32)
        delta_ic_pipeline = np.asarray(pipeline["ics_grf"], dtype=np.float32)
        pipeline_meta = dict(pipeline["metadata"])

    mas_name = mas_from_worder(int(args.mas_worder))
    mas_model = "None" if bool(args.deconvolve_mas) else mas_name
    mas_target = "None" if bool(args.target_deconvolved) else mas_name
    k_nyq = np.pi * int(args.res) / float(args.boxsize)

    print("Computing P(k) diagnostics...", flush=True)
    k_ic_ref, pk_ic_ref = compute_pk(delta=np.asarray(delta_ic_ref_res, dtype=np.float32), boxsize=float(args.boxsize), mas="None", threads=int(args.pk_threads))
    k_ic_pipe, pk_ic_pipe = compute_pk(delta=np.asarray(delta_ic_pipeline, dtype=np.float32), boxsize=float(args.boxsize), mas=mas_model, threads=int(args.pk_threads))
    pk_ic_pipe_i = align_to_reference_k(k_ic_ref, k_ic_pipe, pk_ic_pipe)

    k_fin_true, pk_fin_true = compute_pk(delta=np.asarray(delta_final_true, dtype=np.float32), boxsize=float(args.boxsize), mas=mas_target, threads=int(args.pk_threads))
    k_fin_emu, pk_fin_emu = compute_pk(delta=np.asarray(delta_emu, dtype=np.float32), boxsize=float(args.boxsize), mas=mas_model, threads=int(args.pk_threads))
    pk_fin_emu_i = align_to_reference_k(k_fin_true, k_fin_emu, pk_fin_emu)

    ratio_ic = np.divide(pk_ic_pipe_i, pk_ic_ref, out=np.full_like(pk_ic_ref, np.nan), where=pk_ic_ref > 0.0)
    ratio_fin = np.divide(pk_fin_emu_i, pk_fin_true, out=np.full_like(pk_fin_true, np.nan), where=pk_fin_true > 0.0)

    plot_pk_ratio(
        k=np.asarray(k_ic_ref, dtype=np.float64),
        y_ref=np.asarray(pk_ic_ref, dtype=np.float64),
        y_test=np.asarray(pk_ic_pipe_i, dtype=np.float64),
        label_ref="True IC (input, scaled, on summary grid)",
        label_test="Pipeline IC summary (after upsample/projection)",
        k_nyq=float(k_nyq),
        title="IC control: input vs pipeline IC summary",
        out_path=out_dir / "control_ic_pk.png",
    )
    plot_pk_ratio(
        k=np.asarray(k_fin_true, dtype=np.float64),
        y_ref=np.asarray(pk_fin_true, dtype=np.float64),
        y_test=np.asarray(pk_fin_emu_i, dtype=np.float64),
        label_ref="True final field",
        label_test="Emulated final field (from true IC upsampling test)",
        k_nyq=float(k_nyq),
        title="Final-field control: emulator vs truth",
        out_path=out_dir / "control_final_pk.png",
    )

    k_at_nyq_ic, r_at_nyq_ic = nearest_ratio(k=np.asarray(k_ic_ref, dtype=np.float64), ratio=np.asarray(ratio_ic, dtype=np.float64), kval=float(k_nyq))
    k_at_nyq_fin, r_at_nyq_fin = nearest_ratio(k=np.asarray(k_fin_true, dtype=np.float64), ratio=np.asarray(ratio_fin, dtype=np.float64), kval=float(k_nyq))
    k_at_08nyq_fin, r_at_08nyq_fin = nearest_ratio(
        k=np.asarray(k_fin_true, dtype=np.float64),
        ratio=np.asarray(ratio_fin, dtype=np.float64),
        kval=0.8 * float(k_nyq),
    )

    payload = {
        "target_path": str(args.target_path),
        "ic_key": str(args.ic_key),
        "final_key": str(args.final_key),
        "z_ic": float(args.z_ic),
        "z_target": float(args.z_target),
        "ic_rescale_mode": str(args.ic_rescale_mode),
        "ic_scale_factor": float(scale),
        "n_part": int(args.n_part),
        "res": int(args.res),
        "boxsize": float(args.boxsize),
        "ndiv": [int(x) for x in args.ndiv],
        "emu_precision": str(args.emu_precision),
        "delta_upsample_method": str(args.delta_upsample_method),
        "downsample_method": str(args.downsample_method),
        "mas_worder": int(args.mas_worder),
        "deconvolve_mas": bool(args.deconvolve_mas),
        "target_deconvolved": bool(args.target_deconvolved),
        "mas_model_for_pk": str(mas_model),
        "mas_target_for_pk": str(mas_target),
        "k_nyq_h_per_mpc": float(k_nyq),
        "ratio_ic_at_k_near_nyq": {"k": float(k_at_nyq_ic), "ratio": float(r_at_nyq_ic)},
        "ratio_final_at_k_near_0p8_nyq": {"k": float(k_at_08nyq_fin), "ratio": float(r_at_08nyq_fin)},
        "ratio_final_at_k_near_nyq": {"k": float(k_at_nyq_fin), "ratio": float(r_at_nyq_fin)},
        "pipeline_metadata": pipeline_meta,
        "plot_ic_pk": str(out_dir / "control_ic_pk.png"),
        "plot_final_pk": str(out_dir / "control_final_pk.png"),
    }
    (out_dir / "control_test_metadata.json").write_text(json.dumps(payload, indent=2))

    np.savez_compressed(
        out_dir / "control_test_spectra.npz",
        k_ic=np.asarray(k_ic_ref, dtype=np.float64),
        pk_ic_ref=np.asarray(pk_ic_ref, dtype=np.float64),
        pk_ic_pipeline=np.asarray(pk_ic_pipe_i, dtype=np.float64),
        ratio_ic=np.asarray(ratio_ic, dtype=np.float64),
        k_final=np.asarray(k_fin_true, dtype=np.float64),
        pk_final_true=np.asarray(pk_fin_true, dtype=np.float64),
        pk_final_emu=np.asarray(pk_fin_emu_i, dtype=np.float64),
        ratio_final=np.asarray(ratio_fin, dtype=np.float64),
    )

    print("Control test complete.", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(f"Final ratio near 0.8*k_Nyq: k={k_at_08nyq_fin:.6f}, ratio={r_at_08nyq_fin:.4f}", flush=True)
    print(f"Final ratio near k_Nyq:    k={k_at_nyq_fin:.6f}, ratio={r_at_nyq_fin:.4f}", flush=True)


if __name__ == "__main__":
    main()
