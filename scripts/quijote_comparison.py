#!/usr/bin/env python3
"""
Run the pretrained emulator from Quijote ICs and compare to a Quijote target field.

This script:
1. Loads a Quijote sample `.pt` file (e.g. fields `delta_z127` and `delta_z0`).
2. Rescales the IC field from `z_ic` to `z_target` using the linear growth factor.
3. Resizes fields when needed:
   - IC field to `n_part` resolution for LPT/emulator input (via `--delta-upsample-method`).
   - Target field to `res` for comparison (via `--target-upsample-method`).
4. Builds 1LPT initial displacement from the rescaled IC field.
5. Runs the pretrained emulator forward to `z_target`.
6. Compares emulator outputs against the Quijote target via summary diagnostics.
7. Saves a Minkowski-functional comparison plot for target/LPT/emulator fields.
8. Applies MAS deconvolution to generated fields by default, and optionally to the target field.

Example usage with all CLI arguments:

python scripts/quijote_comparison.py \
  --target-path /home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt \
  --ic-key delta_z127 \
  --target-key delta_z0 \
  --z-ic 127 \
  --z-target 0 \
  --n-part 512 \
  --res 128 \
  --boxsize 1000 \
  --ndiv 2,2,2 \
  --emu-precision f16 \
  --num-sims 1 \
  --mas-worder 4 \
  --delta-upsample-method mode_inject \
  --target-upsample-method fourier \
  --output-dir outputs/quijote_sample0_emulator_compare

Plot-only mode (reuse saved fields in output dir, no emulator run; regenerates slices/P(k)+summary/Minkowski):
python scripts/quijote_comparison.py \
  --plot-only \
  --output-dir outputs/quijote_sample0_emulator_compare

MAS deconvolution flags:
  - Generated fields (LPT/emulator): enabled by default (`--no-deconvolve-mas` to disable)
  - Target field: disabled by default (`--deconvolve-mas-target` to enable)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from core import QUIJOTE_FIDUCIAL, run_lpt_emulator_pipeline
from utils import (
    add_import_paths,
    deconvolve_mas_kernel,
    growth_D_approx,
    mas_from_worder,
    parse_ndiv,
    plot_emulator_vs_target_summary,
    plot_minkowski_functionals,
    plot_quijote_emulator_slices,
    resize_density_grid,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
add_import_paths(repo_root=REPO_ROOT)


PLOT_SLICE_INDICES: list[int] | None = None  # None -> [N/4, N/2, 3N/4]


def save_array(path: Path, array: np.ndarray) -> None:
    """Save one array and raise a clearer message on partial-write failure."""
    arr = np.asarray(array)
    try:
        np.save(path, arr)
    except OSError as exc:
        gib = arr.nbytes / (1024.0**3)
        raise OSError(
            f"Failed to write {path} ({gib:.2f} GiB). "
            "This usually indicates disk quota or filesystem-space limits."
        ) from exc


def parse_args() -> argparse.Namespace:
    """Define and validate CLI options for Quijote-emulator comparison."""
    parser = argparse.ArgumentParser(
        description=(
            "Load Quijote .pt IC/target fields, renormalize IC by linear growth, run emulator, "
            "and compare summaries. MAS deconvolution is enabled by default for generated fields."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--target-path",
        type=Path,
        default=Path("/home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt"),
        help="Path to Quijote sample .pt file containing IC and target fields.",
    )
    parser.add_argument(
        "--ic-key",
        type=str,
        default="delta_z127",
        help="Key for the initial density field in the .pt file.",
    )
    parser.add_argument(
        "--target-key",
        type=str,
        default="delta_z0",
        help="Key for the target final density field in the .pt file.",
    )
    parser.add_argument("--z-ic", type=float, default=127.0, help="Redshift corresponding to --ic-key.")
    parser.add_argument("--z-target", type=float, default=0.0, help="Target redshift for emulator output.")

    parser.add_argument(
        "--n-part",
        type=int,
        default=512,
        help="Particle/displacement lattice resolution for LPT/emulator input.",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=128,
        help="Density grid resolution used for summaries and saved density outputs.",
    )
    parser.add_argument("--boxsize", type=float, default=1000.0, help="Box size in Mpc/h.")

    parser.add_argument(
        "--ndiv",
        type=parse_ndiv,
        default=(2, 2, 2),
        help="Subbox divisions for emulator. Example: 4 or 2,2,2",
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
        help="Number of emulator runs in this process (only last run is saved).",
    )

    parser.add_argument(
        "--mas-worder",
        type=int,
        choices=(2, 3, 4),
        default=4,
        help="Mass-assignment order for density mapping (2=CIC, 3=TSC, 4=PCS).",
    )
    parser.add_argument(
        "--deconvolve-mas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Deconvolve MAS kernel for generated fields (LPT/emulator) before summary comparisons."
        ),
    )
    parser.add_argument(
        "--deconvolve-mas-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply MAS deconvolution to the target input field. Default is disabled, assuming "
            "the target file is already deconvolved (e.g. *_wout_MAK.pt)."
        ),
    )
    parser.add_argument(
        "--delta-upsample-method",
        choices=("mode_inject", "fourier", "linear"),
        default="mode_inject",
        help="Interpolation method when resizing density fields to higher resolution.",
    )
    parser.add_argument(
        "--target-upsample-method",
        choices=("mode_inject", "fourier", "linear"),
        default="fourier",
        help=(
            "Interpolation used only for resizing the target field to --res. "
            "Default is fourier to avoid stochastic high-k injection in comparisons."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/quijote_sample0_emulator_compare"),
        help="Directory where outputs are saved.",
    )
    parser.add_argument(
        "--save-large-fields",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save high-resolution n_part and displacement arrays (can be multi-GB). "
            "Disable to avoid quota/space issues."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip IC/LPT/emulator execution and generate plots only from saved fields in --output-dir "
            "(expects quijote_target.npy, delta_lpt_target.npy, emu_delta_target.npy)."
        ),
    )

    args = parser.parse_args()
    if args.num_sims < 1:
        parser.error("--num-sims must be >= 1")
    if args.res < 1:
        parser.error("--res must be >= 1")
    if args.n_part is not None and args.n_part < 1:
        parser.error("--n-part must be >= 1")
    return args


def main() -> None:
    """Run Quijote IC loading, emulator inference, and summary generation."""
    args = parse_args()

    add_import_paths(repo_root=REPO_ROOT)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        target_field_path = out_dir / "quijote_target.npy"
        lpt_field_path = out_dir / "delta_lpt_target.npy"
        emu_field_path = out_dir / "emu_delta_target.npy"
        missing = [p for p in [target_field_path, lpt_field_path, emu_field_path] if not p.is_file()]
        if missing:
            missing_str = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(
                "Plot-only mode requires saved fields in --output-dir. Missing: "
                f"{missing_str}"
            )

        delta_target = np.asarray(np.load(target_field_path), dtype=np.float32)
        delta_lpt = np.asarray(np.load(lpt_field_path), dtype=np.float32)
        delta_emu = np.asarray(np.load(emu_field_path), dtype=np.float32)

        omega_m = float(QUIJOTE_FIDUCIAL["Omega_m"])
        omega_b = float(QUIJOTE_FIDUCIAL["Omega_b"])
        class_params = {
            "Omega_cdm": omega_m - omega_b,
            "Omega_b": omega_b,
            "h": float(QUIJOTE_FIDUCIAL["h"]),
            "n_s": float(QUIJOTE_FIDUCIAL["n_s"]),
            "sigma8": float(QUIJOTE_FIDUCIAL["sigma8"]),
        }
        target_assumed_predeconvolved = not bool(args.deconvolve_mas_target)
        mas_name = mas_from_worder(args.mas_worder)

        slices_path = out_dir / "0_slices.png"
        plot_quijote_emulator_slices(
            delta_target=np.asarray(delta_target, dtype=np.float32),
            delta_lpt=np.asarray(delta_lpt, dtype=np.float32),
            delta_emu=np.asarray(delta_emu, dtype=np.float32),
            out_path=slices_path,
            slice_indices=PLOT_SLICE_INDICES,
        )

        summary_plot_path = out_dir / "2_power_spectrum.png"
        pdf_plot_path = out_dir / "1_1pt_pdf.png"
        bispec_plot_path = out_dir / "3_bispectrum.png"
        minkowski_plot_path = out_dir / "4_minkowski.png"
        summary = plot_emulator_vs_target_summary(
            delta_target=np.asarray(delta_target, dtype=np.float32),
            delta_lpt=np.asarray(delta_lpt, dtype=np.float32),
            delta_emu=np.asarray(delta_emu, dtype=np.float32),
            boxsize=float(args.boxsize),
            out_path=summary_plot_path,
            pdf_out_path=pdf_plot_path,
            bispec_out_path=bispec_plot_path,
            mas=mas_name,
            mas_deconvolved_fields=bool(args.deconvolve_mas),
            mas_deconvolved_target_field=bool(target_assumed_predeconvolved or args.deconvolve_mas_target),
            class_cosmo_params=class_params,
            z=float(args.z_target),
        )
        plot_minkowski_functionals(
            fields={
                "Quijote target": np.asarray(delta_target, dtype=np.float32),
                "LPT": np.asarray(delta_lpt, dtype=np.float32),
                "Emulator": np.asarray(delta_emu, dtype=np.float32),
            },
            boxsize=float(args.boxsize),
            out_path=minkowski_plot_path,
            standardize=True,
        )

        metadata_path = out_dir / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
        else:
            metadata = {}
        metadata.update(
            {
                "plot_only": True,
                "mas_worder": int(args.mas_worder),
                "deconvolve_mas": bool(args.deconvolve_mas),
                "deconvolve_mas_target": bool(args.deconvolve_mas_target),
                "target_assumed_predeconvolved": bool(target_assumed_predeconvolved),
                "summary_metrics": summary,
                "summary_plot": str(summary_plot_path),
                "summary_1pt_pdf_plot": str(pdf_plot_path),
                "summary_bispectrum_plot": str(bispec_plot_path),
                "summary_minkowski_plot": str(minkowski_plot_path),
                "slices_plot": str(slices_path),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2))

        print("Plot-only comparison finished.")
        print(f"Loaded fields from: {out_dir}")
        print("Saved plots:")
        print(f"  - {slices_path.name}")
        print(f"  - {summary_plot_path.name}")
        print(f"  - {pdf_plot_path.name}")
        print(f"  - {bispec_plot_path.name}")
        print(f"  - {minkowski_plot_path.name}")
        print(f"Updated metadata: {metadata_path}")
        return

    if not args.target_path.is_file():
        raise FileNotFoundError(f"Target file not found: {args.target_path}")

    sample = torch.load(args.target_path, map_location="cpu", weights_only=False)
    if not isinstance(sample, dict):
        raise TypeError(f"Expected a dict in {args.target_path}, got {type(sample)}")
    if args.ic_key not in sample:
        raise KeyError(f"Missing IC key {args.ic_key!r} in {args.target_path}")
    if args.target_key not in sample:
        raise KeyError(f"Missing target key {args.target_key!r} in {args.target_path}")

    delta_ic_in = np.asarray(sample[args.ic_key], dtype=np.float32)
    delta_target_in = np.asarray(sample[args.target_key], dtype=np.float32)

    if delta_ic_in.ndim != 3 or delta_target_in.ndim != 3:
        raise ValueError(
            f"Both IC and target fields must be 3D: got {delta_ic_in.shape} and {delta_target_in.shape}"
        )
    if len(set(delta_ic_in.shape)) != 1:
        raise ValueError(f"IC field must be cubic, got shape={delta_ic_in.shape}")
    if len(set(delta_target_in.shape)) != 1:
        raise ValueError(f"Target field must be cubic, got shape={delta_target_in.shape}")

    n_part = int(args.n_part) if args.n_part is not None else int(delta_ic_in.shape[0])
    density_res = int(args.res)

    omega_m = float(QUIJOTE_FIDUCIAL["Omega_m"])
    omega_b = float(QUIJOTE_FIDUCIAL["Omega_b"])
    class_params = {
        "Omega_cdm": omega_m - omega_b,
        "Omega_b": omega_b,
        "h": float(QUIJOTE_FIDUCIAL["h"]),
        "n_s": float(QUIJOTE_FIDUCIAL["n_s"]),
        "sigma8": float(QUIJOTE_FIDUCIAL["sigma8"]),
    }

    growth = {
        "a_from": float(1.0 / (1.0 + float(args.z_ic))),
        "a_to": float(1.0 / (1.0 + float(args.z_target))),
        "D_from": float(growth_D_approx(class_params, float(args.z_ic))),
        "D_to": float(growth_D_approx(class_params, float(args.z_target))),
    }
    growth["scale"] = float(growth["D_to"] / growth["D_from"])
    delta_ic_scaled = np.asarray(delta_ic_in, dtype=np.float32) * np.float32(growth["scale"])

    pipeline = run_lpt_emulator_pipeline(
        delta_linear=delta_ic_scaled,
        delta_input_scale=1.0,
        n_part=n_part,
        density_res=density_res,
        boxsize=float(args.boxsize),
        ndiv=tuple(args.ndiv),
        emu_precision=str(args.emu_precision),
        num_sims=int(args.num_sims),
        mas_worder=int(args.mas_worder),
        deconvolve_mas=bool(args.deconvolve_mas),
        z_target=float(args.z_target),
        compute_vel=False,
        delta_upsample_method=str(args.delta_upsample_method),
        quijote_fiducial=QUIJOTE_FIDUCIAL,
        output_dir=None,
        save_outputs=False,
        show_progress=True,
        progress_desc_prefix="Processing subboxes",
    )

    delta_ic_npart = np.asarray(pipeline["ics_grf_npart"], dtype=np.float32)
    delta_lpt = np.asarray(pipeline["delta_lpt"], dtype=np.float32)
    psi_lpt = np.asarray(pipeline["psi_lpt"], dtype=np.float32)
    delta_emu = np.asarray(pipeline["delta_emu"], dtype=np.float32)
    psi_emu = np.asarray(pipeline["psi_emu"], dtype=np.float32)

    print(
        f"Resizing IC field to summary grid ({delta_ic_scaled.shape[0]}^3 -> {density_res}^3) "
        f"with '{args.delta_upsample_method}'..."
    )
    t_resize_ic = time.perf_counter()
    delta_ic_density = resize_density_grid(
        delta_in=delta_ic_scaled,
        target_res=density_res,
        boxsize=float(args.boxsize),
        upsample_method=args.delta_upsample_method,
        class_cosmo_params=class_params,
        class_z=float(args.z_target),
    )
    dt_resize_ic = time.perf_counter() - t_resize_ic
    print(f"Finished IC resizing in {dt_resize_ic:.2f} s.")

    print(
        f"Resizing target field to summary grid ({delta_target_in.shape[0]}^3 -> {density_res}^3) "
        f"with '{args.target_upsample_method}'..."
    )
    t_resize_target = time.perf_counter()
    delta_target = resize_density_grid(
        delta_in=delta_target_in,
        target_res=density_res,
        boxsize=float(args.boxsize),
        upsample_method=args.target_upsample_method,
        class_cosmo_params=class_params,
        class_z=float(args.z_target),
    )
    dt_resize_target = time.perf_counter() - t_resize_target
    print(f"Finished target resizing in {dt_resize_target:.2f} s.")

    if args.deconvolve_mas_target:
        delta_target = deconvolve_mas_kernel(delta_target, worder=int(args.mas_worder), with_jax=False)

    target_assumed_predeconvolved = not bool(args.deconvolve_mas_target)
    if target_assumed_predeconvolved and "wout_MAK" not in args.target_path.name:
        print(
            "Warning: target MAS deconvolution is disabled, so the script assumes the target is already "
            "deconvolved. If this is not true for your input file, pass --deconvolve-mas-target."
        )

    mas_name = mas_from_worder(args.mas_worder)
    slices_path = out_dir / "0_slices.png"
    plot_quijote_emulator_slices(
        delta_target=np.asarray(delta_target, dtype=np.float32),
        delta_lpt=np.asarray(delta_lpt, dtype=np.float32),
        delta_emu=np.asarray(delta_emu, dtype=np.float32),
        out_path=slices_path,
        slice_indices=PLOT_SLICE_INDICES,
    )

    summary_plot_path = out_dir / "2_power_spectrum.png"
    pdf_plot_path = out_dir / "1_1pt_pdf.png"
    bispec_plot_path = out_dir / "3_bispectrum.png"
    minkowski_plot_path = out_dir / "4_minkowski.png"
    summary = plot_emulator_vs_target_summary(
        delta_target=np.asarray(delta_target, dtype=np.float32),
        delta_lpt=np.asarray(delta_lpt, dtype=np.float32),
        delta_emu=np.asarray(delta_emu, dtype=np.float32),
        boxsize=float(args.boxsize),
        out_path=summary_plot_path,
        pdf_out_path=pdf_plot_path,
        bispec_out_path=bispec_plot_path,
        mas=mas_name,
        mas_deconvolved_fields=bool(args.deconvolve_mas),
        mas_deconvolved_target_field=bool(target_assumed_predeconvolved or args.deconvolve_mas_target),
        class_cosmo_params=class_params,
        z=float(args.z_target),
    )
    plot_minkowski_functionals(
        fields={
            "Quijote target": np.asarray(delta_target, dtype=np.float32),
            "LPT": np.asarray(delta_lpt, dtype=np.float32),
            "Emulator": np.asarray(delta_emu, dtype=np.float32),
        },
        boxsize=float(args.boxsize),
        out_path=minkowski_plot_path,
        standardize=True,
    )

    save_array(out_dir / "quijote_target.npy", np.asarray(delta_target, dtype=np.float32))
    save_array(out_dir / "quijote_ic_input.npy", np.asarray(delta_ic_in, dtype=np.float32))
    save_array(out_dir / "quijote_ic_scaled_to_target.npy", np.asarray(delta_ic_density, dtype=np.float32))
    if n_part != density_res and args.save_large_fields:
        save_array(out_dir / "quijote_ic_scaled_to_target_npart.npy", np.asarray(delta_ic_npart, dtype=np.float32))
    if args.save_large_fields:
        save_array(out_dir / "dis_lpt_target.npy", np.asarray(psi_lpt, dtype=np.float32))
    save_array(out_dir / "delta_lpt_target.npy", np.asarray(delta_lpt, dtype=np.float32))
    if args.save_large_fields:
        save_array(out_dir / "emu_dis_target.npy", np.asarray(psi_emu, dtype=np.float32))
    save_array(out_dir / "emu_delta_target.npy", np.asarray(delta_emu, dtype=np.float32))

    metadata = dict(pipeline["metadata"])
    metadata.update(
        {
            "target_path": str(args.target_path),
            "ic_key": args.ic_key,
            "target_key": args.target_key,
            "z_ic": float(args.z_ic),
            "z_target": float(args.z_target),
            "a_ic": float(growth["a_from"]),
            "a_target": float(growth["a_to"]),
            "D_ic": float(growth["D_from"]),
            "D_target": float(growth["D_to"]),
            "growth_rescale_factor_Dtarget_over_Dic": float(growth["scale"]),
            "n_part": n_part,
            "res": density_res,
            "save_large_fields": bool(args.save_large_fields),
            "boxsize_mpc_over_h": float(args.boxsize),
            "input_ic_shape": list(delta_ic_in.shape),
            "input_target_shape": list(delta_target_in.shape),
            "delta_upsample_method": args.delta_upsample_method,
            "target_upsample_method": args.target_upsample_method,
            "ic_resize_seconds": float(dt_resize_ic),
            "target_resize_seconds": float(dt_resize_target),
            "mas_worder": int(args.mas_worder),
            "deconvolve_mas": bool(args.deconvolve_mas),
            "deconvolve_mas_target": bool(args.deconvolve_mas_target),
            "target_assumed_predeconvolved": bool(target_assumed_predeconvolved),
            "mas_name_for_plot": mas_name,
            "quijote_fiducial": QUIJOTE_FIDUCIAL,
            "emu_ndiv": list(args.ndiv),
            "emu_precision": args.emu_precision,
            "num_sims": int(args.num_sims),
            "summary_metrics": summary,
            "summary_plot": str(summary_plot_path),
            "summary_1pt_pdf_plot": str(pdf_plot_path),
            "summary_bispectrum_plot": str(bispec_plot_path),
            "summary_minkowski_plot": str(minkowski_plot_path),
            "slices_plot": str(slices_path),
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Quijote emulator comparison finished.")
    print(f"Saved outputs to: {out_dir}")
    print(f"Emulator runtime (last sim): {metadata['emulator_runtime_hms']}")
    print("Emulator runtimes by simulation:")
    for idx, rt_hms in enumerate(pipeline["emulator_runtime_hms_by_sim"], start=1):
        print(f"  - sim {idx}: {rt_hms}")

    saved_files = [
        "quijote_target.npy",
        "quijote_ic_input.npy",
        "quijote_ic_scaled_to_target.npy",
    ]
    if n_part != density_res and args.save_large_fields:
        saved_files.append("quijote_ic_scaled_to_target_npart.npy")
    if args.save_large_fields:
        saved_files.append("dis_lpt_target.npy")
    saved_files.extend(
        [
            "delta_lpt_target.npy",
            "emu_delta_target.npy",
            slices_path.name,
            summary_plot_path.name,
            pdf_plot_path.name,
            bispec_plot_path.name,
            minkowski_plot_path.name,
            "metadata.json",
        ]
    )
    if args.save_large_fields:
        saved_files.append("emu_dis_target.npy")

    print("Files:")
    for fname in saved_files:
        print(f"  - {fname}")


if __name__ == "__main__":
    main()
