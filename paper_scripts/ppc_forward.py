#!/usr/bin/env python3
"""
Forward-only posterior predictive run for IC posterior samples.

This script is intended for GPU jobs. It:
1. Loads `sample_XXXX.npy` IC posterior samples from `--samples-dir`.
2. Optionally rescales IC amplitudes from `z_ic` to `z_target` with linear growth.
3. Runs the emulator pipeline for each sample.
4. Saves per-sample emulator outputs under `--samples-dir/<sample_folder_template>`.
5. Writes an aggregate forward manifest in `--aggregate-dir`.

Example:
  python paper_scripts/ppc_forward.py \
    --samples-dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD \
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
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ppc_common import extract_sample_id, sample_output_dir, select_sample_files, z_file_tag

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for forward-only PPC generation."""
    parser = argparse.ArgumentParser(
        description="Run emulator forward propagation for IC posterior samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("/gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD"),
        help="Directory containing posterior IC samples sample_XXXX.npy.",
    )
    parser.add_argument("--sample-pattern", type=str, default="sample_*.npy", help="IC sample glob pattern.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of selected samples to process.")
    parser.add_argument("--sample-start", type=int, default=0, help="Starting index in sorted sample list.")
    parser.add_argument("--sample-stride", type=int, default=1, help="Stride in sorted sample list.")
    parser.add_argument(
        "--sample-folder-template",
        type=str,
        default="sample_{sample_id:04d}",
        help="Per-sample output folder template under --samples-dir.",
    )

    parser.add_argument("--z-ic", type=float, default=127.0, help="Redshift of IC posterior samples.")
    parser.add_argument("--z-target", type=float, default=0.0, help="Target redshift for emulator output.")
    parser.add_argument(
        "--ic-rescale-mode",
        choices=("growth", "none"),
        default="growth",
        help="How to scale input IC samples before forward run.",
    )

    parser.add_argument("--n-part", type=int, default=512, help="Emulator particle lattice resolution.")
    parser.add_argument("--res", type=int, default=128, help="Summary density grid resolution.")
    parser.add_argument("--boxsize", type=float, default=1000.0, help="Box size in Mpc/h.")
    parser.add_argument(
        "--ndiv",
        type=parse_ndiv if callable(parse_ndiv) else str,
        default=(4, 2, 2),
        help="Subbox partition for emulator.",
    )
    parser.add_argument("--emu-precision", choices=("f16", "f32"), default="f16", help="Emulator precision.")
    parser.add_argument("--num-sims", type=int, default=1, help="Internal emulator repeats per sample.")
    parser.add_argument("--mas-worder", type=int, choices=(2, 3, 4), default=4, help="MAS order.")
    parser.add_argument(
        "--deconvolve-mas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deconvolve MAS when meshing emulator/LPT densities.",
    )
    parser.add_argument(
        "--delta-upsample-method",
        choices=("mode_inject", "fourier", "linear"),
        default="mode_inject",
        help="Upsampling method when input IC grid is below n_part.",
    )
    parser.add_argument(
        "--mode-inject-seed-policy",
        choices=("sample_id", "fixed", "none"),
        default="sample_id",
        help=(
            "Seed policy for mode_inject upsampling RNG. "
            "'sample_id' uses deterministic per-sample seeds; "
            "'fixed' reuses one seed for all samples; "
            "'none' keeps backend default behavior."
        ),
    )
    parser.add_argument(
        "--mode-inject-fixed-seed",
        type=int,
        default=0,
        help="Seed used when --mode-inject-seed-policy=fixed.",
    )
    parser.add_argument(
        "--downsample-method",
        choices=("fourier_truncation", "block_average", "gaussian"),
        default="fourier_truncation",
        help="Downsampling method when input IC grid is above n_part.",
    )
    parser.add_argument(
        "--downsample-gaussian-sigma-mpc-h",
        type=float,
        default=None,
        help="Gaussian smoothing scale for downsample_method=gaussian.",
    )
    parser.add_argument("--downsample-threads", type=int, default=1, help="Threads for gaussian downsampling.")
    parser.add_argument("--class-n-modes", type=int, default=4000, help="Number of CLASS modes.")
    parser.add_argument(
        "--compute-vel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run the emulator velocity head.",
    )
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing emu_delta/emu_dis files when present.",
    )

    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=None,
        help="Manifest output directory (default: <samples-dir>/ppc_results).",
    )
    parser.add_argument("--aggregate-prefix", type=str, default="posterior_predictive", help="Manifest filename prefix.")

    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.sample_start < 0:
        parser.error("--sample-start must be >= 0")
    if args.sample_stride < 1:
        parser.error("--sample-stride must be >= 1")
    if args.res < 1 or args.n_part < 1:
        parser.error("--res and --n-part must be >= 1")
    return args


def format_hms(seconds: float) -> str:
    """Format seconds as HHh MMm SSs."""
    total = int(round(float(seconds)))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}h {mm:02d}m {ss:02d}s"


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


def main() -> None:
    """Run the forward emulator stage for selected IC posterior samples."""
    args = parse_args()

    if _UTILS_IMPORT_ERROR is not None:
        raise ImportError("Failed to import utilities from scripts/utils.py") from _UTILS_IMPORT_ERROR
    if _CORE_IMPORT_ERROR is not None:
        raise ImportError("Failed to import core emulator pipeline from scripts/core.py") from _CORE_IMPORT_ERROR

    samples_dir = args.samples_dir.resolve()
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    aggregate_dir = args.aggregate_dir.resolve() if args.aggregate_dir is not None else (samples_dir / "ppc_results")
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    files_sel = select_sample_files(
        samples_dir=samples_dir,
        pattern=str(args.sample_pattern),
        sample_start=int(args.sample_start),
        sample_stride=int(args.sample_stride),
        num_samples=int(args.num_samples),
    )
    print(f"Selected {len(files_sel)} sample files for forward run.")

    scale = 1.0
    if args.ic_rescale_mode == "growth":
        scale = ic_growth_scale(z_ic=float(args.z_ic), z_target=float(args.z_target))
    print(f"IC scaling mode: {args.ic_rescale_mode}, scale factor={scale:.8f}")

    z_tag = z_file_tag(float(args.z_target))
    records: list[dict[str, object]] = []

    n_total = int(len(files_sel))
    for i, sample_path in enumerate(tqdm(files_sel, desc="PPC forward", ncols=100)):
        sample_id = extract_sample_id(sample_path, fallback_index=i)
        out_dir = sample_output_dir(
            samples_dir=samples_dir,
            template=str(args.sample_folder_template),
            sample_id=int(sample_id),
            sample_path=sample_path,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{i + 1}/{n_total}] Running sample_id={int(sample_id)} from {sample_path.name}",
            flush=True,
        )

        emu_delta_path = out_dir / f"emu_delta_{z_tag}.npy"
        emu_dis_path = out_dir / f"emu_dis_{z_tag}.npy"
        metadata_path = out_dir / "metadata.json"

        reused = False
        runtime_seconds = np.nan
        if bool(args.reuse_existing) and emu_delta_path.is_file() and emu_dis_path.is_file():
            reused = True
            if metadata_path.is_file():
                metadata_local = json.loads(metadata_path.read_text())
                runtime_seconds = float(metadata_local.get("emulator_runtime_seconds", np.nan))
            print(
                f"  Reused existing outputs in {out_dir.name} "
                f"(runtime in metadata: {format_hms(runtime_seconds) if np.isfinite(runtime_seconds) else 'n/a'}).",
                flush=True,
            )
        else:
            ic_sample = np.asarray(np.load(sample_path), dtype=np.float32)
            if ic_sample.ndim != 3 or len(set(ic_sample.shape)) != 1:
                raise ValueError(f"IC sample must be cubic 3D, got {sample_path} with shape={ic_sample.shape}")
            ic_scaled = np.asarray(ic_sample * np.float32(scale), dtype=np.float32)

            mode_inject_seed: int | None = None
            if str(args.delta_upsample_method) == "mode_inject":
                if str(args.mode_inject_seed_policy) == "sample_id":
                    mode_inject_seed = int(sample_id)
                elif str(args.mode_inject_seed_policy) == "fixed":
                    mode_inject_seed = int(args.mode_inject_fixed_seed)
                elif str(args.mode_inject_seed_policy) == "none":
                    mode_inject_seed = None
                else:  # pragma: no cover - guarded by argparse
                    raise ValueError(f"Unknown mode_inject_seed_policy: {args.mode_inject_seed_policy}")

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
                mode_inject_seed=mode_inject_seed,
                quijote_fiducial=QUIJOTE_FIDUCIAL,
                output_dir=out_dir,
                save_outputs=True,
                show_progress=True,
                progress_desc_prefix="PPC subboxes",
            )
            runtime_seconds = float(pipeline["metadata"]["emulator_runtime_seconds"])
            print(
                f"  Emulator finished for sample_id={int(sample_id)} in "
                f"{format_hms(runtime_seconds)} ({runtime_seconds:.2f} s).",
                flush=True,
            )

        records.append(
            {
                "sample_id": int(sample_id),
                "input_path": str(sample_path),
                "output_dir": str(out_dir),
                "emu_delta_path": str(emu_delta_path),
                "emu_dis_path": str(emu_dis_path),
                "reused": bool(reused),
                "emulator_runtime_seconds": float(runtime_seconds),
                "mode_inject_seed": (
                    None
                    if str(args.delta_upsample_method) != "mode_inject"
                    else (
                        int(sample_id)
                        if str(args.mode_inject_seed_policy) == "sample_id"
                        else (
                            int(args.mode_inject_fixed_seed)
                            if str(args.mode_inject_seed_policy) == "fixed"
                            else None
                        )
                    )
                ),
            }
        )

    prefix = str(args.aggregate_prefix)
    manifest_path = aggregate_dir / f"{prefix}_forward_manifest.json"
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples_dir": str(samples_dir),
        "num_samples_processed": int(len(records)),
        "sample_pattern": str(args.sample_pattern),
        "sample_start": int(args.sample_start),
        "sample_stride": int(args.sample_stride),
        "sample_folder_template": str(args.sample_folder_template),
        "z_ic": float(args.z_ic),
        "z_target": float(args.z_target),
        "ic_rescale_mode": str(args.ic_rescale_mode),
        "ic_scale_factor": float(scale),
        "n_part": int(args.n_part),
        "res": int(args.res),
        "boxsize": float(args.boxsize),
        "ndiv": [int(x) for x in args.ndiv],
        "emu_precision": str(args.emu_precision),
        "num_sims_per_sample": int(args.num_sims),
        "mode_inject_seed_policy": str(args.mode_inject_seed_policy),
        "mode_inject_fixed_seed": int(args.mode_inject_fixed_seed),
        "records": records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2))

    print("Forward stage complete.")
    print(f"Samples processed: {len(records)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
