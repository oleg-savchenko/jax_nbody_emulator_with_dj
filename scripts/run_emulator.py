#!/usr/bin/env python3
"""
End-to-end IC -> emulator -> density pipeline.

This script:
1. Uses DISCO-DJ to generate N-GenIC-compatible white noise for a fixed seed.
2. Builds the corresponding linear Gaussian density field at z=0 (scaled by linear P(k)).
3. Builds ZA/1LPT displacement at z=0 for fiducial Quijote cosmology.
4. Runs the pretrained jax_nbody_emulator forward from the initial displacement.
5. Computes density fields from LPT and emulated displacements (via DISCO-DJ MAS).
6. Deconvolves MAS kernels from generated density fields by default.
7. Saves diagnostic plots and output arrays.

Outputs are written as .npy arrays plus a metadata JSON.

Example usage with all CLI arguments:

python scripts/run_emulator.py \
  --seed 42 \
  --n-part 128 \
  --res 128 \
  --boxsize 1000 \
  --ndiv 1,1,1 \
  --num-sims 3 \
  --emu-precision f32 \
  --output-dir outputs/discodj_emulator_seed42 \
  --no-compute-vel \
  --mas-worder 2

To disable MAS deconvolution:
  --no-deconvolve-mas

Alternative mode (external linear GRF density field):

python scripts/run_emulator.py \
  --input-delta-path /path/to/delta_linear_z0.npy \
  --delta-upsample-method mode_inject \
  --n-part 512 \
  --res 512 \
  --boxsize 1000 \
  --ndiv 2,2,2 \
  --num-sims 1 \
  --output-dir outputs/discodj_emulator_from_delta

Plot-only mode (reuse saved fields in output dir, no emulator run):

python scripts/run_emulator.py \
  --plot-only \
  --output-dir outputs/discodj_emulator_seed42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core import QUIJOTE_FIDUCIAL, run_lpt_emulator_pipeline
from utils import (
    add_import_paths,
    deconvolve_mas_kernel,
    mas_from_worder,
    parse_ndiv,
    plot_density_power_spectra,
    plot_density_slices,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
add_import_paths(repo_root=REPO_ROOT)


PLOT_SLICE_INDICES: list[int] | None = None  # None -> [N/4, N/2, 3N/4]
PLOT_NONLINEAR_THEORY = False


def parse_args() -> argparse.Namespace:
    """Define and validate command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate DISCO-DJ ICs and run jax_nbody_emulator end-to-end at z=0. "
            "MAS deconvolution is enabled by default for generated density fields."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--seed", type=int, default=42, help="N-GenIC-compatible random seed.")
    parser.add_argument(
        "--input-delta-path",
        type=Path,
        default=None,
        help=(
            "Optional path to an external linear GRF density field at z=0 (numpy .npy, shape [N,N,N]). "
            "If set, this field is used instead of generating N-GenIC white noise from --seed."
        ),
    )
    parser.add_argument(
        "--delta-upsample-method",
        choices=("mode_inject", "fourier", "linear"),
        default="mode_inject",
        help="Interpolation used when a field must be upsampled to higher resolution.",
    )

    parser.add_argument(
        "--n-part",
        type=int,
        default=None,
        help=(
            "Particle/displacement lattice resolution for DISCO-DJ + emulator input "
            "(defaults to --res if omitted)."
        ),
    )
    parser.add_argument(
        "--res",
        type=int,
        default=128,
        help="Output density grid resolution N (for N^3 density fields and plots).",
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
        default="f32",
        help="Emulator compute precision.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/discodj_emulator_pipeline"),
        help="Directory where outputs are saved.",
    )
    parser.add_argument(
        "--compute-vel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run emulator velocity head and save emulated velocity.",
    )
    parser.add_argument(
        "--mas-worder",
        type=int,
        choices=(2, 3, 4),
        default=2,
        help="Mass-assignment kernel order for density mapping (2=CIC, 3=TSC, 4=PCS).",
    )
    parser.add_argument(
        "--deconvolve-mas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deconvolve MAS kernel from generated density fields.",
    )
    parser.add_argument(
        "--class-n-modes",
        type=int,
        default=4000,
        help="Number of k samples for CLASS linear P(k) table passed to DISCO-DJ.",
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=1,
        help="Number of emulator runs to execute in this process (saves only the last run).",
    )
    parser.add_argument(
        "--plot-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip IC/LPT/emulator execution and regenerate plots only from saved fields in --output-dir "
            "(expects delta_lpt_z0.npy and emu_delta_z0.npy plus IC linear field)."
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
    """Run the end-to-end IC -> LPT -> emulator pipeline and save diagnostics."""
    args = parse_args()

    add_import_paths(repo_root=REPO_ROOT)

    n_part = int(args.n_part) if args.n_part is not None else int(args.res)
    density_res = int(args.res)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        linear_z0_path = out_dir / "delta_linear_z0.npy"
        ics_grf_path = out_dir / "ics_grf.npy"
        white_noise_path = out_dir / "white_noise_ngenic.npy"
        delta_init_path = out_dir / "delta_lpt_z0.npy"
        delta_emu_path = out_dir / "emu_delta_z0.npy"
        metadata_path = out_dir / "metadata.json"

        if linear_z0_path.is_file():
            ics_grf = np.load(linear_z0_path)
        elif ics_grf_path.is_file():
            ics_grf = np.load(ics_grf_path)
        elif white_noise_path.is_file():
            ics_grf = np.load(white_noise_path)
        else:
            raise FileNotFoundError(
                "Missing IC GRF file for plot-only mode: expected one of "
                f"{linear_z0_path}, {ics_grf_path}, or {white_noise_path}"
            )

        if not delta_init_path.is_file():
            raise FileNotFoundError(f"Missing file for plot-only mode: {delta_init_path}")
        if not delta_emu_path.is_file():
            raise FileNotFoundError(f"Missing file for plot-only mode: {delta_emu_path}")

        delta_lpt = np.load(delta_init_path)
        delta_emu = np.load(delta_emu_path)

        metadata: dict = {}
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())

        q = metadata.get("quijote_fiducial", QUIJOTE_FIDUCIAL)
        omega_m = float(q["Omega_m"])
        omega_b = float(q["Omega_b"])
        class_params = {
            "Omega_cdm": omega_m - omega_b,
            "Omega_b": omega_b,
            "h": float(q["h"]),
            "n_s": float(q["n_s"]),
            "sigma8": float(q["sigma8"]),
        }

        boxsize = float(metadata.get("boxsize_mpc_over_h", args.boxsize))
        if "mas_name_for_plot" in metadata:
            mas_name = str(metadata["mas_name_for_plot"])
        else:
            mas_name = mas_from_worder(int(metadata.get("mas_worder", args.mas_worder)))
        worder = int(metadata.get("mas_worder", args.mas_worder))

        saved_fields_deconvolved = bool(metadata.get("deconvolve_mas", False))
        apply_deconvolution = bool(args.deconvolve_mas) and not saved_fields_deconvolved
        if apply_deconvolution:
            delta_lpt = deconvolve_mas_kernel(delta_lpt, worder=worder, with_jax=False)
            delta_emu = deconvolve_mas_kernel(delta_emu, worder=worder, with_jax=False)
        fields_are_deconvolved = saved_fields_deconvolved or bool(args.deconvolve_mas)

        slice_plot_path = out_dir / "density_slices_z0.png"
        plot_density_slices(
            ics_grf=np.asarray(ics_grf, dtype=np.float32),
            delta_init=np.asarray(delta_lpt, dtype=np.float32),
            delta_emu=np.asarray(delta_emu, dtype=np.float32),
            out_path=slice_plot_path,
            slice_indices=PLOT_SLICE_INDICES,
        )

        pk_plot_path = out_dir / "density_pk_vs_class_z0.png"
        plot_density_power_spectra(
            delta_grf=np.asarray(ics_grf, dtype=np.float32),
            delta_init=np.asarray(delta_lpt, dtype=np.float32),
            delta_emu=np.asarray(delta_emu, dtype=np.float32),
            boxsize=float(boxsize),
            out_path=pk_plot_path,
            class_cosmo_params=class_params,
            z=0.0,
            mas=mas_name,
            mas_deconvolved_fields=fields_are_deconvolved,
            non_linear_theory=PLOT_NONLINEAR_THEORY,
        )

        metadata.update(
            {
                "plot_only": True,
                "mas_worder": int(worder),
                "deconvolve_mas": bool(fields_are_deconvolved),
                "mas_name_for_plot": mas_name,
                "plot_slice_indices": PLOT_SLICE_INDICES,
                "plot_nonlinear_theory": PLOT_NONLINEAR_THEORY,
                "density_slice_plot": str(slice_plot_path),
                "density_pk_plot": str(pk_plot_path),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2))

        print("Plot-only pipeline finished.")
        print(f"Loaded fields from: {out_dir}")
        print("Saved plots:")
        print(f"  - {slice_plot_path.name}")
        print(f"  - {pk_plot_path.name}")
        print(f"Updated metadata: {metadata_path}")
        return

    source_kwargs = (
        {"seed": int(args.seed)}
        if args.input_delta_path is None
        else {"delta_linear_path": Path(args.input_delta_path)}
    )

    pipeline = run_lpt_emulator_pipeline(
        **source_kwargs,
        n_part=n_part,
        density_res=density_res,
        boxsize=float(args.boxsize),
        ndiv=tuple(args.ndiv),
        emu_precision=str(args.emu_precision),
        num_sims=int(args.num_sims),
        mas_worder=int(args.mas_worder),
        deconvolve_mas=bool(args.deconvolve_mas),
        z_target=0.0,
        compute_vel=bool(args.compute_vel),
        delta_upsample_method=str(args.delta_upsample_method),
        class_n_modes=int(args.class_n_modes),
        quijote_fiducial=QUIJOTE_FIDUCIAL,
        output_dir=out_dir,
        save_outputs=True,
        show_progress=True,
        progress_desc_prefix="Processing subboxes",
    )

    ics_grf = np.asarray(pipeline["ics_grf"], dtype=np.float32)
    delta_lpt = np.asarray(pipeline["delta_lpt"], dtype=np.float32)
    delta_emu = np.asarray(pipeline["delta_emu"], dtype=np.float32)

    omega_m = float(QUIJOTE_FIDUCIAL["Omega_m"])
    omega_b = float(QUIJOTE_FIDUCIAL["Omega_b"])
    class_params = {
        "Omega_cdm": omega_m - omega_b,
        "Omega_b": omega_b,
        "h": float(QUIJOTE_FIDUCIAL["h"]),
        "n_s": float(QUIJOTE_FIDUCIAL["n_s"]),
        "sigma8": float(QUIJOTE_FIDUCIAL["sigma8"]),
    }

    mas_name = mas_from_worder(args.mas_worder)
    slice_plot_path = out_dir / "density_slices_z0.png"
    plot_density_slices(
        ics_grf=ics_grf,
        delta_init=delta_lpt,
        delta_emu=delta_emu,
        out_path=slice_plot_path,
        slice_indices=PLOT_SLICE_INDICES,
    )

    pk_plot_path = out_dir / "density_pk_vs_class_z0.png"
    plot_density_power_spectra(
        delta_grf=ics_grf,
        delta_init=delta_lpt,
        delta_emu=delta_emu,
        boxsize=float(args.boxsize),
        out_path=pk_plot_path,
        class_cosmo_params=class_params,
        z=0.0,
        mas=mas_name,
        mas_deconvolved_fields=bool(args.deconvolve_mas),
        non_linear_theory=PLOT_NONLINEAR_THEORY,
    )

    metadata = dict(pipeline["metadata"])
    saved_paths = dict(pipeline.get("saved_paths", {}))
    metadata.update(
        {
            "mas_name_for_plot": mas_name,
            "plot_slice_indices": PLOT_SLICE_INDICES,
            "plot_nonlinear_theory": PLOT_NONLINEAR_THEORY,
            "deconvolve_mas": bool(args.deconvolve_mas),
            "density_slice_plot": str(slice_plot_path),
            "density_pk_plot": str(pk_plot_path),
            "ics_grf_field": saved_paths.get("ics_grf", str(out_dir / "ics_grf.npy")),
            "delta_linear_z0_field": saved_paths.get("delta_linear", str(out_dir / "delta_linear_z0.npy")),
            "delta_linear_z0_npart_field": saved_paths.get("delta_linear_npart"),
            "white_noise_ngenic_field": saved_paths.get("white_noise_ngenic"),
            "class_linear_pk_table_file": pipeline.get("class_pk_table_path"),
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Pipeline finished.")
    print(f"Saved outputs to: {out_dir}")
    print(f"Emulator runtime (last sim): {pipeline['metadata']['emulator_runtime_hms']}")
    print("Emulator runtimes by simulation:")
    for idx, rt_hms in enumerate(pipeline["emulator_runtime_hms_by_sim"], start=1):
        print(f"  - sim {idx}: {rt_hms}")

    saved_files = [
        "ics_grf.npy",
        "delta_linear_z0.npy",
    ]
    if n_part != density_res:
        saved_files.append("delta_linear_z0_npart.npy")
    if pipeline["white_noise_ngenic"] is not None:
        saved_files.append("white_noise_ngenic.npy")
    saved_files.extend(
        [
            "dis_lpt_z0.npy",
            "delta_lpt_z0.npy",
            "emu_dis_z0.npy",
            "emu_delta_z0.npy",
        ]
    )
    if pipeline["vel_emu"] is not None:
        saved_files.append("emu_vel_z0.npy")
    saved_files.extend(
        [
            "density_slices_z0.png",
            "density_pk_vs_class_z0.png",
        ]
    )
    if pipeline.get("class_pk_table_path") is not None:
        saved_files.append("class_linear_pk_z0_table.txt")
    saved_files.append("metadata.json")

    print("Files:")
    for fname in saved_files:
        print(f"  - {fname}")


if __name__ == "__main__":
    main()
