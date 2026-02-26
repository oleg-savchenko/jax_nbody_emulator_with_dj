#!/usr/bin/env python3
"""
Reusable IC -> LPT -> emulator -> density pipeline core.

This module centralizes the common workflow shared by run/compare scripts:
1. Build DISCO-DJ state from either N-GenIC seed or external linear GRF field.
2. Generate 1LPT displacement and corresponding density field.
3. Run pretrained jax_nbody_emulator forward.
4. Convert emulated displacement back to density using DISCO-DJ meshing, with optional
   built-in MAS-kernel deconvolution (enabled by default).
5. Optionally save all generated fields and metadata to an output directory.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from utils import (
    add_import_paths,
    format_hms,
    get_pk_class,
    project_field_from_particles,
    resize_density_grid,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
add_import_paths(repo_root=REPO_ROOT)

import jax
import jax.numpy as jnp
from discodj import DiscoDJ
from jax_nbody_emulator import SubboxConfig, create_emulator


QUIJOTE_FIDUCIAL = {
    "Omega_m": 0.3175,
    "Omega_b": 0.0490,
    "h": 0.6711,
    "n_s": 0.9624,
    "sigma8": 0.8340,
}


def _validate_ic_source(
    seed: int | None,
    delta_linear: np.ndarray | None,
    delta_linear_path: Path | None,
) -> None:
    """Validate that exactly one IC source mode is provided."""
    has_seed = seed is not None
    has_delta = delta_linear is not None or delta_linear_path is not None
    if has_seed == has_delta:
        raise ValueError("Provide exactly one IC source: either `seed` or (`delta_linear`/`delta_linear_path`).")
    if delta_linear is not None and delta_linear_path is not None:
        raise ValueError("Provide only one of `delta_linear` or `delta_linear_path`, not both.")


def _validate_pipeline_args(
    n_part: int,
    density_res: int,
    ndiv: tuple[int, int, int],
    emu_precision: str,
    num_sims: int,
    mas_worder: int,
    class_n_modes: int,
    delta_upsample_method: str,
    downsample_method: str,
) -> None:
    """Validate key pipeline argument ranges and enum-like values."""
    if n_part < 1:
        raise ValueError("`n_part` must be >= 1.")
    if density_res < 1:
        raise ValueError("`density_res` must be >= 1.")
    if len(ndiv) != 3 or any(int(x) < 1 for x in ndiv):
        raise ValueError("`ndiv` must be a 3-tuple of positive integers.")
    if emu_precision not in {"f16", "f32"}:
        raise ValueError("`emu_precision` must be one of: 'f16', 'f32'.")
    if num_sims < 1:
        raise ValueError("`num_sims` must be >= 1.")
    if mas_worder not in {2, 3, 4}:
        raise ValueError("`mas_worder` must be one of: 2(CIC), 3(TSC), 4(PCS).")
    if class_n_modes < 2:
        raise ValueError("`class_n_modes` must be >= 2.")
    if delta_upsample_method not in {"mode_inject", "fourier", "linear"}:
        raise ValueError("`delta_upsample_method` must be one of: 'mode_inject', 'fourier', 'linear'.")
    if downsample_method not in {"gaussian", "block_average"}:
        raise ValueError("`downsample_method` must be one of: 'gaussian', 'block_average'.")


def _cosmo_dicts(quijote_fiducial: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    """Build DISCO-DJ and CLASS cosmology dictionaries from Quijote-style parameters."""
    omega_m = float(quijote_fiducial["Omega_m"])
    omega_b = float(quijote_fiducial["Omega_b"])
    omega_c = omega_m - omega_b
    h_val = float(quijote_fiducial["h"])
    ns_val = float(quijote_fiducial["n_s"])
    sigma8_val = float(quijote_fiducial["sigma8"])

    cosmo = {
        "Omega_c": omega_c,
        "Omega_b": omega_b,
        "h": h_val,
        "n_s": ns_val,
        "sigma8": sigma8_val,
    }
    class_params = {
        "Omega_cdm": omega_c,
        "Omega_b": omega_b,
        "h": h_val,
        "n_s": ns_val,
        "sigma8": sigma8_val,
    }
    return cosmo, class_params


def _z_file_tag(z_value: float) -> str:
    """Format redshift for stable file naming."""
    return f"z{z_value:g}".replace(".", "p")


def run_lpt_emulator_pipeline(
    *,
    seed: int | None = None,
    delta_linear: np.ndarray | None = None,
    delta_linear_path: Path | None = None,
    delta_input_scale: float = 1.0,
    n_part: int = 512,
    density_res: int = 512,
    boxsize: float = 1000.0,
    ndiv: tuple[int, int, int] = (2, 2, 2),
    emu_precision: str = "f16",
    num_sims: int = 1,
    mas_worder: int = 2,
    deconvolve_mas: bool = True,
    z_target: float = 0.0,
    compute_vel: bool = False,
    delta_upsample_method: str = "mode_inject",
    downsample_method: str = "gaussian",
    downsample_gaussian_sigma_mpc_h: float | None = None,
    downsample_threads: int = 1,
    class_n_modes: int = 4000,
    quijote_fiducial: dict[str, float] | None = None,
    output_dir: Path | None = None,
    save_outputs: bool = True,
    show_progress: bool = True,
    progress_desc_prefix: str = "Processing subboxes",
) -> dict[str, Any]:
    """
    Run IC generation (seed or GRF), 1LPT construction, emulator forward pass, and density mapping.

    Parameters
    ----------
    seed
        N-GenIC-compatible random seed. Use this IC mode or external-delta mode, not both.
    delta_linear
        External linear GRF density field in real space (numpy array, cubic 3D).
    delta_linear_path
        Path to an external linear GRF density field `.npy` file (cubic 3D).
    delta_input_scale
        Multiplicative factor applied to external `delta_linear` data before resizing.
    n_part
        Particle/displacement lattice resolution used by DISCO-DJ and emulator input.
    density_res
        Output density grid resolution used for `delta_lpt` and `delta_emu`.
    boxsize
        Periodic box size in Mpc/h.
    ndiv
        Emulator subbox partition tuple `(nx, ny, nz)`.
    emu_precision
        Emulator compute dtype (`\"f16\"` or `\"f32\"`).
    num_sims
        Number of emulator forward passes in the same process (saves/returns the last).
    mas_worder
        DISCO-DJ mass-assignment order for density mapping: 2=CIC, 3=TSC, 4=PCS.
    deconvolve_mas
        If True, MAS-deconvolve `delta_lpt` and `delta_emu` after meshing.
    z_target
        Target redshift used for LPT evaluation and emulator forward pass.
    compute_vel
        If True, run emulator velocity head and return/save `vel_emu`.
    delta_upsample_method
        Upsampling mode for resizing external IC fields: `\"mode_inject\"`, `\"fourier\"`, or `\"linear\"`.
    downsample_method
        Downsampling mode used only when resizing external IC fields to a lower resolution:
        `\"gaussian\"` (Pylians smoothing + block average) or `\"block_average\"`.
    downsample_gaussian_sigma_mpc_h
        Gaussian smoothing radius in Mpc/h used for external-IC downsampling when
        `downsample_method=\"gaussian\"`. If None, defaults internally to one target-grid cell.
    downsample_threads
        Thread count passed to Pylians smoothing for external-IC Gaussian downsampling.
    class_n_modes
        Number of k-samples for CLASS linear P(k) table when seed-based IC mode is used.
    quijote_fiducial
        Cosmology dictionary with keys `Omega_m`, `Omega_b`, `h`, `n_s`, `sigma8`.
        If None, uses built-in fiducial Quijote values.
    output_dir
        Output directory path for optional file saving. If None, no files are written.
    save_outputs
        If True and `output_dir` is provided, save fields/metadata to disk.
    show_progress
        If True, show tqdm progress for emulator subbox processing.
    progress_desc_prefix
        Prefix string for emulator progress-bar description.

    Returns
    -------
    dict[str, Any]
        Dictionary containing generated arrays (`ics_grf`, `psi_lpt`, `delta_lpt`,
        `psi_emu`, `delta_emu`, optional `vel_emu`), timing summaries, metadata, and
        `saved_paths` if output saving is enabled.
    """
    _validate_ic_source(seed=seed, delta_linear=delta_linear, delta_linear_path=delta_linear_path)
    _validate_pipeline_args(
        n_part=int(n_part),
        density_res=int(density_res),
        ndiv=tuple(int(x) for x in ndiv),
        emu_precision=str(emu_precision),
        num_sims=int(num_sims),
        mas_worder=int(mas_worder),
        class_n_modes=int(class_n_modes),
        delta_upsample_method=str(delta_upsample_method),
        downsample_method=str(downsample_method),
    )

    add_import_paths(repo_root=REPO_ROOT)

    n_part = int(n_part)
    density_res = int(density_res)
    z_target = float(z_target)
    a_target = 1.0 / (1.0 + z_target)

    fiducial = dict(QUIJOTE_FIDUCIAL if quijote_fiducial is None else quijote_fiducial)
    cosmo, class_params = _cosmo_dicts(fiducial)

    out_dir: Path | None = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    dj = DiscoDJ(
        dim=3,
        res=n_part,
        boxsize=float(boxsize),
        precision="single",
        cosmo=cosmo,
    ).with_timetables()

    white_noise = None
    input_delta_original_shape = None
    delta_upsample_ratio = None
    resize_input_to_npart_seconds: float | None = None
    resize_npart_to_density_seconds: float | None = None
    class_pk_table_path: Path | None = None
    temp_class_table_path: Path | None = None

    try:
        if seed is not None:
            k_fund = 2.0 * np.pi / float(boxsize)
            k_nyq = np.pi * n_part / float(boxsize)
            kmin = min(0.5 * k_fund, 1.0e-5)
            kmax = max(2.0 * k_nyq * np.sqrt(3.0), 10.0)
            k_class, pk_class = get_pk_class(
                cosmo_params=class_params,
                z=0.0,
                kmin=kmin,
                kmax=kmax,
                n_modes=int(class_n_modes),
                non_lin=False,
            )

            if out_dir is not None and save_outputs:
                class_pk_table_path = out_dir / f"class_linear_pk_{_z_file_tag(0.0)}_table.txt"
            else:
                with tempfile.NamedTemporaryFile(prefix="class_linear_pk_", suffix=".txt", delete=False) as tmp:
                    temp_class_table_path = Path(tmp.name)
                class_pk_table_path = temp_class_table_path

            np.savetxt(
                class_pk_table_path,
                np.column_stack([k_class, pk_class]),
                header="k_h_per_Mpc Pk_Mpc_over_h_cubed",
            )

            dj = dj.with_linear_ps(
                transfer_function="from_file",
                filename=str(class_pk_table_path),
                fix_sigma8=False,
            )

            white_noise = np.asarray(dj.get_ngenic_noise(seed=int(seed)), dtype=np.float32)
            dj = dj.with_ics(
                white_noise_space="real",
                white_noise_field=white_noise,
                try_to_jit=False,
            )
            ics_grf_npart = np.asarray(dj.get_delta_linear(a=a_target), dtype=np.float32)
        else:
            if delta_linear is not None:
                delta_src = np.asarray(delta_linear, dtype=np.float32)
            else:
                if delta_linear_path is None or not Path(delta_linear_path).is_file():
                    raise FileNotFoundError(f"Input GRF file not found: {delta_linear_path}")
                delta_src = np.asarray(np.load(Path(delta_linear_path)), dtype=np.float32)

            if delta_src.ndim != 3 or len(set(delta_src.shape)) != 1:
                raise ValueError(f"Input GRF must be cubic 3D, got shape={delta_src.shape}")

            input_delta_original_shape = list(delta_src.shape)
            delta_scaled = delta_src * np.float32(delta_input_scale)
            in_res = int(delta_scaled.shape[0])
            if in_res != n_part:
                print(
                    f"Resizing input IC field ({in_res}^3 -> {n_part}^3) "
                    f"with '{delta_upsample_method}'...",
                    flush=True,
                )
                t_resize = time.perf_counter()
                ics_grf_npart = resize_density_grid(
                    delta_in=delta_scaled,
                    target_res=n_part,
                    boxsize=float(boxsize),
                    upsample_method=delta_upsample_method,
                    class_cosmo_params=class_params,
                    class_z=float(z_target),
                    class_n_modes=int(class_n_modes),
                    downsample_method=str(downsample_method),
                    downsample_gaussian_sigma_mpc_h=downsample_gaussian_sigma_mpc_h,
                    downsample_threads=int(downsample_threads),
                )
                resize_input_to_npart_seconds = time.perf_counter() - t_resize
                print(
                    f"Finished resizing input IC field in {resize_input_to_npart_seconds:.2f} s.",
                    flush=True,
                )
            else:
                print(f"Input IC field already at n_part={n_part}; skipping IC resize.", flush=True)
                ics_grf_npart = np.asarray(delta_scaled, dtype=np.float32)
            if n_part >= in_res:
                delta_upsample_ratio = n_part // in_res

            dj = dj.with_external_ics(delta=ics_grf_npart)

        if n_part > density_res:
            print(
                f"Projecting linear field to summary grid via PM/MAS ({n_part}^3 -> {density_res}^3), "
                f"worder={int(mas_worder)}, deconvolve={bool(deconvolve_mas)}...",
                flush=True,
            )
            t_resize = time.perf_counter()
            ics_grf = project_field_from_particles(
                dj=dj,
                field_npart=np.asarray(ics_grf_npart, dtype=np.float32),
                target_res=int(density_res),
                mas_worder=int(mas_worder),
                deconvolve_mas=bool(deconvolve_mas),
            )
            resize_npart_to_density_seconds = time.perf_counter() - t_resize
            print(
                f"Finished PM/MAS projection for linear summary field in {resize_npart_to_density_seconds:.2f} s.",
                flush=True,
            )
        elif n_part < density_res:
            print(
                f"Resizing linear field for summaries ({n_part}^3 -> {density_res}^3) "
                f"with '{delta_upsample_method}'...",
                flush=True,
            )
            t_resize = time.perf_counter()
            ics_grf = resize_density_grid(
                delta_in=ics_grf_npart,
                target_res=density_res,
                boxsize=float(boxsize),
                upsample_method=delta_upsample_method,
                class_cosmo_params=class_params,
                class_z=float(z_target),
                class_n_modes=int(class_n_modes),
                downsample_method=str(downsample_method),
                downsample_gaussian_sigma_mpc_h=downsample_gaussian_sigma_mpc_h,
                downsample_threads=int(downsample_threads),
            )
            resize_npart_to_density_seconds = time.perf_counter() - t_resize
            print(
                f"Finished resizing linear field for summaries in {resize_npart_to_density_seconds:.2f} s.",
                flush=True,
            )
        else:
            print(f"Density summary grid already equals n_part={n_part}; skipping summary resize.", flush=True)
            ics_grf = np.asarray(ics_grf_npart, dtype=np.float32)

        dj = dj.with_lpt(n_order=1, try_to_jit=False)
        psi_init_mesh = dj.evaluate_lpt_psi_at_a(a=a_target, n_order=1)
        delta_lpt = np.asarray(
            dj.get_delta_from_psi(
                psi_init_mesh,
                method="pm",
                res=density_res,
                worder=int(mas_worder),
                deconvolve=bool(deconvolve_mas),
                try_to_jit=False,
            ),
            dtype=np.float32,
        )
        psi_lpt = np.moveaxis(np.asarray(psi_init_mesh, dtype=np.float32), -1, 0)

        emu_dtype = jnp.float16 if str(emu_precision) == "f16" else jnp.float32
        sb_config = SubboxConfig(
            size=(n_part, n_part, n_part),
            ndiv=tuple(int(x) for x in ndiv),
            dtype=emu_dtype,
            output_dtype=np.float32,
        )
        emulator = create_emulator(
            premodulate=True,
            premodulate_z=z_target,
            premodulate_Om=float(fiducial["Omega_m"]),
            compute_vel=bool(compute_vel),
            processor_config=sb_config,
        )

        emu_runtimes_s: list[float] = []
        psi_emu = None
        vel_emu = None
        for sim_idx in range(int(num_sims)):
            t0 = time.perf_counter()
            emu_out = emulator.process_box(
                psi_lpt,
                z=z_target,
                Om=float(fiducial["Omega_m"]),
                show_progress=bool(show_progress),
                desc=f"{progress_desc_prefix} (sim {sim_idx + 1}/{int(num_sims)})",
            )
            elapsed = time.perf_counter() - t0
            emu_runtimes_s.append(elapsed)
            if compute_vel:
                psi_emu, vel_emu = emu_out
            else:
                psi_emu = emu_out
                vel_emu = None

        psi_emu = np.asarray(psi_emu, dtype=np.float32)
        psi_emu_mesh = np.moveaxis(psi_emu, 0, -1)
        delta_emu = np.asarray(
            dj.get_delta_from_psi(
                jnp.asarray(psi_emu_mesh, dtype=jnp.float32),
                method="pm",
                res=density_res,
                worder=int(mas_worder),
                deconvolve=bool(deconvolve_mas),
                try_to_jit=False,
            ),
            dtype=np.float32,
        )

        last_runtime_s = float(emu_runtimes_s[-1])
        metadata: dict[str, Any] = {
            "seed": None if seed is None else int(seed),
            "input_delta_path": None if delta_linear_path is None else str(delta_linear_path),
            "input_delta_original_shape": input_delta_original_shape,
            "delta_input_scale": float(delta_input_scale),
            "delta_upsample_method": delta_upsample_method,
            "downsample_method": str(downsample_method),
            "downsample_gaussian_sigma_mpc_h": (
                None if downsample_gaussian_sigma_mpc_h is None else float(downsample_gaussian_sigma_mpc_h)
            ),
            "downsample_threads": int(downsample_threads),
            "delta_upsample_ratio": delta_upsample_ratio,
            "linear_summary_mapping_method": (
                "particle_mas_projection" if n_part > density_res else ("resize_density_grid" if n_part < density_res else "identity")
            ),
            "n_part": n_part,
            "res": density_res,
            "boxsize_mpc_over_h": float(boxsize),
            "z_target": z_target,
            "a_target": a_target,
            "quijote_fiducial": fiducial,
            "discodj_cosmology_passed": cosmo,
            "num_sims": int(num_sims),
            "emu_ndiv": [int(x) for x in ndiv],
            "emu_precision": str(emu_precision),
            "compute_vel": bool(compute_vel),
            "mas_worder": int(mas_worder),
            "deconvolve_mas": bool(deconvolve_mas),
            "class_n_modes": int(class_n_modes),
            "resize_input_to_npart_seconds": (
                None if resize_input_to_npart_seconds is None else float(resize_input_to_npart_seconds)
            ),
            "resize_npart_to_density_seconds": (
                None if resize_npart_to_density_seconds is None else float(resize_npart_to_density_seconds)
            ),
            "class_linear_pk_table_file": None if class_pk_table_path is None else str(class_pk_table_path),
            "emulator_runtime_seconds": last_runtime_s,
            "emulator_runtime_hms": format_hms(last_runtime_s),
            "emulator_runtime_seconds_by_sim": [float(x) for x in emu_runtimes_s],
            "emulator_runtime_hms_by_sim": [format_hms(x) for x in emu_runtimes_s],
            "jax_devices": [str(d) for d in jax.devices()],
        }

        saved_paths: dict[str, str] = {}
        if out_dir is not None and save_outputs:
            z_tag = _z_file_tag(z_target)
            paths = {
                "ics_grf": out_dir / "ics_grf.npy",
                "delta_linear": out_dir / f"delta_linear_{z_tag}.npy",
                "psi_lpt": out_dir / f"dis_lpt_{z_tag}.npy",
                "delta_lpt": out_dir / f"delta_lpt_{z_tag}.npy",
                "psi_emu": out_dir / f"emu_dis_{z_tag}.npy",
                "delta_emu": out_dir / f"emu_delta_{z_tag}.npy",
            }
            np.save(paths["ics_grf"], np.asarray(ics_grf, dtype=np.float32))
            np.save(paths["delta_linear"], np.asarray(ics_grf, dtype=np.float32))
            np.save(paths["psi_lpt"], np.asarray(psi_lpt, dtype=np.float32))
            np.save(paths["delta_lpt"], np.asarray(delta_lpt, dtype=np.float32))
            np.save(paths["psi_emu"], np.asarray(psi_emu, dtype=np.float32))
            np.save(paths["delta_emu"], np.asarray(delta_emu, dtype=np.float32))

            if n_part != density_res:
                paths["delta_linear_npart"] = out_dir / f"delta_linear_{z_tag}_npart.npy"
                np.save(paths["delta_linear_npart"], np.asarray(ics_grf_npart, dtype=np.float32))
            if white_noise is not None:
                paths["white_noise_ngenic"] = out_dir / "white_noise_ngenic.npy"
                np.save(paths["white_noise_ngenic"], np.asarray(white_noise, dtype=np.float32))
            if vel_emu is not None:
                paths["vel_emu"] = out_dir / f"emu_vel_{z_tag}.npy"
                np.save(paths["vel_emu"], np.asarray(vel_emu, dtype=np.float32))

            for key, path in paths.items():
                saved_paths[key] = str(path)
            metadata["saved_field_paths"] = saved_paths
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            saved_paths["metadata"] = str(out_dir / "metadata.json")

        return {
            "ics_grf": np.asarray(ics_grf, dtype=np.float32),
            "ics_grf_npart": np.asarray(ics_grf_npart, dtype=np.float32),
            "white_noise_ngenic": None if white_noise is None else np.asarray(white_noise, dtype=np.float32),
            "psi_lpt": np.asarray(psi_lpt, dtype=np.float32),
            "delta_lpt": np.asarray(delta_lpt, dtype=np.float32),
            "psi_emu": np.asarray(psi_emu, dtype=np.float32),
            "delta_emu": np.asarray(delta_emu, dtype=np.float32),
            "vel_emu": None if vel_emu is None else np.asarray(vel_emu, dtype=np.float32),
            "class_pk_table_path": None if class_pk_table_path is None else str(class_pk_table_path),
            "emulator_runtime_seconds_by_sim": [float(x) for x in emu_runtimes_s],
            "emulator_runtime_hms_by_sim": [format_hms(float(x)) for x in emu_runtimes_s],
            "metadata": metadata,
            "saved_paths": saved_paths,
        }
    finally:
        if temp_class_table_path is not None and temp_class_table_path.is_file():
            temp_class_table_path.unlink()
