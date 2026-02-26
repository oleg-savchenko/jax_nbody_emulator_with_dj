#!/usr/bin/env python3
"""
Build a FoF halo catalog from saved emulator displacements and plot halo diagnostics.

This script supports two modes:
1. Full FoF mode (default):
   - Loads emulator displacement (`emu_dis_z0.npy`), reconstructs Eulerian particles,
     runs nbodykit FoF, and saves a halo catalog (`CMPosition`, `Npart`, `Mass`) as `.npz`.
2. Plot-only mode (`--plot-only`):
   - Reuses a saved halo catalog and regenerates plots without rerunning FoF.

Plots produced:
- Halo slice map (with matched density slice from `emu_delta_z0.npy`).
- Halo mass function (HMF), dn/dlog10M, with optional Pylians overlays.

In plot-only mode, the HMF panel can include:
- Pylians empirical (FoF-corrected).
- Pylians empirical (no FoF correction).
- Pylians theoretical fit (e.g. Tinker), when CLASS P(k) table is available.

Environment note:
- Full FoF mode requires `nbodykit` + `mpi4py` (typically `.venv_nbodykit`).
- Plot-only mode does not import nbodykit and can run in your regular `.venv`.

Example:
  module load 2024
  module load Python/3.12.3-GCCcore-13.3.0
  module load OpenMPI/5.0.3-GCC-13.3.0
  source /home/osavchenko/jax_nbody_emulator/.venv_nbodykit/bin/activate
  python scripts/halos.py \
    --emulator-output-dir outputs/discodj_emulator_seed42_res512_box1000 \
    --linking-length 0.2 \
    --nmin 20 \
    --slice-axis x \
    --slice-width 20.0

Plot-only mode (reuse saved catalog, regenerate plots only):
  python scripts/halos.py \
    --emulator-output-dir outputs/discodj_emulator_seed42_res512_box1000 \
    --plot-only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency check
    matplotlib = None
    plt = None
    _MPL_IMPORT_ERROR = exc
else:
    _MPL_IMPORT_ERROR = None

try:
    import mass_function_library as MFL
except Exception as exc:  # pragma: no cover - optional dependency at import time
    MFL = None
    _MFL_IMPORT_ERROR = exc
else:
    _MFL_IMPORT_ERROR = None


RHO_CRIT_H2_MSUN_MPC3 = 2.77536627e11


def log_progress(message: str, *, rank: int, root_only: bool = True) -> None:
    """Print a timestamped progress message (root-only by default)."""
    if root_only and rank != 0:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [rank {rank}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for FoF cataloging and halo summary plots."""
    parser = argparse.ArgumentParser(
        description="Build FoF halos from saved emulator displacements and plot halo summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--emulator-output-dir",
        type=Path,
        default=Path("outputs/discodj_emulator_seed42_res512_box1000"),
        help="Directory with emulator outputs (contains displacement and metadata files).",
    )
    parser.add_argument(
        "--displacement-file",
        type=str,
        default="emu_dis_z0.npy",
        help="Displacement filename inside --emulator-output-dir.",
    )
    parser.add_argument(
        "--density-file",
        type=str,
        default="emu_delta_z0.npy",
        help="Final density-field filename inside --emulator-output-dir used for slice comparison.",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="metadata.json",
        help="Metadata filename inside --emulator-output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for halo catalog + plots (defaults to --emulator-output-dir).",
    )
    parser.add_argument(
        "--n-part",
        type=int,
        default=None,
        help="Override particle/displacement resolution; otherwise inferred from metadata/file.",
    )
    parser.add_argument(
        "--boxsize",
        type=float,
        default=None,
        help="Override box size in Mpc/h; otherwise read from metadata.",
    )
    parser.add_argument(
        "--omega-m",
        type=float,
        default=None,
        help="Override Omega_m used for particle mass and HMF mass units.",
    )
    parser.add_argument(
        "--linking-length",
        type=float,
        default=0.2,
        help="FoF linking length in units of mean inter-particle spacing (unless --absolute-linking).",
    )
    parser.add_argument(
        "--absolute-linking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat --linking-length as absolute Mpc/h instead of relative to mean spacing.",
    )
    parser.add_argument("--nmin", type=int, default=20, help="Minimum FoF particle count per halo.")
    parser.add_argument(
        "--slice-axis",
        choices=("x", "y", "z"),
        default="x",
        help="Axis normal to the halo slice slab.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Slice center index on --slice-axis; if omitted, center of the box is used.",
    )
    parser.add_argument(
        "--slice-width",
        type=float,
        default=20.0,
        help="Slice slab width in Mpc/h.",
    )
    parser.add_argument("--hmf-nbins", type=int, default=25, help="Number of logarithmic HMF bins.")
    parser.add_argument("--hmf-mmin", type=float, default=None, help="Lower HMF mass bound [Msun/h].")
    parser.add_argument("--hmf-mmax", type=float, default=None, help="Upper HMF mass bound [Msun/h].")
    parser.add_argument(
        "--catalog-file",
        type=str,
        default="fof_catalog.npz",
        help="Output FoF catalog filename.",
    )
    parser.add_argument(
        "--slice-plot-file",
        type=str,
        default="fof_slice.png",
        help="Output halo slice plot filename.",
    )
    parser.add_argument(
        "--hmf-plot-file",
        type=str,
        default="fof_hmf.png",
        help="Output HMF plot filename.",
    )
    parser.add_argument(
        "--pylians-hmf-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Overlay Pylians theoretical HMF (from mass_function_library) using CLASS linear P(k) "
            "if available in metadata/output directory. Applied only in --plot-only mode."
        ),
    )
    parser.add_argument(
        "--pylians-hmf-model",
        type=str,
        choices=("ST", "Tinker", "Tinker10", "Crocce", "Jenkins", "Warren", "Watson", "Watson_FOF", "Angulo"),
        default="Tinker",
        help="Pylians HMF fitting function author/model.",
    )
    parser.add_argument(
        "--pylians-hmf-integration-bins",
        type=int,
        default=10000,
        help="Number of log-k integration bins for Pylians MF_theory.",
    )
    parser.add_argument(
        "--plot-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip FoF generation and only make plots from a precomputed halo catalog.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> dict:
    """Load metadata JSON if present, otherwise return an empty dictionary."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def infer_boxsize(metadata: dict, boxsize_override: float | None) -> float:
    """Infer box size in Mpc/h from CLI override or metadata."""
    if boxsize_override is not None:
        return float(boxsize_override)
    if "boxsize_mpc_over_h" in metadata:
        return float(metadata["boxsize_mpc_over_h"])
    raise ValueError("Box size is required: pass --boxsize or provide metadata with boxsize_mpc_over_h.")


def infer_npart(dis_shape: tuple[int, ...], metadata: dict, npart_override: int | None) -> int:
    """Infer displacement lattice resolution from CLI override, metadata, or file shape."""
    if npart_override is not None:
        return int(npart_override)
    if "n_part" in metadata:
        return int(metadata["n_part"])
    if "res" in metadata:
        return int(metadata["res"])

    if len(dis_shape) != 4:
        raise ValueError(f"Displacement field must be 4D, got shape={dis_shape}.")
    if dis_shape[0] == 3:
        return int(dis_shape[1])
    if dis_shape[-1] == 3:
        return int(dis_shape[0])
    raise ValueError(f"Could not infer n_part from displacement shape={dis_shape}.")


def infer_omega_m(metadata: dict, omega_m_override: float | None) -> float:
    """Infer Omega_m from CLI override or pipeline metadata."""
    if omega_m_override is not None:
        return float(omega_m_override)
    q = metadata.get("quijote_fiducial", {})
    if "Omega_m" in q:
        return float(q["Omega_m"])
    return 0.3175


def find_class_pk_table_path(metadata: dict, emu_dir: Path) -> Path | None:
    """Find a CLASS linear P(k) table path from metadata or standard output naming."""
    candidates: list[Path] = []
    meta_pk = metadata.get("class_linear_pk_table_file")
    if meta_pk:
        p = Path(str(meta_pk))
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(p)
            candidates.append(emu_dir / p)
            candidates.append(emu_dir / p.name)
    candidates.append(emu_dir / "class_linear_pk_z0_table.txt")
    for p in candidates:
        if p.is_file():
            return p
    return None


def compute_pylians_hmf_theory(
    *,
    mass_centers: np.ndarray,
    omega_m: float,
    class_pk_table_path: Path,
    author: str,
    integration_bins: int,
) -> np.ndarray:
    """Compute dn/dlog10M via Pylians `MF_theory`, using a CLASS linear P(k) table."""
    if MFL is None:
        raise ImportError("mass_function_library is unavailable.") from _MFL_IMPORT_ERROR
    table = np.loadtxt(class_pk_table_path)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f"Invalid CLASS P(k) table format in {class_pk_table_path}")
    k = np.asarray(table[:, 0], dtype=np.float64)
    pk = np.asarray(table[:, 1], dtype=np.float64)
    dndm = np.asarray(
        MFL.MF_theory(
            k_in=k,
            Pk_in=pk,
            OmegaM=float(omega_m),
            Masses=np.asarray(mass_centers, dtype=np.float64),
            author=str(author),
            bins=int(integration_bins),
            z=0.0,
            delta=200.0,
        ),
        dtype=np.float64,
    )
    return dndm * np.asarray(mass_centers, dtype=np.float64) * np.log(10.0)


def compute_pylians_empirical_hmf(
    *,
    n_part_halos: np.ndarray,
    mass_centers: np.ndarray,
    log_edges: np.ndarray,
    boxsize: float,
    particle_mass: float,
    apply_fof_correction: bool,
) -> np.ndarray:
    """Compute a Pylians-convention empirical dn/dlog10M from FoF halo particle counts."""
    n_h = np.asarray(n_part_halos, dtype=np.float64)
    n_h = n_h[n_h > 0.0]
    if n_h.size == 0:
        return np.full_like(np.asarray(mass_centers, dtype=np.float64), np.nan, dtype=np.float64)

    if apply_fof_correction:
        masses = float(particle_mass) * (n_h * (1.0 - n_h ** (-0.6)))
    else:
        masses = float(particle_mass) * n_h

    edges = 10.0 ** np.asarray(log_edges, dtype=np.float64)
    counts, _ = np.histogram(masses, bins=edges)
    dM = edges[1:] - edges[:-1]
    volume = float(boxsize) ** 3
    dndM = counts.astype(np.float64) / (dM * volume)
    return dndM * np.asarray(mass_centers, dtype=np.float64) * np.log(10.0)


def particle_mass_msun_h(omega_m: float, boxsize: float, n_part: int) -> float:
    """Compute particle mass in Msun/h for a uniform N^3 particle load."""
    n_particles = float(n_part) ** 3
    volume = float(boxsize) ** 3
    return float(omega_m) * RHO_CRIT_H2_MSUN_MPC3 * volume / n_particles


def xslab_for_rank(n_part: int, rank: int, size: int) -> tuple[int, int]:
    """Return rank-local x-slab bounds [start, stop) for an N^3 mesh."""
    i0 = (rank * n_part) // size
    i1 = ((rank + 1) * n_part) // size
    return i0, i1


def load_local_positions_from_displacement(
    displacement_file: Path,
    boxsize: float,
    n_part: int,
    rank: int,
    size: int,
) -> np.ndarray:
    """Load rank-local displacement slab and convert it to Eulerian positions."""
    disp = np.load(displacement_file, mmap_mode="r")
    shape = tuple(int(x) for x in disp.shape)
    if len(shape) != 4:
        raise ValueError(f"Displacement field must be 4D, got shape={shape}.")

    if shape[0] == 3:
        axis_format = "channels_first"
        n_file = int(shape[1])
    elif shape[-1] == 3:
        axis_format = "channels_last"
        n_file = int(shape[0])
    else:
        raise ValueError(f"Displacement shape must be (3,N,N,N) or (N,N,N,3), got {shape}.")

    if n_file != int(n_part):
        raise ValueError(f"n_part mismatch: inferred/argument={n_part}, file has N={n_file}.")

    i0, i1 = xslab_for_rank(n_part=n_part, rank=rank, size=size)
    nx_local = i1 - i0
    if nx_local == 0:
        return np.empty((0, 3), dtype=np.float32)

    if axis_format == "channels_first":
        psi_local = np.asarray(disp[:, i0:i1, :, :], dtype=np.float32)
    else:
        psi_local = np.asarray(np.moveaxis(disp[i0:i1, :, :, :], -1, 0), dtype=np.float32)

    dx = float(boxsize) / float(n_part)
    x = (np.arange(i0, i1, dtype=np.float32) * dx)[:, None, None]
    y = (np.arange(n_part, dtype=np.float32) * dx)[None, :, None]
    z = (np.arange(n_part, dtype=np.float32) * dx)[None, None, :]

    n_local = nx_local * n_part * n_part
    pos = np.empty((n_local, 3), dtype=np.float32)
    pos[:, 0] = np.mod(x + psi_local[0], boxsize).reshape(-1)
    pos[:, 1] = np.mod(y + psi_local[1], boxsize).reshape(-1)
    pos[:, 2] = np.mod(z + psi_local[2], boxsize).reshape(-1)
    return pos


def run_fof(
    local_positions: np.ndarray,
    boxsize: float,
    n_part: int,
    linking_length: float,
    nmin: int,
    absolute_linking: bool,
    comm: Any,
    fof_cls: Any,
    fof_catalog_fn: Any,
    array_catalog_cls: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Run nbodykit FoF and return local halo center positions and particle counts."""
    source = array_catalog_cls(
        {"Position": np.asarray(local_positions, dtype=np.float32)},
        BoxSize=np.array([boxsize, boxsize, boxsize], dtype=np.float64),
        Nmesh=np.array([n_part, n_part, n_part], dtype=np.int64),
        comm=comm,
    )
    fof = fof_cls(
        source,
        linking_length=float(linking_length),
        nmin=int(nmin),
        absolute=bool(absolute_linking),
        periodic=True,
    )
    halos = fof_catalog_fn(
        source,
        fof.labels,
        comm=comm,
        position="Position",
        velocity="Position",
        periodic=True,
    )
    # nbodykit.fof_catalog usually returns a structured numpy-like array here.
    # Keep a fallback for versions that may return a lazy catalog object.
    if hasattr(halos, "compute"):
        cm_pos = np.asarray(halos.compute(halos["CMPosition"]), dtype=np.float32)
        length = np.asarray(halos.compute(halos["Length"]), dtype=np.int32)
    else:
        cm_pos = np.asarray(halos["CMPosition"], dtype=np.float32)
        length = np.asarray(halos["Length"], dtype=np.int32)
    mask = length > 0
    return cm_pos[mask], length[mask]


def gather_halos_to_root(
    cm_pos_local: np.ndarray,
    length_local: np.ndarray,
    comm: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Gather distributed halo arrays to root rank."""
    cm_chunks = comm.gather(np.asarray(cm_pos_local, dtype=np.float32), root=0)
    len_chunks = comm.gather(np.asarray(length_local, dtype=np.int32), root=0)
    if comm.rank != 0:
        return None, None
    cm_pos = np.concatenate(cm_chunks, axis=0) if cm_chunks else np.empty((0, 3), dtype=np.float32)
    length = np.concatenate(len_chunks, axis=0) if len_chunks else np.empty((0,), dtype=np.int32)
    return cm_pos, length


def project_density_slab(
    *,
    density_path: Path,
    boxsize: float,
    axis: str,
    slice_center: float,
    slice_width: float,
) -> tuple[np.ndarray, int]:
    """Load a density field and project the selected slab to a 2D slice map."""
    delta = np.load(density_path, mmap_mode="r")
    if delta.ndim != 3 or not (delta.shape[0] == delta.shape[1] == delta.shape[2]):
        raise ValueError(f"Density field must be cubic 3D, got shape={delta.shape}")

    n = int(delta.shape[0])
    dx = float(boxsize) / float(n)
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]

    coords = (np.arange(n, dtype=np.float64) + 0.5) * dx
    dist = np.abs(coords - float(slice_center))
    dist = np.minimum(dist, float(boxsize) - dist)
    mask = dist <= (0.5 * float(slice_width))
    n_cells = int(np.count_nonzero(mask))
    if n_cells == 0:
        nearest = int(np.argmin(dist))
        mask[nearest] = True
        n_cells = 1

    if axis_idx == 0:
        proj = np.asarray(delta[mask, :, :], dtype=np.float32).mean(axis=0)
    elif axis_idx == 1:
        proj = np.asarray(delta[:, mask, :], dtype=np.float32).mean(axis=1)
    else:
        proj = np.asarray(delta[:, :, mask], dtype=np.float32).mean(axis=2)
    return np.asarray(proj, dtype=np.float32), n_cells


def make_halo_slice_plot(
    cm_pos: np.ndarray,
    masses: np.ndarray,
    *,
    boxsize: float,
    axis: str,
    slice_center: float,
    slice_width: float,
    density_proj: np.ndarray | None,
    out_path: Path,
) -> int:
    """Create a halo slice map plus optional matched density subplot and save them."""
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    plane_axes = [i for i in (0, 1, 2) if i != axis_idx]

    dist = np.abs(cm_pos[:, axis_idx] - float(slice_center))
    dist = np.minimum(dist, float(boxsize) - dist)
    in_slice = dist <= (0.5 * float(slice_width))

    labels = ["x", "y", "z"]
    title_slice = (
        f"{labels[axis_idx]} in [{slice_center - slice_width/2:.1f}, "
        f"{slice_center + slice_width/2:.1f}] Mpc/h"
    )

    if density_proj is None:
        fig, ax_halo = plt.subplots(figsize=(7, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_halo, ax_den = axes

    if not np.any(in_slice):
        ax_halo.text(0.5, 0.5, "No halos in selected slice", ha="center", va="center")
        ax_halo.set_axis_off()
    else:
        pos = cm_pos[in_slice]
        m = masses[in_slice]
        logm = np.log10(m)
        logm_min = float(np.nanmin(logm))
        logm_max = float(np.nanmax(logm))
        if logm_max <= logm_min:
            size = np.full_like(logm, 8.0)
        else:
            size = 5.0 + 20.0 * (logm - logm_min) / (logm_max - logm_min)

        sc = ax_halo.scatter(
            pos[:, plane_axes[1]],
            pos[:, plane_axes[0]],
            c=logm,
            s=size,
            cmap="viridis",
            alpha=0.8,
            linewidths=0.0,
        )
        cbar = fig.colorbar(sc, ax=ax_halo)
        cbar.set_label(r"$\log_{10}(M_{\rm halo}\,[M_\odot/h])$")
        ax_halo.set_xlabel(f"{labels[plane_axes[1]]} [Mpc/h]")
        ax_halo.set_ylabel(f"{labels[plane_axes[0]]} [Mpc/h]")
        ax_halo.set_xlim(0.0, boxsize)
        ax_halo.set_ylim(0.0, boxsize)
        ax_halo.set_aspect("equal", adjustable="box")
        ax_halo.set_title(f"FoF halos slice: {title_slice}")

    if density_proj is not None:
        dproj = np.asarray(density_proj, dtype=np.float32)
        dmin = float(np.nanpercentile(dproj, 1.0))
        dmax = float(np.nanpercentile(dproj, 99.0))
        if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
            dmin = float(np.nanmin(dproj))
            dmax = float(np.nanmax(dproj))
        im = ax_den.imshow(
            dproj,
            origin="lower",
            extent=[0.0, boxsize, 0.0, boxsize],
            cmap="inferno",
            vmin=dmin,
            vmax=dmax,
            aspect="equal",
        )
        cbar = fig.colorbar(im, ax=ax_den)
        cbar.set_label(r"$\delta_{\rm emu}$")
        ax_den.set_xlabel(f"{labels[plane_axes[1]]} [Mpc/h]")
        ax_den.set_ylabel(f"{labels[plane_axes[0]]} [Mpc/h]")
        ax_den.set_xlim(0.0, boxsize)
        ax_den.set_ylim(0.0, boxsize)
        ax_den.set_aspect("equal", adjustable="box")
        ax_den.set_title(f"Emulated density slice: {title_slice}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return int(np.count_nonzero(in_slice))


def make_hmf_plot(
    masses: np.ndarray,
    *,
    boxsize: float,
    nbins: int,
    mmin: float | None,
    mmax: float | None,
    pylians_empirical_curve: np.ndarray | None,
    pylians_empirical_label: str | None,
    pylians_empirical_nocorr_curve: np.ndarray | None,
    pylians_empirical_nocorr_label: str | None,
    pylians_curve: np.ndarray | None,
    pylians_label: str | None,
    out_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute and plot halo mass function dn/dlog10M."""
    masses = np.asarray(masses, dtype=np.float64)
    masses = masses[masses > 0.0]
    if masses.size == 0:
        raise ValueError("No positive halo masses found for HMF.")

    if mmin is None:
        mmin_val = float(np.min(masses))
    else:
        mmin_val = float(mmin)
    if mmax is None:
        mmax_val = float(np.max(masses))
    else:
        mmax_val = float(mmax)
    if not (mmax_val > mmin_val > 0.0):
        raise ValueError(f"Invalid HMF bounds: mmin={mmin_val}, mmax={mmax_val}.")

    log_edges = np.linspace(np.log10(mmin_val), np.log10(mmax_val), int(nbins) + 1)
    counts, _ = np.histogram(np.log10(masses), bins=log_edges)
    dlogm = log_edges[1] - log_edges[0]

    volume = float(boxsize) ** 3
    dndlogm = counts.astype(np.float64) / (volume * dlogm)
    log_centers = 0.5 * (log_edges[1:] + log_edges[:-1])
    m_centers = 10.0**log_centers

    fig, ax = plt.subplots(figsize=(7, 5))
    mask = counts > 0
    ax.plot(m_centers[mask], dndlogm[mask], marker="o", lw=1.5, label="Measured HMF")
    if pylians_empirical_curve is not None:
        py_emp = np.asarray(pylians_empirical_curve, dtype=np.float64)
        py_emp_mask = np.isfinite(py_emp) & (py_emp > 0.0)
        if np.any(py_emp_mask):
            ax.plot(
                m_centers[py_emp_mask],
                py_emp[py_emp_mask],
                lw=1.5,
                ls="-.",
                label=(pylians_empirical_label or "Pylians empirical"),
            )
    if pylians_empirical_nocorr_curve is not None:
        py_emp_nocorr = np.asarray(pylians_empirical_nocorr_curve, dtype=np.float64)
        py_emp_nocorr_mask = np.isfinite(py_emp_nocorr) & (py_emp_nocorr > 0.0)
        if np.any(py_emp_nocorr_mask):
            ax.plot(
                m_centers[py_emp_nocorr_mask],
                py_emp_nocorr[py_emp_nocorr_mask],
                lw=1.5,
                ls=":",
                label=(pylians_empirical_nocorr_label or "Pylians empirical (no FoF correction)"),
            )
    if pylians_curve is not None:
        py = np.asarray(pylians_curve, dtype=np.float64)
        py_mask = np.isfinite(py) & (py > 0.0)
        if np.any(py_mask):
            ax.plot(
                m_centers[py_mask],
                py[py_mask],
                lw=1.5,
                ls="--",
                label=(pylians_label or "Pylians theory"),
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_{\rm halo}\ [M_\odot/h]$")
    ax.set_ylabel(r"$dn/d\log_{10}M\ [(h/{\rm Mpc})^3]$")
    ax.set_title("FoF Halo Mass Function")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return m_centers, dndlogm, counts


def main() -> None:
    """Run FoF halo finding from emulator displacement outputs and write summaries."""
    if _MPL_IMPORT_ERROR is not None:
        raise ImportError(
            "matplotlib is required for halo slice/HMF plots. Install it in your nbodykit environment."
        ) from _MPL_IMPORT_ERROR

    args = parse_args()
    if args.plot_only:
        comm = None
        rank = 0
        size = 1
        fof_cls = None
        fof_catalog_fn = None
        array_catalog_cls = None
    else:
        try:
            from mpi4py import MPI
            from nbodykit.algorithms.fof import FOF, fof_catalog
            from nbodykit.source.catalog import ArrayCatalog
        except ImportError as exc:
            raise ImportError(
                "Full FoF mode requires nbodykit+mpi4py. "
                "Use .venv_nbodykit for full mode, or --plot-only in .venv."
            ) from exc

        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
        fof_cls = FOF
        fof_catalog_fn = fof_catalog
        array_catalog_cls = ArrayCatalog
    t_total = time.perf_counter()

    mode = "plot-only" if bool(args.plot_only) else "full"
    log_progress(f"Starting FoF halo pipeline ({mode} mode).", rank=rank)
    log_progress(f"MPI world size={size}", rank=rank)

    emu_dir = Path(args.emulator_output_dir)
    if not emu_dir.is_dir():
        raise FileNotFoundError(f"Emulator output directory not found: {emu_dir}")

    out_dir = emu_dir if args.output_dir is None else Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if comm is not None:
        comm.Barrier()

    density_path = emu_dir / args.density_file
    if not density_path.is_file():
        raise FileNotFoundError(f"Missing density file for slice plotting: {density_path}")
    metadata_path = emu_dir / args.metadata_file
    log_progress(f"Loading metadata from {metadata_path}", rank=rank)
    metadata = load_metadata(metadata_path)

    cm_pos: np.ndarray | None = None
    length: np.ndarray | None = None
    masses: np.ndarray | None = None
    boxsize: float | None = None
    n_part: int | None = None
    omega_m: float | None = None
    catalog_path = out_dir / args.catalog_file
    linking_length_out: float = float(args.linking_length)
    absolute_linking_out: bool = bool(args.absolute_linking)
    nmin_out: int = int(args.nmin)

    if args.plot_only:
        if rank != 0:
            return
        if not catalog_path.is_file():
            fallback_catalog = emu_dir / args.catalog_file
            if fallback_catalog.is_file():
                catalog_path = fallback_catalog
            else:
                raise FileNotFoundError(
                    f"Missing precomputed halo catalog for plot-only mode: {catalog_path}"
                )
        log_progress(f"Loading precomputed halo catalog from {catalog_path}", rank=rank)
        cat = np.load(catalog_path)
        cm_pos = np.asarray(cat["CMPosition"], dtype=np.float32)
        length = np.asarray(cat["Npart"], dtype=np.int32)
        masses = np.asarray(cat["Mass"], dtype=np.float64) if "Mass" in cat.files else None

        if args.boxsize is not None:
            boxsize = float(args.boxsize)
        elif "BoxSize" in cat.files:
            boxsize_arr = np.asarray(cat["BoxSize"], dtype=np.float64).reshape(-1)
            boxsize = float(boxsize_arr[0])
        else:
            boxsize = infer_boxsize(metadata=metadata, boxsize_override=None)

        if args.n_part is not None:
            n_part = int(args.n_part)
        elif "NpartPerDim" in cat.files:
            n_part = int(np.asarray(cat["NpartPerDim"]).reshape(()))
        elif "n_part" in metadata:
            n_part = int(metadata["n_part"])
        elif "res" in metadata:
            n_part = int(metadata["res"])
        else:
            raise ValueError("Could not infer n_part for plot-only mode; pass --n-part.")

        omega_m = infer_omega_m(metadata=metadata, omega_m_override=args.omega_m)
        if masses is None:
            m_particle = particle_mass_msun_h(omega_m=omega_m, boxsize=boxsize, n_part=n_part)
            masses = np.asarray(length, dtype=np.float64) * float(m_particle)

        if "LinkingLength" in cat.files:
            linking_length_out = float(np.asarray(cat["LinkingLength"]).reshape(()))
        if "AbsoluteLinking" in cat.files:
            absolute_linking_out = bool(np.asarray(cat["AbsoluteLinking"]).reshape(()))
        if "Nmin" in cat.files:
            nmin_out = int(np.asarray(cat["Nmin"]).reshape(()))
    else:
        displacement_path = emu_dir / args.displacement_file
        if not displacement_path.is_file():
            raise FileNotFoundError(f"Missing displacement file: {displacement_path}")

        boxsize = infer_boxsize(metadata=metadata, boxsize_override=args.boxsize)

        log_progress(f"Reading displacement header from {displacement_path}", rank=rank)
        disp_shape = tuple(np.load(displacement_path, mmap_mode="r").shape)
        n_part = infer_npart(dis_shape=disp_shape, metadata=metadata, npart_override=args.n_part)
        omega_m = infer_omega_m(metadata=metadata, omega_m_override=args.omega_m)
        m_particle = particle_mass_msun_h(omega_m=omega_m, boxsize=boxsize, n_part=n_part)
        log_progress(
            (
                f"Inferred setup: n_part={n_part}, boxsize={boxsize:.3f} Mpc/h, "
                f"Omega_m={omega_m:.5f}, m_particle={m_particle:.6e} Msun/h"
            ),
            rank=rank,
        )

        t_positions = time.perf_counter()
        log_progress("Building Eulerian particle positions from displacement...", rank=rank, root_only=False)
        local_pos = load_local_positions_from_displacement(
            displacement_file=displacement_path,
            boxsize=boxsize,
            n_part=n_part,
            rank=rank,
            size=size,
        )
        log_progress(
            f"Local positions built: {local_pos.shape[0]} particles in {time.perf_counter() - t_positions:.2f} s",
            rank=rank,
            root_only=False,
        )

        t_fof = time.perf_counter()
        log_progress(
            (
                f"Running FoF with linking_length={args.linking_length}, "
                f"absolute_linking={args.absolute_linking}, nmin={args.nmin}..."
            ),
            rank=rank,
        )
        cm_local, length_local = run_fof(
            local_positions=local_pos,
            boxsize=boxsize,
            n_part=n_part,
            linking_length=float(args.linking_length),
            nmin=int(args.nmin),
            absolute_linking=bool(args.absolute_linking),
            comm=comm,
            fof_cls=fof_cls,
            fof_catalog_fn=fof_catalog_fn,
            array_catalog_cls=array_catalog_cls,
        )
        log_progress(
            f"FoF finished in {time.perf_counter() - t_fof:.2f} s; local halos={length_local.size}",
            rank=rank,
            root_only=False,
        )

        log_progress("Gathering halos to rank 0...", rank=rank)
        cm_pos, length = gather_halos_to_root(cm_pos_local=cm_local, length_local=length_local, comm=comm)
        if rank != 0:
            return

        log_progress(f"Gather complete: total halos={length.size}", rank=rank)
        masses = np.asarray(length, dtype=np.float64) * float(m_particle)
        log_progress(f"Saving FoF catalog to {catalog_path}", rank=rank)
        np.savez_compressed(
            catalog_path,
            CMPosition=np.asarray(cm_pos, dtype=np.float32),
            Npart=np.asarray(length, dtype=np.int32),
            Mass=np.asarray(masses, dtype=np.float64),
            BoxSize=np.array([boxsize, boxsize, boxsize], dtype=np.float64),
            NpartPerDim=np.int32(n_part),
            LinkingLength=np.float64(linking_length_out),
            AbsoluteLinking=np.bool_(absolute_linking_out),
            Nmin=np.int32(nmin_out),
        )

    if rank != 0:
        return
    if boxsize is None or n_part is None or omega_m is None or cm_pos is None or length is None or masses is None:
        raise RuntimeError("Internal error: missing required halo data on rank 0.")

    if args.slice_index is None:
        slice_center = 0.5 * float(boxsize)
    else:
        if not (0 <= int(args.slice_index) < n_part):
            raise ValueError(f"--slice-index must be in [0, {n_part - 1}]")
        slice_center = (float(args.slice_index) + 0.5) * float(boxsize) / float(n_part)

    slice_plot_path = out_dir / args.slice_plot_file
    log_progress(f"Creating halo slice plot: {slice_plot_path}", rank=rank)
    log_progress(f"Projecting matching density slab from {density_path}", rank=rank)
    density_proj, density_slab_cells = project_density_slab(
        density_path=density_path,
        boxsize=float(boxsize),
        axis=str(args.slice_axis),
        slice_center=float(slice_center),
        slice_width=float(args.slice_width),
    )
    halos_in_slice = make_halo_slice_plot(
        cm_pos=np.asarray(cm_pos, dtype=np.float32),
        masses=np.asarray(masses, dtype=np.float64),
        boxsize=float(boxsize),
        axis=str(args.slice_axis),
        slice_center=float(slice_center),
        slice_width=float(args.slice_width),
        density_proj=np.asarray(density_proj, dtype=np.float32),
        out_path=slice_plot_path,
    )

    hmf_plot_path = out_dir / args.hmf_plot_file
    log_progress(f"Computing and saving HMF plot: {hmf_plot_path}", rank=rank)
    pylians_emp_hmf = None
    pylians_emp_hmf_label = None
    pylians_emp_hmf_nocorr = None
    pylians_emp_hmf_nocorr_label = None
    pylians_emp_hmf_status = "disabled_non_plot_mode"
    pylians_emp_hmf_nocorr_status = "disabled_non_plot_mode"
    pylians_hmf = None
    pylians_hmf_label = None
    pylians_hmf_status = "disabled_non_plot_mode"
    pylians_pk_table_path = None
    masses_pos = np.asarray(masses, dtype=np.float64)
    masses_pos = masses_pos[masses_pos > 0.0]
    mmin_eff = float(np.min(masses_pos)) if args.hmf_mmin is None else float(args.hmf_mmin)
    mmax_eff = float(np.max(masses_pos)) if args.hmf_mmax is None else float(args.hmf_mmax)
    log_edges_hmf = np.linspace(np.log10(mmin_eff), np.log10(mmax_eff), int(args.hmf_nbins) + 1)
    m_centers_hmf = 10.0 ** (0.5 * (log_edges_hmf[1:] + log_edges_hmf[:-1]))

    enable_pylians_empirical = bool(args.plot_only)
    enable_pylians_empirical_fof_correction = bool(args.plot_only)
    if enable_pylians_empirical:
        ratio = np.asarray(masses, dtype=np.float64) / np.clip(np.asarray(length, dtype=np.float64), 1.0, None)
        ratio = ratio[np.isfinite(ratio) & (ratio > 0.0)]
        if ratio.size == 0:
            pylians_emp_hmf_status = "failed: ValueError('cannot infer particle mass from catalog Mass/Npart.')"
            pylians_emp_hmf_nocorr_status = pylians_emp_hmf_status
            log_progress("Pylians empirical HMF overlays failed: cannot infer particle mass.", rank=rank)
        else:
            m_particle_emp = float(np.median(ratio))
            try:
                pylians_emp_hmf = compute_pylians_empirical_hmf(
                    n_part_halos=np.asarray(length, dtype=np.float64),
                    mass_centers=np.asarray(m_centers_hmf, dtype=np.float64),
                    log_edges=np.asarray(log_edges_hmf, dtype=np.float64),
                    boxsize=float(boxsize),
                    particle_mass=float(m_particle_emp),
                    apply_fof_correction=True,
                )
                pylians_emp_hmf_label = "Pylians empirical (FoF-corrected)"
                pylians_emp_hmf_status = "ok"
            except Exception as exc:
                pylians_emp_hmf_status = f"failed: {repr(exc)}"
                log_progress(f"Pylians empirical (FoF-corrected) overlay failed: {exc!r}", rank=rank)
            try:
                pylians_emp_hmf_nocorr = compute_pylians_empirical_hmf(
                    n_part_halos=np.asarray(length, dtype=np.float64),
                    mass_centers=np.asarray(m_centers_hmf, dtype=np.float64),
                    log_edges=np.asarray(log_edges_hmf, dtype=np.float64),
                    boxsize=float(boxsize),
                    particle_mass=float(m_particle_emp),
                    apply_fof_correction=False,
                )
                pylians_emp_hmf_nocorr_label = "Pylians empirical (no FoF correction)"
                pylians_emp_hmf_nocorr_status = "ok"
            except Exception as exc:
                pylians_emp_hmf_nocorr_status = f"failed: {repr(exc)}"
                log_progress(f"Pylians empirical (no FoF correction) overlay failed: {exc!r}", rank=rank)
            if pylians_emp_hmf_status == "ok" or pylians_emp_hmf_nocorr_status == "ok":
                log_progress("Computed Pylians-convention empirical HMF overlays.", rank=rank)

    enable_pylians_theory = bool(args.plot_only) and bool(args.pylians_hmf_check)
    if enable_pylians_theory:
        pylians_pk_table = find_class_pk_table_path(metadata=metadata, emu_dir=emu_dir)
        if pylians_pk_table is None:
            pylians_hmf_status = "missing_class_pk_table"
            log_progress(
                "Pylians HMF overlay skipped: CLASS linear P(k) table not found in metadata/output dir.",
                rank=rank,
            )
        elif MFL is None:
            pylians_hmf_status = f"pylians_import_failed: {repr(_MFL_IMPORT_ERROR)}"
            log_progress(
                f"Pylians HMF overlay skipped: mass_function_library import failed ({_MFL_IMPORT_ERROR!r}).",
                rank=rank,
            )
        else:
            try:
                pylians_pk_table_path = str(pylians_pk_table)
                pylians_hmf = compute_pylians_hmf_theory(
                    mass_centers=np.asarray(m_centers_hmf, dtype=np.float64),
                    omega_m=float(omega_m),
                    class_pk_table_path=pylians_pk_table,
                    author=str(args.pylians_hmf_model),
                    integration_bins=int(args.pylians_hmf_integration_bins),
                )
                pylians_hmf_label = f"Pylians {args.pylians_hmf_model}"
                pylians_hmf_status = "ok"
                log_progress(
                    f"Computed Pylians HMF overlay using {pylians_pk_table} ({args.pylians_hmf_model}).",
                    rank=rank,
                )
            except Exception as exc:
                pylians_hmf_status = f"failed: {repr(exc)}"
                log_progress(f"Pylians HMF overlay failed: {exc!r}", rank=rank)

    m_centers, dndlogm, counts = make_hmf_plot(
        masses=np.asarray(masses, dtype=np.float64),
        boxsize=float(boxsize),
        nbins=int(args.hmf_nbins),
        mmin=args.hmf_mmin,
        mmax=args.hmf_mmax,
        pylians_empirical_curve=(
            None if pylians_emp_hmf is None else np.asarray(pylians_emp_hmf, dtype=np.float64)
        ),
        pylians_empirical_label=pylians_emp_hmf_label,
        pylians_empirical_nocorr_curve=(
            None if pylians_emp_hmf_nocorr is None else np.asarray(pylians_emp_hmf_nocorr, dtype=np.float64)
        ),
        pylians_empirical_nocorr_label=pylians_emp_hmf_nocorr_label,
        pylians_curve=None if pylians_hmf is None else np.asarray(pylians_hmf, dtype=np.float64),
        pylians_label=pylians_hmf_label,
        out_path=hmf_plot_path,
    )

    summary = {
        "emulator_output_dir": str(emu_dir),
        "displacement_file": None if args.plot_only else str(emu_dir / args.displacement_file),
        "density_file": str(density_path),
        "metadata_file": str(emu_dir / args.metadata_file),
        "boxsize_mpc_over_h": float(boxsize),
        "n_part": int(n_part),
        "omega_m": float(omega_m),
        "particle_mass_msun_over_h": (
            float(particle_mass_msun_h(omega_m=float(omega_m), boxsize=float(boxsize), n_part=int(n_part)))
        ),
        "linking_length": float(linking_length_out),
        "absolute_linking": bool(absolute_linking_out),
        "nmin": int(nmin_out),
        "n_halos": int(length.size),
        "plot_only": bool(args.plot_only),
        "slice_axis": str(args.slice_axis),
        "slice_center_mpc_over_h": float(slice_center),
        "slice_width_mpc_over_h": float(args.slice_width),
        "n_halos_in_slice": int(halos_in_slice),
        "n_density_cells_in_slice": int(density_slab_cells),
        "catalog_file": str(catalog_path),
        "slice_plot_file": str(slice_plot_path),
        "hmf_plot_file": str(hmf_plot_path),
        "pylians_empirical_check": bool(enable_pylians_empirical),
        "pylians_empirical_fof_correction": True,
        "pylians_empirical_hmf_status": pylians_emp_hmf_status,
        "pylians_empirical_nocorr_hmf_status": pylians_emp_hmf_nocorr_status,
        "pylians_hmf_check": bool(enable_pylians_theory),
        "pylians_hmf_model": str(args.pylians_hmf_model),
        "pylians_hmf_integration_bins": int(args.pylians_hmf_integration_bins),
        "pylians_hmf_status": pylians_hmf_status,
        "pylians_class_pk_table_file": pylians_pk_table_path,
        "hmf_bin_centers_msun_over_h": [float(x) for x in m_centers],
        "hmf_dn_dlog10m_h3_mpc_minus3": [float(x) for x in dndlogm],
        "hmf_pylians_empirical_dn_dlog10m_h3_mpc_minus3": (
            None
            if pylians_emp_hmf is None
            else [float(x) for x in np.asarray(pylians_emp_hmf, dtype=np.float64)]
        ),
        "hmf_pylians_empirical_nocorr_dn_dlog10m_h3_mpc_minus3": (
            None
            if pylians_emp_hmf_nocorr is None
            else [float(x) for x in np.asarray(pylians_emp_hmf_nocorr, dtype=np.float64)]
        ),
        "hmf_pylians_dn_dlog10m_h3_mpc_minus3": (
            None if pylians_hmf is None else [float(x) for x in np.asarray(pylians_hmf, dtype=np.float64)]
        ),
        "hmf_counts": [int(x) for x in counts],
    }
    summary_path = out_dir / "fof_summary.json"
    log_progress(f"Writing summary to {summary_path}", rank=rank)
    summary_path.write_text(json.dumps(summary, indent=2))
    log_progress(f"Pipeline wall time: {time.perf_counter() - t_total:.2f} s", rank=rank)

    print("FoF halo pipeline finished.")
    print(f"Total halos: {length.size}")
    print(f"Catalog: {catalog_path}")
    print(f"Slice plot: {slice_plot_path}")
    print(f"HMF plot: {hmf_plot_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
