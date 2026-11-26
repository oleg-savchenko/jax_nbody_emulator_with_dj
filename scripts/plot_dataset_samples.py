#!/usr/bin/env python3
"""
Plot random-sample diagnostics for emulator datasets.

This script creates:
  1) dataset_slices_3sims.png
     3x3 panel for 3 random simulations:
       - row 1: IC slices (delta_linear_z0.npy)
       - row 2: final density slices (emu_delta_z0.npy)
       - row 3: halo slab scatter (fof_catalog.npz)
  2) dataset_pk_ensemble_vs_class.png
     Ensemble P(k) over first num_sims simulations:
       - mean IC and final-field P(k)
       - shaded ±1σ and ±2σ bands
       - CLASS linear and Halofit curves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import get_pk_class

try:
    import Pk_library as PKL
except ImportError as exc:  # pragma: no cover - runtime dependency
    PKL = None
    _PKL_IMPORT_ERROR = exc
else:
    _PKL_IMPORT_ERROR = None


QUIJOTE_FIDUCIAL_CLASS = {
    "Omega_cdm": 0.3175 - 0.0490,
    "Omega_b": 0.0490,
    "h": 0.6711,
    "n_s": 0.9624,
    "sigma8": 0.8340,
}

AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    """Parse CLI args for dataset plotting."""
    parser = argparse.ArgumentParser(
        description="Plot random simulation slices and P(k) diagnostics from a generated emulator dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory containing numbered simulation subfolders.",
    )
    parser.add_argument(
        "--n-sims-plot",
        type=int,
        default=3,
        help="Number of random simulations to visualize.",
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=100,
        help="Number of first simulations (sorted by ID) used for ensemble P(k).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used for random simulation selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (defaults to ./outputs/<dataset-folder-name>).",
    )
    parser.add_argument(
        "--boxsize",
        type=float,
        default=None,
        help="Box size [Mpc/h]. If omitted, uses dataset_manifest.json config.boxsize.",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=None,
        help="Density grid size per axis. If omitted, uses manifest config.res or infers from field shape.",
    )
    parser.add_argument(
        "--slice-axis",
        choices=("x", "y", "z"),
        default="x",
        help="Axis normal to the plotted slice/slab.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Slice index (cell index along --slice-axis). Defaults to middle index.",
    )
    parser.add_argument(
        "--halo-slab-width",
        type=float,
        default=None,
        help="Halo slab thickness [Mpc/h]. Defaults to boxsize/8.",
    )
    parser.add_argument(
        "--halo-size-by-mass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale halo marker size by log10(Mass).",
    )
    parser.add_argument(
        "--pk-mas",
        choices=("None", "CIC", "TSC", "PCS"),
        default="None",
        help="MAS keyword passed to Pk_library for gridded fields.",
    )
    args = parser.parse_args()
    if args.n_sims_plot < 1:
        parser.error("--n-sims-plot must be >= 1")
    if args.num_sims < 1:
        parser.error("--num-sims must be >= 1")
    return args


def load_manifest(dataset_dir: Path) -> dict | None:
    """Load dataset manifest if available."""
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text())


def discover_valid_sim_dirs(dataset_dir: Path) -> list[tuple[int, Path]]:
    """Find numeric simulation folders that contain required files."""
    valid: list[tuple[int, Path]] = []
    for p in sorted(dataset_dir.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        if not p.name.isdigit():
            continue
        sid = int(p.name)
        required = [p / "delta_linear_z0.npy", p / "emu_delta_z0.npy", p / "fof_catalog.npz"]
        if all(x.is_file() for x in required):
            valid.append((sid, p))
    return valid


def extract_slice(field: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """Extract a 2D slice from a 3D field along a given axis at index idx."""
    if axis == 0:
        return np.asarray(field[idx, :, :], dtype=np.float32)
    if axis == 1:
        return np.asarray(field[:, idx, :], dtype=np.float32)
    return np.asarray(field[:, :, idx], dtype=np.float32)


def periodic_slab_mask(coords: np.ndarray, center: float, width: float, boxsize: float) -> np.ndarray:
    """Return mask for periodic slab selection |x-center|_periodic <= width/2."""
    dist = np.abs(np.asarray(coords, dtype=np.float64) - float(center))
    dist = np.minimum(dist, float(boxsize) - dist)
    return dist <= (0.5 * float(width))


def compute_pk(field: np.ndarray, boxsize: float, mas: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute isotropic 3D power spectrum using Pylians Pk_library."""
    if PKL is None:
        raise ImportError("Pk_library is unavailable. Install Pylians3.") from _PKL_IMPORT_ERROR
    pk_obj = PKL.Pk(np.asarray(field, dtype=np.float32), float(boxsize), axis=0, MAS=str(mas), threads=1, verbose=False)
    k = np.asarray(pk_obj.k3D, dtype=np.float64)
    pk = np.asarray(pk_obj.Pk[:, 0], dtype=np.float64)
    valid = k > 0.0
    return k[valid], pk[valid]


def mean_sigma_band(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean and +-1/2 sigma envelopes for a stack of curves."""
    arr = np.asarray(stack, dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    low1 = mean - std
    high1 = mean + std
    low2 = mean - 2.0 * std
    high2 = mean + 2.0 * std
    return mean, low1, high1, low2, high2


def resolve_boxsize_res(
    args: argparse.Namespace,
    manifest: dict | None,
    example_ic: np.ndarray,
) -> tuple[float, int]:
    """Resolve boxsize and resolution from CLI overrides or manifest/fallbacks."""
    cfg = {} if manifest is None else dict(manifest.get("config", {}))
    boxsize = float(args.boxsize) if args.boxsize is not None else cfg.get("boxsize", None)
    if boxsize is None:
        raise ValueError(
            "Cannot infer boxsize. Provide --boxsize or ensure dataset_manifest.json has config.boxsize."
        )
    boxsize = float(boxsize)

    if args.res is not None:
        res = int(args.res)
    elif cfg.get("res", None) is not None:
        res = int(cfg["res"])
    else:
        res = int(np.asarray(example_ic).shape[0])
    return boxsize, res


def main() -> None:
    """Run dataset plotting workflow for selected random simulations."""
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    if args.output_dir is None:
        output_dir = (REPO_ROOT / "outputs" / dataset_dir.name).resolve()
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(dataset_dir=dataset_dir)
    sim_dirs = discover_valid_sim_dirs(dataset_dir=dataset_dir)
    if len(sim_dirs) < int(args.n_sims_plot):
        raise RuntimeError(
            f"Found only {len(sim_dirs)} valid simulation folders with required files in {dataset_dir}, "
            f"but --n-sims-plot={args.n_sims_plot}."
        )

    rng = np.random.default_rng(int(args.random_seed))
    chosen_idx = np.asarray(rng.choice(len(sim_dirs), size=int(args.n_sims_plot), replace=False), dtype=int)
    chosen = [sim_dirs[i] for i in chosen_idx]

    # Load fields and halos.
    data: list[dict] = []
    for sid, sim_path in chosen:
        ic = np.asarray(np.load(sim_path / "delta_linear_z0.npy"), dtype=np.float32)
        final = np.asarray(np.load(sim_path / "emu_delta_z0.npy"), dtype=np.float32)
        halos = np.load(sim_path / "fof_catalog.npz")
        cm = np.asarray(halos["CMPosition"], dtype=np.float32)
        mass = np.asarray(halos["Mass"], dtype=np.float64)
        data.append({"sid": sid, "path": sim_path, "ic": ic, "final": final, "cm": cm, "mass": mass})

    boxsize, res = resolve_boxsize_res(args=args, manifest=manifest, example_ic=data[0]["ic"])
    axis = AXIS_TO_INDEX[str(args.slice_axis)]
    slice_index = int(args.slice_index) if args.slice_index is not None else (int(res) // 2)
    if slice_index < 0 or slice_index >= int(res):
        raise ValueError(f"--slice-index must be in [0, {int(res)-1}], got {slice_index}.")
    slab_width = float(args.halo_slab_width) if args.halo_slab_width is not None else (float(boxsize) / 8.0)
    if slab_width <= 0.0:
        raise ValueError("--halo-slab-width must be > 0.")

    dx = float(boxsize) / float(res)
    slice_center = (float(slice_index) + 0.5) * dx

    # Build slice panels.
    ic_slices = [extract_slice(d["ic"], axis=axis, idx=slice_index) for d in data]
    final_slices = [extract_slice(d["final"], axis=axis, idx=slice_index) for d in data]
    ic_abs = np.abs(np.concatenate([x.ravel() for x in ic_slices]))
    vmax_ic = float(np.percentile(ic_abs, 99.0)) if ic_abs.size > 0 else 1.0
    if vmax_ic <= 0.0:
        vmax_ic = 1.0
    final_vals = np.concatenate([x.ravel() for x in final_slices])
    vmax_final = float(np.percentile(final_vals, 99.5)) if final_vals.size > 0 else 1.0
    if vmax_final <= 0.0:
        vmax_final = 1.0

    plane_axes = [i for i in (0, 1, 2) if i != axis]
    axis_labels = ["x", "y", "z"]
    xlabel = axis_labels[plane_axes[1]]
    ylabel = axis_labels[plane_axes[0]]

    ncols = int(args.n_sims_plot)
    fig = plt.figure(figsize=(4.4 * ncols + 0.8, 11.0), constrained_layout=True)
    gs = fig.add_gridspec(3, ncols + 1, width_ratios=[1.0] * ncols + [0.05])
    axes = np.empty((3, ncols), dtype=object)
    for row in range(3):
        for col in range(ncols):
            axes[row, col] = fig.add_subplot(gs[row, col])
    cax_ic = fig.add_subplot(gs[0, ncols])
    cax_final = fig.add_subplot(gs[1, ncols])
    cax_halo = fig.add_subplot(gs[2, ncols])
    cax_halo.axis("off")

    im_ic = None
    im_final = None
    for col, d in enumerate(data):
        sid = int(d["sid"])
        ic2d = ic_slices[col]
        final2d = final_slices[col]

        ax = axes[0, col]
        im_ic = ax.imshow(
            ic2d,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax_ic,
            vmax=vmax_ic,
            extent=[0.0, boxsize, 0.0, boxsize],
            aspect="equal",
        )
        ax.set_title(f"Sim {sid}")
        ax.set_xlabel(xlabel)
        if col == 0:
            ax.set_ylabel(f"IC ({ylabel})")

        ax = axes[1, col]
        im_final = ax.imshow(
            final2d,
            origin="lower",
            cmap="inferno",
            vmin=0.0,
            vmax=vmax_final,
            extent=[0.0, boxsize, 0.0, boxsize],
            aspect="equal",
        )
        ax.set_xlabel(xlabel)
        if col == 0:
            ax.set_ylabel(f"Final ({ylabel})")

        cm = np.asarray(d["cm"], dtype=np.float64)
        mass = np.asarray(d["mass"], dtype=np.float64)
        mask = periodic_slab_mask(cm[:, axis], center=slice_center, width=slab_width, boxsize=boxsize)
        proj0 = cm[mask, plane_axes[0]]
        proj1 = cm[mask, plane_axes[1]]
        ax = axes[2, col]
        if proj0.size > 0:
            if bool(args.halo_size_by_mass):
                m = np.log10(np.maximum(mass[mask], 1.0))
                mmin, mmax = float(np.min(m)), float(np.max(m))
                denom = max(mmax - mmin, 1e-6)
                sizes = 14.0 + 38.0 * (m - mmin) / denom
            else:
                sizes = np.full(proj0.shape, 16.0, dtype=np.float64)
            ax.scatter(proj1, proj0, s=sizes, c="deepskyblue", alpha=0.8, linewidths=0.0)
        else:
            ax.text(0.5, 0.5, "No halos in slab", transform=ax.transAxes, ha="center", va="center")
        ax.set_xlim(0.0, boxsize)
        ax.set_ylim(0.0, boxsize)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        if col == 0:
            ax.set_ylabel(f"Halos ({ylabel})")

    if im_ic is not None:
        fig.colorbar(im_ic, cax=cax_ic, label=r"$\delta_{\rm IC}$")
    if im_final is not None:
        fig.colorbar(im_final, cax=cax_final, label=r"$\delta_{\rm final}$")
    fig.suptitle(
        f"Random dataset samples (axis={args.slice_axis}, slice_index={slice_index}, slab_width={slab_width:.2f} Mpc/h)",
        fontsize=13,
    )
    slices_path = output_dir / "dataset_slices_3sims.png"
    fig.savefig(slices_path, dpi=180)
    plt.close(fig)

    # Ensemble power spectra over first num_sims (sorted by sim ID).
    n_pk = min(int(args.num_sims), len(sim_dirs))
    if n_pk < 1:
        raise RuntimeError("No valid simulation folders available for P(k) ensemble.")
    pk_sims = sim_dirs[:n_pk]

    k_ref = None
    pk_ic_curves: list[np.ndarray] = []
    pk_final_curves: list[np.ndarray] = []
    max_k_seen = 0.0
    for sid, sim_path in pk_sims:
        ic = np.asarray(np.load(sim_path / "delta_linear_z0.npy"), dtype=np.float32)
        final = np.asarray(np.load(sim_path / "emu_delta_z0.npy"), dtype=np.float32)
        k_ic, pk_ic = compute_pk(ic, boxsize=boxsize, mas=str(args.pk_mas))
        k_final, pk_final = compute_pk(final, boxsize=boxsize, mas=str(args.pk_mas))
        max_k_seen = max(max_k_seen, float(np.max(k_ic)), float(np.max(k_final)))
        if k_ref is None:
            k_ref = k_ic
            pk_ic_curves.append(pk_ic)
            pk_final_curves.append(pk_final)
            continue
        if k_ic.shape == k_ref.shape and np.allclose(k_ic, k_ref, rtol=1e-8, atol=1e-12):
            pk_ic_curves.append(pk_ic)
            pk_final_curves.append(pk_final)
        else:
            pk_ic_curves.append(np.interp(k_ref, k_ic, pk_ic))
            pk_final_curves.append(np.interp(k_ref, k_final, pk_final))

    if k_ref is None:
        raise RuntimeError("No P(k) curves were computed.")

    stack_ic = np.stack(pk_ic_curves, axis=0)
    stack_final = np.stack(pk_final_curves, axis=0)
    mean_ic, low1_ic, high1_ic, low2_ic, high2_ic = mean_sigma_band(stack_ic)
    mean_final, low1_final, high1_final, low2_final, high2_final = mean_sigma_band(stack_final)

    fig, ax = plt.subplots(figsize=(8.6, 6.6), constrained_layout=True)
    ic_color = "forestgreen"
    final_color = "mediumpurple"
    tiny = np.finfo(np.float64).tiny

    ax.fill_between(
        k_ref,
        np.maximum(low2_ic, tiny),
        np.maximum(high2_ic, tiny),
        color=ic_color,
        alpha=0.14,
        label=r"IC $\pm2\sigma$",
    )
    ax.fill_between(
        k_ref,
        np.maximum(low1_ic, tiny),
        np.maximum(high1_ic, tiny),
        color=ic_color,
        alpha=0.28,
        label=r"IC $\pm1\sigma$",
    )
    ax.plot(k_ref, np.maximum(mean_ic, tiny), color=ic_color, lw=2.2, label="IC mean")

    ax.fill_between(
        k_ref,
        np.maximum(low2_final, tiny),
        np.maximum(high2_final, tiny),
        color=final_color,
        alpha=0.14,
        label=r"Final $\pm2\sigma$",
    )
    ax.fill_between(
        k_ref,
        np.maximum(low1_final, tiny),
        np.maximum(high1_final, tiny),
        color=final_color,
        alpha=0.28,
        label=r"Final $\pm1\sigma$",
    )
    ax.plot(k_ref, np.maximum(mean_final, tiny), color=final_color, lw=2.2, label="Final mean")

    k_theory = np.logspace(np.log10(max(1.0e-4, float(np.min(k_ref)))), np.log10(max(10.0, 1.2 * max_k_seen)), 400)
    pk_lin = get_pk_class(QUIJOTE_FIDUCIAL_CLASS, z=0.0, k=k_theory, non_lin=False)
    pk_nl = get_pk_class(QUIJOTE_FIDUCIAL_CLASS, z=0.0, k=k_theory, non_lin=True)
    ax.plot(k_theory, pk_lin, color="black", lw=1.4, ls="--", alpha=0.9, label="CLASS linear")
    ax.plot(k_theory, pk_nl, color="dimgray", lw=1.4, ls="-.", alpha=0.95, label="CLASS Halofit")

    k_nyq = np.pi * float(res) / float(boxsize)
    ax.axvline(k_nyq, color="red", lw=1.0, ls="--", alpha=0.7, label=r"$k_{\rm Nyq}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    k_pylians_max = float(max_k_seen)
    if k_pylians_max > 0.0:
        ax.set_xlim(right=1.1 * k_pylians_max)
        pk_class_lin_at_kmax = float(np.interp(k_pylians_max, k_theory, pk_lin))
        pk_floor = 0.9 * pk_class_lin_at_kmax
        if pk_floor > 0.0:
            ax.set_ylim(bottom=pk_floor)
    pk_class_lin_at_kmin = float(pk_lin[0]) if pk_lin.size > 0 else None
    if pk_class_lin_at_kmin is not None and np.isfinite(pk_class_lin_at_kmin) and pk_class_lin_at_kmin > 0.0:
        ymin_now, _ = ax.get_ylim()
        if ymin_now < pk_class_lin_at_kmin:
            ax.set_ylim(top=2 * pk_class_lin_at_kmin)
    ax.set_xlabel(r"$k\ [h/{\rm Mpc}]$")
    ax.set_ylabel(r"$P(k)$")
    ax.grid(which="both", alpha=0.15)
    ax.legend(framealpha=0.9, fontsize=9, ncol=2)
    ax.set_title(f"Ensemble P(k): first {n_pk} sims (MAS={args.pk_mas})")

    pk_path = output_dir / "dataset_pk_ensemble_vs_class.png"
    fig.savefig(pk_path, dpi=180)
    plt.close(fig)

    selected_ids = [int(d["sid"]) for d in data]
    meta = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "selected_sim_ids": selected_ids,
        "n_sims_plot": int(args.n_sims_plot),
        "num_sims_pk": int(n_pk),
        "random_seed": int(args.random_seed),
        "boxsize": float(boxsize),
        "res": int(res),
        "slice_axis": str(args.slice_axis),
        "slice_index": int(slice_index),
        "slice_center_mpc_h": float(slice_center),
        "halo_slab_width_mpc_h": float(slab_width),
        "halo_size_by_mass": bool(args.halo_size_by_mass),
        "pk_mas": str(args.pk_mas),
        "plots": {
            "slices": str(slices_path),
            "power_spectrum": str(pk_path),
        },
    }
    meta_path = output_dir / "dataset_plot_selection.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Selected sim IDs: {selected_ids}")
    print(f"Saved: {slices_path}")
    print(f"Saved: {pk_path}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
