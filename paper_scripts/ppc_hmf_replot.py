#!/usr/bin/env python3
"""
Fast HMF replot from saved aggregate arrays.

This script re-renders the HMF figure directly from a previously saved
`*_hmf_arrays.npz` file, without rerunning FoF halo finding.

It can also compute a Tinker theoretical HMF line from a provided
CLASS linear P(k) table and overlay it on the plot.

Example:
  python paper_scripts/ppc_hmf_replot.py \
    --arrays-path /gpfs/home5/osavchenko/gaussian_npe/paper_plots_scripts/260303_224627_net_IsotropicD/posterior_resimulation/posterior_predictive_hmf_arrays.npz \
    --class-pk-table-path /path/to/class_linear_pk_z0_table.txt \
    --omega-m 0.3175
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import mass_function_library as MFL
except Exception as exc:  # pragma: no cover - optional dependency
    MFL = None
    _MFL_IMPORT_ERROR = exc
else:
    _MFL_IMPORT_ERROR = None

try:
    from classy import Class
except Exception as exc:  # pragma: no cover - optional dependency
    Class = None
    _CLASS_IMPORT_ERROR = exc
else:
    _CLASS_IMPORT_ERROR = None

QUIJOTE_FIDUCIAL_CLASS = {
    "Omega_cdm": 0.3175 - 0.0490,
    "Omega_b": 0.0490,
    "h": 0.6711,
    "n_s": 0.9624,
    "sigma8": 0.8340,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for fast HMF replotting."""
    parser = argparse.ArgumentParser(
        description="Replot HMF from saved aggregate arrays (no FoF recomputation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arrays-path",
        type=Path,
        default=Path(
            "/gpfs/home5/osavchenko/gaussian_npe/paper_plots_scripts/260303_224627_net_IsotropicD/"
            "posterior_resimulation/posterior_predictive_hmf_arrays.npz"
        ),
        help="Path to *_hmf_arrays.npz produced by paper_scripts/ppc_hmf.py.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output figure path. Default: sibling PNG of arrays file, replacing "
            "'_arrays.npz' with '.png'."
        ),
    )
    parser.add_argument(
        "--save-arrays-path",
        type=Path,
        default=None,
        help="Optional output arrays path. If set, saves arrays with the (possibly computed) hmf_tinker.",
    )

    parser.add_argument(
        "--title",
        type=str,
        default="Friend-of-Friends mass",
        help="Title for the top HMF panel.",
    )
    parser.add_argument(
        "--ratio-ymin",
        type=float,
        default=0.6,
        help="Lower y-limit of the ratio panel.",
    )
    parser.add_argument(
        "--ratio-ymax",
        type=float,
        default=1.4,
        help="Upper y-limit of the ratio panel.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output figure DPI.",
    )

    parser.add_argument(
        "--show-truth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay truth HMF if present in arrays file.",
    )
    parser.add_argument(
        "--show-tinker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay Tinker HMF (from arrays or computed from CLASS table).",
    )

    parser.add_argument(
        "--class-pk-table-path",
        type=Path,
        default=None,
        help=(
            "Optional CLASS linear P(k) table path. "
            "If provided, computes Tinker HMF and overrides any existing hmf_tinker in arrays."
        ),
    )
    parser.add_argument(
        "--build-class-pk-fiducial",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Build a CLASS linear P(k) table on the fly using fiducial Quijote cosmology "
            "and use it for the Tinker overlay."
        ),
    )
    parser.add_argument("--class-z", type=float, default=0.0, help="Redshift for generated fiducial CLASS P(k).")
    parser.add_argument("--class-kmin", type=float, default=1.0e-5, help="kmin [h/Mpc] for generated fiducial CLASS P(k).")
    parser.add_argument("--class-kmax", type=float, default=10.0, help="kmax [h/Mpc] for generated fiducial CLASS P(k).")
    parser.add_argument("--class-n-modes", type=int, default=4000, help="Number of k-samples for generated fiducial CLASS P(k).")
    parser.add_argument(
        "--generated-class-pk-save-path",
        type=Path,
        default=None,
        help="Optional output path to save generated fiducial CLASS P(k) table.",
    )

    parser.add_argument("--omega-m", type=float, default=0.3175, help="Omega_m for Tinker MF_theory.")
    parser.add_argument(
        "--tinker-author",
        type=str,
        choices=("ST", "Tinker", "Tinker10", "Crocce", "Jenkins", "Warren", "Watson", "Watson_FOF", "Angulo"),
        default="Tinker",
        help="Pylians mass-function fitting function.",
    )
    parser.add_argument(
        "--tinker-integration-bins",
        type=int,
        default=1000,
        help="Number of log-k bins used by Pylians MF_theory integration.",
    )

    args = parser.parse_args()

    if args.ratio_ymax <= args.ratio_ymin:
        parser.error("--ratio-ymax must be greater than --ratio-ymin")
    if args.dpi < 50:
        parser.error("--dpi must be >= 50")
    if args.tinker_integration_bins < 100:
        parser.error("--tinker-integration-bins must be >= 100")
    if args.class_pk_table_path is not None and bool(args.build_class_pk_fiducial):
        parser.error("Use either --class-pk-table-path or --build-class-pk-fiducial, not both")
    if args.class_n_modes < 2:
        parser.error("--class-n-modes must be >= 2")
    if args.class_kmin <= 0.0:
        parser.error("--class-kmin must be > 0")
    if args.class_kmax <= args.class_kmin:
        parser.error("--class-kmax must be greater than --class-kmin")

    return args


def _default_output_path(arrays_path: Path) -> Path:
    """Infer default output PNG path from the arrays file path."""
    name = arrays_path.name
    if name.endswith("_arrays.npz"):
        return arrays_path.with_name(name.replace("_arrays.npz", ".png"))
    return arrays_path.with_suffix(".png")


def _optional_curve(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    """Return optional 1D curve or None when key is missing/empty."""
    if key not in data.files:
        return None
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.size == 0:
        return None
    return arr


def compute_tinker_curve_from_class_pk(
    *,
    mass_centers: np.ndarray,
    class_pk_table_path: Path,
    omega_m: float,
    author: str,
    integration_bins: int,
) -> np.ndarray:
    """Compute dn/dlog10M using Pylians MF_theory from a CLASS linear P(k) table."""
    if MFL is None:
        raise ImportError("mass_function_library is unavailable.") from _MFL_IMPORT_ERROR

    table = np.loadtxt(class_pk_table_path)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f"Invalid CLASS P(k) table format in {class_pk_table_path}")

    k = np.asarray(table[:, 0], dtype=np.float64)
    pk = np.asarray(table[:, 1], dtype=np.float64)
    masses = np.asarray(mass_centers, dtype=np.float64)

    dndm = np.asarray(
        MFL.MF_theory(
            k_in=k,
            Pk_in=pk,
            OmegaM=float(omega_m),
            Masses=masses,
            author=str(author),
            bins=int(integration_bins),
            z=0.0,
            delta=200.0,
        ),
        dtype=np.float64,
    )

    return dndm * masses * np.log(10.0)


def _build_fiducial_class_pk_table(
    *,
    z: float,
    kmin: float,
    kmax: float,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate fiducial Quijote linear CLASS P(k) sampled on a geometric k-grid."""
    if Class is None:
        raise ImportError("classy is unavailable.") from _CLASS_IMPORT_ERROR

    k_eval = np.geomspace(float(kmin), float(kmax), int(n_modes)).astype(np.float64)
    params = dict(QUIJOTE_FIDUCIAL_CLASS)
    h = float(params["h"])
    params.update(
        {
            "output": "mPk",
            "P_k_max_h/Mpc": float(np.max(k_eval)),
            "z_max_pk": float(z),
        }
    )

    cosmo = Class()
    try:
        cosmo.set(params)
        cosmo.compute()
        pk = h**3 * np.asarray([cosmo.pk_lin(h * float(ki), float(z)) for ki in k_eval], dtype=np.float64)
    finally:
        try:
            cosmo.struct_cleanup()
            cosmo.empty()
        except Exception:
            pass

    return k_eval, pk


def compute_tinker_curve_from_fiducial_class(
    *,
    mass_centers: np.ndarray,
    omega_m: float,
    author: str,
    integration_bins: int,
    class_z: float,
    class_kmin: float,
    class_kmax: float,
    class_n_modes: int,
    generated_class_pk_save_path: Path | None,
) -> tuple[np.ndarray, Path | None]:
    """Compute Tinker dn/dlog10M from an on-the-fly fiducial Quijote CLASS P(k) table."""
    if MFL is None:
        raise ImportError("mass_function_library is unavailable.") from _MFL_IMPORT_ERROR

    k, pk = _build_fiducial_class_pk_table(
        z=float(class_z),
        kmin=float(class_kmin),
        kmax=float(class_kmax),
        n_modes=int(class_n_modes),
    )

    saved_path = None
    if generated_class_pk_save_path is not None:
        saved_path = Path(generated_class_pk_save_path).resolve()
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            saved_path,
            np.column_stack([k, pk]),
            header="k_h_per_Mpc Pk_Mpc_over_h_cubed",
        )

    masses = np.asarray(mass_centers, dtype=np.float64)
    dndm = np.asarray(
        MFL.MF_theory(
            k_in=np.asarray(k, dtype=np.float64),
            Pk_in=np.asarray(pk, dtype=np.float64),
            OmegaM=float(omega_m),
            Masses=masses,
            author=str(author),
            bins=int(integration_bins),
            z=float(class_z),
            delta=200.0,
        ),
        dtype=np.float64,
    )
    return dndm * masses * np.log(10.0), saved_path


def plot_hmf_summary(
    *,
    m_centers: np.ndarray,
    hmf_samples: np.ndarray,
    tinker_curve: np.ndarray | None,
    hmf_truth: np.ndarray | None,
    mass_resolution_limit: float | None,
    out_path: Path,
    title: str,
    ratio_ymin: float,
    ratio_ymax: float,
    dpi: int,
) -> None:
    """Plot HMF with posterior bands and optional truth/Tinker overlays."""
    m_centers = np.asarray(m_centers, dtype=np.float64)
    hmf_samples = np.asarray(hmf_samples, dtype=np.float64)
    hmf_samples = np.where(hmf_samples > 0.0, hmf_samples, np.nan)

    mu = np.nanmean(hmf_samples, axis=0)
    sigma = np.nanstd(hmf_samples, axis=0)

    sigma1_lo = np.where((mu - sigma) > 0.0, mu - sigma, np.nan)
    sigma1_hi = np.where((mu + sigma) > 0.0, mu + sigma, np.nan)
    sigma2_lo = np.where((mu - 2.0 * sigma) > 0.0, mu - 2.0 * sigma, np.nan)
    sigma2_hi = np.where((mu + 2.0 * sigma) > 0.0, mu + 2.0 * sigma, np.nan)

    tinker = None if tinker_curve is None else np.asarray(tinker_curve, dtype=np.float64)
    truth = None if hmf_truth is None else np.asarray(hmf_truth, dtype=np.float64)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.4, 7.4),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )

    ax = axes[0]
    if mass_resolution_limit is not None and np.isfinite(float(mass_resolution_limit)) and float(mass_resolution_limit) > 0.0:
        ax.axvspan(
            float(np.min(m_centers)),
            float(mass_resolution_limit),
            color="#efe6e6",
            alpha=0.85,
            label="Mass resolution limit",
            zorder=0,
        )

    ax.fill_between(m_centers, sigma2_lo, sigma2_hi, color="0.7", alpha=0.45, lw=0.0, label=r"$\pm 2\sigma$", zorder=1)
    ax.fill_between(m_centers, sigma1_lo, sigma1_hi, color="0.55", alpha=0.55, lw=0.0, label=r"$\pm 1\sigma$", zorder=2)
    ax.plot(m_centers, mu, color="#5aa2c7", lw=2.0, label="Posterior Resimulations", zorder=3)
    if tinker is not None:
        ax.plot(m_centers, tinker, color="0.5", lw=2.0, ls=":", label="Tinker", zorder=4)
    if truth is not None:
        ax.plot(
            m_centers,
            truth,
            color="k",
            ls="--",
            lw=1.8,
            marker="x",
            ms=6,
            mew=1.6,
            label="Ground Truth",
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$dn/d\log_{10}M\ [(h/{\rm Mpc})^3]$")
    ax.set_title(str(title), fontsize=16)
    ax.grid(alpha=0.25, which="both", ls=":")
    ax.legend(framealpha=0.95, fontsize=9, loc="best")

    ax = axes[1]
    if truth is not None:
        denom = np.where(truth > 0.0, truth, np.nan)
        ratio_ref = np.ones_like(m_centers, dtype=np.float64)
        ax.plot(m_centers, np.ones_like(m_centers), color="k", ls="--", lw=1.4, marker="x", ms=5, mew=1.4)
    elif tinker is not None:
        denom = np.where(tinker > 0.0, tinker, np.nan)
        ratio_ref = np.ones_like(m_centers, dtype=np.float64)
        ax.axhline(1.0, color="k", ls="--", lw=1.4)
    else:
        denom = np.where(mu > 0.0, mu, np.nan)
        ratio_ref = np.ones_like(m_centers, dtype=np.float64)
        ax.axhline(1.0, color="k", ls="--", lw=1.4)

    ratio_mu = mu / denom
    ratio_sigma1_lo = sigma1_lo / denom
    ratio_sigma1_hi = sigma1_hi / denom
    ratio_sigma2_lo = sigma2_lo / denom
    ratio_sigma2_hi = sigma2_hi / denom

    if mass_resolution_limit is not None and np.isfinite(float(mass_resolution_limit)) and float(mass_resolution_limit) > 0.0:
        ax.axvspan(float(np.min(m_centers)), float(mass_resolution_limit), color="#efe6e6", alpha=0.85, zorder=0)
    ax.fill_between(m_centers, ratio_sigma2_lo, ratio_sigma2_hi, color="#7eb6d6", alpha=0.45, lw=0.0, zorder=1)
    ax.fill_between(m_centers, ratio_sigma1_lo, ratio_sigma1_hi, color="#5aa2c7", alpha=0.55, lw=0.0, zorder=2)
    ax.plot(m_centers, ratio_mu, color="#2d7ea8", lw=1.8, zorder=3)
    ax.plot(m_centers, ratio_ref, color="0.5", lw=1.6, ls=":", zorder=4)
    ax.set_xscale("log")
    ax.set_ylim(float(ratio_ymin), float(ratio_ymax))
    ax.set_ylabel("Ratio")
    ax.set_xlabel(r"Mass [$M_\odot h^{-1}$]")
    ax.grid(alpha=0.25, which="both", ls=":")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def main() -> None:
    """Load saved HMF arrays and render a new HMF figure."""
    args = parse_args()

    arrays_path = Path(args.arrays_path).resolve()
    if not arrays_path.is_file():
        raise FileNotFoundError(f"Arrays file not found: {arrays_path}")

    output_path = _default_output_path(arrays_path) if args.output_path is None else Path(args.output_path).resolve()

    data = np.load(arrays_path)

    required = ("hmf_mass_centers", "hmf_samples", "mass_resolution_limit")
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"Missing required keys in {arrays_path}: {missing}")

    m_centers = np.asarray(data["hmf_mass_centers"], dtype=np.float64)
    hmf_samples = np.asarray(data["hmf_samples"], dtype=np.float64)
    mass_resolution_limit = float(np.asarray(data["mass_resolution_limit"], dtype=np.float64))

    hmf_truth = _optional_curve(data, "hmf_truth") if bool(args.show_truth) else None
    hmf_tinker = _optional_curve(data, "hmf_tinker") if bool(args.show_tinker) else None

    class_pk_table_path = None if args.class_pk_table_path is None else Path(args.class_pk_table_path).resolve()
    generated_class_pk_saved_path = None
    used_generated_class_pk = False

    if class_pk_table_path is not None:
        if not class_pk_table_path.is_file():
            raise FileNotFoundError(f"CLASS P(k) table not found: {class_pk_table_path}")
        hmf_tinker = compute_tinker_curve_from_class_pk(
            mass_centers=np.asarray(m_centers, dtype=np.float64),
            class_pk_table_path=class_pk_table_path,
            omega_m=float(args.omega_m),
            author=str(args.tinker_author),
            integration_bins=int(args.tinker_integration_bins),
        )
    elif bool(args.build_class_pk_fiducial):
        hmf_tinker, generated_class_pk_saved_path = compute_tinker_curve_from_fiducial_class(
            mass_centers=np.asarray(m_centers, dtype=np.float64),
            omega_m=float(args.omega_m),
            author=str(args.tinker_author),
            integration_bins=int(args.tinker_integration_bins),
            class_z=float(args.class_z),
            class_kmin=float(args.class_kmin),
            class_kmax=float(args.class_kmax),
            class_n_modes=int(args.class_n_modes),
            generated_class_pk_save_path=(
                None
                if args.generated_class_pk_save_path is None
                else Path(args.generated_class_pk_save_path)
            ),
        )
        used_generated_class_pk = True

    if hmf_truth is not None and hmf_truth.shape != m_centers.shape:
        raise ValueError(
            f"hmf_truth shape mismatch: expected {m_centers.shape}, got {hmf_truth.shape}"
        )
    if hmf_tinker is not None and hmf_tinker.shape != m_centers.shape:
        raise ValueError(
            f"hmf_tinker shape mismatch: expected {m_centers.shape}, got {hmf_tinker.shape}"
        )

    plot_hmf_summary(
        m_centers=m_centers,
        hmf_samples=hmf_samples,
        tinker_curve=hmf_tinker,
        hmf_truth=hmf_truth,
        mass_resolution_limit=mass_resolution_limit,
        out_path=output_path,
        title=str(args.title),
        ratio_ymin=float(args.ratio_ymin),
        ratio_ymax=float(args.ratio_ymax),
        dpi=int(args.dpi),
    )

    if args.save_arrays_path is not None:
        save_arrays_path = Path(args.save_arrays_path).resolve()
        payload = {k: data[k] for k in data.files}
        payload["hmf_tinker"] = np.asarray([] if hmf_tinker is None else hmf_tinker, dtype=np.float64)
        save_arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_arrays_path, **payload)
        print(f"Saved arrays: {save_arrays_path}")

    print("HMF replot complete.")
    print(f"Arrays: {arrays_path}")
    if class_pk_table_path is not None:
        print(f"Tinker source P(k): {class_pk_table_path}")
        print(f"Tinker author: {args.tinker_author}")
        print(f"Omega_m: {float(args.omega_m):.6f}")
    elif used_generated_class_pk:
        print("Tinker source P(k): generated on the fly (fiducial Quijote cosmology)")
        print(f"Tinker author: {args.tinker_author}")
        print(f"Omega_m: {float(args.omega_m):.6f}")
        print(f"CLASS z: {float(args.class_z):.3f}")
        print(f"CLASS k-range: [{float(args.class_kmin):.3e}, {float(args.class_kmax):.3e}] h/Mpc")
        print(f"CLASS n-modes: {int(args.class_n_modes)}")
        if generated_class_pk_saved_path is not None:
            print(f"Saved generated CLASS P(k): {generated_class_pk_saved_path}")
    print(f"Plot:   {output_path}")


if __name__ == "__main__":
    main()
