#!/usr/bin/env python3
"""
CPU HMF stage for posterior predictive emulator outputs using nbodykit FoF.

This script expects per-sample displacement outputs already present in
`--samples-dir/<sample_folder_template>/emu_dis_<ztag>.npy`. It:
1. Runs FoF halo finding for each sample.
2. Computes dn/dlog10M curves on common mass bins.
3. Aggregates HMF over samples and plots mean ±1sigma/±2sigma.
4. Optionally overlays a truth HMF from `--truth-displacement-path`.

Example:
  python paper_scripts/ppc_hmf.py \
    --samples-dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD \
    --num-samples 100 \
    --n-part 512 \
    --boxsize 1000
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

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):  # type: ignore[override]
        """Fallback iterator when tqdm is unavailable."""
        return iterable

from ppc_common import extract_sample_id, mean_sigma, sample_output_dir, save_summary_npz, select_sample_files, z_file_tag

try:
    from halos import (
        compute_pylians_hmf_theory,
        find_class_pk_table_path,
        gather_halos_to_root,
        load_local_positions_from_displacement,
        particle_mass_msun_h,
        run_fof,
    )
except ImportError as exc:  # pragma: no cover - runtime dependency
    compute_pylians_hmf_theory = None
    find_class_pk_table_path = None
    gather_halos_to_root = None
    load_local_positions_from_displacement = None
    particle_mass_msun_h = None
    run_fof = None
    _HALOS_IMPORT_ERROR = exc
else:
    _HALOS_IMPORT_ERROR = None

try:
    from mpi4py import MPI
    from nbodykit.algorithms.fof import FOF, fof_catalog
    from nbodykit.source.catalog import ArrayCatalog
except ImportError as exc:  # pragma: no cover - runtime dependency
    MPI = None
    FOF = None
    fof_catalog = None
    ArrayCatalog = None
    _NBodyKit_IMPORT_ERROR = exc
else:
    _NBodyKit_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for PPC FoF/HMF aggregation."""
    parser = argparse.ArgumentParser(
        description="Compute posterior predictive HMF from precomputed emulator displacements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("/gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD"),
        help="Directory containing IC samples and per-sample emulator folders.",
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

    parser.add_argument("--z-target", type=float, default=0.0, help="Target redshift used for emu_dis_<ztag>.npy lookup.")
    parser.add_argument("--n-part", type=int, default=512, help="Particle lattice resolution.")
    parser.add_argument("--boxsize", type=float, default=1000.0, help="Simulation box size in Mpc/h.")
    parser.add_argument("--omega-m", type=float, default=0.3175, help="Omega_m used for particle mass conversion.")

    parser.add_argument("--fof-linking-length", type=float, default=0.2, help="FoF linking length parameter.")
    parser.add_argument(
        "--fof-absolute-linking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat linking length as absolute Mpc/h instead of b relative to mean spacing.",
    )
    parser.add_argument("--fof-nmin", type=int, default=20, help="FoF minimum particle count per halo.")

    parser.add_argument("--hmf-nbins", type=int, default=25, help="Number of log10 mass bins.")
    parser.add_argument("--hmf-mmin", type=float, default=None, help="Lower HMF bound [Msun/h].")
    parser.add_argument("--hmf-mmax", type=float, default=None, help="Upper HMF bound [Msun/h].")
    parser.add_argument(
        "--truth-displacement-path",
        type=Path,
        default=None,
        help="Optional truth displacement file to overlay truth HMF.",
    )
    parser.add_argument(
        "--truth-fof-tab-path",
        type=Path,
        default=None,
        help=(
            "Optional Quijote FoF group_tab file path (e.g. group_tab_004.0). "
            "When provided, truth HMF is read from GroupLen instead of rerunning FoF on a truth displacement."
        ),
    )

    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=None,
        help=(
            "Aggregate output directory. "
            "Default: /home/osavchenko/gaussian_npe/paper_plots_scripts/<RUN_NAME>/posterior_resimulation, "
            "where RUN_NAME is basename(--samples-dir)."
        ),
    )
    parser.add_argument("--aggregate-prefix", type=str, default="posterior_predictive", help="Output filename prefix.")
    parser.add_argument(
        "--save-per-sample-hmf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-sample FoF/HMF files inside each sample output folder.",
    )

    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.sample_start < 0:
        parser.error("--sample-start must be >= 0")
    if args.sample_stride < 1:
        parser.error("--sample-stride must be >= 1")
    if args.n_part < 1:
        parser.error("--n-part must be >= 1")
    if args.hmf_nbins < 3:
        parser.error("--hmf-nbins must be >= 3")
    if args.truth_displacement_path is not None and args.truth_fof_tab_path is not None:
        parser.error("Use only one truth source: --truth-displacement-path or --truth-fof-tab-path")
    return args


def run_distributed_fof_lengths(
    displacement_path: Path,
    *,
    boxsize: float,
    n_part: int,
    linking_length: float,
    nmin: int,
    absolute_linking: bool,
) -> np.ndarray | None:
    """Run MPI-distributed FoF for one displacement file and gather halo lengths to rank 0."""
    if _NBodyKit_IMPORT_ERROR is not None:
        raise ImportError("nbodykit + mpi4py are required for HMF/FoF computation") from _NBodyKit_IMPORT_ERROR

    comm = MPI.COMM_WORLD
    rank = int(comm.rank)
    size = int(comm.size)
    local_positions = load_local_positions_from_displacement(
        displacement_file=Path(displacement_path),
        boxsize=float(boxsize),
        n_part=int(n_part),
        rank=rank,
        size=size,
    )
    cm_local, length_local = run_fof(
        local_positions=np.asarray(local_positions, dtype=np.float32),
        boxsize=float(boxsize),
        n_part=int(n_part),
        linking_length=float(linking_length),
        nmin=int(nmin),
        absolute_linking=bool(absolute_linking),
        comm=comm,
        fof_cls=FOF,
        fof_catalog_fn=fof_catalog,
        array_catalog_cls=ArrayCatalog,
    )
    _, length = gather_halos_to_root(
        cm_pos_local=np.asarray(cm_local, dtype=np.float32),
        length_local=np.asarray(length_local, dtype=np.int32),
        comm=comm,
    )
    if comm.rank != 0:
        return None
    return np.asarray(length, dtype=np.int32)


def compute_hmf_curve(masses: np.ndarray, log_edges: np.ndarray, boxsize: float) -> np.ndarray:
    """Compute dn/dlog10M for fixed logarithmic mass bins."""
    masses = np.asarray(masses, dtype=np.float64)
    masses = masses[np.isfinite(masses) & (masses > 0.0)]
    counts, _ = np.histogram(np.log10(masses), bins=log_edges)
    dlogm = float(log_edges[1] - log_edges[0])
    volume = float(boxsize) ** 3
    return counts.astype(np.float64) / (volume * dlogm)


def _group_tab_prefix(path: Path) -> Path:
    """Return `.../group_tab_XXX` prefix from a path like `.../group_tab_XXX.0`."""
    name = path.name
    if "." not in name:
        raise ValueError(f"Expected a shard file like group_tab_XXX.0, got: {path}")
    stem, shard = name.rsplit(".", 1)
    if not shard.isdigit():
        raise ValueError(f"Expected numeric shard suffix in {path}")
    return path.with_name(stem)


def _read_group_tab_lengths_single(path: Path) -> tuple[np.ndarray, int]:
    """Read GroupLen from one Gadget FoF group_tab shard and return (lengths, nfiles)."""
    with path.open("rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=6)
        if header.size != 6:
            raise ValueError(f"Invalid group_tab header in {path}")
        ngroups = int(header[0])
        nfiles = int(header[5])
        if nfiles < 1 and int(header[4]) > 0:
            nfiles = int(header[4])
        if ngroups < 0:
            raise ValueError(f"Invalid Ngroups={ngroups} in {path}")
        lengths = np.fromfile(f, dtype=np.int32, count=ngroups)
        if lengths.size != ngroups:
            raise ValueError(
                f"Failed to read GroupLen array in {path}: expected {ngroups}, got {lengths.size}"
            )
    return np.asarray(lengths, dtype=np.int32), int(max(1, nfiles))


def load_group_tab_lengths(group_tab_path: Path) -> np.ndarray:
    """Load concatenated GroupLen from all shards of a Quijote/Gadget FoF group_tab catalog."""
    p = Path(group_tab_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"FoF group_tab file not found: {p}")

    prefix = _group_tab_prefix(p)

    # Probe either the provided shard or shard 0 to discover shard count.
    probe = p
    shard0 = Path(f"{prefix}.0")
    if shard0.is_file():
        probe = shard0
    _, nfiles = _read_group_tab_lengths_single(probe)

    lengths_all: list[np.ndarray] = []
    for i in range(int(nfiles)):
        shard = Path(f"{prefix}.{i}")
        if not shard.is_file():
            raise FileNotFoundError(
                f"Missing FoF shard {shard}. Expected {nfiles} shard(s) from header."
            )
        lengths_i, nfiles_i = _read_group_tab_lengths_single(shard)
        if int(nfiles_i) != int(nfiles):
            raise ValueError(
                f"Inconsistent shard count in {shard}: header says {nfiles_i}, expected {nfiles}"
            )
        lengths_all.append(np.asarray(lengths_i, dtype=np.int32))

    if len(lengths_all) == 0:
        return np.empty((0,), dtype=np.int32)
    return np.concatenate(lengths_all, axis=0).astype(np.int32, copy=False)


def compute_tinker_curve_from_samples(
    *,
    sample_output_dirs: list[str],
    mass_centers: np.ndarray,
    omega_m: float,
) -> tuple[np.ndarray | None, str | None, str]:
    """Compute a Tinker HMF curve using CLASS P(k) table discovered in sample metadata."""
    for out_dir_str in sample_output_dirs:
        out_dir = Path(out_dir_str)
        metadata_path = out_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception:
            continue
        class_pk_path = find_class_pk_table_path(metadata=metadata, emu_dir=out_dir)
        if class_pk_path is None:
            continue
        try:
            tinker = compute_pylians_hmf_theory(
                mass_centers=np.asarray(mass_centers, dtype=np.float64),
                omega_m=float(omega_m),
                class_pk_table_path=Path(class_pk_path),
                author="Tinker",
                integration_bins=1000,
            )
            return np.asarray(tinker, dtype=np.float64), str(class_pk_path), "ok"
        except Exception as exc:
            return None, str(class_pk_path), f"failed: {repr(exc)}"
    return None, None, "missing_class_pk_table_or_metadata"


def plot_hmf_summary(
    *,
    m_centers: np.ndarray,
    hmf_samples: np.ndarray,
    tinker_curve: np.ndarray | None,
    hmf_truth: np.ndarray | None,
    mass_resolution_limit: float | None,
    out_path: Path,
) -> None:
    """Plot HMF in a paper-style two-panel layout with ±1σ/±2σ bands and Tinker line."""
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
    hmf_truth_arr = None if hmf_truth is None else np.asarray(hmf_truth, dtype=np.float64)

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
            np.min(m_centers),
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
    if hmf_truth_arr is not None:
        ax.plot(
            m_centers,
            hmf_truth_arr,
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
    ax.set_title("Friend-of-Friends mass", fontsize=16)
    ax.grid(alpha=0.25, which="both", ls=":")
    ax.legend(framealpha=0.95, fontsize=9, loc="best")

    ax = axes[1]
    if hmf_truth_arr is not None:
        denom = np.where(hmf_truth_arr > 0.0, hmf_truth_arr, np.nan)
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
        ax.axvspan(np.min(m_centers), float(mass_resolution_limit), color="#efe6e6", alpha=0.85, zorder=0)
    ax.fill_between(m_centers, ratio_sigma2_lo, ratio_sigma2_hi, color="#7eb6d6", alpha=0.45, lw=0.0, zorder=1)
    ax.fill_between(m_centers, ratio_sigma1_lo, ratio_sigma1_hi, color="#5aa2c7", alpha=0.55, lw=0.0, zorder=2)
    ax.plot(m_centers, ratio_mu, color="#2d7ea8", lw=1.8, zorder=3)
    ax.plot(m_centers, ratio_ref, color="0.5", lw=1.6, ls=":", zorder=4)
    ax.set_xscale("log")
    ax.set_ylim(0.6, 1.4)
    ax.set_ylabel("Ratio")
    ax.set_xlabel(r"Mass [$M_\odot h^{-1}$]")
    ax.grid(alpha=0.25, which="both", ls=":")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Compute FoF catalogs and aggregate HMF curves for selected samples."""
    args = parse_args()

    if _HALOS_IMPORT_ERROR is not None:
        raise ImportError("Failed to import FoF helpers from scripts/halos.py") from _HALOS_IMPORT_ERROR
    if _NBodyKit_IMPORT_ERROR is not None:
        raise ImportError("nbodykit + mpi4py are required for this script") from _NBodyKit_IMPORT_ERROR

    comm = MPI.COMM_WORLD
    rank = int(comm.rank)
    size = int(comm.size)

    samples_dir = args.samples_dir.resolve()
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    if args.aggregate_dir is None:
        run_name = samples_dir.name
        aggregate_dir = (
            Path("/home/osavchenko/gaussian_npe/paper_plots_scripts")
            / run_name
            / "posterior_resimulation"
        )
    else:
        aggregate_dir = args.aggregate_dir.resolve()
    if rank == 0:
        aggregate_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    files_sel = select_sample_files(
        samples_dir=samples_dir,
        pattern=str(args.sample_pattern),
        sample_start=int(args.sample_start),
        sample_stride=int(args.sample_stride),
        num_samples=int(args.num_samples),
    )
    if rank == 0:
        print(f"Selected {len(files_sel)} sample files for HMF computation.")
        print(f"MPI world size={size}")

    z_tag = z_file_tag(float(args.z_target))
    m_particle = particle_mass_msun_h(omega_m=float(args.omega_m), boxsize=float(args.boxsize), n_part=int(args.n_part))
    truth_displacement_path = (
        None if args.truth_displacement_path is None else Path(args.truth_displacement_path).resolve()
    )
    truth_fof_tab_path = (
        None if args.truth_fof_tab_path is None else Path(args.truth_fof_tab_path).resolve()
    )
    if truth_displacement_path is not None and not truth_displacement_path.is_file():
        raise FileNotFoundError(f"Truth displacement file not found: {truth_displacement_path}")
    if truth_fof_tab_path is not None and not truth_fof_tab_path.is_file():
        raise FileNotFoundError(f"Truth FoF group_tab file not found: {truth_fof_tab_path}")

    sample_ids: list[int] = []
    sample_output_dirs: list[str] = []
    hmf_samples: list[np.ndarray] = []
    hmf_log_edges: np.ndarray | None = None
    hmf_centers: np.ndarray | None = None
    hmf_truth: np.ndarray | None = None

    bins_initialized = False
    truth_done = (truth_displacement_path is None) and (truth_fof_tab_path is None)
    iterator = tqdm(files_sel, desc="PPC HMF", ncols=100) if rank == 0 else files_sel
    for idx, sample_path in enumerate(iterator):
        sample_id = extract_sample_id(sample_path, fallback_index=idx)
        out_dir = sample_output_dir(
            samples_dir=samples_dir,
            template=str(args.sample_folder_template),
            sample_id=int(sample_id),
            sample_path=sample_path,
        )
        emu_dis_path = out_dir / f"emu_dis_{z_tag}.npy"
        if not emu_dis_path.is_file():
            raise FileNotFoundError(
                f"Missing forward displacement output: {emu_dis_path}. "
                "Run paper_scripts/ppc_forward.py first or adjust --z-target/--sample-folder-template."
            )

        length = run_distributed_fof_lengths(
            displacement_path=Path(emu_dis_path),
            boxsize=float(args.boxsize),
            n_part=int(args.n_part),
            linking_length=float(args.fof_linking_length),
            nmin=int(args.fof_nmin),
            absolute_linking=bool(args.fof_absolute_linking),
        )

        hmf_log_edges_root = None
        hmf_centers_root = None
        if rank == 0:
            if length is None:
                raise RuntimeError("Internal error: rank 0 did not receive FoF halo lengths.")
            masses = np.asarray(length, dtype=np.float64) * float(m_particle)
            masses = masses[np.isfinite(masses) & (masses > 0.0)]
            if masses.size == 0:
                raise ValueError(f"No halos found for sample {sample_path.name}.")

            if not bins_initialized:
                mmin = float(args.hmf_mmin) if args.hmf_mmin is not None else float(np.min(masses))
                mmax = float(args.hmf_mmax) if args.hmf_mmax is not None else float(np.max(masses))
                if not (mmax > mmin > 0.0):
                    raise ValueError(f"Invalid HMF bounds from first sample: mmin={mmin}, mmax={mmax}.")
                hmf_log_edges_root = np.linspace(np.log10(mmin), np.log10(mmax), int(args.hmf_nbins) + 1)
                hmf_centers_root = 10.0 ** (0.5 * (hmf_log_edges_root[1:] + hmf_log_edges_root[:-1]))

        if not bins_initialized:
            hmf_log_edges = np.asarray(comm.bcast(hmf_log_edges_root, root=0), dtype=np.float64)
            hmf_centers = np.asarray(comm.bcast(hmf_centers_root, root=0), dtype=np.float64)
            bins_initialized = True

        if not truth_done:
            hmf_truth_root = None
            if truth_displacement_path is not None:
                truth_length = run_distributed_fof_lengths(
                    displacement_path=truth_displacement_path,
                    boxsize=float(args.boxsize),
                    n_part=int(args.n_part),
                    linking_length=float(args.fof_linking_length),
                    nmin=int(args.fof_nmin),
                    absolute_linking=bool(args.fof_absolute_linking),
                )
                if rank == 0:
                    if truth_length is None:
                        raise RuntimeError("Internal error: rank 0 did not receive truth FoF halo lengths.")
                    truth_masses = np.asarray(truth_length, dtype=np.float64) * float(m_particle)
                    hmf_truth_root = compute_hmf_curve(
                        masses=np.asarray(truth_masses, dtype=np.float64),
                        log_edges=np.asarray(hmf_log_edges, dtype=np.float64),
                        boxsize=float(args.boxsize),
                    )
            elif truth_fof_tab_path is not None:
                if rank == 0:
                    truth_lengths = load_group_tab_lengths(truth_fof_tab_path)
                    if truth_lengths.size == 0:
                        raise ValueError(f"No FoF groups found in truth catalog: {truth_fof_tab_path}")
                    truth_masses = np.asarray(truth_lengths, dtype=np.float64) * float(m_particle)
                    hmf_truth_root = compute_hmf_curve(
                        masses=np.asarray(truth_masses, dtype=np.float64),
                        log_edges=np.asarray(hmf_log_edges, dtype=np.float64),
                        boxsize=float(args.boxsize),
                    )
            hmf_truth_bcast = comm.bcast(hmf_truth_root, root=0)
            hmf_truth = None if hmf_truth_bcast is None else np.asarray(hmf_truth_bcast, dtype=np.float64)
            truth_done = True

        if rank == 0:
            hmf_curve = compute_hmf_curve(
                masses=np.asarray(masses, dtype=np.float64),
                log_edges=np.asarray(hmf_log_edges, dtype=np.float64),
                boxsize=float(args.boxsize),
            )
            hmf_samples.append(np.asarray(hmf_curve, dtype=np.float64))
            sample_ids.append(int(sample_id))
            sample_output_dirs.append(str(out_dir))

            if bool(args.save_per_sample_hmf):
                np.savez_compressed(
                    out_dir / "ppc_sample_hmf.npz",
                    sample_id=np.int32(sample_id),
                    hmf_mass_centers=np.asarray(hmf_centers, dtype=np.float64),
                    hmf_dn_dlog10m=np.asarray(hmf_curve, dtype=np.float64),
                )

    if rank != 0:
        return

    if hmf_centers is None or hmf_log_edges is None or len(hmf_samples) == 0:
        raise RuntimeError("No HMF samples were produced.")

    hmf_samples_arr = np.asarray(hmf_samples, dtype=np.float64)
    prefix = str(args.aggregate_prefix)
    mass_resolution_limit = float(m_particle) * float(args.fof_nmin)
    tinker_curve = None
    tinker_pk_table_path = None
    tinker_status = "disabled"
    if compute_pylians_hmf_theory is not None and find_class_pk_table_path is not None:
        tinker_curve, tinker_pk_table_path, tinker_status = compute_tinker_curve_from_samples(
            sample_output_dirs=sample_output_dirs,
            mass_centers=np.asarray(hmf_centers, dtype=np.float64),
            omega_m=float(args.omega_m),
        )
    else:
        tinker_status = "helpers_unavailable_from_halos_module"

    payload = {
        "sample_ids": np.asarray(sample_ids, dtype=np.int32),
        "sample_output_dirs": np.asarray(sample_output_dirs),
        "hmf_mass_centers": np.asarray(hmf_centers, dtype=np.float64),
        "hmf_log_edges": np.asarray(hmf_log_edges, dtype=np.float64),
        "hmf_samples": np.asarray(hmf_samples_arr, dtype=np.float64),
        "hmf_truth": np.asarray([] if hmf_truth is None else hmf_truth, dtype=np.float64),
        "hmf_tinker": np.asarray([] if tinker_curve is None else tinker_curve, dtype=np.float64),
        "mass_resolution_limit": np.asarray(mass_resolution_limit, dtype=np.float64),
    }
    arrays_path = aggregate_dir / f"{prefix}_hmf_arrays.npz"
    save_summary_npz(arrays_path, payload)

    plot_hmf_path = aggregate_dir / f"{prefix}_hmf.png"
    plot_hmf_summary(
        m_centers=np.asarray(hmf_centers, dtype=np.float64),
        hmf_samples=np.asarray(hmf_samples_arr, dtype=np.float64),
        tinker_curve=None if tinker_curve is None else np.asarray(tinker_curve, dtype=np.float64),
        hmf_truth=None if hmf_truth is None else np.asarray(hmf_truth, dtype=np.float64),
        mass_resolution_limit=mass_resolution_limit,
        out_path=plot_hmf_path,
    )

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples_dir": str(samples_dir),
        "num_samples_requested": int(args.num_samples),
        "num_samples_processed": int(len(sample_ids)),
        "sample_ids": [int(x) for x in sample_ids],
        "sample_output_dirs": [str(x) for x in sample_output_dirs],
        "z_target": float(args.z_target),
        "mpi_world_size": int(size),
        "n_part": int(args.n_part),
        "boxsize": float(args.boxsize),
        "omega_m": float(args.omega_m),
        "fof_linking_length": float(args.fof_linking_length),
        "fof_absolute_linking": bool(args.fof_absolute_linking),
        "fof_nmin": int(args.fof_nmin),
        "mass_resolution_limit_msun_h": float(mass_resolution_limit),
        "hmf_nbins": int(args.hmf_nbins),
        "hmf_mmin": None if args.hmf_mmin is None else float(args.hmf_mmin),
        "hmf_mmax": None if args.hmf_mmax is None else float(args.hmf_mmax),
        "tinker_status": str(tinker_status),
        "tinker_pk_table_path": None if tinker_pk_table_path is None else str(tinker_pk_table_path),
        "truth_hmf_available": bool(hmf_truth is not None),
        "truth_hmf_source": (
            "displacement"
            if truth_displacement_path is not None
            else ("fof_group_tab" if truth_fof_tab_path is not None else "none")
        ),
        "truth_displacement_path": None if truth_displacement_path is None else str(truth_displacement_path),
        "truth_fof_tab_path": None if truth_fof_tab_path is None else str(truth_fof_tab_path),
        "aggregate_hmf_arrays": str(arrays_path),
        "aggregate_hmf_plot": str(plot_hmf_path),
    }
    metadata_path = aggregate_dir / f"{prefix}_hmf_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("HMF stage complete.")
    print(f"Samples processed: {len(sample_ids)}")
    print(f"HMF arrays: {arrays_path}")
    print(f"HMF plot: {plot_hmf_path}")
    print(f"HMF metadata: {metadata_path}")


if __name__ == "__main__":
    main()
