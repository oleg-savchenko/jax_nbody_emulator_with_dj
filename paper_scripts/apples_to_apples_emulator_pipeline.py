#!/usr/bin/env python3
"""
End-to-end apples-to-apples emulator diagnostic pipeline.

This script builds a fully controlled comparison with matched high-k treatment:

1) Build a reference IC field from the true IC by low-pass filtering at k_cut.
2) Inject Gaussian high-k modes (conditional on retained low-k modes) to n_part.
3) Run emulator forward to obtain a reference emulator final field.
4) Run posterior inference against this emulator final field to generate IC samples.
5) Apply the same low-pass + high-k injection recipe to sampled ICs, run forward,
   and compare IC/final two-point summaries against the reference branches.

Outputs are written under a dedicated run directory in --samples-root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ppc_common import extract_sample_id, mean_sigma, select_sample_files

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
    from core import QUIJOTE_FIDUCIAL, run_lpt_emulator_pipeline
except ImportError as exc:  # pragma: no cover - runtime dependency
    QUIJOTE_FIDUCIAL = None
    run_lpt_emulator_pipeline = None
    _CORE_IMPORT_ERROR = exc
else:
    _CORE_IMPORT_ERROR = None

try:
    from utils import get_pk_class, growth_D_approx, mas_from_worder, parse_ndiv, resize_density_grid
except ImportError as exc:  # pragma: no cover - runtime dependency
    get_pk_class = None
    growth_D_approx = None
    mas_from_worder = None
    parse_ndiv = None
    resize_density_grid = None
    _UTILS_IMPORT_ERROR = exc
else:
    _UTILS_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Run apples-to-apples emulator + inference + resimulation diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=Path("/gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD"),
        help="Base directory where pipeline run folders are created.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run subfolder name under --samples-root. If omitted, generated from timestamp.",
    )
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing stage outputs where possible.",
    )

    parser.add_argument(
        "--target-path",
        type=Path,
        default=Path("/home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt"),
        help="True target .pt file containing IC/final fields.",
    )
    parser.add_argument("--ic-key", type=str, default="delta_z127", help="IC key in target .pt file.")
    parser.add_argument("--target-key", type=str, default="delta_z0", help="Final-field key in target .pt file.")
    parser.add_argument("--z-ic", type=float, default=127.0, help="Redshift of IC field in target/sample files.")
    parser.add_argument("--z-target", type=float, default=0.0, help="Forward-model target redshift.")

    parser.add_argument("--res", type=int, default=128, help="Coarse/reference density resolution.")
    parser.add_argument("--n-part", type=int, default=512, help="Emulator particle/displacement resolution.")
    parser.add_argument("--boxsize", type=float, default=1000.0, help="Simulation box size [Mpc/h].")
    parser.add_argument(
        "--ndiv",
        type=parse_ndiv if callable(parse_ndiv) else str,
        default=(4, 2, 2),
        help="Emulator subbox partition.",
    )
    parser.add_argument("--emu-precision", choices=("f16", "f32"), default="f16", help="Emulator precision.")
    parser.add_argument("--num-sims", type=int, default=1, help="Internal emulator repeats per forward call.")
    parser.add_argument("--mas-worder", type=int, choices=(2, 3, 4), default=4, help="MAS order for meshing.")
    parser.add_argument(
        "--deconvolve-mas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether generated emulator/LPT fields are MAS-deconvolved.",
    )
    parser.add_argument("--class-n-modes", type=int, default=4000, help="CLASS k-modes when using class P(k).")

    parser.add_argument(
        "--k-cut-fraction",
        type=float,
        default=1.0,
        help="k_cut as fraction of coarse-grid Nyquist (ignored when --k-cut-hmpc is set).",
    )
    parser.add_argument(
        "--k-cut-hmpc",
        type=float,
        default=None,
        help="Absolute k_cut [h/Mpc]. Overrides --k-cut-fraction when provided.",
    )
    parser.add_argument(
        "--inject-pk-source",
        choices=("class", "reference_measured"),
        default="class",
        help="Target P(k) source for high-k mode injection.",
    )
    parser.add_argument("--reference-inject-seed", type=int, default=0, help="RNG seed for reference IC injection.")
    parser.add_argument(
        "--sample-inject-seed-policy",
        choices=("paired", "sample_id", "fixed", "none"),
        default="paired",
        help="RNG seed policy for posterior-sample high-k injection.",
    )
    parser.add_argument("--sample-inject-fixed-seed", type=int, default=0, help="Seed used for policy=fixed.")

    parser.add_argument(
        "--run-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run external inference script to generate posterior IC samples.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Trained gaussian_npe model directory. Auto-discovered from samples-root name if omitted.",
    )
    parser.add_argument(
        "--inference-script",
        type=Path,
        default=Path("/home/osavchenko/gaussian_npe/paper_plots_scripts/save_samples.py"),
        help="Path to gaussian_npe posterior sampling script.",
    )
    parser.add_argument(
        "--inference-repo-dir",
        type=Path,
        default=Path("/home/osavchenko/gaussian_npe"),
        help="Working directory used when running the inference script.",
    )
    parser.add_argument(
        "--inference-python",
        type=str,
        default="python",
        help="Python executable used for the external inference script.",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of posterior samples to generate/use.")
    parser.add_argument("--noise-seed", type=int, default=42, help="Inference noise seed.")
    parser.add_argument(
        "--existing-samples-dir",
        type=Path,
        default=None,
        help="Use an existing IC sample directory instead of running inference.",
    )
    parser.add_argument("--sample-pattern", type=str, default="sample_*.npy", help="Posterior IC sample glob pattern.")
    parser.add_argument("--sample-start", type=int, default=0, help="Starting index in sorted sample list.")
    parser.add_argument("--sample-stride", type=int, default=1, help="Stride in sorted sample list.")

    parser.add_argument("--pk-threads", type=int, default=1, help="Thread count for Pk_library calls.")
    parser.add_argument(
        "--save-injected-npart",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save large n_part injected IC fields for each sample.",
    )

    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only build config/stage paths and exit (no heavy computation).",
    )

    args = parser.parse_args()
    if args.res < 1 or args.n_part < 1:
        parser.error("--res and --n-part must be >= 1.")
    if args.n_part < args.res:
        parser.error("--n-part must be >= --res for this pipeline.")
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1.")
    if args.sample_start < 0:
        parser.error("--sample-start must be >= 0.")
    if args.sample_stride < 1:
        parser.error("--sample-stride must be >= 1.")
    if args.k_cut_fraction <= 0.0:
        parser.error("--k-cut-fraction must be > 0.")
    return args


def require_torch() -> None:
    """Raise a clear error when torch is missing."""
    if torch is None:
        raise ImportError("torch is required for .pt I/O and inference target generation.") from _TORCH_IMPORT_ERROR


def _z_file_tag(z_value: float) -> str:
    """Format redshift for file naming."""
    return f"z{z_value:g}".replace(".", "p")


def format_hms(seconds: float) -> str:
    """Format seconds as HHh MMm SSs."""
    total = int(round(float(seconds)))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}h {mm:02d}m {ss:02d}s"


def maybe_resize_to_res(field: np.ndarray, *, res: int, boxsize: float) -> np.ndarray:
    """Resize field to target resolution only when needed."""
    arr = np.asarray(field, dtype=np.float32)
    if int(arr.shape[0]) == int(res):
        return arr
    return resize_density_grid(
        delta_in=arr,
        target_res=int(res),
        boxsize=float(boxsize),
        upsample_method="mode_inject",
    )


def load_field_from_pt(path: Path, key: str) -> np.ndarray:
    """Load one field from a .pt dictionary."""
    require_torch()
    if not path.is_file():
        raise FileNotFoundError(f"Target file not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data)}")
    if key not in data:
        raise KeyError(f"Missing key {key!r} in {path}")
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim != 3 or len(set(arr.shape)) != 1:
        raise ValueError(f"Field {key!r} must be cubic 3D, got shape={arr.shape}")
    return arr


def lowpass_field(delta: np.ndarray, *, boxsize: float, k_cut: float) -> np.ndarray:
    """Low-pass filter a cubic periodic field by zeroing Fourier modes with |k| > k_cut."""
    arr = np.asarray(delta, dtype=np.float32)
    n = int(arr.shape[0])
    k_nyq = np.pi * n / float(boxsize)
    k_use = min(float(k_cut), float(k_nyq))
    if k_use >= (1.0 - 1e-7) * float(k_nyq):
        return np.asarray(arr, dtype=np.float32)

    f = np.fft.fftn(arr).astype(np.complex64, copy=False)
    k1d = (2.0 * np.pi * np.fft.fftfreq(n, d=float(boxsize) / n)).astype(np.float32, copy=False)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz).astype(np.float32, copy=False)
    f = np.where(kmag <= np.float32(k_use), f, np.complex64(0.0))
    out = np.fft.ifftn(f).real.astype(np.float32, copy=False)
    return np.asarray(out, dtype=np.float32)


def build_pk_table(
    *,
    source: str,
    reference_field: np.ndarray,
    boxsize: float,
    target_res: int,
    class_params: dict[str, float],
    z_target: float,
    class_n_modes: int,
    pk_threads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build target P(k) table used for high-k injection."""
    if source == "class":
        k_fund = 2.0 * np.pi / float(boxsize)
        k_nyq = np.pi * int(target_res) / float(boxsize)
        kmin = min(0.5 * k_fund, 1.0e-5)
        kmax = max(2.0 * k_nyq * np.sqrt(3.0), 10.0)
        k_tab, pk_tab = get_pk_class(
            cosmo_params=class_params,
            z=float(z_target),
            kmin=float(kmin),
            kmax=float(kmax),
            n_modes=int(class_n_modes),
            non_lin=False,
        )
        return np.asarray(k_tab, dtype=np.float64), np.asarray(pk_tab, dtype=np.float64)

    if source != "reference_measured":
        raise ValueError(f"Unknown P(k) source: {source!r}")
    if PKL is None:
        raise ImportError("Pk_library is required for inject-pk-source=reference_measured.") from _PKL_IMPORT_ERROR

    obj = PKL.Pk(
        np.asarray(reference_field, dtype=np.float32),
        float(boxsize),
        axis=0,
        MAS="None",
        threads=int(pk_threads),
        verbose=False,
    )
    k = np.asarray(obj.k3D, dtype=np.float64)
    pk = np.asarray(obj.Pk[:, 0], dtype=np.float64)
    valid = np.isfinite(k) & (k > 0.0) & np.isfinite(pk) & (pk > 0.0)
    return k[valid], pk[valid]


def inject_modes_with_cutoff(
    *,
    delta_in: np.ndarray,
    target_res: int,
    boxsize: float,
    k_copy_cut: float,
    k_target: np.ndarray,
    pk_target: np.ndarray,
    seed: int | None,
) -> np.ndarray:
    """Upsample by copying low-k modes up to k_copy_cut and injecting random high-k modes."""
    delta_in = np.asarray(delta_in, dtype=np.float32)
    if delta_in.ndim != 3 or len(set(delta_in.shape)) != 1:
        raise ValueError(f"`delta_in` must be cubic 3D, got shape={delta_in.shape}")

    in_res = int(delta_in.shape[0])
    out_res = int(target_res)
    if out_res < in_res:
        raise ValueError(f"`target_res` must be >= input resolution ({in_res}), got {out_res}.")
    if out_res == in_res:
        return np.asarray(delta_in, dtype=np.float32)
    if out_res % in_res != 0:
        raise ValueError(f"Cannot upsample from {in_res} to {out_res}: integer factor required.")

    k_target = np.asarray(k_target, dtype=np.float64).ravel()
    pk_target = np.asarray(pk_target, dtype=np.float64).ravel()
    if k_target.size < 2 or pk_target.size != k_target.size:
        raise ValueError("Invalid P(k) table for injection.")
    order = np.argsort(k_target)
    k_target = k_target[order]
    pk_target = pk_target[order]
    if np.any(np.diff(k_target) <= 0.0):
        raise ValueError("k_target must be strictly increasing.")

    lowk_scale = (float(out_res) / float(in_res)) ** 3
    volume = float(boxsize) ** 3
    rng = np.random.default_rng(seed)

    f_low_shift = np.fft.fftshift(np.fft.fftn(delta_in)).astype(np.complex64, copy=False)
    f_out_shift = np.empty((out_res, out_res, out_res), dtype=np.complex64)

    start = (out_res - in_res) // 2
    stop = start + in_res
    low_block = np.asarray(f_low_shift * np.float32(lowk_scale), dtype=np.complex64)

    k1d_shift = np.fft.fftshift(
        2.0 * np.pi * np.fft.fftfreq(out_res, d=float(boxsize) / out_res)
    ).astype(np.float32, copy=False)
    ky2z2 = (k1d_shift[:, None] ** 2 + k1d_shift[None, :] ** 2).astype(np.float32, copy=False)
    k_copy_max = min(float(k_copy_cut), float(np.pi * in_res / float(boxsize)))

    low_mask_subcube = np.zeros((in_res, in_res, in_res), dtype=bool)
    for ix in range(start, stop):
        kmag_sub = np.sqrt(float(k1d_shift[ix]) ** 2 + ky2z2[start:stop, start:stop]).astype(np.float32, copy=False)
        low_mask_subcube[ix - start, :, :] = kmag_sub <= np.float32(k_copy_max)

    tail = min(8, int(k_target.size))
    slope, intercept = np.polyfit(np.log(k_target[-tail:]), np.log(pk_target[-tail:]), 1)
    k_hi = float(k_target[-1])

    for ix, kx_val in enumerate(k1d_shift):
        kmag = np.sqrt(float(kx_val) ** 2 + ky2z2).astype(np.float64, copy=False)
        pk_slice = np.interp(kmag.ravel(), k_target, pk_target, left=pk_target[0], right=pk_target[-1]).reshape(kmag.shape)
        high = kmag > k_hi
        if np.any(high):
            pk_slice[high] = np.exp(intercept + slope * np.log(kmag[high]))
        pk_slice = np.maximum(pk_slice, 0.0)

        sigma_slice = (out_res**3) * np.sqrt(pk_slice / volume)
        rand_re = rng.normal(size=(out_res, out_res))
        rand_im = rng.normal(size=(out_res, out_res))
        f_out_shift[ix, :, :] = np.asarray((rand_re + 1j * rand_im) * sigma_slice, dtype=np.complex64)

    out_subcube = f_out_shift[start:stop, start:stop, start:stop]
    out_subcube = np.where(low_mask_subcube, low_block, out_subcube)
    f_out_shift[start:stop, start:stop, start:stop] = out_subcube

    f_out = np.fft.ifftshift(f_out_shift)
    delta_real = np.fft.ifftn(f_out).real.astype(np.float32, copy=False)
    f_out_shift = np.fft.fftshift(np.fft.fftn(delta_real)).astype(np.complex64, copy=False)
    out_subcube = f_out_shift[start:stop, start:stop, start:stop]
    out_subcube = np.where(low_mask_subcube, low_block, out_subcube)
    f_out_shift[start:stop, start:stop, start:stop] = out_subcube

    return np.asarray(np.fft.ifftn(np.fft.ifftshift(f_out_shift)).real, dtype=np.float32)


def infer_model_dir(samples_root: Path, explicit: Path | None) -> Path:
    """Resolve gaussian_npe model directory."""
    if explicit is not None:
        model_dir = explicit.expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"--model-dir does not exist: {model_dir}")
        return model_dir

    run_name = samples_root.name
    candidates = [
        Path("/home/osavchenko/gaussian_npe/runs") / run_name,
        Path("/home/osavchenko/gaussian_npe/paper_test_runs/runs") / run_name,
    ]
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.pt").is_file():
            return candidate.resolve()

    root = Path("/home/osavchenko/gaussian_npe/paper_test_runs/runs")
    if root.is_dir():
        found = sorted(
            p
            for p in root.rglob(run_name)
            if p.is_dir() and (p / "config.json").is_file() and (p / "model.pt").is_file()
        )
        if found:
            return found[0].resolve()
    raise FileNotFoundError(
        "Could not auto-discover model directory. Pass --model-dir explicitly."
    )


def run_inference_sampling(
    *,
    inference_python: str,
    inference_script: Path,
    inference_repo_dir: Path,
    model_dir: Path,
    target_path: Path,
    output_base: Path,
    num_samples: int,
    noise_seed: int,
    reuse_existing: bool,
) -> Path:
    """Run gaussian_npe save_samples.py and return the produced sample directory."""
    samples_dir = output_base / model_dir.name
    existing = list(samples_dir.glob("sample_*.npy")) if samples_dir.is_dir() else []
    if reuse_existing and len(existing) >= int(num_samples):
        print(f"[Inference] Reusing existing samples in {samples_dir} (found {len(existing)} files).", flush=True)
        return samples_dir

    output_base.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(inference_python),
        str(inference_script),
        "--model_dir",
        str(model_dir),
        "--target_path",
        str(target_path),
        "--num_samples",
        str(int(num_samples)),
        "--output_dir",
        str(output_base),
        "--noise_seed",
        str(int(noise_seed)),
    ]
    print("[Inference] Running:", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(inference_repo_dir),
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"[Inference] Finished in {format_hms(elapsed)} ({elapsed:.2f} s).", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Inference script failed.\n"
            f"Return code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    existing = list(samples_dir.glob("sample_*.npy")) if samples_dir.is_dir() else []
    if len(existing) < int(num_samples):
        raise RuntimeError(
            f"Inference completed but expected >= {num_samples} samples in {samples_dir}; "
            f"found {len(existing)}."
        )
    return samples_dir


def compute_pk_truth(
    *,
    delta_true: np.ndarray,
    boxsize: float,
    mas_target: str,
    pk_threads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute truth P(k)."""
    obj = PKL.Pk(
        np.asarray(delta_true, dtype=np.float32),
        float(boxsize),
        axis=0,
        MAS=str(mas_target),
        threads=int(pk_threads),
        verbose=False,
    )
    k = np.asarray(obj.k3D, dtype=np.float64)
    pk = np.asarray(obj.Pk[:, 0], dtype=np.float64)
    valid = np.isfinite(k) & (k > 0.0) & np.isfinite(pk) & (pk > 0.0)
    return k[valid], pk[valid]


def compute_pk_xcorr_sample(
    *,
    field_sample: np.ndarray,
    field_true: np.ndarray,
    boxsize: float,
    mas_model: str,
    mas_target: str,
    pk_threads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sample P(k) and C(k) against truth."""
    obj = PKL.XPk(
        [np.asarray(field_sample, dtype=np.float32), np.asarray(field_true, dtype=np.float32)],
        float(boxsize),
        axis=0,
        MAS=[str(mas_model), str(mas_target)],
        threads=int(pk_threads),
    )
    k = np.asarray(obj.k3D, dtype=np.float64)
    pk_sample = np.asarray(obj.Pk[:, 0, 0], dtype=np.float64)
    pk_true = np.asarray(obj.Pk[:, 0, 1], dtype=np.float64)
    xpk = np.asarray(obj.XPk[:, 0, 0], dtype=np.float64)
    denom = np.sqrt(pk_sample * pk_true)
    c = np.divide(xpk, denom, out=np.full_like(xpk, np.nan), where=denom > 0.0)
    valid = np.isfinite(k) & (k > 0.0) & np.isfinite(pk_sample) & (pk_sample > 0.0) & np.isfinite(c)
    return k[valid], pk_sample[valid], c[valid]


def collect_pk_xcorr_stack(
    *,
    sample_fields: list[np.ndarray],
    truth_field: np.ndarray,
    boxsize: float,
    mas_model: str,
    mas_target: str,
    pk_threads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build stacked sample arrays of P(k) and C(k)."""
    k_ref, pk_true = compute_pk_truth(
        delta_true=np.asarray(truth_field, dtype=np.float32),
        boxsize=float(boxsize),
        mas_target=str(mas_target),
        pk_threads=int(pk_threads),
    )
    pk_samples: list[np.ndarray] = []
    c_samples: list[np.ndarray] = []
    for field in sample_fields:
        k_loc, pk_loc, c_loc = compute_pk_xcorr_sample(
            field_sample=np.asarray(field, dtype=np.float32),
            field_true=np.asarray(truth_field, dtype=np.float32),
            boxsize=float(boxsize),
            mas_model=str(mas_model),
            mas_target=str(mas_target),
            pk_threads=int(pk_threads),
        )
        if k_loc.shape != k_ref.shape or not np.allclose(k_loc, k_ref, rtol=1e-7, atol=1e-10):
            pk_loc = np.interp(k_ref, k_loc, pk_loc)
            c_loc = np.interp(k_ref, k_loc, c_loc)
        pk_samples.append(np.asarray(pk_loc, dtype=np.float64))
        c_samples.append(np.asarray(c_loc, dtype=np.float64))
    return (
        np.asarray(k_ref, dtype=np.float64),
        np.asarray(pk_true, dtype=np.float64),
        np.asarray(pk_samples, dtype=np.float64),
        np.asarray(c_samples, dtype=np.float64),
    )


def _draw_pk_ratio_xcorr_column(
    *,
    axes_col: np.ndarray,
    k: np.ndarray,
    pk_true: np.ndarray,
    pk_samples: np.ndarray,
    xcorr_samples: np.ndarray,
    column_title: str,
    truth_label: str,
    sample_label: str,
    ratio_label: str,
    xcorr_label: str,
    k_nyq: float,
    sample_color: str,
) -> None:
    """Draw one 3-row column: P(k), ratio, and C(k)."""
    k = np.asarray(k, dtype=np.float64)
    pk_true = np.asarray(pk_true, dtype=np.float64)
    pk_samples = np.asarray(pk_samples, dtype=np.float64)
    xcorr_samples = np.asarray(xcorr_samples, dtype=np.float64)

    pk_mu, pk_sigma = mean_sigma(pk_samples)
    ratio_samples = np.divide(
        pk_samples,
        pk_true[None, :],
        out=np.full_like(pk_samples, np.nan),
        where=pk_true[None, :] > 0,
    )
    ratio_mu, ratio_sigma = mean_sigma(ratio_samples)
    xcorr_mu, xcorr_sigma = mean_sigma(xcorr_samples)

    ax = axes_col[0]
    ax.plot(k, pk_true, color="black", lw=1.4, label=truth_label)
    ax.plot(k, pk_mu, color=sample_color, lw=1.2, label=sample_label)
    ax.fill_between(k, pk_mu - pk_sigma, pk_mu + pk_sigma, color=sample_color, alpha=0.35, label=r"$\pm 1\sigma$")
    ax.fill_between(k, pk_mu - 2.0 * pk_sigma, pk_mu + 2.0 * pk_sigma, color=sample_color, alpha=0.16, label=r"$\pm 2\sigma$")
    ax.axvline(float(k_nyq), color="red", ls="--", lw=0.8, alpha=0.6, label=r"$k_{\rm Nyq}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$P(k)$")
    ax.set_title(column_title)
    ax.grid(alpha=0.15, which="both")
    ax.legend(framealpha=0.9, fontsize=8, loc="best")

    ax = axes_col[1]
    ax.plot(k, ratio_mu, color=sample_color, lw=1.2, label=ratio_label)
    ax.fill_between(k, ratio_mu - ratio_sigma, ratio_mu + ratio_sigma, color=sample_color, alpha=0.35)
    ax.fill_between(k, ratio_mu - 2.0 * ratio_sigma, ratio_mu + 2.0 * ratio_sigma, color=sample_color, alpha=0.16)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.axvline(float(k_nyq), color="red", ls="--", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0.0, 2.0)
    ax.grid(alpha=0.15, which="both")

    ax = axes_col[2]
    ax.plot(k, xcorr_mu, color="tab:green", lw=1.2, label=xcorr_label)
    ax.fill_between(k, xcorr_mu - xcorr_sigma, xcorr_mu + xcorr_sigma, color="tab:green", alpha=0.35, label=r"$\pm 1\sigma$")
    ax.fill_between(k, xcorr_mu - 2.0 * xcorr_sigma, xcorr_mu + 2.0 * xcorr_sigma, color="tab:green", alpha=0.16, label=r"$\pm 2\sigma$")
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Truth self-correlation")
    ax.axvline(float(k_nyq), color="red", ls="--", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel(r"$k$ [$h/{\rm Mpc}$]")
    ax.set_ylabel(r"$C(k)$")
    ax.grid(alpha=0.15, which="both")
    ax.legend(framealpha=0.9, fontsize=8, loc="best")


def plot_pk_ratio_crosscorr_two_column(
    *,
    k_ic: np.ndarray,
    pk_true_ic: np.ndarray,
    pk_samples_ic: np.ndarray,
    xcorr_samples_ic: np.ndarray,
    k_final: np.ndarray,
    pk_true_final: np.ndarray,
    pk_samples_final: np.ndarray,
    xcorr_samples_final: np.ndarray,
    boxsize: float,
    res: int,
    out_path: Path,
    left_title: str,
    right_title: str,
) -> None:
    """Plot 3x2 diagnostics with IC on the left and final on the right."""
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13.4, 9.0),
        constrained_layout=True,
        sharex="col",
        height_ratios=(2.0, 1.0, 1.0),
    )
    k_nyq = np.pi * int(res) / float(boxsize)

    _draw_pk_ratio_xcorr_column(
        axes_col=axes[:, 0],
        k=np.asarray(k_ic, dtype=np.float64),
        pk_true=np.asarray(pk_true_ic, dtype=np.float64),
        pk_samples=np.asarray(pk_samples_ic, dtype=np.float64),
        xcorr_samples=np.asarray(xcorr_samples_ic, dtype=np.float64),
        column_title=str(left_title),
        truth_label="Reference true IC",
        sample_label="Posterior IC mean",
        ratio_label="Posterior IC / reference",
        xcorr_label="Posterior IC mean C(k)",
        k_nyq=float(k_nyq),
        sample_color="tab:orange",
    )

    _draw_pk_ratio_xcorr_column(
        axes_col=axes[:, 1],
        k=np.asarray(k_final, dtype=np.float64),
        pk_true=np.asarray(pk_true_final, dtype=np.float64),
        pk_samples=np.asarray(pk_samples_final, dtype=np.float64),
        xcorr_samples=np.asarray(xcorr_samples_final, dtype=np.float64),
        column_title=str(right_title),
        truth_label="Reference emulator final",
        sample_label="Posterior resim mean",
        ratio_label="Resim / reference",
        xcorr_label="Resim mean C(k)",
        k_nyq=float(k_nyq),
        sample_color="tab:blue",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Run the full apples-to-apples pipeline."""
    args = parse_args()

    samples_root = args.samples_root.expanduser().resolve()
    samples_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name
    if run_name is None:
        run_name = f"apples_to_apples_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    run_dir = samples_root / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    stage1_dir = run_dir / "stage1_reference_ic"
    stage2_dir = run_dir / "stage2_reference_forward"
    stage3_dir = run_dir / "stage3_inference"
    stage4_dir = run_dir / "stage4_posterior_forward"
    stage5_dir = run_dir / "stage5_diagnostics"
    for path in (stage1_dir, stage2_dir, stage3_dir, stage4_dir, stage5_dir):
        path.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_manifest = {
            "run_dir": str(run_dir),
            "stage_dirs": {
                "stage1_reference_ic": str(stage1_dir),
                "stage2_reference_forward": str(stage2_dir),
                "stage3_inference": str(stage3_dir),
                "stage4_posterior_forward": str(stage4_dir),
                "stage5_diagnostics": str(stage5_dir),
            },
        }
        (run_dir / "dry_run_manifest.json").write_text(json.dumps(dry_manifest, indent=2))
        print(f"Dry run complete. Manifest: {run_dir / 'dry_run_manifest.json'}", flush=True)
        return

    if _CORE_IMPORT_ERROR is not None:
        raise ImportError("Failed to import scripts/core.py dependencies.") from _CORE_IMPORT_ERROR
    if _UTILS_IMPORT_ERROR is not None:
        raise ImportError("Failed to import scripts/utils.py dependencies.") from _UTILS_IMPORT_ERROR
    if PKL is None:
        raise ImportError("Pk_library is required for diagnostics.") from _PKL_IMPORT_ERROR

    t_all = time.perf_counter()

    omega_m = float(QUIJOTE_FIDUCIAL["Omega_m"])
    omega_b = float(QUIJOTE_FIDUCIAL["Omega_b"])
    class_params = {
        "Omega_cdm": omega_m - omega_b,
        "Omega_b": omega_b,
        "h": float(QUIJOTE_FIDUCIAL["h"]),
        "n_s": float(QUIJOTE_FIDUCIAL["n_s"]),
        "sigma8": float(QUIJOTE_FIDUCIAL["sigma8"]),
    }
    scale = float(
        growth_D_approx(class_params, float(args.z_target))
        / growth_D_approx(class_params, float(args.z_ic))
    )
    k_nyq_res = float(np.pi * int(args.res) / float(args.boxsize))
    k_cut = float(args.k_cut_hmpc) if args.k_cut_hmpc is not None else float(args.k_cut_fraction) * k_nyq_res
    if k_cut > k_nyq_res:
        print(
            f"Requested k_cut={k_cut:.6f} exceeds coarse Nyquist={k_nyq_res:.6f}; clipping to Nyquist.",
            flush=True,
        )
        k_cut = k_nyq_res

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Growth scaling D(z={args.z_target})/D(z={args.z_ic}) = {scale:.8f}", flush=True)
    print(f"k_cut = {k_cut:.6f} h/Mpc (k_Nyq_res={k_nyq_res:.6f})", flush=True)

    stage1_files = {
        "true_ic_z127_res": stage1_dir / "true_ic_z127_res.npy",
        "true_ic_z127_cut": stage1_dir / "true_ic_z127_cut.npy",
        "true_ic_z0_cut": stage1_dir / "true_ic_z0_cut.npy",
        "reference_ic_z0_npart": stage1_dir / "reference_ic_z0_npart.npy",
        "pk_table": stage1_dir / "injection_pk_table.txt",
        "metadata": stage1_dir / "stage1_metadata.json",
    }

    if (
        bool(args.reuse_existing)
        and all(path.is_file() for path in stage1_files.values())
    ):
        print("[Stage 1] Reusing reference IC products.", flush=True)
        true_ic_z127_res = np.asarray(np.load(stage1_files["true_ic_z127_res"]), dtype=np.float32)
        true_ic_z127_cut = np.asarray(np.load(stage1_files["true_ic_z127_cut"]), dtype=np.float32)
        true_ic_z0_cut = np.asarray(np.load(stage1_files["true_ic_z0_cut"]), dtype=np.float32)
        reference_ic_z0_npart = np.asarray(np.load(stage1_files["reference_ic_z0_npart"]), dtype=np.float32)
        table = np.loadtxt(stage1_files["pk_table"])
        k_table = np.asarray(table[:, 0], dtype=np.float64)
        pk_table = np.asarray(table[:, 1], dtype=np.float64)
    else:
        print("[Stage 1] Building reference IC fields with cutoff + injection...", flush=True)
        true_ic_raw = load_field_from_pt(args.target_path, args.ic_key)
        true_ic_z127_res = maybe_resize_to_res(true_ic_raw, res=int(args.res), boxsize=float(args.boxsize))
        true_ic_z127_cut = lowpass_field(
            np.asarray(true_ic_z127_res, dtype=np.float32),
            boxsize=float(args.boxsize),
            k_cut=float(k_cut),
        )
        true_ic_z0_cut = np.asarray(true_ic_z127_cut * np.float32(scale), dtype=np.float32)

        k_table, pk_table = build_pk_table(
            source=str(args.inject_pk_source),
            reference_field=np.asarray(true_ic_z0_cut, dtype=np.float32),
            boxsize=float(args.boxsize),
            target_res=int(args.n_part),
            class_params=class_params,
            z_target=float(args.z_target),
            class_n_modes=int(args.class_n_modes),
            pk_threads=int(args.pk_threads),
        )
        if int(args.n_part) == int(args.res):
            reference_ic_z0_npart = np.asarray(true_ic_z0_cut, dtype=np.float32)
        else:
            reference_ic_z0_npart = inject_modes_with_cutoff(
                delta_in=np.asarray(true_ic_z0_cut, dtype=np.float32),
                target_res=int(args.n_part),
                boxsize=float(args.boxsize),
                k_copy_cut=float(k_cut),
                k_target=np.asarray(k_table, dtype=np.float64),
                pk_target=np.asarray(pk_table, dtype=np.float64),
                seed=int(args.reference_inject_seed),
            )

        np.save(stage1_files["true_ic_z127_res"], np.asarray(true_ic_z127_res, dtype=np.float32))
        np.save(stage1_files["true_ic_z127_cut"], np.asarray(true_ic_z127_cut, dtype=np.float32))
        np.save(stage1_files["true_ic_z0_cut"], np.asarray(true_ic_z0_cut, dtype=np.float32))
        np.save(stage1_files["reference_ic_z0_npart"], np.asarray(reference_ic_z0_npart, dtype=np.float32))
        np.savetxt(
            stage1_files["pk_table"],
            np.column_stack([np.asarray(k_table, dtype=np.float64), np.asarray(pk_table, dtype=np.float64)]),
            header="k_h_per_Mpc Pk_Mpc_over_h_cubed",
        )
        stage1_meta = {
            "target_path": str(args.target_path),
            "ic_key": str(args.ic_key),
            "res": int(args.res),
            "n_part": int(args.n_part),
            "boxsize": float(args.boxsize),
            "z_ic": float(args.z_ic),
            "z_target": float(args.z_target),
            "growth_scale": float(scale),
            "k_cut_h_per_mpc": float(k_cut),
            "k_nyq_res_h_per_mpc": float(k_nyq_res),
            "inject_pk_source": str(args.inject_pk_source),
            "reference_inject_seed": int(args.reference_inject_seed),
        }
        stage1_files["metadata"].write_text(json.dumps(stage1_meta, indent=2))

    stage2_files = {
        "reference_delta_emu": stage2_dir / f"reference_emu_delta_{_z_file_tag(float(args.z_target))}.npy",
        "reference_delta_lpt": stage2_dir / f"reference_lpt_delta_{_z_file_tag(float(args.z_target))}.npy",
        "reference_ic_summary": stage2_dir / "reference_ic_summary.npy",
        "metadata": stage2_dir / "stage2_metadata.json",
    }
    if (
        bool(args.reuse_existing)
        and all(path.is_file() for path in stage2_files.values())
    ):
        print("[Stage 2] Reusing reference forward outputs.", flush=True)
        reference_delta_emu = np.asarray(np.load(stage2_files["reference_delta_emu"]), dtype=np.float32)
        reference_delta_lpt = np.asarray(np.load(stage2_files["reference_delta_lpt"]), dtype=np.float32)
        reference_ic_summary = np.asarray(np.load(stage2_files["reference_ic_summary"]), dtype=np.float32)
        stage2_meta = json.loads(stage2_files["metadata"].read_text())
    else:
        print("[Stage 2] Running emulator forward for the reference branch...", flush=True)
        t0 = time.perf_counter()
        pipeline_ref = run_lpt_emulator_pipeline(
            delta_linear=np.asarray(reference_ic_z0_npart, dtype=np.float32),
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
            delta_upsample_method="mode_inject",
            downsample_method="fourier_truncation",
            class_n_modes=int(args.class_n_modes),
            mode_inject_seed=None,
            quijote_fiducial=QUIJOTE_FIDUCIAL,
            output_dir=None,
            save_outputs=False,
            show_progress=True,
            progress_desc_prefix="A2A reference subboxes",
        )
        elapsed = time.perf_counter() - t0
        print(f"[Stage 2] Reference forward runtime: {format_hms(elapsed)} ({elapsed:.2f} s).", flush=True)

        reference_delta_emu = np.asarray(pipeline_ref["delta_emu"], dtype=np.float32)
        reference_delta_lpt = np.asarray(pipeline_ref["delta_lpt"], dtype=np.float32)
        reference_ic_summary = np.asarray(pipeline_ref["ics_grf"], dtype=np.float32)

        np.save(stage2_files["reference_delta_emu"], reference_delta_emu)
        np.save(stage2_files["reference_delta_lpt"], reference_delta_lpt)
        np.save(stage2_files["reference_ic_summary"], reference_ic_summary)
        stage2_meta = {
            "forward_runtime_seconds_wall": float(elapsed),
            "forward_runtime_hms_wall": format_hms(elapsed),
            "pipeline_metadata": pipeline_ref["metadata"],
        }
        stage2_files["metadata"].write_text(json.dumps(stage2_meta, indent=2))

    stage3_target_path = stage3_dir / "reference_target_emulator.pt"
    if torch is not None:
        if not stage3_target_path.is_file() or not bool(args.reuse_existing):
            payload = {
                "delta_z0": np.asarray(reference_delta_emu, dtype=np.float32),
                "delta_z127_reference_cut": np.asarray(true_ic_z127_cut, dtype=np.float32),
            }
            torch.save(payload, stage3_target_path)
            print(f"[Stage 3] Saved inference target at {stage3_target_path}", flush=True)
    elif bool(args.run_inference):
        require_torch()

    if bool(args.run_inference):
        model_dir = infer_model_dir(samples_root=samples_root, explicit=args.model_dir)
        print(f"[Stage 3] Using inference model_dir: {model_dir}", flush=True)
        inference_samples_base = stage3_dir / "samples"
        inferred_samples_dir = run_inference_sampling(
            inference_python=str(args.inference_python),
            inference_script=args.inference_script.expanduser().resolve(),
            inference_repo_dir=args.inference_repo_dir.expanduser().resolve(),
            model_dir=model_dir,
            target_path=stage3_target_path,
            output_base=inference_samples_base,
            num_samples=int(args.num_samples),
            noise_seed=int(args.noise_seed),
            reuse_existing=bool(args.reuse_existing),
        )
    else:
        if args.existing_samples_dir is None:
            default_parent = stage3_dir / "samples"
            if not default_parent.is_dir():
                raise ValueError(
                    "--run-inference is disabled and no --existing-samples-dir was provided."
                )
            if any(default_parent.glob(str(args.sample_pattern))):
                inferred_samples_dir = default_parent
            else:
                candidate_dirs = sorted(
                    p for p in default_parent.iterdir() if p.is_dir() and any(p.glob(str(args.sample_pattern)))
                )
                if len(candidate_dirs) == 1:
                    inferred_samples_dir = candidate_dirs[0]
                else:
                    raise ValueError(
                        "Could not infer sample directory under "
                        f"{default_parent}. Pass --existing-samples-dir explicitly."
                    )
        else:
            inferred_samples_dir = args.existing_samples_dir.expanduser().resolve()
    print(f"[Stage 3] Posterior IC samples dir: {inferred_samples_dir}", flush=True)

    sample_files = select_sample_files(
        samples_dir=inferred_samples_dir,
        pattern=str(args.sample_pattern),
        sample_start=int(args.sample_start),
        sample_stride=int(args.sample_stride),
        num_samples=int(args.num_samples),
    )
    print(f"[Stage 3] Selected {len(sample_files)} IC samples for diagnostics/forward.", flush=True)

    stage4_sample_dirs: list[Path] = []
    stage4_final_fields: list[np.ndarray] = []
    stage4_records: list[dict[str, object]] = []
    z_tag = _z_file_tag(float(args.z_target))

    print("[Stage 4] Forwarding posterior samples with matched cutoff/injection...", flush=True)
    for idx, sample_path in enumerate(sample_files):
        sample_id = int(extract_sample_id(sample_path, fallback_index=idx))
        sample_dir = stage4_dir / f"sample_{sample_id:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        stage4_sample_dirs.append(sample_dir)
        emu_path = sample_dir / f"emu_delta_{z_tag}.npy"
        meta_path = sample_dir / "metadata.json"

        reused = False
        if bool(args.reuse_existing) and emu_path.is_file() and meta_path.is_file():
            reused = True
            delta_emu = np.asarray(np.load(emu_path), dtype=np.float32)
        else:
            sample_ic = np.asarray(np.load(sample_path), dtype=np.float32)
            if sample_ic.ndim != 3 or len(set(sample_ic.shape)) != 1:
                raise ValueError(f"IC sample must be cubic 3D, got {sample_path} shape={sample_ic.shape}")
            sample_ic_res = maybe_resize_to_res(sample_ic, res=int(args.res), boxsize=float(args.boxsize))
            sample_ic_cut = lowpass_field(
                np.asarray(sample_ic_res, dtype=np.float32),
                boxsize=float(args.boxsize),
                k_cut=float(k_cut),
            )
            sample_ic_z0_cut = np.asarray(sample_ic_cut * np.float32(scale), dtype=np.float32)

            if str(args.sample_inject_seed_policy) == "paired":
                sample_seed = int(args.reference_inject_seed)
            elif str(args.sample_inject_seed_policy) == "sample_id":
                sample_seed = int(sample_id)
            elif str(args.sample_inject_seed_policy) == "fixed":
                sample_seed = int(args.sample_inject_fixed_seed)
            elif str(args.sample_inject_seed_policy) == "none":
                sample_seed = None
            else:  # pragma: no cover - guarded by argparse
                raise ValueError(f"Unknown sample_inject_seed_policy: {args.sample_inject_seed_policy}")

            if int(args.n_part) == int(args.res):
                sample_ic_npart = np.asarray(sample_ic_z0_cut, dtype=np.float32)
            else:
                sample_ic_npart = inject_modes_with_cutoff(
                    delta_in=np.asarray(sample_ic_z0_cut, dtype=np.float32),
                    target_res=int(args.n_part),
                    boxsize=float(args.boxsize),
                    k_copy_cut=float(k_cut),
                    k_target=np.asarray(k_table, dtype=np.float64),
                    pk_target=np.asarray(pk_table, dtype=np.float64),
                    seed=None if sample_seed is None else int(sample_seed),
                )

            t0 = time.perf_counter()
            pipeline = run_lpt_emulator_pipeline(
                delta_linear=np.asarray(sample_ic_npart, dtype=np.float32),
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
                delta_upsample_method="mode_inject",
                downsample_method="fourier_truncation",
                class_n_modes=int(args.class_n_modes),
                mode_inject_seed=None,
                quijote_fiducial=QUIJOTE_FIDUCIAL,
                output_dir=None,
                save_outputs=False,
                show_progress=True,
                progress_desc_prefix="A2A posterior subboxes",
            )
            elapsed = time.perf_counter() - t0
            delta_emu = np.asarray(pipeline["delta_emu"], dtype=np.float32)
            np.save(emu_path, delta_emu)
            np.save(sample_dir / "ic_sample_z127_res.npy", np.asarray(sample_ic_res, dtype=np.float32))
            np.save(sample_dir / "ic_sample_z127_cut.npy", np.asarray(sample_ic_cut, dtype=np.float32))
            np.save(sample_dir / "ic_sample_z0_cut.npy", np.asarray(sample_ic_z0_cut, dtype=np.float32))
            if bool(args.save_injected_npart):
                np.save(sample_dir / "ic_sample_z0_npart_injected.npy", np.asarray(sample_ic_npart, dtype=np.float32))
            sample_meta = {
                "sample_id": int(sample_id),
                "input_sample_path": str(sample_path),
                "sample_inject_seed_policy": str(args.sample_inject_seed_policy),
                "sample_inject_seed": None if sample_seed is None else int(sample_seed),
                "runtime_seconds_wall": float(elapsed),
                "runtime_hms_wall": format_hms(elapsed),
                "pipeline_metadata": pipeline["metadata"],
            }
            meta_path.write_text(json.dumps(sample_meta, indent=2))
            print(
                f"[Stage 4] sample_id={sample_id} finished in {format_hms(elapsed)} ({elapsed:.2f} s).",
                flush=True,
            )

        stage4_final_fields.append(np.asarray(delta_emu, dtype=np.float32))
        stage4_records.append(
            {
                "sample_id": int(sample_id),
                "input_sample_path": str(sample_path),
                "sample_dir": str(sample_dir),
                "emu_delta_path": str(emu_path),
                "reused": bool(reused),
            }
        )

    print("[Stage 5] Computing IC/final P(k), ratio, and C(k) diagnostics...", flush=True)
    mas_name = mas_from_worder(int(args.mas_worder))
    mas_model_final = "None" if bool(args.deconvolve_mas) else mas_name
    mas_target_final = str(mas_model_final)
    mas_model_ic = "None"
    mas_target_ic = "None"

    sampled_ic_fields: list[np.ndarray] = []
    sampled_ic_ids: list[int] = []
    for idx, sample_path in enumerate(sample_files):
        sample_id = int(extract_sample_id(sample_path, fallback_index=idx))
        sampled_ic_ids.append(sample_id)
        sample_ic = np.asarray(np.load(sample_path), dtype=np.float32)
        sample_ic_res = maybe_resize_to_res(sample_ic, res=int(args.res), boxsize=float(args.boxsize))
        sample_ic_cut = lowpass_field(
            np.asarray(sample_ic_res, dtype=np.float32),
            boxsize=float(args.boxsize),
            k_cut=float(k_cut),
        )
        sampled_ic_fields.append(np.asarray(sample_ic_cut, dtype=np.float32))

    k_ic, pk_true_ic, pk_samples_ic, c_samples_ic = collect_pk_xcorr_stack(
        sample_fields=sampled_ic_fields,
        truth_field=np.asarray(true_ic_z127_cut, dtype=np.float32),
        boxsize=float(args.boxsize),
        mas_model=str(mas_model_ic),
        mas_target=str(mas_target_ic),
        pk_threads=int(args.pk_threads),
    )
    k_fin, pk_true_fin, pk_samples_fin, c_samples_fin = collect_pk_xcorr_stack(
        sample_fields=stage4_final_fields,
        truth_field=np.asarray(reference_delta_emu, dtype=np.float32),
        boxsize=float(args.boxsize),
        mas_model=str(mas_model_final),
        mas_target=str(mas_target_final),
        pk_threads=int(args.pk_threads),
    )

    ratio_ic = np.divide(
        pk_samples_ic,
        pk_true_ic[None, :],
        out=np.full_like(pk_samples_ic, np.nan),
        where=pk_true_ic[None, :] > 0.0,
    )
    ratio_fin = np.divide(
        pk_samples_fin,
        pk_true_fin[None, :],
        out=np.full_like(pk_samples_fin, np.nan),
        where=pk_true_fin[None, :] > 0.0,
    )

    fig_path = stage5_dir / "ic_and_final_pk_ratio_crosscorr.png"
    plot_pk_ratio_crosscorr_two_column(
        k_ic=np.asarray(k_ic, dtype=np.float64),
        pk_true_ic=np.asarray(pk_true_ic, dtype=np.float64),
        pk_samples_ic=np.asarray(pk_samples_ic, dtype=np.float64),
        xcorr_samples_ic=np.asarray(c_samples_ic, dtype=np.float64),
        k_final=np.asarray(k_fin, dtype=np.float64),
        pk_true_final=np.asarray(pk_true_fin, dtype=np.float64),
        pk_samples_final=np.asarray(pk_samples_fin, dtype=np.float64),
        xcorr_samples_final=np.asarray(c_samples_fin, dtype=np.float64),
        boxsize=float(args.boxsize),
        res=int(args.res),
        out_path=fig_path,
        left_title="ICs: posterior samples vs cutoff true reference IC",
        right_title="Finals: posterior resimulations vs reference emulator final",
    )

    arrays_path = stage5_dir / "ic_and_final_pk_ratio_crosscorr_arrays.npz"
    np.savez_compressed(
        arrays_path,
        k_ic=np.asarray(k_ic, dtype=np.float64),
        pk_true_ic=np.asarray(pk_true_ic, dtype=np.float64),
        pk_samples_ic=np.asarray(pk_samples_ic, dtype=np.float64),
        ratio_samples_ic=np.asarray(ratio_ic, dtype=np.float64),
        c_samples_ic=np.asarray(c_samples_ic, dtype=np.float64),
        k_final=np.asarray(k_fin, dtype=np.float64),
        pk_true_final=np.asarray(pk_true_fin, dtype=np.float64),
        pk_samples_final=np.asarray(pk_samples_fin, dtype=np.float64),
        ratio_samples_final=np.asarray(ratio_fin, dtype=np.float64),
        c_samples_final=np.asarray(c_samples_fin, dtype=np.float64),
        sample_ids=np.asarray(sampled_ic_ids, dtype=np.int64),
    )

    diagnostics_meta = {
        "num_samples": int(len(sample_files)),
        "sample_ids": [int(x) for x in sampled_ic_ids],
        "boxsize_mpc_over_h": float(args.boxsize),
        "res": int(args.res),
        "k_cut_h_per_mpc": float(k_cut),
        "k_nyq_h_per_mpc": float(k_nyq_res),
        "mas_worder": int(args.mas_worder),
        "deconvolve_mas": bool(args.deconvolve_mas),
        "mas_model_final_for_pk": str(mas_model_final),
        "mas_target_final_for_pk": str(mas_target_final),
        "mas_model_ic_for_pk": str(mas_model_ic),
        "mas_target_ic_for_pk": str(mas_target_ic),
        "figure_path": str(fig_path),
        "arrays_path": str(arrays_path),
    }
    diagnostics_meta_path = stage5_dir / "diagnostics_metadata.json"
    diagnostics_meta_path.write_text(json.dumps(diagnostics_meta, indent=2))

    manifest = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "k_cut_h_per_mpc": float(k_cut),
        "k_nyq_res_h_per_mpc": float(k_nyq_res),
        "growth_scale": float(scale),
        "stage_dirs": {
            "stage1_reference_ic": str(stage1_dir),
            "stage2_reference_forward": str(stage2_dir),
            "stage3_inference": str(stage3_dir),
            "stage4_posterior_forward": str(stage4_dir),
            "stage5_diagnostics": str(stage5_dir),
        },
        "stage1_files": {key: str(path) for key, path in stage1_files.items()},
        "stage2_files": {key: str(path) for key, path in stage2_files.items()},
        "stage4_records": stage4_records,
        "stage5_files": {
            "figure": str(fig_path),
            "arrays": str(arrays_path),
            "metadata": str(diagnostics_meta_path),
        },
        "total_runtime_seconds": float(time.perf_counter() - t_all),
        "total_runtime_hms": format_hms(time.perf_counter() - t_all),
    }
    manifest_path = run_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Pipeline finished. Manifest: {manifest_path}", flush=True)
    print(f"Total runtime: {manifest['total_runtime_hms']} ({manifest['total_runtime_seconds']:.2f} s)", flush=True)


if __name__ == "__main__":
    main()
