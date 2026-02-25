"""
Plotting and summary helpers for DISCO-DJ + emulator workflows.

This module contains:
  - Slice visualization for IC/LPT/emulated density fields
  - MAS-kernel deconvolution helpers
  - P(k) comparisons against CLASS
  - Quijote target-vs-model summary diagnostics
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from classy import Class
except ImportError as exc:  # pragma: no cover - optional dependency at import time
    Class = None
    _CLASS_IMPORT_ERROR = exc
else:
    _CLASS_IMPORT_ERROR = None

try:
    from discodj.core.scatter_and_gather import compensate_mak, interpolate_field
except ImportError as exc:  # pragma: no cover - optional dependency at import time
    interpolate_field = None
    compensate_mak = None
    _DISCODJ_SCATTER_GATHER_IMPORT_ERROR = exc
else:
    _DISCODJ_SCATTER_GATHER_IMPORT_ERROR = None

try:
    import Pk_library as PKL
except ImportError as exc:  # pragma: no cover - optional dependency at import time
    PKL = None
    _PKL_IMPORT_ERROR = exc
else:
    _PKL_IMPORT_ERROR = None

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
except ImportError as exc:  # pragma: no cover - optional dependency at import time
    jax = None
    jnp = None
    jrandom = None
    _JAX_IMPORT_ERROR = exc
else:
    _JAX_IMPORT_ERROR = None


QUIJOTE_FIDUCIAL_CLASS = {
    "Omega_cdm": 0.3175 - 0.0490,
    "Omega_b": 0.0490,
    "h": 0.6711,
    "n_s": 0.9624,
    "sigma8": 0.8340,
}


def growth_D_approx(cosmo_params: dict[str, float], z: float) -> float:
    """Compute Quijote's approximate linear growth factor D(z)."""
    om0_m = float(cosmo_params["Omega_cdm"]) + float(cosmo_params["Omega_b"])
    om0_l = 1.0 - om0_m
    zp1 = 1.0 + float(z)
    om_m = om0_m * (zp1**3) / (om0_l + om0_m * (zp1**3))
    om_l = om0_l / (om0_l + om0_m * (zp1**3))
    growth = (zp1 ** (-1.0)) * (5.0 * om_m / 2.0) / (
        om_m ** (4.0 / 7.0) - om_l + (1.0 + om_m / 2.0) * (1.0 + om_l / 70.0)
    )
    return float(growth)


def parse_ndiv(value: str) -> tuple[int, int, int]:
    """Parse `--ndiv` as either one integer or a 3-tuple."""
    parts = [int(x.strip()) for x in value.strip("()").split(",") if x.strip()]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    raise argparse.ArgumentTypeError(
        f"Expected one integer or three comma-separated integers, got: {value!r}"
    )


def add_import_paths(repo_root: Path) -> None:
    """Add local repository source folder to `sys.path` for runtime imports."""
    jax_emu_src = repo_root / "src"

    if jax_emu_src.is_dir():
        sys.path.insert(0, str(jax_emu_src))


def mas_from_worder(worder: int) -> str:
    """Map DISCO-DJ mass-assignment order to the Pylians MAS name."""
    mapper = {2: "CIC", 3: "TSC", 4: "PCS"}
    if worder not in mapper:
        raise ValueError(f"Unsupported mass-assignment order: {worder}")
    return mapper[worder]


def format_hms(seconds: float) -> str:
    """Format seconds as `HHh MMm SSs`."""
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"


def deconvolve_mas_kernel(
    delta: np.ndarray,
    worder: int,
    with_jax: bool = False,
) -> np.ndarray:
    """Deconvolve DISCO-DJ MAS kernel from a gridded density field."""
    if compensate_mak is None:
        raise ImportError(
            "DISCO-DJ scatter/gather utilities are unavailable. Install DISCO-DJ or fix PYTHONPATH."
        ) from _DISCODJ_SCATTER_GATHER_IMPORT_ERROR
    delta = np.asarray(delta, dtype=np.float32)
    out = compensate_mak(delta, worder=int(worder), with_jax=bool(with_jax))
    return np.asarray(out, dtype=np.float32)


def upsample_density_with_discodj(
    delta_in: np.ndarray,
    target_res: int,
    boxsize: float,
    method: str,
) -> np.ndarray:
    """
    Upsample a periodic density field using DISCO-DJ interpolation.

    Supports integer upscale factors only (target_res = in_res * r).
    """
    delta_in = np.asarray(delta_in, dtype=np.float32)
    if delta_in.ndim != 3:
        raise ValueError(f"Input density must be 3D, got shape={delta_in.shape}")
    if not (delta_in.shape[0] == delta_in.shape[1] == delta_in.shape[2]):
        raise ValueError(f"Input density must be cubic, got shape={delta_in.shape}")

    in_res = int(delta_in.shape[0])
    if in_res == target_res:
        return delta_in
    if target_res % in_res != 0:
        raise ValueError(
            f"Cannot upsample from {in_res} to {target_res}: target must be an integer multiple."
        )
    ratio = target_res // in_res

    if interpolate_field is None:
        raise ImportError(
            "DISCO-DJ interpolation is unavailable. Install DISCO-DJ or fix PYTHONPATH."
        ) from _DISCODJ_SCATTER_GATHER_IMPORT_ERROR

    out = np.empty((target_res, target_res, target_res), dtype=np.float32)
    dvals = np.arange(ratio, dtype=np.float32) / float(ratio)

    for ix, dx in enumerate(dvals):
        for iy, dy in enumerate(dvals):
            for iz, dz in enumerate(dvals):
                shifted = interpolate_field(
                    dim=3,
                    field=delta_in,
                    dshift=[float(dx), float(dy), float(dz)],
                    boxsize=boxsize,
                    dtype_num=32,
                    with_jax=False,
                    which=method,
                )
                out[ix::ratio, iy::ratio, iz::ratio] = np.asarray(shifted, dtype=np.float32)

    return out


def _build_class_linear_pk_table(
    *,
    cosmo_params: dict[str, float],
    z: float,
    boxsize: float,
    target_res: int,
    n_modes: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a tabulated linear CLASS P(k) on a k-range suited for the target grid."""
    k_fund = 2.0 * np.pi / float(boxsize)
    k_nyq = np.pi * int(target_res) / float(boxsize)
    kmin = min(0.5 * k_fund, 1.0e-5)
    kmax = max(2.0 * k_nyq * np.sqrt(3.0), 10.0)
    k_tab, pk_tab = get_pk_class(
        cosmo_params=cosmo_params,
        z=float(z),
        kmin=float(kmin),
        kmax=float(kmax),
        n_modes=int(n_modes),
        non_lin=False,
    )
    return np.asarray(k_tab, dtype=np.float64), np.asarray(pk_tab, dtype=np.float64)


def _upsample_density_mode_inject_numpy(
    *,
    delta_in: np.ndarray,
    target_res: int,
    boxsize: float,
    k_target: np.ndarray | None,
    pk_target: np.ndarray | None,
    pk_func: Callable[[np.ndarray], np.ndarray] | None,
    seed: int | None,
) -> np.ndarray:
    """Numpy backend for mode-injection upsampling."""
    in_res = int(delta_in.shape[0])
    out_res = int(target_res)
    lowk_scale = (float(out_res) / float(in_res)) ** 3

    rng = np.random.default_rng(seed)
    volume = float(boxsize) ** 3

    # Embed coarse Fourier modes into the center of the higher-resolution grid (shifted layout).
    f_low_shift = np.fft.fftshift(np.fft.fftn(delta_in)).astype(np.complex64, copy=False)
    f_out_shift = np.empty((out_res, out_res, out_res), dtype=np.complex64)
    start = (out_res - in_res) // 2
    stop = start + in_res
    # Numpy FFT coefficients scale with grid cell volume: F_N ~ delta_cont / dV ~ N^3.
    # To preserve the same physical low-k modes on a finer grid, rescale by (N_out / N_in)^3.
    low_block = np.asarray(f_low_shift * np.float32(lowk_scale), dtype=np.complex64)

    # Build shifted k-grid components in h/Mpc units (assuming boxsize is in Mpc/h).
    k1d_shift = np.fft.fftshift(
        2.0 * np.pi * np.fft.fftfreq(out_res, d=float(boxsize) / out_res)
    ).astype(np.float32, copy=False)
    ky2z2 = (k1d_shift[:, None] ** 2 + k1d_shift[None, :] ** 2).astype(np.float32, copy=False)
    k_cut = np.float32(np.pi * in_res / float(boxsize))

    # Spherical low-k mask in the embedded coarse cube: copy only |k| <= k_Nyq(coarse).
    low_mask_subcube = np.zeros((in_res, in_res, in_res), dtype=bool)
    for ix in range(start, stop):
        kmag_sub = np.sqrt(
            float(k1d_shift[ix]) ** 2 + ky2z2[start:stop, start:stop]
        ).astype(np.float32, copy=False)
        low_mask_subcube[ix - start, :, :] = kmag_sub <= k_cut

    slope = None
    intercept = None
    k_hi = None
    if pk_func is None:
        tail = min(8, int(k_target.size))
        slope, intercept = np.polyfit(np.log(k_target[-tail:]), np.log(pk_target[-tail:]), 1)
        k_hi = float(k_target[-1])

    # Fill the whole Fourier cube with random modes slice-by-slice to reduce peak memory.
    for ix, kx_val in enumerate(k1d_shift):
        kmag = np.sqrt(float(kx_val) ** 2 + ky2z2).astype(np.float64, copy=False)
        if pk_func is not None:
            pk_slice = np.asarray(pk_func(kmag), dtype=np.float64)
        else:
            pk_slice = np.interp(kmag.ravel(), k_target, pk_target, left=pk_target[0], right=pk_target[-1]).reshape(
                kmag.shape
            )
            high = kmag > k_hi
            if np.any(high):
                pk_slice[high] = np.exp(intercept + slope * np.log(kmag[high]))
        pk_slice = np.maximum(pk_slice, 0.0)

        sigma_slice = (out_res**3) * np.sqrt(pk_slice / volume)
        re = rng.normal(size=(out_res, out_res))
        im = rng.normal(size=(out_res, out_res))
        # We later project to a real field, which symmetrizes Fourier pairs and halves
        # coefficient variance for generic k/-k pairs. Draw raw complex modes with 2x
        # target variance so the post-projection variance matches target P(k).
        f_out_shift[ix, :, :] = np.asarray((re + 1j * im) * sigma_slice, dtype=np.complex64)

    # Restore copied spherical low-k modes exactly.
    out_subcube = f_out_shift[start:stop, start:stop, start:stop]
    out_subcube = np.where(low_mask_subcube, low_block, out_subcube)
    f_out_shift[start:stop, start:stop, start:stop] = out_subcube

    # Project to a real field to enforce Hermitian symmetry exactly.
    f_out = np.fft.ifftshift(f_out_shift)
    delta_real = np.fft.ifftn(f_out).real.astype(np.float32, copy=False)
    f_out_shift = np.fft.fftshift(np.fft.fftn(delta_real)).astype(np.complex64, copy=False)
    out_subcube = f_out_shift[start:stop, start:stop, start:stop]
    out_subcube = np.where(low_mask_subcube, low_block, out_subcube)
    f_out_shift[start:stop, start:stop, start:stop] = out_subcube

    return np.asarray(np.fft.ifftn(np.fft.ifftshift(f_out_shift)).real, dtype=np.float32)


def _upsample_density_mode_inject_jax(
    *,
    delta_in: np.ndarray,
    target_res: int,
    boxsize: float,
    k_target: np.ndarray,
    pk_target: np.ndarray,
    seed: int | None,
) -> np.ndarray:
    """JAX backend for mode-injection upsampling (GPU/CPU depending on active JAX backend)."""
    if jax is None or jnp is None or jrandom is None:
        raise ImportError("JAX is unavailable for mode injection backend.") from _JAX_IMPORT_ERROR

    in_res = int(delta_in.shape[0])
    out_res = int(target_res)
    lowk_scale = (float(out_res) / float(in_res)) ** 3
    volume = float(boxsize) ** 3

    delta = jnp.asarray(delta_in, dtype=jnp.float32)
    k_tab = jnp.asarray(k_target, dtype=jnp.float32)
    pk_tab = jnp.asarray(pk_target, dtype=jnp.float32)

    start = (out_res - in_res) // 2
    stop = start + in_res

    # Match coarse-grid Fourier coefficients to fine-grid FFT normalization (F_N ~ N^3).
    f_low_shift = (
        jnp.fft.fftshift(jnp.fft.fftn(delta)).astype(jnp.complex64)
        * jnp.asarray(lowk_scale, dtype=jnp.float32)
    )
    low_embedded = jnp.zeros((out_res, out_res, out_res), dtype=jnp.complex64)
    low_embedded = low_embedded.at[start:stop, start:stop, start:stop].set(f_low_shift)

    k1d = jnp.fft.fftshift(2.0 * jnp.pi * jnp.fft.fftfreq(out_res, d=float(boxsize) / out_res)).astype(jnp.float32)
    kx = k1d[:, None, None]
    ky = k1d[None, :, None]
    kz = k1d[None, None, :]
    kmag = jnp.sqrt(kx * kx + ky * ky + kz * kz)
    k_cut = jnp.asarray(np.pi * in_res / float(boxsize), dtype=jnp.float32)

    i = jnp.arange(out_res)
    j = jnp.arange(out_res)
    k = jnp.arange(out_res)
    cube_i = (i >= start) & (i < stop)
    cube_j = (j >= start) & (j < stop)
    cube_k = (k >= start) & (k < stop)
    cube_mask = cube_i[:, None, None] & cube_j[None, :, None] & cube_k[None, None, :]
    # Spherical low-k mask in the embedded coarse cube.
    low_mask = cube_mask & (kmag <= k_cut)
    kflat = kmag.reshape(-1)

    pk_flat = jnp.interp(kflat, k_tab, pk_tab, left=pk_tab[0], right=pk_tab[-1])
    tail = min(8, int(k_target.size))
    slope, intercept = np.polyfit(np.log(k_target[-tail:]), np.log(pk_target[-tail:]), 1)
    k_hi = float(k_target[-1])
    high = kflat > k_hi
    pk_high = jnp.exp(jnp.asarray(intercept, dtype=jnp.float32) + jnp.asarray(slope, dtype=jnp.float32) * jnp.log(kflat))
    pk_flat = jnp.where(high, pk_high, pk_flat)
    pk_grid = jnp.maximum(pk_flat.reshape((out_res, out_res, out_res)), 0.0)

    sigma = (out_res**3) * jnp.sqrt(pk_grid / volume)
    key = jrandom.PRNGKey(0 if seed is None else int(seed))
    key_re, key_im = jrandom.split(key)
    re = jrandom.normal(key_re, sigma.shape, dtype=jnp.float32)
    im = jrandom.normal(key_im, sigma.shape, dtype=jnp.float32)
    # Match target variance after real-field projection (which halves generic pair variance).
    random_modes = (re + 1j * im) * sigma

    f_out_shift = jnp.where(low_mask, low_embedded, random_modes.astype(jnp.complex64))

    # Project to a real field to enforce Hermitian symmetry exactly, then restore low-k block.
    delta_real = jnp.fft.ifftn(jnp.fft.ifftshift(f_out_shift)).real.astype(jnp.float32)
    f_out_shift = jnp.fft.fftshift(jnp.fft.fftn(delta_real)).astype(jnp.complex64)
    f_out_shift = jnp.where(low_mask, low_embedded, f_out_shift)
    delta_out = jnp.fft.ifftn(jnp.fft.ifftshift(f_out_shift)).real.astype(jnp.float32)

    return np.asarray(jax.device_get(delta_out), dtype=np.float32)


def upsample_density_mode_inject(
    delta_in: np.ndarray,
    target_res: int,
    boxsize: float,
    *,
    k_target: np.ndarray | None = None,
    pk_target: np.ndarray | None = None,
    pk_func: Callable[[np.ndarray], np.ndarray] | None = None,
    class_cosmo_params: dict[str, float] | None = None,
    class_z: float = 0.0,
    class_n_modes: int = 4096,
    backend: str = "auto",
    seed: int | None = None,
) -> np.ndarray:
    """
    Upsample a GRF using conditional Gaussian refinement with Fourier mode injection.

    Steps:
      1) Copy low-k modes from the coarse realization using a spherical mask
         (|k| <= k_Nyq of the coarse grid).
      2) Sample new high-k modes from a target P(k).
      3) Impose Hermitian symmetry (project to a real field).
      4) Return the inverse-FFT real-space field.

    Default target P(k):
      If no `pk_func` or (`k_target`, `pk_target`) is provided, builds a tabulated
      CLASS linear P(k) at `class_z` with `class_cosmo_params` (Quijote fiducial by default).

    backend:
      "auto" (default) uses JAX if available and `pk_func` is not used; otherwise numpy.
      "jax" forces JAX backend (requires `k_target/pk_target` and JAX installation).
      "numpy" forces numpy backend.
    """
    delta_in = np.asarray(delta_in, dtype=np.float32)
    if delta_in.ndim != 3:
        raise ValueError(f"Input density must be 3D, got shape={delta_in.shape}")
    if not (delta_in.shape[0] == delta_in.shape[1] == delta_in.shape[2]):
        raise ValueError(f"Input density must be cubic, got shape={delta_in.shape}")

    in_res = int(delta_in.shape[0])
    out_res = int(target_res)
    if out_res == in_res:
        return np.asarray(delta_in, dtype=np.float32)
    if out_res < in_res:
        raise ValueError(f"`target_res` must be >= input resolution ({in_res}), got {out_res}.")
    if out_res % in_res != 0:
        raise ValueError(
            f"Cannot upsample from {in_res} to {out_res}: target must be an integer multiple."
        )
    if backend not in {"auto", "jax", "numpy"}:
        raise ValueError("`backend` must be one of: 'auto', 'jax', 'numpy'.")

    has_table = (k_target is not None) or (pk_target is not None)
    has_func = pk_func is not None
    if has_table and has_func:
        raise ValueError("Provide either (`k_target`, `pk_target`) or `pk_func`, not both.")
    if has_table:
        if k_target is None or pk_target is None:
            raise ValueError("Both `k_target` and `pk_target` are required when using tabulated P(k).")
        k_target = np.asarray(k_target, dtype=np.float64).ravel()
        pk_target = np.asarray(pk_target, dtype=np.float64).ravel()
        if k_target.size < 2 or pk_target.size != k_target.size:
            raise ValueError("Invalid tabulated P(k): need matching arrays with at least two points.")
        order = np.argsort(k_target)
        k_target = k_target[order]
        pk_target = pk_target[order]
        if np.any(np.diff(k_target) <= 0):
            raise ValueError("`k_target` values must be strictly increasing.")
    elif not has_func:
        # Default behavior: use a CLASS-based callable linear P(k).
        if class_cosmo_params is None:
            class_cosmo_params = QUIJOTE_FIDUCIAL_CLASS
        k_target, pk_target = _build_class_linear_pk_table(
            cosmo_params=class_cosmo_params,
            z=float(class_z),
            boxsize=float(boxsize),
            target_res=out_res,
            n_modes=int(class_n_modes),
        )
    use_jax = backend == "jax" or (backend == "auto" and (jax is not None) and (not has_func))
    if use_jax:
        if k_target is None or pk_target is None:
            raise ValueError("JAX backend requires tabulated target P(k): provide `k_target` and `pk_target`.")
        return _upsample_density_mode_inject_jax(
            delta_in=delta_in,
            target_res=out_res,
            boxsize=float(boxsize),
            k_target=np.asarray(k_target, dtype=np.float64),
            pk_target=np.asarray(pk_target, dtype=np.float64),
            seed=seed,
        )

    return _upsample_density_mode_inject_numpy(
        delta_in=delta_in,
        target_res=out_res,
        boxsize=float(boxsize),
        k_target=None if k_target is None else np.asarray(k_target, dtype=np.float64),
        pk_target=None if pk_target is None else np.asarray(pk_target, dtype=np.float64),
        pk_func=pk_func,
        seed=seed,
    )


def downsample_density_block_average(
    delta_in: np.ndarray,
    target_res: int,
) -> np.ndarray:
    """
    Downsample a cubic periodic density field by block averaging.
    """
    delta_in = np.asarray(delta_in, dtype=np.float32)
    in_res = int(delta_in.shape[0])
    if in_res == target_res:
        return delta_in
    if in_res % target_res != 0:
        raise ValueError(
            f"Cannot downsample from {in_res} to {target_res}: input must be an integer multiple."
        )
    ratio = in_res // target_res
    out = delta_in.reshape(
        target_res,
        ratio,
        target_res,
        ratio,
        target_res,
        ratio,
    ).mean(axis=(1, 3, 5))
    return np.asarray(out, dtype=np.float32)


def resize_density_grid(
    delta_in: np.ndarray,
    target_res: int,
    boxsize: float,
    upsample_method: str,
    class_cosmo_params: dict[str, float] | None = None,
    class_z: float = 0.0,
    class_n_modes: int = 4096,
) -> np.ndarray:
    """
    Resize a cubic periodic density field between resolutions.
    Upsampling can use DISCO-DJ interpolation or mode injection; downsampling uses block averaging.
    """
    in_res = int(np.asarray(delta_in).shape[0])
    if in_res == target_res:
        return np.asarray(delta_in, dtype=np.float32)
    if target_res > in_res:
        if upsample_method == "mode_inject":
            return upsample_density_mode_inject(
                delta_in=delta_in,
                target_res=target_res,
                boxsize=boxsize,
                class_cosmo_params=class_cosmo_params,
                class_z=float(class_z),
                class_n_modes=int(class_n_modes),
            )
        if upsample_method not in {"fourier", "linear"}:
            raise ValueError(
                f"Unknown upsample method: {upsample_method!r}. "
                "Expected one of: 'mode_inject', 'fourier', 'linear'."
            )
        return upsample_density_with_discodj(
            delta_in=delta_in,
            target_res=target_res,
            boxsize=boxsize,
            method=upsample_method,
        )
    return downsample_density_block_average(delta_in=delta_in, target_res=target_res)


def plot_density_slices(
    ics_grf: np.ndarray,
    delta_init: np.ndarray,
    delta_emu: np.ndarray,
    out_path: Path,
    slice_indices: list[int] | None = None,
) -> None:
    """
    Plot slices of initial and emulated density fields.
    """
    ics_grf = np.asarray(ics_grf, dtype=np.float32)
    delta_init = np.asarray(delta_init, dtype=np.float32)
    delta_emu = np.asarray(delta_emu, dtype=np.float32)

    n = ics_grf.shape[0]
    if delta_init.shape[0] != n or delta_emu.shape[0] != n:
        raise ValueError(
            "All fields must have the same grid size in the first axis "
            f"(got {ics_grf.shape}, {delta_init.shape}, {delta_emu.shape})."
        )
    if slice_indices is None:
        slice_indices = [n // 4, n // 2, (3 * n) // 4]
    slice_indices = sorted(set(int(i) for i in slice_indices))

    for sidx in slice_indices:
        if sidx < 0 or sidx >= n:
            raise ValueError(f"Slice index {sidx} is out of bounds for res={n}.")

    rows = [
        ("ICs GRF (linear z=0)", ics_grf, "seismic", -3.0, 3.0),
        ("LPT field", delta_init, "inferno", 0.0, 15.0),
        ("Emulated field", delta_emu, "inferno", 0.0, 15.0),
    ]

    fig, axes = plt.subplots(
        len(rows),
        len(slice_indices),
        figsize=(3.4 * len(slice_indices), 3.6 * len(rows)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(slice_indices) == 1:
        axes = np.asarray(axes).reshape(len(rows), 1)

    for row_idx, (label, field, cmap, vmin, vmax) in enumerate(rows):
        for col_idx, sidx in enumerate(slice_indices):
            imshow_kwargs = {
                "origin": "lower",
                "cmap": cmap,
            }
            if vmin is not None and vmax is not None:
                imshow_kwargs["vmin"] = vmin
                imshow_kwargs["vmax"] = vmax
            im = axes[row_idx, col_idx].imshow(
                field[sidx, :, :],
                **imshow_kwargs,
            )
            axes[row_idx, col_idx].set_title(f"{label}\nslice x={sidx}")
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel("y")
            if row_idx == len(rows) - 1:
                axes[row_idx, col_idx].set_xlabel("z")
        fig.colorbar(im, ax=list(axes[row_idx, :]), shrink=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_quijote_emulator_slices(
    delta_target: np.ndarray,
    delta_lpt: np.ndarray,
    delta_emu: np.ndarray,
    out_path: Path,
    slice_indices: list[int] | None = None,
) -> None:
    """Plot target/LPT/emulated slices plus a residual row."""
    delta_target = np.asarray(delta_target, dtype=np.float32)
    delta_lpt = np.asarray(delta_lpt, dtype=np.float32)
    delta_emu = np.asarray(delta_emu, dtype=np.float32)

    n = delta_target.shape[0]
    if delta_lpt.shape != delta_target.shape or delta_emu.shape != delta_target.shape:
        raise ValueError(
            "All fields must have identical shape for slice plotting "
            f"(got target={delta_target.shape}, lpt={delta_lpt.shape}, emu={delta_emu.shape})."
        )

    if slice_indices is None:
        slice_indices = [n // 4, n // 2, (3 * n) // 4]
    slice_indices = sorted(set(int(i) for i in slice_indices))

    residual = delta_emu - delta_target
    rows = [
        ("Quijote target", delta_target, "inferno", 0.0, 15.0),
        ("LPT field", delta_lpt, "inferno", 0.0, 15.0),
        ("Emulated field", delta_emu, "inferno", 0.0, 15.0),
        ("Emulated - target", residual, "seismic", -3.0, 3.0),
    ]

    fig, axes = plt.subplots(
        len(rows),
        len(slice_indices),
        figsize=(3.4 * len(slice_indices), 3.4 * len(rows)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(slice_indices) == 1:
        axes = np.asarray(axes).reshape(len(rows), 1)

    for row_idx, (label, field, cmap, vmin, vmax) in enumerate(rows):
        for col_idx, sidx in enumerate(slice_indices):
            im = axes[row_idx, col_idx].imshow(
                field[sidx, :, :],
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            axes[row_idx, col_idx].set_title(f"{label}\nslice x={sidx}")
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel("y")
            if row_idx == len(rows) - 1:
                axes[row_idx, col_idx].set_xlabel("z")
        fig.colorbar(im, ax=list(axes[row_idx, :]), shrink=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def get_pk_class(
    cosmo_params: dict,
    z: float,
    k: np.ndarray | None = None,
    *,
    kmin: float | None = None,
    kmax: float | None = None,
    n_modes: int | None = None,
    non_lin: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute theory P(k) in (Mpc/h)^3 from CLASS.

    Two modes are supported:
      1) Provide `k` (array in h/Mpc) -> returns `pk` on that grid.
      2) Provide `kmin` and `kmax` (and optional `n_modes`) -> returns `(k, pk)`.
    """
    if Class is None:
        raise ImportError("classy is unavailable. Install CLASS python bindings.") from _CLASS_IMPORT_ERROR

    if k is not None:
        if kmin is not None or kmax is not None or n_modes is not None:
            raise ValueError("Provide either `k` OR (`kmin`, `kmax`, `n_modes`), not both.")
        k_eval = np.asarray(k, dtype=np.float64)
        return_with_k = False
    else:
        if kmin is None or kmax is None:
            raise ValueError("When `k` is not provided, both `kmin` and `kmax` are required.")
        n_modes = 256 if n_modes is None else int(n_modes)
        if n_modes < 2:
            raise ValueError("`n_modes` must be >= 2.")
        k_eval = np.geomspace(float(kmin), float(kmax), n_modes).astype(np.float64)
        return_with_k = True

    h = float(cosmo_params["h"])
    params = dict(cosmo_params)
    params.update(
        {
            "output": "mPk",
            "P_k_max_h/Mpc": float(np.max(k_eval)),
            "z_max_pk": float(z),
        }
    )
    if non_lin:
        params["non linear"] = "halofit"

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    if non_lin:
        pk = h**3 * np.asarray([cosmo.pk(h * ki, z) for ki in k_eval], dtype=np.float64)
    else:
        pk = h**3 * np.asarray([cosmo.pk_lin(h * ki, z) for ki in k_eval], dtype=np.float64)

    cosmo.struct_cleanup()
    cosmo.empty()
    if return_with_k:
        return k_eval, pk
    return pk


def plot_density_power_spectra(
    delta_grf: np.ndarray,
    delta_init: np.ndarray,
    delta_emu: np.ndarray,
    boxsize: float,
    out_path: Path,
    class_cosmo_params: dict,
    z: float = 0.0,
    mas: str = "CIC",
    mas_deconvolved_fields: bool = False,
    non_linear_theory: bool = False,
) -> None:
    """
    Plot P(k) of GRF, LPT, and emulated density fields and compare to CLASS linear and Halofit.

    If `mas_deconvolved_fields=True`, model fields are treated as already MAS-deconvolved.
    """
    if PKL is None:
        raise ImportError("Pk_library is unavailable. Install Pylians3.") from _PKL_IMPORT_ERROR

    delta_grf = np.asarray(delta_grf, dtype=np.float32)
    delta_init = np.asarray(delta_init, dtype=np.float32)
    delta_emu = np.asarray(delta_emu, dtype=np.float32)

    mas_for_model = "None" if mas_deconvolved_fields else mas

    # GRF is already a grid field (not particle-deposited), so no MAS deconvolution here.
    pk_grf_obj = PKL.Pk(delta_grf, boxsize, axis=0, MAS="None", threads=1, verbose=False)
    pk_init_obj = PKL.Pk(delta_init, boxsize, axis=0, MAS=mas_for_model, threads=1, verbose=False)
    pk_emu_obj = PKL.Pk(delta_emu, boxsize, axis=0, MAS=mas_for_model, threads=1, verbose=False)

    k = np.asarray(pk_init_obj.k3D, dtype=np.float64)
    pk_grf = np.asarray(pk_grf_obj.Pk[:, 0], dtype=np.float64)
    pk_init = np.asarray(pk_init_obj.Pk[:, 0], dtype=np.float64)
    pk_emu = np.asarray(pk_emu_obj.Pk[:, 0], dtype=np.float64)

    valid = k > 0
    k = k[valid]
    pk_grf = pk_grf[valid]
    pk_init = pk_init[valid]
    pk_emu = pk_emu[valid]

    k_theory = np.logspace(np.log10(max(1e-4, float(k.min()))), np.log10(10.0), 256)
    pk_theory_lin = get_pk_class(class_cosmo_params, z=z, k=k_theory, non_lin=False)
    pk_theory_nl = get_pk_class(class_cosmo_params, z=z, k=k_theory, non_lin=True)
    pk_theory_lin_at_k = np.interp(k, k_theory, pk_theory_lin)
    pk_theory_nl_at_k = np.interp(k, k_theory, pk_theory_nl)

    n = delta_init.shape[0]
    k_nyq = np.pi * n / boxsize

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.4),
        sharex=True,
        constrained_layout=True,
        height_ratios=(2, 1),
    )

    ax = axes[0]
    ax.plot(k, pk_grf, lw=1.2, label="GRF density")
    ax.plot(k, pk_init, lw=1.2, label="LPT density")
    ax.plot(k, pk_emu, lw=1.2, label="Emulated density")
    ax.plot(
        k_theory,
        pk_theory_lin,
        lw=1.2,
        ls="--",
        color="black",
        alpha=0.8,
        label="CLASS linear",
    )
    ax.plot(
        k_theory,
        pk_theory_nl,
        lw=1.2,
        ls="-.",
        color="dimgray",
        alpha=0.9,
        label="CLASS Halofit",
    )
    ax.axvline(x=k_nyq, color="red", ls="--", lw=0.8, alpha=0.6, label=r"$k_{\rm Nyq}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=2.0e2)
    ax.set_ylabel(r"$P(k)$")
    ax.grid(which="both", alpha=0.15)
    ax.legend(framealpha=0.9, loc="lower right")

    ax = axes[1]
    ax.plot(k, pk_grf / pk_theory_lin_at_k, lw=1.2, label="GRF / CLASS linear")
    ax.plot(k, pk_init / pk_theory_lin_at_k, lw=1.2, label="LPT / CLASS linear")
    ax.plot(k, pk_emu / pk_theory_nl_at_k, lw=1.2, label="Emulated / CLASS Halofit")
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.axvline(x=k_nyq, color="red", ls="--", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [$h/{\rm Mpc}$]")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0.0, 2.0)
    ax.grid(which="both", alpha=0.15)
    ax.legend(framealpha=0.9, loc="lower right", bbox_to_anchor=(0.98, 0.03))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _field_moments(field: np.ndarray) -> dict[str, float]:
    """Compute mean, std, skewness, and excess kurtosis for one field."""
    x = np.asarray(field, dtype=np.float64).ravel()
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma <= 0.0:
        return {
            "mean": mu,
            "std": sigma,
            "skewness": 0.0,
            "kurtosis_excess": 0.0,
        }

    xc = x - mu
    m3 = float(np.mean(xc**3))
    m4 = float(np.mean(xc**4))
    skewness = m3 / (sigma**3)
    kurtosis_excess = m4 / (sigma**4) - 3.0
    return {
        "mean": mu,
        "std": sigma,
        "skewness": float(skewness),
        "kurtosis_excess": float(kurtosis_excess),
    }


def _plot_target_lpt_emu_slices(
    delta_target: np.ndarray,
    delta_lpt: np.ndarray,
    delta_emu: np.ndarray,
    out_path: Path,
    slice_indices: list[int] | None = None,
) -> None:
    """Plot target, LPT, and emulated density slices on matched color limits."""
    delta_target = np.asarray(delta_target, dtype=np.float32)
    delta_lpt = np.asarray(delta_lpt, dtype=np.float32)
    delta_emu = np.asarray(delta_emu, dtype=np.float32)

    n = int(delta_target.shape[0])
    if slice_indices is None:
        slice_indices = [n // 4, n // 2, (3 * n) // 4]
    slice_indices = sorted(set(int(i) for i in slice_indices))

    vmin = float(np.percentile(np.concatenate([delta_target.ravel(), delta_lpt.ravel(), delta_emu.ravel()]), 1.0))
    vmax = float(np.percentile(np.concatenate([delta_target.ravel(), delta_lpt.ravel(), delta_emu.ravel()]), 99.0))

    rows = [
        ("Quijote target", delta_target),
        ("LPT field", delta_lpt),
        ("Emulated field", delta_emu),
    ]

    fig, axes = plt.subplots(
        len(rows),
        len(slice_indices),
        figsize=(3.4 * len(slice_indices), 3.5 * len(rows)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(slice_indices) == 1:
        axes = np.asarray(axes).reshape(len(rows), 1)

    for row_idx, (label, field) in enumerate(rows):
        for col_idx, sidx in enumerate(slice_indices):
            im = axes[row_idx, col_idx].imshow(
                field[sidx, :, :],
                origin="lower",
                cmap="inferno",
                vmin=vmin,
                vmax=vmax,
            )
            axes[row_idx, col_idx].set_title(f"{label}\nslice x={sidx}")
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel("y")
            if row_idx == len(rows) - 1:
                axes[row_idx, col_idx].set_xlabel("z")
        fig.colorbar(im, ax=list(axes[row_idx, :]), shrink=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_1pt_pdf_target_lpt_emu(
    delta_target: np.ndarray,
    delta_lpt: np.ndarray,
    delta_emu: np.ndarray,
    out_path: Path,
    nbins: int = 120,
) -> None:
    """Plot the one-point PDF comparison for target, LPT, and emulator fields."""
    delta_target = np.asarray(delta_target, dtype=np.float64).ravel()
    delta_lpt = np.asarray(delta_lpt, dtype=np.float64).ravel()
    delta_emu = np.asarray(delta_emu, dtype=np.float64).ravel()

    all_vals = np.concatenate([delta_target, delta_lpt, delta_emu])
    lo = float(np.percentile(all_vals, 0.1))
    hi = float(np.percentile(all_vals, 99.9))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -5.0, 5.0

    bins = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    h_target, _ = np.histogram(delta_target[np.isfinite(delta_target)], bins=bins, density=True)
    h_lpt, _ = np.histogram(delta_lpt[np.isfinite(delta_lpt)], bins=bins, density=True)
    h_emu, _ = np.histogram(delta_emu[np.isfinite(delta_emu)], bins=bins, density=True)

    mt = _field_moments(delta_target)
    ml = _field_moments(delta_lpt)
    me = _field_moments(delta_emu)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=160)
    ax.plot(centers, h_target, lw=1.8, color="black", alpha=0.75, label="Quijote target")
    ax.plot(centers, h_lpt, lw=1.4, color="tab:blue", alpha=0.75, label="LPT")
    ax.plot(centers, h_emu, lw=1.4, color="tab:orange", alpha=0.75, label="Emulator")
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel("PDF")
    ax.set_title("1-point PDF comparison")
    ax.legend(framealpha=0.9, loc="lower right", bbox_to_anchor=(0.98, 0.03))
    ax.grid(alpha=0.15)

    col_labels = ["Statistic", "Target", "Emu", "LPT"]
    table_values = [
        ["mean", f"{mt['mean']:.3e}", f"{me['mean']:.3e}", f"{ml['mean']:.3e}"],
        ["std", f"{mt['std']:.3f}", f"{me['std']:.3f}", f"{ml['std']:.3f}"],
        ["skewness", f"{mt['skewness']:.3f}", f"{me['skewness']:.3f}", f"{ml['skewness']:.3f}"],
        [
            "kurtosis_excess",
            f"{mt['kurtosis_excess']:.3f}",
            f"{me['kurtosis_excess']:.3f}",
            f"{ml['kurtosis_excess']:.3f}",
        ],
    ]
    table = ax.table(
        cellText=table_values,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        colWidths=[0.34, 0.22, 0.22, 0.22],
        bbox=[0.46, 0.54, 0.52, 0.42],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_bispectrum_target_lpt_emu(
    delta_target: np.ndarray,
    delta_lpt: np.ndarray,
    delta_emu: np.ndarray,
    boxsize: float,
    mas: str,
    mas_deconvolved_model_fields: bool,
    mas_deconvolved_target_field: bool,
    out_path: Path,
) -> dict[str, float]:
    """Compare reduced bispectra Q(theta) for target, LPT, and emulator fields."""
    if PKL is None:
        raise ImportError("Pk_library is unavailable. Install Pylians3.") from _PKL_IMPORT_ERROR

    delta_target = np.asarray(delta_target, dtype=np.float32)
    delta_lpt = np.asarray(delta_lpt, dtype=np.float32)
    delta_emu = np.asarray(delta_emu, dtype=np.float32)

    theta = np.linspace(0.0, np.pi, 25)
    theta_deg = np.degrees(theta)
    bispec_configs = [
        {"k1": 0.1, "k2": 0.1, "label": r"$k_1 = k_2 = 0.1\;h/\mathrm{Mpc}$"},
        {"k1": 0.05, "k2": 0.1, "label": r"$k_1 = 0.05,\; k_2 = 0.1\;h/\mathrm{Mpc}$"},
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 7.2),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    metrics: dict[str, float] = {}
    mas_for_model = "None" if mas_deconvolved_model_fields else mas
    mas_for_target = "None" if mas_deconvolved_target_field else mas
    for idx, cfg in enumerate(bispec_configs, start=1):
        col = idx - 1
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        q_target = np.asarray(
            PKL.Bk(delta_target, boxsize, cfg["k1"], cfg["k2"], theta, mas_for_target, threads=1).Q
        )
        q_lpt = np.asarray(PKL.Bk(delta_lpt, boxsize, cfg["k1"], cfg["k2"], theta, mas_for_model, threads=1).Q)
        q_emu = np.asarray(PKL.Bk(delta_emu, boxsize, cfg["k1"], cfg["k2"], theta, mas_for_model, threads=1).Q)

        ax_top.plot(theta_deg, q_target, lw=1.4, color="black", alpha=0.75, label="Quijote target")
        ax_top.plot(theta_deg, q_lpt, lw=1.2, color="tab:blue", alpha=0.75, label="LPT")
        ax_top.plot(theta_deg, q_emu, lw=1.2, color="tab:orange", alpha=0.75, label="Emulator")
        ax_top.set_title(cfg["label"], fontsize=11)
        ax_top.grid(alpha=0.15)
        ax_top.legend(framealpha=0.9, fontsize=8)

        eps = 1.0e-12
        ratio_lpt = np.divide(
            q_lpt,
            q_target,
            out=np.full_like(q_lpt, np.nan, dtype=np.float64),
            where=np.abs(q_target) > eps,
        )
        ratio_emu = np.divide(
            q_emu,
            q_target,
            out=np.full_like(q_emu, np.nan, dtype=np.float64),
            where=np.abs(q_target) > eps,
        )

        ax_bot.plot(theta_deg, ratio_lpt, lw=1.2, color="tab:blue", label="LPT / target")
        ax_bot.plot(theta_deg, ratio_emu, lw=1.2, color="tab:orange", label="Emulator / target")
        ax_bot.axhline(1.0, color="black", ls="--", lw=0.8)
        ax_bot.set_ylim(0.5, 1.5)
        ax_bot.set_xlabel(r"$\theta$ [deg]")
        ax_bot.grid(alpha=0.15)
        ax_bot.legend(framealpha=0.9, fontsize=8)

        metrics[f"bispectrum_cfg{idx}_mae_lpt_vs_target"] = float(np.mean(np.abs(q_lpt - q_target)))
        metrics[f"bispectrum_cfg{idx}_mae_emu_vs_target"] = float(np.mean(np.abs(q_emu - q_target)))

    axes[0, 0].set_ylabel(r"$Q(\theta)$")
    axes[1, 0].set_ylabel("Ratio")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return metrics


def plot_emulator_vs_target_summary(
    delta_target: np.ndarray,
    delta_lpt: np.ndarray,
    delta_emu: np.ndarray,
    boxsize: float,
    out_path: Path,
    pdf_out_path: Path | None = None,
    bispec_out_path: Path | None = None,
    mas: str = "None",
    mas_deconvolved_fields: bool = False,
    mas_deconvolved_target_field: bool | None = None,
    class_cosmo_params: dict | None = None,
    z: float = 0.0,
) -> dict[str, object]:
    """
    Compare emulator and LPT fields to a target field with summary statistics.

    The saved figure has three panels:
      1) P(k)
      2) transfer function T(k)=sqrt(P_pred/P_target)
      3) cross-correlation C(k)=P_cross/sqrt(P_pred*P_target)

    If `mas_deconvolved_fields=True`, model fields (`delta_lpt`, `delta_emu`) are treated as
    already MAS-deconvolved. `mas_deconvolved_target_field` can override this convention for
    `delta_target`; if omitted, it follows `mas_deconvolved_fields`.
    """
    if PKL is None:
        raise ImportError("Pk_library is unavailable. Install Pylians3.") from _PKL_IMPORT_ERROR

    delta_target = np.asarray(delta_target, dtype=np.float32)
    delta_lpt = np.asarray(delta_lpt, dtype=np.float32)
    delta_emu = np.asarray(delta_emu, dtype=np.float32)

    if delta_target.shape != delta_lpt.shape or delta_target.shape != delta_emu.shape:
        raise ValueError(
            "All fields must have identical shape for summary comparison "
            f"(got target={delta_target.shape}, lpt={delta_lpt.shape}, emu={delta_emu.shape})."
        )

    if mas_deconvolved_target_field is None:
        mas_deconvolved_target_field = bool(mas_deconvolved_fields)

    mas_for_model = "None" if mas_deconvolved_fields else mas
    mas_for_target = "None" if mas_deconvolved_target_field else mas

    pk_target_obj = PKL.Pk(delta_target, boxsize, axis=0, MAS=mas_for_target, threads=1, verbose=False)
    pk_lpt_obj = PKL.Pk(delta_lpt, boxsize, axis=0, MAS=mas_for_model, threads=1, verbose=False)
    pk_emu_obj = PKL.Pk(delta_emu, boxsize, axis=0, MAS=mas_for_model, threads=1, verbose=False)

    xpk_lpt_obj = PKL.XPk(
        [delta_lpt, delta_target], boxsize, axis=0, MAS=[mas_for_model, mas_for_target], threads=1
    )
    xpk_emu_obj = PKL.XPk(
        [delta_emu, delta_target], boxsize, axis=0, MAS=[mas_for_model, mas_for_target], threads=1
    )

    k = np.asarray(pk_target_obj.k3D, dtype=np.float64)
    pk_target = np.asarray(pk_target_obj.Pk[:, 0], dtype=np.float64)
    pk_lpt = np.asarray(pk_lpt_obj.Pk[:, 0], dtype=np.float64)
    pk_emu = np.asarray(pk_emu_obj.Pk[:, 0], dtype=np.float64)

    xcorr_lpt = np.asarray(xpk_lpt_obj.XPk[:, 0, 0], dtype=np.float64) / np.sqrt(
        np.asarray(xpk_lpt_obj.Pk[:, 0, 0], dtype=np.float64)
        * np.asarray(xpk_lpt_obj.Pk[:, 0, 1], dtype=np.float64)
    )
    xcorr_emu = np.asarray(xpk_emu_obj.XPk[:, 0, 0], dtype=np.float64) / np.sqrt(
        np.asarray(xpk_emu_obj.Pk[:, 0, 0], dtype=np.float64)
        * np.asarray(xpk_emu_obj.Pk[:, 0, 1], dtype=np.float64)
    )

    valid = (
        (k > 0)
        & np.isfinite(pk_target)
        & np.isfinite(pk_lpt)
        & np.isfinite(pk_emu)
        & np.isfinite(xcorr_lpt)
        & np.isfinite(xcorr_emu)
        & (pk_target > 0)
        & (pk_lpt > 0)
        & (pk_emu > 0)
    )
    k = k[valid]
    pk_target = pk_target[valid]
    pk_lpt = pk_lpt[valid]
    pk_emu = pk_emu[valid]
    xcorr_lpt = xcorr_lpt[valid]
    xcorr_emu = xcorr_emu[valid]

    transfer_lpt = np.sqrt(np.clip(pk_lpt / pk_target, a_min=0.0, a_max=None))
    transfer_emu = np.sqrt(np.clip(pk_emu / pk_target, a_min=0.0, a_max=None))

    n = int(delta_target.shape[0])
    k_nyq = np.pi * n / float(boxsize)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 8.6),
        sharex=True,
        constrained_layout=True,
        height_ratios=(2.2, 1.0, 1.0),
    )

    ax = axes[0]
    ax.plot(k, pk_target, lw=1.2, alpha=0.75, label="Quijote target")
    ax.plot(k, pk_lpt, lw=1.2, alpha=0.75, label="LPT")
    ax.plot(k, pk_emu, lw=1.2, alpha=0.75, label="Emulator")
    if class_cosmo_params is not None:
        k_theory_max = 2.0
        k_theory = np.logspace(np.log10(max(1e-4, float(k.min()))), np.log10(k_theory_max), 256)
        pk_lin = get_pk_class(class_cosmo_params, z=z, k=k_theory, non_lin=False)
        pk_nl = get_pk_class(class_cosmo_params, z=z, k=k_theory, non_lin=True)
        ax.plot(k_theory, pk_lin, ls="--", lw=1.1, color="black", alpha=0.8, label="CLASS linear")
        ax.plot(k_theory, pk_nl, ls="-.", lw=1.1, color="dimgray", alpha=0.9, label="CLASS Halofit")
    ax.axvline(x=k_nyq, color="red", ls="--", lw=0.8, alpha=0.6, label=r"$k_{\rm Nyq}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$P(k)$")
    ax.set_ylim(bottom=2.0e2)
    ax.grid(which="both", alpha=0.15)
    ax.legend(framealpha=0.9)

    ax = axes[1]
    ax.plot(k, transfer_lpt, lw=1.2, label="LPT / target")
    ax.plot(k, transfer_emu, lw=1.2, label="Emulator / target")
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.axvline(x=k_nyq, color="red", ls="--", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_ylabel(r"$T(k)$")
    ax.set_ylim(0.5, 1.5)
    ax.grid(which="both", alpha=0.15)
    ax.legend(framealpha=0.9)

    ax = axes[2]
    ax.plot(k, xcorr_lpt, lw=1.2, label="corr(LPT, target)")
    ax.plot(k, xcorr_emu, lw=1.2, label="corr(Emu, target)")
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.axvline(x=k_nyq, color="red", ls="--", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [$h/{\rm Mpc}$]")
    ax.set_ylabel(r"$C(k)$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(right=2.0)
    ax.grid(which="both", alpha=0.15)
    ax.legend(framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    pdf_path = Path(pdf_out_path) if pdf_out_path is not None else out_path.parent / f"{out_path.stem}_1pt_pdf.png"
    bispec_path = (
        Path(bispec_out_path) if bispec_out_path is not None else out_path.parent / f"{out_path.stem}_bispectrum.png"
    )

    _plot_1pt_pdf_target_lpt_emu(
        delta_target=delta_target,
        delta_lpt=delta_lpt,
        delta_emu=delta_emu,
        out_path=pdf_path,
    )
    bispec_metrics = _plot_bispectrum_target_lpt_emu(
        delta_target=delta_target,
        delta_lpt=delta_lpt,
        delta_emu=delta_emu,
        boxsize=boxsize,
        mas=mas,
        mas_deconvolved_model_fields=mas_deconvolved_fields,
        mas_deconvolved_target_field=bool(mas_deconvolved_target_field),
        out_path=bispec_path,
    )

    flat_target = np.asarray(delta_target, dtype=np.float64).ravel()
    flat_lpt = np.asarray(delta_lpt, dtype=np.float64).ravel()
    flat_emu = np.asarray(delta_emu, dtype=np.float64).ravel()

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        """Return Pearson correlation with zero-variance protection."""
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    summary = {
        "moments_target": _field_moments(delta_target),
        "moments_lpt": _field_moments(delta_lpt),
        "moments_emulator": _field_moments(delta_emu),
        "rmse_lpt_vs_target": float(np.sqrt(np.mean((flat_lpt - flat_target) ** 2))),
        "rmse_emu_vs_target": float(np.sqrt(np.mean((flat_emu - flat_target) ** 2))),
        "corrcoef_lpt_vs_target": _corr(flat_lpt, flat_target),
        "corrcoef_emu_vs_target": _corr(flat_emu, flat_target),
        "median_abs_transfer_error_lpt": float(np.median(np.abs(transfer_lpt - 1.0))),
        "median_abs_transfer_error_emu": float(np.median(np.abs(transfer_emu - 1.0))),
        "mean_crosscorr_lpt": float(np.mean(xcorr_lpt)),
        "mean_crosscorr_emu": float(np.mean(xcorr_emu)),
        "k_nyquist_h_per_mpc": float(k_nyq),
        "summary_pk_plot": str(out_path),
        "summary_slices_plot": None,
        "summary_1pt_pdf_plot": str(pdf_path),
        "summary_bispectrum_plot": str(bispec_path),
        **bispec_metrics,
    }
    return summary
