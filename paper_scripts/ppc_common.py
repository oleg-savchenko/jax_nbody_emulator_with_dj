#!/usr/bin/env python3
"""Shared helpers for posterior predictive check scripts."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def z_file_tag(z_value: float) -> str:
    """Format redshift value for stable file naming."""
    return f"z{z_value:g}".replace(".", "p")


def sorted_sample_files(samples_dir: Path, pattern: str) -> list[Path]:
    """Return sample files sorted by numeric id found in filename stem."""
    files = list(samples_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching pattern {pattern!r} in {samples_dir}")

    def _key(path: Path) -> tuple[int, str]:
        match = re.search(r"(\d+)", path.stem)
        if match is None:
            return (10**9, path.name)
        return (int(match.group(1)), path.name)

    return sorted(files, key=_key)


def select_sample_files(
    *,
    samples_dir: Path,
    pattern: str,
    sample_start: int,
    sample_stride: int,
    num_samples: int,
) -> list[Path]:
    """Select a contiguous sample subset after sorted start/stride slicing."""
    files_all = sorted_sample_files(samples_dir=samples_dir, pattern=pattern)
    files_sel = files_all[int(sample_start) :: int(sample_stride)]
    files_sel = files_sel[: int(num_samples)]
    if len(files_sel) == 0:
        raise ValueError("No sample files selected after applying start/stride/num-samples.")
    return files_sel


def extract_sample_id(path: Path, fallback_index: int) -> int:
    """Extract numeric sample id from filename stem when available."""
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        return int(fallback_index)
    return int(match.group(1))


def sample_output_dir(samples_dir: Path, template: str, sample_id: int, sample_path: Path) -> Path:
    """Build per-sample output folder from template and sample metadata."""
    try:
        folder_name = template.format(sample_id=int(sample_id), sample_stem=sample_path.stem)
    except Exception as exc:  # pragma: no cover - formatting guard
        raise ValueError(
            "Invalid sample folder template: "
            f"{template!r}. Expected e.g. 'sample_{{sample_id:04d}}' or "
            "'sample_fourier_{sample_id:04d}'."
        ) from exc
    return samples_dir / folder_name


def mean_sigma(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute sample mean and standard deviation over axis=0."""
    arr = np.asarray(samples, dtype=np.float64)
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def save_summary_npz(path: Path, payload: dict[str, np.ndarray | list | float | int | bool | str]) -> None:
    """Persist heterogeneous summary payload as compressed npz arrays."""
    to_save: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            to_save[key] = value
        elif isinstance(value, (list, tuple)):
            to_save[key] = np.asarray(value)
        elif value is None:
            to_save[key] = np.asarray([])
        else:
            to_save[key] = np.asarray(value)
    np.savez_compressed(path, **to_save)
