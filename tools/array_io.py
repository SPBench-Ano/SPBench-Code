"""Unified volume I/O: .npy, .mat, and .sgy -> (n_shots, n_traces, n_time) ndarray."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


def read_npy_volume(path: Union[str, Path]) -> np.ndarray:
    """Load a .npy file; ensure 3-D output."""
    arr = np.load(str(path))
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"npy volume must be 2-D or 3-D, got {arr.ndim}D.")
    return arr.astype(np.float32, copy=False)


def read_mat_volume(path: Union[str, Path], key: Optional[str] = None) -> np.ndarray:
    """Load a .mat file using scipy.io.loadmat."""
    from scipy.io import loadmat

    data = loadmat(str(path))
    if key is None:
        candidates = [k for k in data.keys() if not k.startswith("__")]
        if not candidates:
            raise ValueError(f"No usable variable found in {path}.")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple variables found in {path}: {candidates}. "
                f"Please specify the 'key' to load."
            )
        key = candidates[0]
    arr = data[key]
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"mat volume must be 2-D or 3-D, got {arr.ndim}D.")
    return arr.astype(np.float32, copy=False)


def read_sgy_volume(
    path: Union[str, Path],
    traces_per_shot: int,
    time_downsample: int = 1,
) -> np.ndarray:
    """Wrapper around tools.segy_read.read_regular_shots; returns shots only."""
    from tools.segy_read import read_regular_shots

    shots, _ = read_regular_shots(
        path=path,
        traces_per_shot=traces_per_shot,
        time_downsample=time_downsample,
        return_headers=False,
    )
    return shots


def load_volume(data_cfg: Dict[str, Any]) -> np.ndarray:
    """Dispatch to the correct reader based on ``path`` suffix.

    Parameters
    ----------
    data_cfg :
        Dict with ``path`` (required) and format-specific keys.
        For ``.sgy`` / ``.segy``: ``traces_per_shot`` (default 201),
        ``time_downsample`` (default 1).
        For ``.mat``: optional ``key`` (variable name).

    Returns
    -------
    shots :
        ``(n_shots, n_traces, n_time)`` float32 ndarray.
    """
    path = str(data_cfg["path"])
    suffix = Path(path).suffix.lower()

    if suffix == ".npy":
        return read_npy_volume(path)
    elif suffix == ".mat":
        return read_mat_volume(path, key=data_cfg.get("key"))
    elif suffix in (".sgy", ".segy"):
        return read_sgy_volume(
            path,
            traces_per_shot=int(data_cfg.get("traces_per_shot", 201)),
            time_downsample=int(data_cfg.get("time_downsample", 1)),
        )
    else:
        raise ValueError(f"Unsupported volume format: {suffix!r} ({path}).")
