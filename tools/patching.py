"""Patchify / unpatchify on the (trace, time) plane (pure numpy).

Inputs accept ``(n_traces, n_time)`` or ``(n_shots, n_traces, n_time)``.
Output ndim is selectable: ``3`` -> ``(P, h, w)``; ``4`` -> ``(P, 1, h, w)``,
with ``P = n_shots * n_per_shot``. Only the uniform mode is invertible.
See ``tools/README.md`` for the per-function summary.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._array_utils import (
    RNGLike,
    as_3d as _as_3d,
    as_generator as _as_generator,
)

__all__ = [
    "patchify_uniform",
    "patchify_random",
    "unpatchify_uniform",
]


# ----------------------------------------------------------------------
# Internal helpers (kept private)
# ----------------------------------------------------------------------

def _gen_uniform_starts(length: int, patch_len: int, stride: int) -> np.ndarray:
    """Regular-grid starts with the last position anchored to ``length - patch_len``."""
    if patch_len <= 0:
        raise ValueError(f"patch length must be positive, got {patch_len}.")
    if stride <= 0:
        raise ValueError(f"patch stride must be positive, got {stride}.")
    if length <= patch_len:
        return np.asarray([0], dtype=np.int64)
    last_valid = length - patch_len
    starts = list(range(0, last_valid + 1, stride))
    if starts[-1] != last_valid:
        starts.append(last_valid)
    return np.asarray(starts, dtype=np.int64)


def _validate_patch_size(
    patch_size: Tuple[int, int], n_traces: int, n_time: int
) -> Tuple[int, int]:
    if (
        not isinstance(patch_size, (tuple, list))
        or len(patch_size) != 2
    ):
        raise ValueError(
            f"patch_size must be a 2-tuple (patch_h, patch_w); got {patch_size!r}."
        )
    h, w = int(patch_size[0]), int(patch_size[1])
    if h <= 0 or w <= 0:
        raise ValueError(
            f"patch_size entries must be positive; got ({h}, {w})."
        )
    if h > n_traces or w > n_time:
        raise ValueError(
            f"patch_size ({h}, {w}) exceeds input ({n_traces}, {n_time})."
        )
    return h, w


def _validate_output_ndim(output_ndim: int) -> None:
    if output_ndim not in (3, 4):
        raise ValueError(
            f"output_ndim must be 3 or 4, got {output_ndim!r}."
        )


def _maybe_add_channel(patches: np.ndarray, output_ndim: int) -> np.ndarray:
    if output_ndim == 4:
        return patches[:, None, :, :]
    return patches


def _drop_channel(patches: np.ndarray) -> np.ndarray:
    """Squeeze the optional singleton channel axis (accepts 3D or 4D)."""
    if patches.ndim == 4:
        if patches.shape[1] != 1:
            raise ValueError(
                "4D patches must have a singleton channel axis (shape "
                f"(N, 1, h, w)); got shape {patches.shape}."
            )
        return patches[:, 0]
    if patches.ndim == 3:
        return patches
    raise ValueError(
        f"patches must be 3D or 4D; got ndim={patches.ndim}."
    )


# ----------------------------------------------------------------------
# 1. Uniform overlapping patching
# ----------------------------------------------------------------------

def patchify_uniform(
    data: np.ndarray,
    patch_size: Tuple[int, int],
    overlap: float = 0.0,
    output_ndim: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Overlapping regular-grid patches; the last patch on each axis is tail-anchored.

    Parameters
    ----------
    data        : ``(n_traces, n_time)`` or ``(n_shots, n_traces, n_time)``.
    patch_size  : ``(patch_h, patch_w)`` along ``(trace, time)``.
    overlap     : fraction in ``[0, 1)``; stride = ``max(1, round(patch*(1-overlap)))``.
    output_ndim : 3 -> ``(P, h, w)``; 4 -> ``(P, 1, h, w)``.

    Returns
    -------
    patches : ndarray of the requested ``output_ndim``.
    info    : bookkeeping consumed by :func:`unpatchify_uniform` (``shape``,
              ``was_2d``, ``patch_size``, ``trace_starts`` / ``time_starts``,
              ``n_shots``, ``n_per_shot``, ``output_ndim``, ``mode``).
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}.")
    _validate_output_ndim(output_ndim)

    x, was_2d = _as_3d(np.ascontiguousarray(data))
    n_shots, n_traces, n_time = x.shape
    h, w = _validate_patch_size(patch_size, n_traces, n_time)

    s_h = max(1, int(round(h * (1.0 - overlap))))
    s_w = max(1, int(round(w * (1.0 - overlap))))

    trace_starts = _gen_uniform_starts(n_traces, h, s_h)
    time_starts = _gen_uniform_starts(n_time, w, s_w)
    n_h, n_w = trace_starts.size, time_starts.size
    n_per_shot = int(n_h * n_w)

    # sliding_window_view returns a view of shape
    # (n_shots, n_traces - h + 1, n_time - w + 1, h, w).
    windows = sliding_window_view(x, (h, w), axis=(1, 2))
    grid = windows[:, trace_starts[:, None], time_starts[None, :], :, :]
    patches = np.ascontiguousarray(grid.reshape(n_shots * n_per_shot, h, w))

    info: Dict[str, Any] = {
        "shape": (n_shots, n_traces, n_time),
        "was_2d": was_2d,
        "patch_size": (h, w),
        "trace_starts": trace_starts,
        "time_starts": time_starts,
        "n_shots": n_shots,
        "n_per_shot": n_per_shot,
        "output_ndim": output_ndim,
        "mode": "uniform",
    }
    return _maybe_add_channel(patches, output_ndim), info


# ----------------------------------------------------------------------
# 2. Uniform reconstruction (overlap-aware averaging)
# ----------------------------------------------------------------------

def unpatchify_uniform(
    patches: np.ndarray,
    info: Dict[str, Any],
) -> np.ndarray:
    """Reconstruct the array produced by :func:`patchify_uniform`; overlaps are averaged.

    Parameters
    ----------
    patches : ``(P, h, w)`` or ``(P, 1, h, w)`` as returned by :func:`patchify_uniform`.
    info    : the bookkeeping dict returned alongside ``patches``.

    Returns
    -------
    reconstructed : original shape (2D if ``info['was_2d']``, else 3D).
    """
    if info.get("mode") != "uniform":
        raise ValueError(
            f"unpatchify_uniform expects info['mode']=='uniform', "
            f"got {info.get('mode')!r}. Random patching is non-invertible."
        )

    p = _drop_channel(patches)
    n_shots = int(info["n_shots"])
    h, w = info["patch_size"]
    trace_starts = np.asarray(info["trace_starts"], dtype=np.int64)
    time_starts = np.asarray(info["time_starts"], dtype=np.int64)
    n_h, n_w = trace_starts.size, time_starts.size
    n_per_shot = int(info["n_per_shot"])

    expected = n_shots * n_per_shot
    if p.shape != (expected, h, w):
        raise ValueError(
            f"patches shape mismatch: expected ({expected}, {h}, {w}), got {p.shape}."
        )

    _, n_traces, n_time = info["shape"]
    out = np.zeros((n_shots, n_traces, n_time), dtype=p.dtype)
    cnt = np.zeros((n_shots, n_traces, n_time), dtype=p.dtype)

    grid = p.reshape(n_shots, n_h, n_w, h, w)

    # Loop is over the patch grid (typically O(10^2)), not over data points.
    for i, h0 in enumerate(trace_starts.tolist()):
        h1 = h0 + h
        for j, w0 in enumerate(time_starts.tolist()):
            w1 = w0 + w
            out[:, h0:h1, w0:w1] += grid[:, i, j]
            cnt[:, h0:h1, w0:w1] += 1.0

    out /= np.maximum(cnt, 1.0)

    if info.get("was_2d"):
        return out[0]
    return out


# ----------------------------------------------------------------------
# 3. Random patching (no inverse)
# ----------------------------------------------------------------------

def patchify_random(
    data: np.ndarray,
    patch_size: Tuple[int, int],
    n_patches: int,
    output_ndim: int = 3,
    rng: RNGLike = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Randomly sampled patches; each shot independently draws ``n_patches`` positions (no inverse).

    Parameters
    ----------
    data        : ``(n_traces, n_time)`` or ``(n_shots, n_traces, n_time)``.
    patch_size  : ``(patch_h, patch_w)``.
    n_patches   : positive int, **per shot** (total output count = ``n_shots * n_patches``).
    output_ndim : 3 -> ``(P, h, w)``; 4 -> ``(P, 1, h, w)``.
    rng         : seed / :class:`numpy.random.Generator`; ``None`` = fresh.

    Returns
    -------
    patches : ndarray of the requested ``output_ndim``.
    info    : ``trace_starts`` / ``time_starts`` shaped ``(n_shots, n_patches)``
              plus the same bookkeeping fields as :func:`patchify_uniform`.
    """
    if n_patches <= 0:
        raise ValueError(f"n_patches must be positive, got {n_patches}.")
    _validate_output_ndim(output_ndim)

    gen = _as_generator(rng)
    x, was_2d = _as_3d(np.ascontiguousarray(data))
    n_shots, n_traces, n_time = x.shape
    h, w = _validate_patch_size(patch_size, n_traces, n_time)

    trace_starts = gen.integers(
        0, n_traces - h + 1, size=(n_shots, n_patches), dtype=np.int64,
    )
    time_starts = gen.integers(
        0, n_time - w + 1, size=(n_shots, n_patches), dtype=np.int64,
    )

    # Vectorised extraction via fancy indexing; no Python loop over patches.
    flat = n_shots * n_patches
    shot_idx = np.repeat(np.arange(n_shots, dtype=np.int64), n_patches)  # (P,)
    t_idx = trace_starts.reshape(flat, 1) + np.arange(h, dtype=np.int64)[None, :]
    w_idx = time_starts.reshape(flat, 1) + np.arange(w, dtype=np.int64)[None, :]

    patches = x[
        shot_idx[:, None, None],
        t_idx[:, :, None],
        w_idx[:, None, :],
    ]
    patches = np.ascontiguousarray(patches)

    info: Dict[str, Any] = {
        "shape": (n_shots, n_traces, n_time),
        "was_2d": was_2d,
        "patch_size": (h, w),
        "trace_starts": trace_starts,
        "time_starts": time_starts,
        "n_shots": n_shots,
        "n_per_shot": int(n_patches),
        "output_ndim": output_ndim,
        "mode": "random",
    }
    return _maybe_add_channel(patches, output_ndim), info
