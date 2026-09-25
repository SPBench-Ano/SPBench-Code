"""Internal numpy helpers shared by ``tools/*``; not for use outside the package."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

__all__ = [
    "RNGLike",
    "as_3d",
    "restore",
    "as_generator",
]

RNGLike = Union[None, int, np.random.Generator]


def as_3d(data: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Promote ``(n_traces, n_time)`` to ``(1, n_traces, n_time)``; pass 3D through. Returns ``(x3d, was_2d)``."""
    if data.ndim == 2:
        return data[None, ...], True
    if data.ndim == 3:
        return data, False
    raise ValueError(
        "Expected shape (n_traces, n_time) or (n_shots, n_traces, n_time); "
        f"got ndim={data.ndim}."
    )


def restore(x: np.ndarray, was_2d: bool) -> np.ndarray:
    """Reverse of :func:`as_3d` based on the ``was_2d`` flag."""
    return x[0] if was_2d else x


def as_generator(rng: "RNGLike") -> np.random.Generator:
    """Coerce ``None`` / ``int`` / :class:`numpy.random.Generator` into a local Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
