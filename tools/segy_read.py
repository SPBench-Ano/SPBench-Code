"""SEG-Y reader: flat trace table -> shot gathers, with optional time downsampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

try:
    import segyio
except ImportError as exc:  # pragma: no cover - surfaced to the user at runtime
    raise ImportError(
        "segyio is required. Install it with `pip install segyio`."
    ) from exc


# Default per-shot headers kept alongside the trace tensor.
_DEFAULT_HEADER_KEYS: Tuple[str, ...] = (
    "FieldRecord",  # FFID, shot identifier
    "SourceX",
    "SourceY",
    "GroupX",
    "GroupY",
)


# ----------------------------------------------------------------------
# Introspection
# ----------------------------------------------------------------------

def inspect_segy(path: Union[str, Path]) -> Dict[str, Any]:
    """Probe SEG-Y metadata for shot-gather planning.

    Returns
    -------
    dict with keys ``n_traces`` / ``n_samples`` / ``sample_interval_us``
    / ``unique_ffid_count`` / ``traces_per_ffid_min_max``.
    """
    path = Path(path)
    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        n_traces = int(f.tracecount)
        n_samples = int(len(f.samples))
        sample_interval_us = int(f.bin[segyio.BinField.Interval])
        ffid = np.asarray(f.attributes(segyio.TraceField.FieldRecord)[:], dtype=np.int64)

    unique, counts = np.unique(ffid, return_counts=True)
    return {
        "n_traces": n_traces,
        "n_samples": n_samples,
        "sample_interval_us": sample_interval_us,
        "unique_ffid_count": int(unique.size),
        "traces_per_ffid_min_max": (int(counts.min()), int(counts.max())),
    }


# ----------------------------------------------------------------------
# Regular data (default path)
# ----------------------------------------------------------------------

def read_regular_shots(
    path: Union[str, Path],
    traces_per_shot: int,
    time_downsample: int = 1,
    dtype: np.dtype = np.float32,
    return_headers: bool = True,
    verify_ffid: bool = True,
    header_keys: Tuple[str, ...] = _DEFAULT_HEADER_KEYS,
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """Read a regular SEG-Y (every shot has the same trace count).

    Parameters
    ----------
    traces_per_shot : run :func:`inspect_segy` if unknown.
    time_downsample : 1 = off; > 1 applies anti-aliased ``scipy.signal.decimate``.
    verify_ffid     : raise if any shot does not share one ``FieldRecord`` value.

    Returns
    -------
    traces  : ``(n_shots, traces_per_shot, time_length)``,
              ``n_shots = n_traces // traces_per_shot``.
    headers : ``dict[str, (n_shots, traces_per_shot)]`` or ``None``.
    """
    path = Path(path)
    if traces_per_shot <= 0:
        raise ValueError(f"traces_per_shot must be positive, got {traces_per_shot}.")
    if time_downsample < 1:
        raise ValueError(f"time_downsample must be >= 1, got {time_downsample}.")

    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        n_traces = int(f.tracecount)
        n_samples = int(len(f.samples))

        if n_traces % traces_per_shot != 0:
            raise ValueError(
                f"File {path.name} has {n_traces} traces, not divisible by "
                f"traces_per_shot={traces_per_shot}. The file is likely "
                f"irregular; use read_irregular_shots_by_header() instead."
            )
        n_shots = n_traces // traces_per_shot

        traces_flat = segyio.tools.collect(f.trace[:]).astype(dtype, copy=False)
        traces = traces_flat.reshape(n_shots, traces_per_shot, n_samples)

        if time_downsample > 1:
            from scipy.signal import decimate  # lazy: only needed here
            traces = decimate(
                traces, q=time_downsample, axis=-1, zero_phase=True,
            ).astype(dtype, copy=False)

        headers: Optional[Dict[str, np.ndarray]] = None
        if return_headers or verify_ffid:
            headers_flat: Dict[str, np.ndarray] = {}
            keys_to_read = set(header_keys) | ({"FieldRecord"} if verify_ffid else set())
            for name in keys_to_read:
                field = getattr(segyio.TraceField, name)
                headers_flat[name] = np.asarray(f.attributes(field)[:], dtype=np.int64)

        if verify_ffid:
            ffid_2d = headers_flat["FieldRecord"].reshape(n_shots, traces_per_shot)
            per_shot_unique = np.array([np.unique(row).size for row in ffid_2d])
            if np.any(per_shot_unique != 1):
                raise ValueError(
                    "Regularity check failed: some shots contain more than one "
                    "FieldRecord value. Use read_irregular_shots_by_header()."
                )

        if return_headers:
            headers = {
                name: headers_flat[name].reshape(n_shots, traces_per_shot)
                for name in header_keys
            }

    return traces, headers


# ----------------------------------------------------------------------
# Irregular data (placeholder)
# ----------------------------------------------------------------------

def read_irregular_shots_by_header(
    path: Union[str, Path],
    header_key: str = "FieldRecord",
    time_downsample: int = 1,
    dtype: np.dtype = np.float32,
    return_headers: bool = True,
    header_keys: Tuple[str, ...] = _DEFAULT_HEADER_KEYS,
) -> Tuple[List[np.ndarray], Optional[List[Dict[str, np.ndarray]]]]:
    """Read a SEG-Y with variable trace count per shot. NOT IMPLEMENTED YET.

    Returns
    -------
    traces  : ``List[(n_traces_i, time_length)]``, length = ``n_shots``.
    headers : ``List[dict[str, (n_traces_i,)]]`` or ``None``.
    """
    raise NotImplementedError(
        "read_irregular_shots_by_header is a placeholder; implement when the "
        "first SEG_C3NA_ffid_*.sgy file is consumed."
    )


# ----------------------------------------------------------------------
# Trace-header introspection helpers
# ----------------------------------------------------------------------


def contiguous_ffid_blocks(ffids: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a flat FFID array into contiguous blocks of repeated values.

    Parameters
    ----------
    ffids : 1-D int array of ``FieldRecord`` values, one per trace.
    name  : human label for the source file (used in error messages).

    Returns
    -------
    values : ``(n_blocks,)`` int64 — unique FFID per block.
    starts : ``(n_blocks,)`` int64 — first trace index of each block.
    stops  : ``(n_blocks,)`` int64 — exclusive end index of each block.
    """
    if ffids.ndim != 1 or ffids.size == 0:
        raise ValueError(f"{name}: FieldRecord array must be non-empty and 1-D.")
    changes = np.flatnonzero(ffids[1:] != ffids[:-1]) + 1
    starts = np.concatenate(([0], changes)).astype(np.int64)
    stops = np.concatenate((changes, [ffids.size])).astype(np.int64)
    values = ffids[starts].astype(np.int64)
    if np.unique(values).size != values.size:
        raise ValueError(
            f"{name}: repeated non-contiguous FieldRecord values found. "
            "This dataset expects each FFID gather to occupy one contiguous block."
        )
    return values, starts, stops


def read_group_coordinates(segy_file: Any, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``GroupX`` and ``GroupY`` trace headers as float64 arrays."""
    try:
        group_x = np.asarray(
            segy_file.attributes(segyio.TraceField.GroupX)[:],
            dtype=np.float64,
        )
        group_y = np.asarray(
            segy_file.attributes(segyio.TraceField.GroupY)[:],
            dtype=np.float64,
        )
    except Exception as exc:
        raise ValueError(
            f"{name}: GroupX/GroupY trace headers are required but could not be read."
        ) from exc
    if group_x.ndim != 1 or group_y.ndim != 1 or group_x.size != group_y.size:
        raise ValueError(f"{name}: GroupX/GroupY headers must be matching 1-D arrays.")
    return group_x, group_y


def group_coordinates_are_usable(group_x: np.ndarray, group_y: np.ndarray) -> bool:
    """Return ``True`` when neighbor distances have at least one finite positive value."""
    distances = np.hypot(np.diff(group_x), np.diff(group_y))
    return bool(np.any(np.isfinite(distances) & (distances > 0)))


def read_line_id_header(
    segy_file: Any,
    name: str,
    *,
    header_name: str,
) -> Optional[np.ndarray]:
    """Read a named SEG-Y trace header field, with graceful fallback.

    Parameters
    ----------
    segy_file   : open ``segyio`` file handle.
    name        : human label for the source file.
    header_name : a SEG-Y trace header name such as ``"INLINE_3D"``.

    Returns
    -------
    1-D int64 array of header values, or ``None`` if the field is unavailable,
    empty, or all zeros.
    """
    trace_field = getattr(segyio.TraceField, header_name, None)
    if trace_field is None:
        warnings.warn(
            f"{name}: line_id_header={header_name!r} is not a known SEG-Y trace "
            "header; inferring receiver lines from GroupX/GroupY geometry.",
            RuntimeWarning,
        )
        return None
    try:
        line_ids = np.asarray(
            segy_file.attributes(trace_field)[:],
            dtype=np.int64,
        )
    except Exception:
        warnings.warn(
            f"{name}: line_id_header={header_name!r} is unavailable; inferring "
            "receiver lines from GroupX/GroupY geometry.",
            RuntimeWarning,
        )
        return None
    if line_ids.ndim != 1 or line_ids.size == 0:
        warnings.warn(
            f"{name}: line_id_header={header_name!r} is empty or not 1-D; "
            "inferring receiver lines from GroupX/GroupY geometry.",
            RuntimeWarning,
        )
        return None
    if not bool(np.any(line_ids != 0)):
        warnings.warn(
            f"{name}: line_id_header={header_name!r} is all zeros; inferring "
            "receiver lines from GroupX/GroupY geometry.",
            RuntimeWarning,
        )
        return None
    return line_ids


# ----------------------------------------------------------------------
# CLI smoke test (read-only; safe to run manually)
# ----------------------------------------------------------------------

def _demo() -> None:  # pragma: no cover - manual inspection helper
    """Read-only smoke test against the regular demo file. Run with ``python -m tools.segy_read``."""
    path = Path("/path/to/SEG_45Shot_shots1-9.sgy")
    if not path.exists():
        print(f"[demo] File not found: {path}")
        return

    info = inspect_segy(path)
    print(f"[demo] inspect_segy -> {info}")

    traces_min, traces_max = info["traces_per_ffid_min_max"]
    if traces_min != traces_max:
        print(
            "[demo] File appears irregular (per-FFID trace count varies); "
            "read_regular_shots would fail. Use read_irregular_shots_by_header."
        )
        return
    traces_per_shot = traces_min
    print(f"[demo] inferred traces_per_shot = {traces_per_shot}")

    traces, headers = read_regular_shots(
        path, traces_per_shot=201, time_downsample=1, return_headers=True,
    )
    print(f"[demo] no downsample  -> shape = {traces.shape}, dtype = {traces.dtype}")

    traces_d2, _ = read_regular_shots(
        path, traces_per_shot=traces_per_shot, time_downsample=2, return_headers=False,
    )
    print(f"[demo] downsample = 2 -> shape = {traces_d2.shape}, dtype = {traces_d2.dtype}")

    if headers is not None:
        for name, arr in headers.items():
            print(f"[demo] headers['{name}'].shape = {arr.shape}")


if __name__ == "__main__":  # pragma: no cover
    _demo()
