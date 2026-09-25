"""Training logger: ``train_log.txt`` + ``loss_history.csv`` + ``metrics_history.csv``.

The text log captures human-readable lines (``info`` + per-epoch summary).
The two CSVs are append-only so resumed runs keep growing the same file.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

from .metrics import format_metric_value

# Reserved key in ``log_epoch(extras=...)`` for the current learning rate;
# stored as a dedicated column in ``loss_history.csv``.
_LR_KEY: str = "lr"


def _format_value(value: Any) -> str:
    """Render numeric values with 6 sig figs; non-finite -> ``nan``."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def _safe_value(value: Any) -> Any:
    """Replace non-finite floats with ``float('nan')`` for CSV serialization."""
    if isinstance(value, float) and not math.isfinite(value):
        return float("nan")
    return value


class TrainingLogger:
    """Append-only text log + per-epoch loss / metrics CSVs + auto-refreshed curves.

    Parameters
    ----------
    log_dir       : directory; created if missing.
    loss_keys     : ordered list of loss column names; default ``["train", "val"]``.
                    An ``lr`` column is always inserted right after ``epoch``.
    metric_keys   : ordered list of metric column names; if empty, no metrics
                    CSV / curve is created.
    plot_interval : refresh ``loss_curve.png`` / ``metrics_curve.png`` every N
                    epochs. ``0`` disables auto-plotting; ``close()`` always
                    does a final refresh. Default ``5``.

    Files
    -----
    ``<log_dir>/train_log.txt``        : timestamped human-readable log lines.
    ``<log_dir>/loss_history.csv``     : columns ``epoch, lr, *loss_keys``.
    ``<log_dir>/metrics_history.csv``  : columns ``epoch, *metric_keys``
                                         (only if ``metric_keys`` is non-empty).
    ``<log_dir>/loss_curve.png``       : redrawn from the in-memory history.
    ``<log_dir>/metrics_curve.png``    : ditto (only if ``metric_keys`` non-empty).

    Resume safety
    -------------
    On instantiation the in-memory history is rebuilt from any existing CSV in
    ``log_dir``, so curves drawn after resuming a run still include earlier
    epochs.
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        loss_keys: Optional[List[str]] = None,
        metric_keys: Optional[List[str]] = None,
        plot_interval: int = 5,
    ) -> None:
        if plot_interval < 0:
            raise ValueError(f"plot_interval must be >= 0, got {plot_interval}.")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._txt_path = self.log_dir / "train_log.txt"
        self._loss_path = self.log_dir / "loss_history.csv"
        self._metric_path = self.log_dir / "metrics_history.csv"
        self._loss_curve_path = self.log_dir / "loss_curve.png"
        self._metric_curve_dir = self.log_dir / "metrics_curves"

        self.loss_keys: List[str] = list(loss_keys) if loss_keys else ["train", "val"]
        self.metric_keys: List[str] = list(metric_keys) if metric_keys else []
        self.plot_interval: int = int(plot_interval)

        self._loss_columns: List[str] = ["epoch", _LR_KEY, *self.loss_keys]
        self._metric_columns: List[str] = ["epoch", *self.metric_keys]

        self._txt: Optional[TextIO] = None
        self._loss_file: Optional[TextIO] = None
        self._metric_file: Optional[TextIO] = None
        self._loss_writer: Optional[csv.DictWriter] = None
        self._metric_writer: Optional[csv.DictWriter] = None
        self._closed: bool = False

        # In-memory curve data; rehydrated from existing CSVs on resume.
        self._loss_history: Dict[str, List[float]] = {k: [] for k in self.loss_keys}
        self._metric_history: Dict[str, List[float]] = {k: [] for k in self.metric_keys}
        self._dirty: bool = False  # any new row since last successful plot refresh.

        self._open()
        self._rehydrate_history()

    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _rehydrate_history(self) -> None:
        """Load any already-on-disk CSV rows into the in-memory history dicts."""
        if self._loss_path.exists() and self._loss_path.stat().st_size > 0:
            with self._loss_path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    for key in self.loss_keys:
                        self._loss_history[key].append(self._coerce_float(row.get(key)))
        if (
            self.metric_keys
            and self._metric_path.exists()
            and self._metric_path.stat().st_size > 0
        ):
            with self._metric_path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    for key in self.metric_keys:
                        self._metric_history[key].append(self._coerce_float(row.get(key)))

    @staticmethod
    def _metric_base_name(key: str) -> str:
        for prefix in ("train_", "test_", "val_"):
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    def _refresh_curves(self) -> None:
        """Redraw ``loss_curve.png`` and one PNG per metric from in-memory history."""
        if not self._dirty:
            return
        # Lazy import: keeps matplotlib out of the bare-import path of utils.logger.
        from .visualization import plot_loss_curve, plot_single_metric_curve

        if any(len(v) > 0 for v in self._loss_history.values()):
            plot_loss_curve(self._loss_history, self._loss_curve_path, title="Loss", log_y=False)
            plot_loss_curve(
                self._loss_history,
                self._loss_curve_path.with_stem(self._loss_curve_path.stem + "_log"),
                title="Loss (log scale)",
                log_y=True,
            )

        if self.metric_keys and any(
            len(v) > 0 for v in self._metric_history.values()
        ):
            self._metric_curve_dir.mkdir(parents=True, exist_ok=True)
            groups: Dict[str, Dict[str, List[float]]] = {}
            for key in self.metric_keys:
                base = self._metric_base_name(key)
                groups.setdefault(base, {})[key] = self._metric_history[key]
            for base, sub_history in groups.items():
                plot_single_metric_curve(
                    sub_history,
                    self._metric_curve_dir / f"{base}_curve.png",
                    title=base.upper(),
                )
        self._dirty = False

    def _open(self) -> None:
        self._txt = self._txt_path.open("a", encoding="utf-8")

        loss_existed = self._loss_path.exists() and self._loss_path.stat().st_size > 0
        self._loss_file = self._loss_path.open("a", encoding="utf-8", newline="")
        self._loss_writer = csv.DictWriter(self._loss_file, fieldnames=self._loss_columns)
        if not loss_existed:
            self._loss_writer.writeheader()
            self._loss_file.flush()

        if self.metric_keys:
            metric_existed = (
                self._metric_path.exists() and self._metric_path.stat().st_size > 0
            )
            self._metric_file = self._metric_path.open("a", encoding="utf-8", newline="")
            self._metric_writer = csv.DictWriter(
                self._metric_file, fieldnames=self._metric_columns
            )
            if not metric_existed:
                self._metric_writer.writeheader()
                self._metric_file.flush()

    def info(self, message: str) -> None:
        """Write ``[YYYY-MM-DD HH:MM:SS] message`` to the text log and stdout."""
        self._ensure_open()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        assert self._txt is not None
        self._txt.write(line + "\n")
        self._txt.flush()
        print(line, flush=True)

    def log_epoch(
        self,
        epoch: int,
        losses: Dict[str, float],
        metrics: Optional[Dict[str, float]] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one row to each CSV and write a one-line summary via ``info``.

        Parameters
        ----------
        epoch   : 0-based epoch index.
        losses  : ``{loss_key: scalar}``; missing keys fill as NaN.
        metrics : ``{metric_key: scalar}``; ignored if ``metric_keys`` is empty.
        extras  : optional; recognized keys: ``"lr"`` (stored in loss CSV),
                  any other entry is appended to the text summary.
        """
        self._ensure_open()
        extras = extras or {}
        metrics = metrics or {}

        lr_value: float = float(extras.get(_LR_KEY, float("nan")))
        loss_row: Dict[str, Any] = {"epoch": epoch, _LR_KEY: _safe_value(lr_value)}
        for key in self.loss_keys:
            value = float(losses.get(key, float("nan")))
            loss_row[key] = _safe_value(value)
            self._loss_history[key].append(value)
        assert self._loss_writer is not None and self._loss_file is not None
        self._loss_writer.writerow(loss_row)
        self._loss_file.flush()

        if self.metric_keys and self._metric_writer is not None:
            assert self._metric_file is not None
            metric_row: Dict[str, Any] = {"epoch": epoch}
            for key in self.metric_keys:
                value = float(metrics.get(key, float("nan")))
                metric_row[key] = _safe_value(value)
                self._metric_history[key].append(value)
            self._metric_writer.writerow(metric_row)
            self._metric_file.flush()

        self._dirty = True

        loss_str = " ".join(
            f"{k}={_format_value(losses.get(k, float('nan')))}" for k in self.loss_keys
        )
        metric_str = (
            " | "
            + " ".join(
                f"{k}={format_metric_value(k, metrics.get(k, float('nan')))}"
                for k in self.metric_keys
            )
            if self.metric_keys
            else ""
        )
        extra_str = ""
        if math.isfinite(lr_value):
            extra_str += f" | lr={_format_value(lr_value)}"
        for k, v in extras.items():
            if k == _LR_KEY:
                continue
            extra_str += f" | {k}={_format_value(v)}"

        self.info(f"[epoch={epoch}] {loss_str}{metric_str}{extra_str}")

        if self.plot_interval > 0 and (epoch + 1) % self.plot_interval == 0:
            self._refresh_curves()

    def flush(self) -> None:
        """Flush all open file handles."""
        for handle in (self._txt, self._loss_file, self._metric_file):
            if handle is not None and not handle.closed:
                handle.flush()

    def close(self) -> None:
        """Close all file handles; safe to call repeatedly. Refreshes curves once."""
        if self._closed:
            return
        try:
            self._refresh_curves()
        except Exception:  # pragma: no cover - never block close on plot errors
            pass
        for handle in (self._txt, self._loss_file, self._metric_file):
            if handle is not None and not handle.closed:
                try:
                    handle.flush()
                finally:
                    handle.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("TrainingLogger is closed; instantiate a new one to continue.")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass


class StepLossLogger:
    """Append per-train-step loss rows and refresh step-level loss curves."""

    _columns = [
        "epoch",
        "step",
        "global_step",
        "loss",
        "lr",
        "step_time_sec",
        "epoch_elapsed_sec",
    ]

    def __init__(self, log_dir: Path, *, flush_interval: int = 50) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flush_interval = max(1, int(flush_interval))
        self._path = self.log_dir / "step_loss_history.csv"
        self._curve_path = self.log_dir / "step_loss_curve.png"
        self._curve_log_path = self.log_dir / "step_loss_curve_log.png"
        self._global_steps: list[int] = []
        self._losses: list[float] = []
        self._rows_since_flush = 0
        self._dirty = False
        self._closed = False
        self._file: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter[str]] = None

        self._rehydrate_history()
        self._open()

    @property
    def next_global_step(self) -> int:
        return (max(self._global_steps) + 1) if self._global_steps else 1

    def _rehydrate_history(self) -> None:
        if not self._path.exists() or self._path.stat().st_size == 0:
            return
        with self._path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    self._global_steps.append(int(row["global_step"]))
                    self._losses.append(float(row["loss"]))
                except (KeyError, TypeError, ValueError):
                    continue

    def _open(self) -> None:
        existed = self._path.exists() and self._path.stat().st_size > 0
        self._file = self._path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._columns)
        if not existed:
            self._writer.writeheader()
            self._file.flush()

    def log_step(
        self,
        *,
        epoch: int,
        step: int,
        global_step: int,
        loss: float,
        lr: float,
        step_time_sec: float,
        epoch_elapsed_sec: float,
    ) -> None:
        if self._closed:
            raise RuntimeError("StepLossLogger is closed.")
        assert self._writer is not None
        self._writer.writerow(
            {
                "epoch": int(epoch),
                "step": int(step),
                "global_step": int(global_step),
                "loss": _safe_value(float(loss)),
                "lr": _safe_value(float(lr)),
                "step_time_sec": _safe_value(float(step_time_sec)),
                "epoch_elapsed_sec": _safe_value(float(epoch_elapsed_sec)),
            }
        )
        self._global_steps.append(int(global_step))
        self._losses.append(float(loss))
        self._dirty = True
        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_interval:
            self.flush()

    def refresh_curves(self) -> None:
        if not self._dirty:
            return
        from .visualization import plot_step_loss_curve

        plot_step_loss_curve(self._global_steps, self._losses, self._curve_path)
        plot_step_loss_curve(
            self._global_steps,
            self._losses,
            self._curve_log_path,
            log_y=True,
        )
        self._dirty = False

    def flush(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.flush()
        self._rows_since_flush = 0

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.refresh_curves()
        except Exception:
            pass
        finally:
            if self._file is not None and not self._file.closed:
                self._file.flush()
                self._file.close()
            self._closed = True
