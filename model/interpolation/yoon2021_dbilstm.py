"""Deep BiLSTM with skip connections for seismic trace interpolation.

Yoon, D., Yeeh, Z., and Byun, J. "Seismic Data Reconstruction Using Deep
Bidirectional Long Short-Term Memory With Skip Connections."
IEEE Geoscience and Remote Sensing Letters, 2021.
DOI: 10.1109/LGRS.2020.2993847

Conservative reproduction profile:
- 3 BiLSTM layers, hidden sizes [64, 128, 128]
- Projected additive skip connections
- Dropout 0.2 after each BiLSTM layer
- Linear output head (256 -> 1)
- Native input: [B, T, 2]  (T = time samples, 2 = left/right trace)
- Native output: [B, T, 1]

The core paradigm is trace-as-time-series: two surrounding traces predict
the midpoint trace.  The 4D gather adapter (regular_midpoint_4d) is a
repository adaptation, not the paper's original input form.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..registry import register_model


# ---------------------------------------------------------------------------
# BiLSTM layer with skip connection
# ---------------------------------------------------------------------------

class _BiLSTMSkipLayer(nn.Module):
    """Single bidirectional LSTM layer with optional projected additive skip.

    Path: BiLSTM -> [+ projected skip] -> Dropout.

    Parameters
    ----------
    input_size : Number of input features per time step.
    hidden_size : Hidden size of the LSTM (per direction).
    dropout : Dropout probability applied after the skip-add (default 0.0).
    use_skip : If True, adds a projected skip connection from input to output.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2 * hidden_size  # bidirectional concatenation

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.use_skip = use_skip
        if use_skip:
            if input_size != self.output_size:
                self.skip_projection: nn.Module = nn.Linear(
                    input_size, self.output_size, bias=False
                )
            else:
                self.skip_projection = nn.Identity()
        else:
            self.skip_projection = None

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(B, T, input_size)``.

        Returns
        -------
        Tensor of shape ``(B, T, output_size)`` where ``output_size = 2 * hidden_size``.
        """
        lstm_out, _ = self.lstm(x)

        if self.use_skip and self.skip_projection is not None:
            skip = self.skip_projection(x)
            out = lstm_out + skip
        else:
            out = lstm_out

        out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# Native core network (no repository dependencies)
# ---------------------------------------------------------------------------

class YoonDBiLSTMCore(nn.Module):
    """Native DBiLSTM core: two surrounding traces -> midpoint trace.

    Input:  ``(B, T, 2)``  — left trace and right trace stacked along the last axis.
    Output: ``(B, T, 1)``  — predicted midpoint trace.

    Parameters
    ----------
    input_features : Number of input traces (default 2: left + right).
    output_features : Number of output values per time step (default 1).
    hidden_sizes : Hidden size of each BiLSTM layer (per direction).
        Length determines the number of layers.
    dropout : Dropout rate after each BiLSTM layer.
    use_skip : Whether to use projected additive skip connections.
    """

    def __init__(
        self,
        input_features: int = 2,
        output_features: int = 1,
        hidden_sizes: Tuple[int, ...] = (64, 128, 128),
        dropout: float = 0.2,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must not be empty.")
        if any(h <= 0 for h in hidden_sizes):
            raise ValueError(f"All hidden sizes must be positive, got {hidden_sizes}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        layers: List[nn.Module] = []
        in_size = input_features
        for h in hidden_sizes:
            layer = _BiLSTMSkipLayer(
                input_size=in_size,
                hidden_size=h,
                dropout=dropout,
                use_skip=use_skip,
            )
            layers.append(layer)
            in_size = layer.output_size  # 2 * h for next layer

        self.layers = nn.ModuleList(layers)
        self.output_head = nn.Linear(in_size, output_features, bias=True)

        self.input_features = input_features
        self.output_features = output_features
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout
        self.use_skip = use_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(B, T, input_features)``.

        Returns
        -------
        Tensor of shape ``(B, T, output_features)``.
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.output_head(h)


# ---------------------------------------------------------------------------
# Repository-registered model (adapter for trainer interface)
# ---------------------------------------------------------------------------

@register_model("yoon2021_dbilstm")
class Yoon2021DBiLSTM(nn.Module):
    """DBiLSTM with skip connections, adapted to the repository trainer interface.

    Supports multiple input modes:

    - ``"pair_btf"``: ``[B, T, 2] -> [B, T, 1]``  (native paper format)
    - ``"pair_bct"``: ``[B, 2, T] -> [B, 1, T]``
    - ``"pair_bct1"``: ``[B, 2, T, 1] -> [B, 1, T, 1]``
    - ``"regular_midpoint_4d"``: ``[B, 1, T, X] -> [B, 1, T, X]``
      Repository adapter that extracts left/right neighbours for target
      positions defined by ``target_stride`` and ``target_offset``.

    Parameters
    ----------
    input_mode : Input layout mode (default ``"regular_midpoint_4d"``).
    hidden_sizes : Hidden sizes per BiLSTM layer (default ``(64, 128, 128)``).
    dropout : Dropout rate after each BiLSTM layer (default 0.2).
    use_skip : Whether to use projected additive skip connections (default True).
    target_stride : Stride between target trace positions (``regular_midpoint_4d`` only).
    target_offset : Offset of the first target trace (``regular_midpoint_4d`` only).
    spatial_chunk_size : Max target traces per chunk, or None for no chunking
        (``regular_midpoint_4d`` only).
    """

    def __init__(
        self,
        input_mode: str = "regular_midpoint_4d",
        hidden_sizes: Tuple[int, ...] = (64, 128, 128),
        dropout: float = 0.2,
        use_skip: bool = True,
        target_stride: int = 2,
        target_offset: int = 1,
        spatial_chunk_size: Optional[int] = 64,
    ) -> None:
        super().__init__()

        # ---- Validation ----
        _valid_modes = {"pair_btf", "pair_bct", "pair_bct1", "regular_midpoint_4d"}
        if input_mode not in _valid_modes:
            raise ValueError(
                f"Unsupported input_mode: '{input_mode}'. "
                f"Must be one of {sorted(_valid_modes)}."
            )
        if target_stride <= 0:
            raise ValueError(f"target_stride must be positive, got {target_stride}.")
        if target_offset < 0:
            raise ValueError(f"target_offset must be non-negative, got {target_offset}.")
        if spatial_chunk_size is not None and spatial_chunk_size <= 0:
            raise ValueError(
                f"spatial_chunk_size must be positive or None, got {spatial_chunk_size}."
            )

        self.input_mode = input_mode
        self.target_stride = target_stride
        self.target_offset = target_offset
        self.spatial_chunk_size = spatial_chunk_size

        self.core = YoonDBiLSTMCore(
            input_features=2,
            output_features=1,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            use_skip=use_skip,
        )

    # ------------------------------------------------------------------
    # Input mode helpers
    # ------------------------------------------------------------------

    def _validate_input(self, x: torch.Tensor) -> None:
        """Raise ValueError with clear message if the input shape is wrong."""
        mode = self.input_mode
        if mode == "pair_btf":
            if x.ndim != 3 or x.shape[-1] != 2:
                raise ValueError(
                    f"pair_btf expects input shape [B, T, 2], got {list(x.shape)}."
                )
        elif mode == "pair_bct":
            if x.ndim != 3 or x.shape[1] != 2:
                raise ValueError(
                    f"pair_bct expects input shape [B, 2, T], got {list(x.shape)}."
                )
        elif mode == "pair_bct1":
            if x.ndim != 4 or x.shape[1] != 2 or x.shape[-1] != 1:
                raise ValueError(
                    f"pair_bct1 expects input shape [B, 2, T, 1], got {list(x.shape)}."
                )
        elif mode == "regular_midpoint_4d":
            if x.ndim != 4 or x.shape[1] != 1:
                raise ValueError(
                    f"regular_midpoint_4d expects input shape [B, 1, T, X], "
                    f"got {list(x.shape)}."
                )
            if x.shape[-1] < 3:
                raise ValueError(
                    f"regular_midpoint_4d requires at least 3 traces (X >= 3), "
                    f"got X={x.shape[-1]}."
                )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Input tensor.  Shape depends on ``input_mode``.
        mask : Ignored (trainer compatibility).
        positions : Ignored.
        **kwargs : Ignored.

        Returns
        -------
        Output tensor with shape matching the input mode convention.
        """
        self._validate_input(x)

        if self.input_mode == "pair_btf":
            return self.core(x)

        elif self.input_mode == "pair_bct":
            # [B, 2, T] -> [B, T, 2] -> core -> [B, T, 1] -> [B, 1, T]
            x_btf = x.transpose(1, 2)
            out_btf = self.core(x_btf)
            return out_btf.transpose(1, 2)

        elif self.input_mode == "pair_bct1":
            # [B, 2, T, 1] -> [B, 2, T] -> core via transposition
            x_sq = x.squeeze(-1)
            x_btf = x_sq.transpose(1, 2)
            out_btf = self.core(x_btf)
            out_bct = out_btf.transpose(1, 2)
            return out_bct.unsqueeze(-1)

        elif self.input_mode == "regular_midpoint_4d":
            return self._forward_regular_midpoint_4d(x)

        raise ValueError(f"Unhandled input_mode: {self.input_mode}")

    # ------------------------------------------------------------------
    # Full-gather adapter
    # ------------------------------------------------------------------

    def _forward_regular_midpoint_4d(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized midpoint prediction over a 4D gather.

        Extracts (left, right) trace pairs for each target position,
        runs the core network on all pairs, and writes predictions back.
        """
        B, C, T, X = x.shape  # C == 1 (validated)

        target_indices = list(
            range(self.target_offset, X - 1, self.target_stride)
        )
        if not target_indices:
            # No valid target positions — return input unchanged.
            return x

        n_targets = len(target_indices)

        # Build pairs: [B, n_targets, T, 2]
        left_traces = torch.stack(
            [x[:, 0, :, j - 1] for j in target_indices], dim=1
        )  # [B, n_targets, T]
        right_traces = torch.stack(
            [x[:, 0, :, j + 1] for j in target_indices], dim=1
        )  # [B, n_targets, T]
        pairs = torch.stack([left_traces, right_traces], dim=-1)  # [B, n_targets, T, 2]

        # Run core (possibly chunked along the target dimension).
        if self.spatial_chunk_size is not None and n_targets > self.spatial_chunk_size:
            pred_chunks: List[torch.Tensor] = []
            for start in range(0, n_targets, self.spatial_chunk_size):
                end = min(start + self.spatial_chunk_size, n_targets)
                chunk_pairs = pairs[:, start:end, :, :]  # [B, chunk, T, 2]
                chunk_flat = chunk_pairs.reshape(B * (end - start), T, 2)
                chunk_pred = self.core(chunk_flat)  # [B*chunk, T, 1]
                chunk_pred = chunk_pred.reshape(B, end - start, T)  # [B, chunk, T]
                pred_chunks.append(chunk_pred)
            pred_traces = torch.cat(pred_chunks, dim=1)  # [B, n_targets, T]
        else:
            pairs_flat = pairs.reshape(B * n_targets, T, 2)
            pred_flat = self.core(pairs_flat)  # [B*n_targets, T, 1]
            pred_traces = pred_flat.reshape(B, n_targets, T)  # [B, n_targets, T]

        # Write predictions back into output gather.
        output = x.clone()
        for k, j in enumerate(target_indices):
            output[:, 0, :, j] = pred_traces[:, k, :]

        return output
