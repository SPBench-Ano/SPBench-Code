"""8-layer residual CNN from Wang et al. "Deep-learning-based seismic data
interpolation: A preliminary result" (Geophysics, 2019).

Conservative reproduction profile:
- 8 Conv2d layers: 1 stem + 6 residual-block convs + 1 output head
- 3 residual blocks with identity shortcuts
- No spatial downsampling / upsampling
- Default: features=64, kernel_size=3, norm=none, activation=relu,
  output_mode=direct, conv_bias=True
- Exact parameter count (default config): 222,785

This is a conservative reproduction.  Only the 2D 8-layer residual CNN concept
is paper-explicit.  Exact width, kernel, normalization, and output residual
mode are reproduction-assumptions pending verification from the original paper.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..registry import register_model


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_norm(norm: str, channels: int, num_groups: int = 8) -> nn.Module:
    """Return a normalization layer for the given type and channel count."""
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels)
    if norm == "group":
        g = min(num_groups, channels)
        return nn.GroupNorm(g, channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm type: {norm}")


def _make_activation(activation: str, channels: int = 0) -> nn.Module:
    """Return an activation module for the given name.

    ``channels`` is only used by ``prelu`` to set the parameter count.
    """
    if activation == "relu":
        return nn.ReLU(inplace=False)
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=False)
    if activation == "prelu":
        return nn.PReLU(num_parameters=channels)
    if activation == "elu":
        return nn.ELU(inplace=False)
    if activation == "gelu":
        return nn.GELU()
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {activation}")


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    """Residual block with identity shortcut (no projection).

    Path: Conv → [Norm] → Activation → [Dropout] → Conv → [Norm] → +x → Activation.

    Paper-default: no norm, no dropout, ReLU, bias=True.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm: str = "none",
        activation: str = "relu",
        dropout: float = 0.0,
        conv_bias: bool = True,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        use_bias = conv_bias if norm == "none" else False

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=use_bias)
        self.norm1 = _make_norm(norm, channels, num_groups)
        self.act1 = _make_activation(activation, channels)

        self.dropout: Optional[nn.Module] = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=use_bias)
        self.norm2 = _make_norm(norm, channels, num_groups)
        self.act2 = _make_activation(activation, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + residual
        out = self.act2(out)
        return out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

@register_model("wang2019_resnet")
class Wang2019ResNet(nn.Module):
    """8-layer residual CNN for regularly-missing seismic trace interpolation.

    Reproduces the architecture concept from Wang et al. (Geophysics, 2019).
    The paper-describes an 8-layer residual learning network for 2D seismic
    interpolation.  This implementation uses a conservative profile:
    1 stem conv + 3 residual blocks (2 convs each) + 1 output conv = 8 Conv2d.

    Parameters
    ----------
    in_channels : Input channels (default 1).
    out_channels : Output channels (default 1).
    features : Feature channels throughout the residual body (default 64).
    num_blocks : Number of residual blocks (paper: 3, yielding 8 Conv2d total).
    kernel_size : Conv kernel size, must be a positive odd integer (default 3).
    activation : Activation function. One of ``"relu"``, ``"leaky_relu"``,
        ``"prelu"``, ``"elu"``, ``"gelu"``, ``"none"`` (default ``"relu"``).
    norm : Normalization type. One of ``"batch"``, ``"instance"``, ``"group"``,
        ``"none"`` (default ``"none"``).
    output_mode : ``"direct"`` — model predicts the complete data.  ``"global_residual"``
        — model predicts a residual added to the input.  Default ``"direct"``.
    conv_bias : Whether conv layers include a bias term (default True).
    dropout : Dropout2d rate inside each residual block (default 0.0).
    num_groups : Group count for GroupNorm (only used when ``norm="group"``).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: int = 64,
        num_blocks: int = 3,
        kernel_size: int = 3,
        activation: str = "relu",
        norm: str = "none",
        output_mode: str = "direct",
        conv_bias: bool = True,
        dropout: float = 0.0,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        # ---- Validation ----
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}."
            )
        if features <= 0:
            raise ValueError(f"features must be positive, got {features}.")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if output_mode not in ("direct", "global_residual"):
            raise ValueError(
                f"output_mode must be 'direct' or 'global_residual', got '{output_mode}'."
            )
        if output_mode == "global_residual" and in_channels != out_channels:
            raise ValueError(
                "output_mode='global_residual' requires in_channels == out_channels, "
                f"got in_channels={in_channels}, out_channels={out_channels}."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_mode = output_mode
        self._num_conv_layers = 2 * num_blocks + 2  # stem + 2*blocks + output

        padding = kernel_size // 2
        use_bias = conv_bias if norm == "none" else False

        block_kwargs = dict(
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            dropout=dropout,
            conv_bias=conv_bias,
            num_groups=num_groups,
        )

        # ---- Stem ----
        self.stem_conv = nn.Conv2d(in_channels, features, kernel_size, padding=padding, bias=use_bias)
        self.stem_norm = _make_norm(norm, features, num_groups)
        self.stem_act = _make_activation(activation, features)

        # ---- Residual blocks ----
        self.blocks = nn.ModuleList([
            _ResidualBlock(features, **block_kwargs) for _ in range(num_blocks)
        ])

        # ---- Output head ----
        self.head = nn.Conv2d(features, out_channels, kernel_size, padding=padding, bias=True)

    @property
    def num_conv_layers(self) -> int:
        """Total number of Conv2d layers (computed from num_blocks)."""
        return self._num_conv_layers

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
        x : Input tensor of shape ``(B, in_channels, H, W)``.
        mask : Ignored (accepted for trainer compatibility).
        positions : Ignored.
        **kwargs : Ignored.

        Returns
        -------
        Output tensor of shape ``(B, out_channels, H, W)``.
        """
        h = self.stem_conv(x)
        h = self.stem_norm(h)
        h = self.stem_act(h)

        for block in self.blocks:
            h = block(h)

        out = self.head(h)

        if self.output_mode == "global_residual":
            # in_channels == out_channels is enforced in __init__.
            out = x + out

        return out
