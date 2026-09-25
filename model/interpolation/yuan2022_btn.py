"""Blind-Trace Network (BTN) from Yuan et al. "Self-Supervised Learning for
Efficient Antialiasing Seismic Data Interpolation" (IEEE TGRS, 2022).

The BTN enforces a strict receptive-field constraint: the prediction for
trace j must NOT depend on the input at trace j.  Two half-plane U-Net
branches (rotated 90 and 270 degrees) process left-context and right-context
independently, followed by 1x1 convolution fusion.

Conservative reproduction profile:
- 4-level HalfSidedBlindUNet, base_channels=32
- kernel_size=3, encoder/bottleneck norm=none, decoder norm=batch
- nearest upsampling, shared branch weights, LeakyReLU(0.01)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_norm(norm: str, channels: int, num_groups: int = 8) -> nn.Module:
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


def _make_activation(activation: str, negative_slope: float = 0.01) -> nn.Module:
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    if activation == "elu":
        return nn.ELU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {activation}")


# ---------------------------------------------------------------------------
# Half-sided causal convolution
# ---------------------------------------------------------------------------

class BlindTraceConv2d(nn.Module):
    """2D convolution with a half-plane causal receptive field.

    Output row *i* depends only on input rows *0 .. i* (upper half-plane).
    Achieved by padding ``k//2`` zero rows on top before the conv and
    cropping ``k//2`` rows from the bottom afterwards.

    Parameters
    ----------
    in_channels : Number of input channels.
    out_channels : Number of output channels.
    kernel_size : Spatial kernel size (must be a positive odd integer).
    bias : Whether the internal Conv2d uses bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}."
            )
        self.kernel_size = kernel_size
        self._pad = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self._pad
        if p > 0:
            x = F.pad(x, (0, 0, p, 0), mode="constant", value=0.0)
        out = self.conv(x)
        if p > 0:
            out = out[..., :-p, :]
        return out


# ---------------------------------------------------------------------------
# Half-sided max pooling
# ---------------------------------------------------------------------------

class BlindTraceMaxPool2d(nn.Module):
    """Max-pooling that preserves the half-plane causality constraint.

    Before pooling, the feature map is shifted DOWN by one row (top row
    zero-filled, bottom row discarded).  This prevents the pool window from
    mixing the current row with content from rows below.
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero_row = torch.zeros_like(x[..., :1, :])
        shifted = torch.cat([zero_row, x[..., :-1, :]], dim=-2)
        return self.pool(shifted)


# ---------------------------------------------------------------------------
# Blind-trace residual block
# ---------------------------------------------------------------------------

class BlindTraceResidualBlock(nn.Module):
    """Residual block using only half-plane (blind-trace) convolutions.

    Structure::

        BlindTraceConv2d -> [Norm] -> ReLU -> BlindTraceConv2d -> [Norm]
           |                                                           |
           +------- Shortcut (Identity or 1x1 Conv2d) -------->  Add -> ReLU

    Parameters
    ----------
    in_channels : Input channels.
    out_channels : Output channels.
    kernel_size : Convolution kernel size (positive odd integer).
    norm : Normalization type (``"batch"``, ``"instance"``, ``"group"``, ``"none"``).
    activation : Activation type (``"relu"``, ``"leaky_relu"``, ``"none"``, ...).
    conv_bias : Whether the internal convolutions use bias.
    num_groups : Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm: str = "none",
        activation: str = "relu",
        conv_bias: bool = True,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        use_bias = conv_bias if norm == "none" else False

        self.conv1 = BlindTraceConv2d(in_channels, out_channels, kernel_size, bias=use_bias)
        self.norm1 = _make_norm(norm, out_channels, num_groups) if norm != "none" else nn.Identity()
        self.act1 = _make_activation(activation)

        self.conv2 = BlindTraceConv2d(out_channels, out_channels, kernel_size, bias=use_bias)
        self.norm2 = _make_norm(norm, out_channels, num_groups) if norm != "none" else nn.Identity()

        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=conv_bias
            )
        else:
            self.shortcut = nn.Identity()

        self.act_out = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + identity
        out = self.act_out(out)
        return out


# ---------------------------------------------------------------------------
# Half-sided blind U-Net
# ---------------------------------------------------------------------------

class HalfSidedBlindUNet(nn.Module):
    """4-level U-Net where every spatial convolution is a ``BlindTraceConv2d``.

    Encoder path: BlindTraceResidualBlock -> BlindTraceMaxPool2d (repeat).
    Bottleneck: BlindTraceResidualBlock (no pool).
    Decoder path: Upsample -> Concat skip -> BlindTraceResidualBlock (repeat).

    Parameters
    ----------
    in_channels : Input channels.
    base_channels : Channel count at the first encoder level (default 32).
    num_levels : Number of encoder/decoder levels (default 4).
    kernel_size : Convolution kernel size (positive odd integer).
    encoder_norm : Norm for encoder and bottleneck blocks.
    decoder_norm : Norm for decoder blocks.
    activation : Activation type.
    upsample_mode : Upsampling mode (``"nearest"``, ``"bilinear"``, ``"transpose"``).
    conv_bias : Whether convolutions use bias.
    num_groups : Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 4,
        kernel_size: int = 3,
        encoder_norm: str = "none",
        decoder_norm: str = "batch",
        activation: str = "relu",
        upsample_mode: str = "nearest",
        conv_bias: bool = True,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}.")
        if upsample_mode not in ("nearest", "bilinear", "transpose"):
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")

        self.num_levels = num_levels
        self.base_channels = base_channels
        self.upsample_mode = upsample_mode

        chans: List[int] = [base_channels * (2 ** i) for i in range(num_levels)]

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for c in chans:
            self.encoders.append(BlindTraceResidualBlock(
                prev, c, kernel_size=kernel_size, norm=encoder_norm,
                activation=activation, conv_bias=conv_bias, num_groups=num_groups,
            ))
            self.pools.append(BlindTraceMaxPool2d())
            prev = c

        # Bottleneck
        self.bottleneck = BlindTraceResidualBlock(
            chans[-1], chans[-1] * 2, kernel_size=kernel_size, norm=encoder_norm,
            activation=activation, conv_bias=conv_bias, num_groups=num_groups,
        )

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in = chans[-1] * 2
        for c in reversed(chans):
            if upsample_mode == "transpose":
                self.upconvs.append(nn.ConvTranspose2d(dec_in, c, kernel_size=2, stride=2))
                dec_block_in = c + c
            else:
                self.upconvs.append(nn.Upsample(
                    scale_factor=2, mode=upsample_mode,
                    align_corners=False if upsample_mode == "bilinear" else None,
                ))
                dec_block_in = dec_in + c
            self.decoders.append(BlindTraceResidualBlock(
                dec_block_in, c, kernel_size=kernel_size, norm=decoder_norm,
                activation=activation, conv_bias=conv_bias, num_groups=num_groups,
            ))
            dec_in = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        h = x
        for enc, pool in zip(self.encoders, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([skip, h], dim=1)
            h = dec(h)

        return h


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------

def _orient_left(x: torch.Tensor) -> torch.Tensor:
    """Rotate 90 degrees CCW so that left-neighbour traces become the
    upward-visible half-plane in the rotated domain."""
    return torch.rot90(x, k=1, dims=(-2, -1))


def _orient_right(x: torch.Tensor) -> torch.Tensor:
    """Rotate 270 degrees CCW (90 CW) so that right-neighbour traces become the
    upward-visible half-plane in the rotated domain."""
    return torch.rot90(x, k=3, dims=(-2, -1))


def _restore_left(x: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_orient_left``."""
    return torch.rot90(x, k=3, dims=(-2, -1))


def _restore_right(x: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_orient_right``."""
    return torch.rot90(x, k=1, dims=(-2, -1))


def _exclude_current_row(features: torch.Tensor) -> torch.Tensor:
    """Shift features DOWN by one row (top row zeroed, bottom row discarded).

    After this operation, output row *i* can only depend on input rows
    *0 .. i-1*, fully excluding the current row from the receptive field.
    """
    zero_row = torch.zeros_like(features[..., :1, :])
    return torch.cat([zero_row, features[..., :-1, :]], dim=-2)


# ---------------------------------------------------------------------------
# Full Blind-Trace Network
# ---------------------------------------------------------------------------

@register_model("yuan2022_btn")
class Yuan2022BTN(nn.Module):
    """Blind-Trace Network for self-supervised seismic interpolation.

    Two half-plane U-Net branches process left-context and right-context
    independently, followed by 1x1 convolution fusion.  The prediction for
    trace *j* cannot access the input at trace *j*.

    Parameters
    ----------
    in_channels : Input channels (default 1).
    out_channels : Output channels (default 1).
    base_channels : First-level channel count (default 32).
    num_levels : Encoder/decoder levels (default 4).
    kernel_size : Convolution kernel size, must be positive odd (default 3).
    encoder_norm : Norm for encoder/bottleneck (default ``"none"``).
    decoder_norm : Norm for decoder (default ``"batch"``).
    activation : Activation type (default ``"relu"``).
    upsample_mode : Upsampling mode (default ``"nearest"``).
    share_branch_weights : If True, a single ``HalfSidedBlindUNet`` is used
        for both orientations (default True).
    conv_bias : Whether convolutions use bias (default True).
    negative_slope : LeakyReLU slope for the output activation (default 0.01).
    num_groups : Group count for GroupNorm (default 8).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 4,
        kernel_size: int = 3,
        encoder_norm: str = "none",
        decoder_norm: str = "batch",
        activation: str = "relu",
        upsample_mode: str = "nearest",
        share_branch_weights: bool = True,
        conv_bias: bool = True,
        negative_slope: float = 0.01,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        # ---- Validation ----
        if base_channels <= 0:
            raise ValueError(f"base_channels must be > 0, got {base_channels}.")
        if num_levels <= 0:
            raise ValueError(f"num_levels must be > 0, got {num_levels}.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
        if negative_slope < 0:
            raise ValueError(f"negative_slope must be >= 0, got {negative_slope}.")
        _valid_norms = {"none", "batch", "instance", "group"}
        if encoder_norm not in _valid_norms:
            raise ValueError(f"Unknown encoder_norm: {encoder_norm}")
        if decoder_norm not in _valid_norms:
            raise ValueError(f"Unknown decoder_norm: {decoder_norm}")
        _valid_activations = {"relu", "leaky_relu", "elu", "gelu", "none"}
        if activation not in _valid_activations:
            raise ValueError(f"Unknown activation: {activation}")
        if upsample_mode not in ("nearest", "bilinear", "transpose"):
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.share_branch_weights = share_branch_weights

        self._stride = 2 ** num_levels

        # Branches
        branch_kwargs = dict(
            in_channels=in_channels,
            base_channels=base_channels,
            num_levels=num_levels,
            kernel_size=kernel_size,
            encoder_norm=encoder_norm,
            decoder_norm=decoder_norm,
            activation=activation,
            upsample_mode=upsample_mode,
            conv_bias=conv_bias,
            num_groups=num_groups,
        )
        self.branch = HalfSidedBlindUNet(**branch_kwargs)
        if not share_branch_weights:
            self.right_branch = HalfSidedBlindUNet(**branch_kwargs)

        # Aggregation
        self.fusion_conv1 = nn.Conv2d(2 * base_channels, base_channels, kernel_size=1, bias=True)
        self.fusion_activation = nn.ReLU(inplace=False)
        self.fusion_conv2 = nn.Conv2d(base_channels, out_channels, kernel_size=1, bias=True)
        self.output_activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=False)

        self.negative_slope = negative_slope

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape [B, C, T, X], got {tuple(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s), got {x.shape[1]}."
            )
        if x.shape[-2] < 2:
            raise ValueError(f"Time dimension must be >= 2, got {x.shape[-2]}.")
        if x.shape[-1] < 3:
            raise ValueError(f"Trace dimension must be >= 3, got {x.shape[-1]}.")

    def _pad_to_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """Pad spatial dims to multiples of ``2**num_levels`` using constant zeros."""
        h, w = x.shape[-2:]
        target_h = ((h + self._stride - 1) // self._stride) * self._stride
        target_w = ((w + self._stride - 1) // self._stride) * self._stride
        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h == 0 and pad_w == 0:
            return x, (0, pad_h, 0, pad_w)
        out = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        return out, (0, pad_h, 0, pad_w)

    def _crop_to_original(
        self, x: torch.Tensor, orig_h: int, orig_w: int
    ) -> torch.Tensor:
        return x[..., :orig_h, :orig_w]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_branch(self, x: torch.Tensor) -> torch.Tensor:
        if self.share_branch_weights:
            return self.branch(x)
        return self.right_branch(x)

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
        x : Input tensor of shape ``(B, C_in, T, X)``.
        mask : Ignored (trainer compatibility).
        positions : Ignored.
        **kwargs : Ignored.

        Returns
        -------
        Output tensor of shape ``(B, C_out, T, X)``.
        """
        self._validate_input(x)

        orig_h, orig_w = x.shape[-2:]
        x_padded, _ = self._pad_to_multiple(x)

        # Left branch
        left_in = _orient_left(x_padded)
        left_feat = self._forward_branch(left_in)
        left_feat = _exclude_current_row(left_feat)
        left_feat = _restore_left(left_feat)

        # Right branch
        right_in = _orient_right(x_padded)
        right_feat = self._forward_branch(right_in)
        right_feat = _exclude_current_row(right_feat)
        right_feat = _restore_right(right_feat)

        # Fusion
        features = torch.cat([left_feat, right_feat], dim=1)
        out = self.fusion_conv1(features)
        out = self.fusion_activation(out)
        out = self.fusion_conv2(out)
        out = self.output_activation(out)

        # Crop back to original size
        out = self._crop_to_original(out, orig_h, orig_w)
        return out
