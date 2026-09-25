"""2D U-Net from Chai et al. "Deep Learning for Regularly Missing Data Reconstruction"
(IEEE TGRS, 2020).

Paper architecture: 50 layers = 1 input + 19 Conv + 18 ReLU + 4 MaxPool + 4 Upsample + 4 Concat.
- 18 hidden convolutions use 5x5 kernels with same padding.
- 1 output convolution uses 1x1 kernel.
- No BatchNorm, no Dropout, non-learned upsampling, Conv bias=True.
- Channels: [64, 128, 256, 512, 1024].  Exact parameter count: 87,149,953.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


# ---------------------------------------------------------------------------
# Helper factories (used when extension norm / activation is requested)
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


def _make_activation(activation: str) -> nn.Module:
    """Return an activation module for the given name."""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if activation == "elu":
        return nn.ELU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {activation}")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """(Conv -> [Norm] -> Activation -> [Dropout]) x 2, same spatial size.

    Paper-faithful defaults: 5x5 conv, no norm, ReLU, no dropout, bias=True.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 5,
        norm: str = "none",
        activation: str = "relu",
        dropout: float = 0.0,
        conv_bias: bool = True,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        # When a norm layer follows, conv bias is redundant.
        use_bias = conv_bias if norm == "none" else False

        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=use_bias),
            _make_norm(norm, out_ch, num_groups),
            _make_activation(activation),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        layers.extend([
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding, bias=use_bias),
            _make_norm(norm, out_ch, num_groups),
            _make_activation(activation),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

@register_model("chai2020_unet")
class Chai2020UNet(nn.Module):
    """2D U-Net for regularly-missing seismic trace reconstruction.

    Reproduces Chai et al. (IEEE TGRS, 2020).  The paper-default constructor
    arguments produce the exact 50-layer network described in the paper:
    19 Conv (18 x 5x5 + 1 x 1x1), 18 ReLU, 4 MaxPool, 4 Upsample, 4 Concat.

    Parameters
    ----------
    in_channels : Input channels (default 1).
    out_channels : Output channels (default 1).
    base_channels : Channels in the first encoder block (paper: 64).
    num_levels : Encoder/decoder depth (paper: 4).
    kernel_size : Conv kernel size (paper: 5).
    norm : Normalization type. One of ``"batch"``, ``"instance"``, ``"group"``,
        ``"none"`` (paper: ``"none"``).
    activation : Activation function. One of ``"relu"``, ``"leaky_relu"``,
        ``"elu"``, ``"gelu"``, ``"none"`` (paper: ``"relu"``).
    upsample : Upsampling mode. ``"nearest"`` / ``"bilinear"`` use non-learned
        upsampling (paper: non-learned, reproduced as ``"nearest"``).
        ``"transpose"`` uses ``ConvTranspose2d`` (extension).
    dropout : Dropout2d rate inside each ``_ConvBlock`` (paper: 0.0).
    residual_learning : If True, ``output = input + residual`` (paper: False).
    conv_bias : Whether conv layers include a bias term (paper: True).
    num_groups : Group count for GroupNorm (only used when ``norm="group"``).
    alignment_pad_mode : Padding mode for size-alignment to a multiple of
        ``2**num_levels``. One of ``"reflect"``, ``"replicate"``, ``"constant"``.
        This is a repository adaptation; the paper uses 112x112 patches which
        are already divisible by 16.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        num_levels: int = 4,
        kernel_size: int = 5,
        norm: str = "none",
        activation: str = "relu",
        upsample: str = "nearest",
        dropout: float = 0.0,
        residual_learning: bool = False,
        conv_bias: bool = True,
        num_groups: int = 8,
        alignment_pad_mode: str = "reflect",
    ) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError(f"num_levels must be >= 2, got {num_levels}.")
        if residual_learning and in_channels != out_channels:
            raise ValueError(
                "residual_learning=True requires in_channels == out_channels, "
                f"got in_channels={in_channels}, out_channels={out_channels}."
            )
        if upsample not in ("nearest", "bilinear", "transpose"):
            raise ValueError(
                f"upsample must be 'nearest', 'bilinear', or 'transpose', got '{upsample}'."
            )
        if alignment_pad_mode not in ("reflect", "replicate", "constant"):
            raise ValueError(
                f"alignment_pad_mode must be 'reflect', 'replicate', or 'constant', "
                f"got '{alignment_pad_mode}'."
            )

        self.num_levels = num_levels
        self.residual_learning = residual_learning
        self.alignment_pad_mode = alignment_pad_mode
        self._stride = 2 ** num_levels

        chans: List[int] = [base_channels * (2 ** i) for i in range(num_levels)]

        block_kwargs = dict(
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            dropout=dropout,
            conv_bias=conv_bias,
            num_groups=num_groups,
        )

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for c in chans:
            self.encoders.append(_ConvBlock(prev, c, **block_kwargs))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = c

        # ---- Bottleneck ----
        self.bottleneck = _ConvBlock(chans[-1], chans[-1] * 2, **block_kwargs)

        # ---- Decoder ----
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in: int = chans[-1] * 2  # bottleneck output channels
        for c in reversed(chans):
            if upsample in ("nearest", "bilinear"):
                # Non-learned upsampling preserves channels.
                align = False if upsample == "bilinear" else None
                self.upconvs.append(
                    nn.Upsample(scale_factor=2, mode=upsample, align_corners=align)  # type: ignore[arg-type]
                )
                dec_block_in = dec_in + c  # e.g. 1024+512=1536, 512+256=768, ...
            elif upsample == "transpose":
                # Learned upsampling changes channels from dec_in to c.
                self.upconvs.append(
                    nn.ConvTranspose2d(dec_in, c, kernel_size=2, stride=2)
                )
                dec_block_in = c + c  # upsampled (now c ch) + skip (c ch)
            else:
                raise ValueError(f"Unsupported upsample mode: {upsample}")
            self.decoders.append(_ConvBlock(dec_block_in, c, **block_kwargs))
            dec_in = c

        # ---- Output head ----
        self.head = nn.Conv2d(chans[0], out_channels, kernel_size=1, bias=True)

    # ------------------------------------------------------------------
    # Size alignment helpers (repository-adaptation)
    # ------------------------------------------------------------------

    def _pad_to_multiple(self, x: torch.Tensor) -> torch.Tensor:
        """Pad spatial dims to the next multiple of ``2**num_levels``."""
        h, w = x.shape[-2:]
        target_h = ((h + self._stride - 1) // self._stride) * self._stride
        target_w = ((w + self._stride - 1) // self._stride) * self._stride
        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h == 0 and pad_w == 0:
            return x
        # reflect padding requires input >= padding amount; if the input is
        # too small in either dimension, fall back to replicate.
        mode = self.alignment_pad_mode
        if mode == "reflect" and (h <= pad_h or w <= pad_w):
            mode = "replicate"
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

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
        mask : Ignored (accepted for compatibility with trainers that pass
            auxiliary arguments).
        positions : Ignored.
        **kwargs : Ignored.

        Returns
        -------
        Output tensor of shape ``(B, out_channels, H, W)``.
        """
        orig_h, orig_w = x.shape[-2:]

        # Pad so that every 2x-downsample path reaches an integer size.
        x = self._pad_to_multiple(x)

        # ---- Encoder ----
        skips: List[torch.Tensor] = []
        h: torch.Tensor = x
        for enc, pool in zip(self.encoders, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)

        # ---- Bottleneck ----
        h = self.bottleneck(h)

        # ---- Decoder ----
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = up(h)
            # Safety: align if upsampled size still mismatches the skip.
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(
                    h, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            h = torch.cat([skip, h], dim=1)
            h = dec(h)

        out = self.head(h)

        # Crop back to original spatial size.
        if out.shape[-2] != orig_h or out.shape[-1] != orig_w:
            out = out[..., :orig_h, :orig_w]

        # ---- Residual learning (extension) ----
        if self.residual_learning:
            # x has been padded; x_input for the residual is the original
            # (cropped) input.  in_channels == out_channels is enforced in __init__.
            out = x[..., :orig_h, :orig_w] + out

        return out
