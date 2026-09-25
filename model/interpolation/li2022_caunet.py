"""CA-Unet from Li et al., "CA-Unet: Coordinate Attention U-Net for Seismic
Data Reconstruction" (IEEE, 2022).

Paper-explicit: standard U-Net encoder-decoder backbone with Coordinate
Attention blocks (directional H/W pooling, shared 1x1 conv, H-Swish, split,
two 1x1 convs, Sigmoid, element-wise multiply).  Hybrid SSIM + L1 loss
(reuses the shared ``ssim_l1`` loss alias).

Reproduction assumptions: 3x3 kernels, Conv->BN->ReLU ordering, depth=3 with
base_channels=32 (~1M params), CA blocks after every double-conv stage
("7-layer CA-U-Net"), CA reduction ratio=16.  Training hyperparameters
inherited from the ANet paper (same task, same domain).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


class _DoubleConv(nn.Module):
    """(Conv->BN->ReLU) x 2 with same spatial size."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CoordAttention2D(nn.Module):
    """Coordinate Attention block from Hou et al. (CVPR 2021).

    Decomposes channel attention into two 1D directional encodings so that
    long-range spatial dependencies along height and width are captured with
    lightweight 1x1 convolutions.  Not SE (no 2D global pool), not CBAM (no
    max-pooling or 2D spatial attention), not an attention gate.

    Parameters
    ----------
    channels         : input/output feature channels.
    reduction_ratio  : channel reduction for the shared middle layer.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if reduction_ratio <= 0:
            raise ValueError(f"reduction_ratio must be positive, got {reduction_ratio}.")
        reduced = max(1, channels // reduction_ratio)
        self.channels = int(channels)
        self.reduction_ratio = int(reduction_ratio)
        self.reduced = reduced

        self.shared_conv = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(reduced)
        self.conv_h = nn.Conv2d(reduced, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(reduced, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(
                f"CoordAttention2D expects {self.channels} channels, got {c}."
            )

        h_pool = F.adaptive_avg_pool2d(x, (h, 1))        # (B, C, H, 1)
        w_pool = F.adaptive_avg_pool2d(x, (1, w))        # (B, C, 1, W)
        h_pool_t = h_pool.permute(0, 1, 3, 2)             # (B, C, 1, H)
        z = torch.cat([h_pool_t, w_pool], dim=-1)         # (B, C, 1, H+W)

        z = self.shared_conv(z)                            # (B, C_r, 1, H+W)
        z = self.bn(z)
        z = F.hardswish(z)

        z_h, z_w = z.split([h, w], dim=-1)                 # (B,C_r,1,H), (B,C_r,1,W)
        a_h = torch.sigmoid(self.conv_h(z_h))              # (B, C, 1, H)
        a_w = torch.sigmoid(self.conv_w(z_w))              # (B, C, 1, W)
        a_h = a_h.permute(0, 1, 3, 2)                     # (B, C, H, 1)

        return x * a_h * a_w


@register_model("li2022_caunet")
class Li2022CAUNet(nn.Module):
    """CA-Unet: U-Net with Coordinate Attention blocks after every double-conv stage.

    Parameters
    ----------
    in_channels        : input channels (paper: 1, no mask channel).
    out_channels       : output channels (paper: 1).
    base_channels      : first encoder channels (paper-inferred: 32).
    depth              : encoder stages (paper-inferred: 3, giving 7 layers).
    ca_reduction_ratio : CA channel reduction (paper-standard: 16).
    kernel_size        : convolution kernel size (reproduction-assumption: 3).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        ca_reduction_ratio: int = 16,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.depth = int(depth)
        self.ca_reduction_ratio = int(ca_reduction_ratio)

        chans: List[int] = [base_channels * (2 ** i) for i in range(depth)]

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        self.encoder_cas = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = int(in_channels)
        for c in chans:
            self.encoders.append(_DoubleConv(prev, c, kernel_size))
            self.encoder_cas.append(CoordAttention2D(c, ca_reduction_ratio))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = c

        # ---- Bottleneck ----
        self.bottleneck = _DoubleConv(chans[-1], chans[-1] * 2, kernel_size)
        self.bottleneck_ca = CoordAttention2D(chans[-1] * 2, ca_reduction_ratio)

        # ---- Decoder ----
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.decoder_cas = nn.ModuleList()
        dec_in = chans[-1] * 2
        for c in reversed(chans):
            self.upconvs.append(nn.ConvTranspose2d(dec_in, c, kernel_size=2, stride=2))
            self.decoders.append(_DoubleConv(c * 2, c, kernel_size))
            self.decoder_cas.append(CoordAttention2D(c, ca_reduction_ratio))
            dec_in = c

        # ---- Output head ----
        self.head = nn.Conv2d(chans[0], out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x    : input tensor of shape ``(B, in_channels, H, W)``.
        mask : ignored (accepted for trainer compatibility).

        Returns
        -------
        Output tensor of shape ``(B, out_channels, H, W)``.
        """
        if x.dim() != 4:
            raise ValueError(
                f"Li2022CAUNet expects a 4D input (B, C, H, W), got ndim={x.dim()}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Li2022CAUNet expects {self.in_channels} input channels, got {x.shape[1]}."
            )

        skips: List[torch.Tensor] = []
        h = x
        for enc, ca, pool in zip(self.encoders, self.encoder_cas, self.pools):
            h = enc(h)
            h = ca(h)
            skips.append(h)
            h = pool(h)

        h = self.bottleneck(h)
        h = self.bottleneck_ca(h)

        for up, dec, ca, skip in zip(
            self.upconvs, self.decoders, self.decoder_cas, reversed(skips)
        ):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([skip, h], dim=1)
            h = dec(h)
            h = ca(h)

        return self.head(h)
