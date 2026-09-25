"""Basic 2D U-Net for seismic interpolation / denoising."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..registry import register_model


class _DoubleConv(nn.Module):
    """(Conv->BN->ReLU) x 2 with same spatial size."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@register_model("unet")
class UNet(nn.Module):
    """Minimal U-Net with configurable depth and channel width."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"UNet depth must be >= 2, got {depth}.")

        chans: List[int] = [base_channels * (2**i) for i in range(depth)]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev = in_channels
        for c in chans:
            self.encoders.append(_DoubleConv(prev, c))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = c

        self.bottleneck = _DoubleConv(chans[-1], chans[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in = chans[-1] * 2
        for c in reversed(chans):
            self.upconvs.append(nn.ConvTranspose2d(dec_in, c, kernel_size=2, stride=2))
            self.decoders.append(_DoubleConv(c * 2, c))
            dec_in = c

        self.head = nn.Conv2d(chans[0], out_channels, kernel_size=1)

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
                h = torch.nn.functional.interpolate(
                    h, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            h = torch.cat([skip, h], dim=1)
            h = dec(h)
        return self.head(h)
