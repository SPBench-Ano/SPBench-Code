"""2D U-Net with additive attention gates on skip connections (Attention U-Net style)."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..registry import register_model


class _AttentionGate(nn.Module):
    """Gate skip features using upsampled coarse features (`g`)."""

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r = self.relu(self.W_g(g) + self.W_x(x))
        return x * torch.sigmoid(self.psi(r))


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


@register_model("atten_unet")
class UNet(nn.Module):
    """U-Net with Attention Gate blocks between upsampled features and skips."""

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
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in = chans[-1] * 2
        for c in reversed(chans):
            self.upconvs.append(nn.ConvTranspose2d(dec_in, c, kernel_size=2, stride=2))
            mid = max(c // 2, 8)
            self.attention_gates.append(_AttentionGate(F_g=c, F_l=c, F_int=mid))
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

        for up, attn, dec, skip in zip(
            self.upconvs, self.attention_gates, self.decoders, reversed(skips)
        ):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = torch.nn.functional.interpolate(
                    h, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            skip_g = attn(g=h, x=skip)
            h = torch.cat([skip_g, h], dim=1)
            h = dec(h)
        return self.head(h)
