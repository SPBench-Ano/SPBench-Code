"""Physics-constrained deep learning for ground-roll attenuation.

Three CNNs (CNN1 for signal, CNN2 for ground-roll, CNN3 for SW constraint)
plus an f-k domain binary classifier.  Asymmetric kernels (7×21, 3×9) encode
geophysical prior: longer time-axis kernels capture dispersion, shorter
trace-axis kernels respect lateral stationarity.  A single skip connection
links the first encoder layer to the last decoder layer.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


# ---------------------------------------------------------------------------
# asymmetric encoder / decoder building blocks
# ---------------------------------------------------------------------------

class _AsymConvBlock(nn.Module):
    """Conv2d(asymmetric kernel) → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: Tuple[int, int]) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=(kernel[0] // 2, kernel[1] // 2), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _EncoderLevel(nn.Module):
    """Conv 3×9 + MaxPool(2,2) — one encoder stage."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _AsymConvBlock(in_ch, out_ch, (3, 9))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        return h, self.pool(h)


class _DecoderLevel(nn.Module):
    """Upsample ×2 + Conv 3×9 — one decoder stage."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _AsymConvBlock(in_ch, out_ch, (3, 9))

    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        h = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if target_size is not None and h.shape[-2:] != target_size:
            h = F.interpolate(h, size=target_size, mode="bilinear", align_corners=False)
        return self.conv(h)


# ---------------------------------------------------------------------------
# CNN1 / CNN2 / CNN3 — signal and ground-roll estimation networks
# ---------------------------------------------------------------------------

class _PhysicsCNN(nn.Module):
    """Encoder-decoder with asymmetric kernels and single skip connection.

    Parameters
    ----------
    in_channels, out_channels : I/O channel counts.
    base_channels : first-level feature width.
    num_levels : number of encoder/decoder stages (excluding initial conv).
    num_output_convs : extra conv layers after decoder (3 for CNN1, 2 for CNN2/3).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 3,
        num_output_convs: int = 2,
    ) -> None:
        super().__init__()

        # initial conv 7×21 (no downsampling)
        self.init_conv = _AsymConvBlock(in_channels, base_channels, (7, 21))

        chs = [base_channels * (2 ** i) for i in range(num_levels)]  # [32, 64, 128] for 3 levels

        # encoder
        self.encoders = nn.ModuleList()
        prev = base_channels
        for c in chs:
            self.encoders.append(_EncoderLevel(prev, c))
            prev = c

        # bottleneck
        bottleneck_ch = chs[-1] * 2
        self.bottleneck = _AsymConvBlock(prev, bottleneck_ch, (3, 9))

        # decoder
        self.decoders = nn.ModuleList()
        dec_in = bottleneck_ch
        for c in reversed(chs):
            self.decoders.append(_DecoderLevel(dec_in, c))
            dec_in = c

        # final fusion: skip from init_conv (first encoder) concat with last decoder
        self.final_fuse = _AsymConvBlock(dec_in + base_channels, base_channels, (3, 9))

        # output convs
        out_layers: List[nn.Module] = []
        prev_out = base_channels
        for i in range(num_output_convs):
            is_last = (i == num_output_convs - 1)
            oc = out_channels if is_last else base_channels
            out_layers.append(
                nn.Conv2d(prev_out, oc, kernel_size=3, padding=1, bias=not is_last)
            )
            if not is_last:
                out_layers.append(nn.BatchNorm2d(oc))
                out_layers.append(nn.ReLU(inplace=False))
            prev_out = oc
        self.output = nn.Sequential(*out_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.init_conv(x)       # (B, base_ch, H, W) — skip source

        # encoder
        skips: List[torch.Tensor] = []
        h = h0
        for enc in self.encoders:
            skip_h, h = enc(h)
            skips.append(skip_h)

        # bottleneck
        h = self.bottleneck(h)

        # decoder
        for dec in self.decoders:
            h = dec(h)

        # single skip: init_conv (h0) → last decoder
        if h.shape[-2:] != h0.shape[-2:]:
            h = F.interpolate(h, size=h0.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.cat([h0, h], dim=1)
        h = self.final_fuse(h)

        return self.output(h)


# ---------------------------------------------------------------------------
# f-k domain classifier
# ---------------------------------------------------------------------------

class FKClassifier(nn.Module):
    """Binary classifier operating on f-k domain (real + imag channels).

    Architecture: Conv3×3 → 3× (Conv3×3 + MaxPool) → Dropout → 3 FC layers.
    Output: logit for signal (1) vs ground-roll (0).
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 32, dropout: float = 0.5) -> None:
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=False),
        )

        chs = [base_channels, base_channels * 2, base_channels * 4]  # 32, 64, 128
        self.down_layers = nn.ModuleList()
        prev = base_channels
        for c in chs:
            self.down_layers.append(
                nn.Sequential(
                    nn.Conv2d(prev, c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
            prev = c

        self.dropout = nn.Dropout(dropout)

        # Adaptive pooling collapses spatial dims to 1×1 so the FC input
        # size is just `prev` regardless of input patch dimensions.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, H, W) — stacked [real, imag] from f-k transform."""
        h = self.init_conv(x)
        for down in self.down_layers:
            h = down(h)
        h = self.dropout(h)
        h = self.pool(h)
        return self.fc(h)


# ---------------------------------------------------------------------------
# physics-constrained separation network (wrapper)
# ---------------------------------------------------------------------------

@register_model("physics_unet")
class PhysicsSeparationNet(nn.Module):
    """Three-CNN physics-constrained network for signal / ground-roll separation.

    CNN1 estimates clean signal X from noisy input Z.
    CNN2 estimates ground-roll Y from noisy input Z.
    CNN3 maps estimated X back to ground-roll (SW constraint).

    At inference only CNN1 (and optionally CNN2) is needed.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 3,
    ) -> None:
        super().__init__()
        self.cnn1 = _PhysicsCNN(in_channels, out_channels, base_channels, num_levels, num_output_convs=3)
        self.cnn2 = _PhysicsCNN(in_channels, out_channels, base_channels, num_levels, num_output_convs=2)
        self.cnn3 = _PhysicsCNN(out_channels, out_channels, base_channels, num_levels, num_output_convs=2)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (X, Y, Y_recover)."""
        x = self.cnn1(z)
        y = self.cnn2(z)
        y_recover = self.cnn3(x)
        return x, y, y_recover

    def denoise(self, z: torch.Tensor) -> torch.Tensor:
        """Inference: return estimated clean signal X."""
        return self.cnn1(z)
