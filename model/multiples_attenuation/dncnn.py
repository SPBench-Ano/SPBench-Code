"""DnCNN with residual output x - F(x) for denoising / restoration (Zhang et al., 2017)."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..registry import register_model


@register_model("dncnn")
class DnCNN(nn.Module):
    """Stacked Conv–BN–ReLU backbone; forward returns the residual reconstruction x - noise."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 17,
        base_channels: int = 64,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError(
                f"DnCNN residual form requires in_channels == out_channels; got {in_channels} vs {out_channels}."
            )
        if depth < 2:
            raise ValueError(f"DnCNN depth must be >= 2, got {depth}.")
        padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(
                    base_channels,
                    base_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(base_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                base_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.net(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
