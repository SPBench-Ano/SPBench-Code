"""SANet: Soft Attention Network with multi-branch feature extraction for ground-roll attenuation.

Multi-branch parallel convolutions with different kernel sizes extract multi-scale
features; a soft-attention mechanism adaptively weights and fuses them before the
residual connection.  The model predicts the additive ground-roll noise component.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..registry import register_model


class _MultiBranchConv(nn.Module):
    """Parallel Conv-BN-ReLU branches with different kernel sizes."""

    def __init__(self, in_ch: int, out_ch: int, kernel_sizes: List[int]) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        branch_out = out_ch // len(kernel_sizes)
        remainder = out_ch - branch_out * (len(kernel_sizes) - 1)
        for i, k in enumerate(kernel_sizes):
            b_out = remainder if i == 0 else branch_out
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, b_out, kernel_size=k, padding=k // 2, bias=False),
                    nn.BatchNorm2d(b_out),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class _SoftAttention(nn.Module):
    """Spatial soft attention: features → Conv → Sigmoid → gate."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class _SANetBlock(nn.Module):
    """Multi-branch convs → fusion → soft attention → residual."""

    def __init__(self, channels: int, kernel_sizes: List[int]) -> None:
        super().__init__()
        self.multi_branch = _MultiBranchConv(channels, channels, kernel_sizes)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.attention = _SoftAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.multi_branch(x)
        h = self.fuse(h)
        h = self.attention(h)
        return x + h


@register_model("sanet")
class SANet(nn.Module):
    """Soft Attention Network for ground-roll noise prediction.

    Parameters
    ----------
    in_channels, out_channels : input / output channel count.
    base_channels : channel width of internal feature maps.
    num_blocks : number of SANet blocks stacked in the backbone.
    kernel_sizes : list of kernel sizes for the multi-branch convs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_blocks: int = 8,
        kernel_sizes: List[int] = [3, 5, 7],  # noqa: B006
    ) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            *[_SANetBlock(base_channels, kernel_sizes) for _ in range(num_blocks)]
        )

        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.blocks(h)
        return self.output_conv(h)
