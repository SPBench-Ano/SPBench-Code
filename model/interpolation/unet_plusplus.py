"""UNet++ for seismic interpolation / denoising.

Reference: Zhou et al., "UNet++: Redesigning Skip Connections to Exploit
Multiscale Features in Image Segmentation", IEEE TMI 2019.
Decoder logic adapted from segmentation_models.pytorch (MIT licence).
"""

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


@register_model("unet_plusplus")
class UNetPlusPlus(nn.Module):
    """UNet++ with nested dense skip connections.

    Parameters match the existing ``unet`` model for drop-in replacement.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"UNetPlusPlus depth must be >= 2, got {depth}.")

        chans: List[int] = [base_channels * (2**i) for i in range(depth)]

        # ----- Encoder -----
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for c in chans:
            self.encoders.append(_DoubleConv(prev, c))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = c

        # ----- Bottleneck -----
        self.bottleneck = _DoubleConv(chans[-1], chans[-1] * 2)

        # ----- Decoder nodes -----
        # Node X^{i,j}: i = resolution level, j = layer index (0 = encoder)
        # For j > 0: upconv from X^{i+1, j-1} then concat all X^{i, k} (k < j)
        self.upconvs: nn.ModuleDict = nn.ModuleDict()
        self.decoders: nn.ModuleDict = nn.ModuleDict()

        for i in range(depth):  # resolution level
            lower_ch = chans[-1] * 2 if i == depth - 1 else chans[i + 1]
            for j in range(1, depth - i + 1):  # layer index
                up_name = f"up_{i}_{j}"
                dec_name = f"dec_{i}_{j}"
                self.upconvs[up_name] = nn.ConvTranspose2d(lower_ch, chans[i], kernel_size=2, stride=2)
                # concat: upconv output (chans[i]) + j previous same-res features (j * chans[i])
                self.decoders[dec_name] = _DoubleConv((j + 1) * chans[i], chans[i])

        self.head = nn.Conv2d(chans[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store encoder features and decoder outputs
        # enc[i] = X^{i,0}
        enc: List[torch.Tensor] = []
        h = x
        for encoder, pool in zip(self.encoders, self.pools):
            h = encoder(h)
            enc.append(h)
            h = pool(h)

        # bottleneck = X^{depth,0}
        bottleneck = self.bottleneck(h)

        # dense[i][j] stores X^{i,j}; dense[i][0] is enc[i]
        dense: List[List[torch.Tensor]] = [[f] for f in enc]
        # Append bottleneck as its own level
        dense.append([bottleneck])

        depth = len(enc)
        for i in range(depth - 1, -1, -1):  # from deepest to shallowest
            for j in range(1, depth - i + 1):
                up = self.upconvs[f"up_{i}_{j}"]
                dec = self.decoders[f"dec_{i}_{j}"]
                # upsample from lower level
                lower = dense[i + 1][j - 1]
                h_up = up(lower)
                # align spatial size (safeguard)
                target_size = dense[i][0].shape[-2:]
                if h_up.shape[-2:] != target_size:
                    h_up = torch.nn.functional.interpolate(
                        h_up, size=target_size, mode="bilinear", align_corners=False
                    )
                # concatenate all same-resolution features
                feats = [h_up] + dense[i]
                h = dec(torch.cat(feats, dim=1))
                dense[i].append(h)

        return self.head(dense[0][-1])
