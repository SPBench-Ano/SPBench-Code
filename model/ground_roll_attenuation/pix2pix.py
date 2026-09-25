"""Pix2Pix cGAN generator and discriminator for ground-roll attenuation.

Generator: U-Net with 7-level encoder/decoder, skip connections, ConvTranspose2d
upsampling. Discriminator: 4-level PatchGAN (Isola et al., 2017).

References
----------
Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks",
CVPR 2017.  https://arxiv.org/abs/1611.07004
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


# ---------------------------------------------------------------------------
# Generator (U-Net)
# ---------------------------------------------------------------------------

@register_model("pix2pix_generator")
class Pix2PixGenerator(nn.Module):
    """Pix2Pix-style U-Net generator for noise prediction.

    Encoder: 7× Conv2d(4, stride=2) + BN + LeakyReLU(0.2); first layer no BN.
    Decoder: 7× ConvTranspose2d(4, stride=2) + BN + [Dropout] + ReLU;
    first two decoder blocks have 50% dropout.
    Skip connections from encoder layer i to decoder layer (6-i).
    Output: Conv2d + Tanh.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        # ---------- Encoder --------------------------------------------------
        # ch progression: 64 → 128 → 256 → 512 → 512 → 512 → 512
        enc_chs = [64, 128, 256, 512, 512, 512, 512]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for i, c in enumerate(enc_chs):
            layers: List[nn.Module] = [
                nn.Conv2d(prev, c, kernel_size=4, stride=2, padding=1, bias=False),
            ]
            if i != 0:  # first encoder layer: no BN
                layers.append(nn.BatchNorm2d(c))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            self.encoders.append(nn.Sequential(*layers))
            prev = c

        # bottleneck channels = 512 (output of last encoder)

        # ---------- Decoder --------------------------------------------------
        # dec ch progression (in order of execution):
        #   CD512 → CD512 → C512 → C512 → C256 → C128 → C64
        dec_chs = [512, 512, 512, 512, 256, 128, 64]
        # skip channels from encoder (reversed, excluding bottleneck):
        #   enc5=512, enc4=512, enc3=512, enc2=256, enc1=128, enc0=64
        skip_chs = [512, 512, 512, 256, 128, 64]   # enc5 .. enc0

        self.decoders = nn.ModuleList()
        dec_in = 512  # bottleneck
        for i, c in enumerate(dec_chs):
            layers = [
                nn.ConvTranspose2d(dec_in, c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c),
            ]
            if i <= 1:  # first two decoder blocks: dropout 50%
                layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU(inplace=False))
            self.decoders.append(nn.Sequential(*layers))
            # next input = this output + skip (except last layer)
            skip = skip_chs[i] if i < len(skip_chs) else 0
            dec_in = c + skip

        # ---------- Output ---------------------------------------------------
        self.head = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=7, padding=3, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips: List[torch.Tensor] = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)

        # Bottleneck = skips[-1] (enc6, 512 ch).  Skips for decoder:
        #   enc5, enc4, enc3, enc2, enc1, enc0 (skips[-2] .. skips[0])
        decoder_skips = list(reversed(skips[:-1]))   # 6 tensors
        h = skips[-1]  # bottleneck

        # first 6 decoder layers with skip connections
        for dec, skip in zip(self.decoders[:-1], decoder_skips):
            h = dec(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(
                    h, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            h = torch.cat([skip, h], dim=1)

        # final decoder layer (C64, no skip)
        h = self.decoders[-1](h)

        return self.head(h)


# ---------------------------------------------------------------------------
# Discriminator (PatchGAN)
# ---------------------------------------------------------------------------

class Pix2PixDiscriminator(nn.Module):
    """4-level PatchGAN discriminator.

    Input: concatenation of condition (noisy) and target (noise) along channel
    dim → ``(B, in_channels * 2, H, W)``. Output: ``(B, 1, H', W')`` patch
    scores (logits, no Sigmoid — use BCEWithLogitsLoss).
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        chs = [64, 128, 256, 512]
        layers: List[nn.Module] = []
        prev = in_channels * 2  # condition + real/fake
        for i, c in enumerate(chs):
            layers.append(
                nn.Conv2d(prev, c, kernel_size=4, stride=2, padding=1, bias=False)
            )
            if i != 0:  # first layer: no BN
                layers.append(nn.BatchNorm2d(c))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            prev = c
        layers.append(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False)
        )
        self.net = nn.Sequential(*layers)

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([condition, target], dim=1)
        return self.net(x)
