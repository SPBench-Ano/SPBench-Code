"""Enhanced U-Net with residual blocks, attention-gated skip connections, and AFM loss.

Combines ResUNet-style residual blocks in encoder/decoder with additive attention
gates on skip connections (Attention U-Net style).  Predicts the additive noise
component (aligned with other denoising models).  Trained with a hybrid MSE +
adaptive-frequency-modulation (AFM) loss that operates in the f-x domain via FFT.

Reference
---------
Ground Roll Attenuation in Seismic Data Based on Enhanced Deep Learning Framework
With Adaptive Frequency Modulation Loss.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from utils.losses import BaseLoss, register_loss


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

class _AttentionGate(nn.Module):
    """Additive attention gate: gate skip features using upsampled coarse features."""

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r = self.relu(self.W_g(g) + self.W_x(x))
        return x * torch.sigmoid(self.psi(r))


class _ResBlock(nn.Module):
    """Conv→BN→ReLU→Conv→BN + identity shortcut."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


# ---------------------------------------------------------------------------
# Enhanced Attention U-Net
# ---------------------------------------------------------------------------

@register_model("enhanced_atten_unet")
class EnhancedAttentionUNet(nn.Module):
    """U-Net with residual encoder/decoder blocks and attention-gated skip connections.

    Predicts the additive ground-roll noise component.
    Denoised signal = noisy input - predicted noise.
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
            raise ValueError(f"depth must be >= 2, got {depth}.")

        chans: List[int] = [base_channels * (2**i) for i in range(depth)]

        # ---- encoder -------------------------------------------------------
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for c in chans:
            self.encoders.append(_ResBlock(prev, c))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = c

        # ---- bottleneck ----------------------------------------------------
        self.bottleneck = _ResBlock(chans[-1], chans[-1] * 2)

        # ---- decoder -------------------------------------------------------
        self.upconvs = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()

        dec_in = chans[-1] * 2
        for c in reversed(chans):
            self.upconvs.append(nn.ConvTranspose2d(dec_in, c, kernel_size=2, stride=2))
            self.attn_gates.append(
                _AttentionGate(F_g=c, F_l=c, F_int=max(c // 2, 8))
            )
            # after attention gate + concatenation: c (gated skip) + c (upsampled) = 2c
            self.decoders.append(_ResBlock(c * 2, c))
            dec_in = c

        # ---- head ----------------------------------------------------------
        self.head = nn.Conv2d(chans[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        h = x
        for enc, pool in zip(self.encoders, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)

        h = self.bottleneck(h)

        for up, gate, dec, skip in zip(
            self.upconvs, self.attn_gates, self.decoders, reversed(skips)
        ):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(
                    h, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            gated_skip = gate(h, skip)
            h = torch.cat([gated_skip, h], dim=1)
            h = dec(h)

        return self.head(h)


# ---------------------------------------------------------------------------
# Hybrid MSE + AFM loss
# ---------------------------------------------------------------------------

@register_loss("hybrid_mse_afm")
class HybridMSEAFMLoss(BaseLoss):
    """MSE in time-space domain + λ × AFM loss in frequency domain.

    Parameters
    ----------
    lambda_afm : weight for the AFM term (default 0.1).
    eps : small constant for numerical stability in adaptive weights.
    """

    def __init__(self, lambda_afm: float = 0.1, eps: float = 1e-3) -> None:
        super().__init__()
        self.lambda_afm = float(lambda_afm)
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **extras: object,
    ) -> torch.Tensor:
        # ---- time-space MSE ------------------------------------------------
        mse = F.mse_loss(pred, target)

        # ---- AFM: adaptive frequency modulation in f-x domain --------------
        # 2D real FFT over spatial dims (H, W)
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        targ_fft = torch.fft.rfft2(target, dim=(-2, -1))

        pred_mag = torch.abs(pred_fft)
        targ_mag = torch.abs(targ_fft)

        diff = torch.abs(pred_mag - targ_mag)

        # Soft energy floor prevents division by near-zero when the target
        # has negligible energy at certain frequencies (common in spectral
        # domain of natural signals).
        safe_mag = torch.clamp(targ_mag, min=self.eps)

        # Adaptive weight: higher where target energy is low (weak signals)
        # and where prediction error is large.
        weight = (1.0 / safe_mag) * (1.0 + diff / safe_mag)
        # Self-normalize so AFM stays O(1) regardless of absolute magnitude
        weight = weight / (weight.mean().detach() + 1e-8)

        afm = torch.mean(weight * diff)

        return mse + self.lambda_afm * afm
