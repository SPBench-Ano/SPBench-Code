"""Liu et al. 2022 WRDL — Wavelet-Based Residual Deep Learning for seismic data reconstruction.

Paper: "Seismic Data Reconstruction via Wavelet-Based Residual Deep Learning",
IEEE TGRS, vol. 60, 2022, article 4508213.
DOI: 10.1109/TGRS.2022.3152984

Architecture: U-Net backbone where pooling is replaced by DWT (Discrete Wavelet
Transform) and upsampling by IWT (Inverse Wavelet Transform) with expansion
convolution. Bottleneck residual blocks (1x1→3x3→1x1) improve training
stability. The hybrid SSIM + Huber loss is defined separately in utils/losses.py.

.. note::

    The exact paper wavelet basis, channel counts, residual block count, and
    training hyperparameters could not be independently verified from the
    paywalled paper (no author code available).  All defaults are conservative
    ``reproduction-assumption`` values.  See ``liu2022_wrdl_notes.md`` for the
    full parameter classification table.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


# ---------------------------------------------------------------------------
# Haar wavelet filter construction
# ---------------------------------------------------------------------------

def _make_haar_analysis_filters() -> torch.Tensor:
    """Return analysis filters ``(4, 1, 2, 2)`` for 2D Haar DWT.

    Subband order: LL, LH, HL, HH.

    Orthonormal 1D kernels (1/sqrt(2) normalised)::

        L = 1/sqrt(2) * [1, 1]
        H = 1/sqrt(2) * [-1, 1]

    2D kernels are ``L^T L``, ``L^T H``, ``H^T L``, ``H^T H``.
    """
    sqrt2 = 2.0 ** 0.5
    L = torch.tensor([1.0, 1.0], dtype=torch.float32) / sqrt2
    H = torch.tensor([-1.0, 1.0], dtype=torch.float32) / sqrt2

    filters = torch.stack([
        torch.outer(L, L),  # LL
        torch.outer(L, H),  # LH
        torch.outer(H, L),  # HL
        torch.outer(H, H),  # HH
    ], dim=0)  # (4, 2, 2)
    return filters.unsqueeze(1)  # (4, 1, 2, 2)


def _make_haar_synthesis_filters() -> torch.Tensor:
    """Return synthesis filters ``(4, 1, 2, 2)`` for 2D Haar IWT.

    For orthogonal Haar the synthesis filters equal the analysis filters:
    conv_transpose2d with the same filters implements the adjoint, and
    orthogonality (A^T A = I) guarantees perfect reconstruction.
    """
    return _make_haar_analysis_filters()


# ---------------------------------------------------------------------------
# FixedWaveletDWT2D
# ---------------------------------------------------------------------------

class FixedWaveletDWT2D(nn.Module):
    """Differentiable 2D Discrete Wavelet Transform with fixed (non-trainable) filters.

    Implemented via grouped ``conv2d`` with stride 2.  Input ``(B, C, H, W)``
    produces output ``(B, 4*C, H/2, W/2)`` with subband order ``LL, LH, HL, HH``.

    Parameters
    ----------
    filters : ``(4, 1, K, K)`` tensor or ``None`` (default: Haar).
    """

    def __init__(self, filters: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if filters is None:
            filters = _make_haar_analysis_filters()
        if filters.dim() != 4 or filters.shape[0] != 4:
            raise ValueError(
                f"Expected filters shape (4, 1, K, K), got {tuple(filters.shape)}."
            )
        self.register_buffer("filters", filters)  # (4, 1, K, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DWT.

        Parameters
        ----------
        x : ``(B, C, H, W)``

        Returns
        -------
        ``(B, 4*C, H/2, W/2)`` — subbands stacked in channel dim: LL, LH, HL, HH.
        """
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(
                f"DWT requires even spatial dimensions, got H={H}, W={W}. "
                f"Pad the input before calling."
            )

        # Grouped conv: each of C input channels is processed independently
        # through 4 analysis filters → 4*C output channels.
        # kernel (4*C, 1, K, K) with groups=C: each group maps 1 ch → 4 ch.
        kernel = self.filters.repeat(C, 1, 1, 1)  # (4*C, 1, K, K)
        return F.conv2d(x, kernel, stride=2, padding=0, groups=C)


# ---------------------------------------------------------------------------
# FixedWaveletIWT2D
# ---------------------------------------------------------------------------

class FixedWaveletIWT2D(nn.Module):
    """Differentiable 2D Inverse Wavelet Transform with fixed (non-trainable) filters.

    Implemented via grouped ``conv_transpose2d`` with stride 2.
    Input ``(B, 4*C, H, W)`` produces output ``(B, C, 2*H, 2*W)``.

    The input subband order must match :class:`FixedWaveletDWT2D`:
    ``LL, LH, HL, HH``.

    Parameters
    ----------
    filters : ``(4, 1, K, K)`` tensor or ``None`` (default: Haar).
    """

    def __init__(self, filters: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if filters is None:
            filters = _make_haar_synthesis_filters()
        if filters.dim() != 4 or filters.shape[0] != 4:
            raise ValueError(
                f"Expected filters shape (4, 1, K, K), got {tuple(filters.shape)}."
            )
        self.register_buffer("filters", filters)  # (4, 1, K, K)

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply IWT.

        Parameters
        ----------
        coeffs : ``(B, 4*C, H, W)`` — LL, LH, HL, HH stacked in channel dim.

        Returns
        -------
        ``(B, C, 2*H, 2*W)``
        """
        B, C4, H, W = coeffs.shape
        if C4 % 4 != 0:
            raise ValueError(
                f"IWT input must have channels divisible by 4, got C={C4}."
            )
        C = C4 // 4

        # Grouped conv_transpose2d: each group maps 4 input subbands → 1 output channel.
        # kernel (4*C, 1, K, K) with groups=C: each group maps 4 ch → 1 ch.
        kernel = self.filters.repeat(C, 1, 1, 1)  # (4*C, 1, K, K)
        return F.conv_transpose2d(coeffs, kernel, stride=2, padding=0, groups=C)


# ---------------------------------------------------------------------------
# WRDLBottleneckResidualBlock
# ---------------------------------------------------------------------------

class WRDLBottleneckResidualBlock(nn.Module):
    """Bottleneck residual block: 1x1 → 3x3 → 1x1 with identity shortcut.

    Structure::

        x → 1x1 Conv(C//R) → BN → ReLU
          → 3x3 Conv(C//R) → BN → ReLU
          → 1x1 Conv(C) → BN
          → + x → ReLU

    Parameters
    ----------
    channels : input and output channels.
    bottleneck_ratio : channel reduction factor in the bottleneck (default 4).
    """

    def __init__(self, channels: int, bottleneck_ratio: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // bottleneck_ratio)

        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.act(out + identity)


# ---------------------------------------------------------------------------
# WRDLEncoderStage: Conv → BN → ReLU → Conv → BN → ReLU → DWT
# ---------------------------------------------------------------------------

class WRDLEncoderStage(nn.Module):
    """One encoder level: two 3x3 Convs + DWT for 2x spatial reduction.

    Returns both the DWT output (for deeper layers) and the pre-DWT features
    (for the skip connection).
    """

    def __init__(
        self, in_channels: int, out_channels: int, dwt: FixedWaveletDWT2D
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dwt = dwt

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(dwt_output, skip_features)``."""
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        skip = out
        return self.dwt(out), skip


# ---------------------------------------------------------------------------
# WaveletExpansionIWTBlock: Expansion Conv → BN → ReLU → IWT
# ---------------------------------------------------------------------------

class WaveletExpansionIWTBlock(nn.Module):
    """Expansion convolution + IWT for 2x spatial upsampling.

    Input ``(B, C_in, H, W)`` → 3x3 Conv producing ``4*C_out`` channels
    → IWT → ``(B, C_out, 2H, 2W)``.

    The expansion conv generates the four subband channels (LL, LH, HL, HH)
    expected by the IWT.
    """

    def __init__(
        self, in_channels: int, out_channels: int, iwt: FixedWaveletIWT2D
    ) -> None:
        super().__init__()
        if (4 * out_channels) % 4 != 0:
            raise ValueError(
                f"Expansion output channels 4*{out_channels} must be divisible by 4."
            )
        self.expand = nn.Conv2d(in_channels, 4 * out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4 * out_channels)
        self.act = nn.ReLU(inplace=True)
        self.iwt = iwt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.iwt(self.act(self.bn(self.expand(x))))


# ---------------------------------------------------------------------------
# WRDLDecoderStage: concat skip → Conv → BN → ReLU → Conv → BN → ReLU
# ---------------------------------------------------------------------------

class WRDLDecoderStage(nn.Module):
    """One decoder level: concat skip features → two 3x3 Convs."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Crop skip if size mismatch (from odd-size inputs)
        if skip.shape[-2:] != x.shape[-2:]:
            skip = skip[:, :, : x.shape[-2], : x.shape[-1]]
        out = torch.cat([x, skip], dim=1)
        out = self.act(self.bn1(self.conv1(out)))
        return self.act(self.bn2(self.conv2(out)))


# ---------------------------------------------------------------------------
# Liu2022WRDL — full model
# ---------------------------------------------------------------------------

@register_model("liu2022_wrdl")
class Liu2022WRDL(nn.Module):
    """WRDL: Wavelet-Based Residual Deep Learning for seismic data reconstruction.

    U-Net backbone where DWT replaces pooling and IWT with expansion convolution
    replaces upsampling. Bottleneck residual blocks improve training stability.

    Parameters
    ----------
    in_channels : input channels (default 1).
    out_channels : output channels (default 1).
    encoder_channels : channels after each encoder conv stage, from shallowest
        to deepest.  Default ``(32, 64, 128, 256, 512)`` (5 levels, DWT at
        each level halves spatial size).
    bottleneck_channels : channels at the bottleneck.  Defaults to
        ``encoder_channels[-1]``.
    bottleneck_blocks : number of bottleneck residual blocks. Default 2.
    bottleneck_ratio : channel reduction in bottleneck blocks. Default 4.
    wavelet : ``"haar"`` (only option currently).
    subband_order : ``"ll_lh_hl_hh"`` (only option currently).
    output_activation : ``"none"``, ``"sigmoid"``, or ``"tanh"``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: Sequence[int] = (32, 64, 128, 256, 512),
        bottleneck_channels: Optional[int] = None,
        bottleneck_blocks: int = 2,
        bottleneck_ratio: int = 4,
        wavelet: str = "haar",
        subband_order: str = "ll_lh_hl_hh",
        output_activation: str = "none",
    ) -> None:
        super().__init__()

        if wavelet != "haar":
            raise ValueError(
                f"Unsupported wavelet: {wavelet!r}. Only 'haar' is implemented."
            )
        if subband_order != "ll_lh_hl_hh":
            raise ValueError(
                f"Unsupported subband_order: {subband_order!r}. "
                f"Only 'll_lh_hl_hh' is implemented."
            )

        self.dwt = FixedWaveletDWT2D()
        self.iwt = FixedWaveletIWT2D()

        enc_chs = list(encoder_channels)
        num_levels = len(enc_chs)
        if num_levels < 2:
            raise ValueError(f"Need at least 2 encoder levels, got {num_levels}.")
        self._pad_multiple = 2 ** (num_levels - 1)

        if bottleneck_channels is None:
            bottleneck_channels = enc_chs[-1]

        # ---- Stem ----
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, enc_chs[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(enc_chs[0]),
            nn.ReLU(inplace=True),
        )

        # ---- Encoder ----
        # DWT stacks LL/LH/HL/HH subbands, so each stage receives 4x the
        # previous stage's conv output channels (except the first stage,
        # which receives stem output directly).
        enc_stages: List[WRDLEncoderStage] = []
        for i in range(num_levels - 1):
            in_ch = enc_chs[i] if i == 0 else 4 * enc_chs[i]
            enc_stages.append(WRDLEncoderStage(in_ch, enc_chs[i + 1], self.dwt))
        self.encoder_stages = nn.ModuleList(enc_stages)

        # ---- Bottleneck ----
        # Bottleneck receives DWT output from the last encoder stage (4x channels).
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(4 * enc_chs[-1], bottleneck_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )
        self.bottleneck_residuals = nn.Sequential(*[
            WRDLBottleneckResidualBlock(bottleneck_channels, bottleneck_ratio)
            for _ in range(bottleneck_blocks)
        ])

        # ---- Decoder ----
        # Reverse channel progression: deepest → shallowest
        dec_chs = list(reversed(enc_chs))  # e.g. [512, 256, 128, 64, 32]

        expansion_blocks: List[WaveletExpansionIWTBlock] = []
        decoder_stages: List[WRDLDecoderStage] = []
        for i in range(num_levels - 1):
            in_ch = dec_chs[i]       # e.g. 512, 256, 128, 64
            out_ch = dec_chs[i + 1]  # e.g. 256, 128, 64,  32
            expansion_blocks.append(WaveletExpansionIWTBlock(in_ch, in_ch, self.iwt))
            # After IWT + skip concat: channels = in_ch (IWT output) + in_ch (skip)
            decoder_stages.append(WRDLDecoderStage(2 * in_ch, out_ch))

        self.expansion_blocks = nn.ModuleList(expansion_blocks)
        self.decoder_stages = nn.ModuleList(decoder_stages)

        # ---- Output ----
        self.output_conv = nn.Conv2d(enc_chs[0], out_channels, 1)

        if output_activation == "sigmoid":
            self.output_act: nn.Module = nn.Sigmoid()
        elif output_activation == "tanh":
            self.output_act = nn.Tanh()
        elif output_activation == "none":
            self.output_act = nn.Identity()
        else:
            raise ValueError(f"Unknown output_activation: {output_activation!r}.")

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : ``(B, C_in, H, W)`` — incomplete seismic data (missing traces zero-filled).

        Returns
        -------
        ``(B, C_out, H, W)`` — reconstructed data (same spatial size as input).
        """
        H_in, W_in = x.shape[-2], x.shape[-1]

        # Pad to a multiple of 2^(num_levels-1) so every DWT halving keeps even
        # spatial dims (each encoder stage DWT halves H and W).
        mult = self._pad_multiple
        pad_h = (mult - H_in % mult) % mult
        pad_w = (mult - W_in % mult) % mult
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Stem
        f = self.stem(x)

        # Encoder: collect skip features
        skips: List[torch.Tensor] = []
        for stage in self.encoder_stages:
            f, skip = stage(f)
            skips.append(skip)

        # Bottleneck
        f = self.bottleneck_conv(f)
        f = self.bottleneck_residuals(f)

        # Decoder: expansion IWT + skip concat
        for expand, decode, skip in zip(
            self.expansion_blocks, self.decoder_stages, reversed(skips)
        ):
            f = expand(f)
            f = decode(f, skip)

        # Output
        out = self.output_conv(f)
        out = self.output_act(out)

        # Crop back to original size
        if pad_h or pad_w:
            out = out[:, :, :H_in, :W_in]

        return out
