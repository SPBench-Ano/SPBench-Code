"""Sparse prior-net for seismic missing-trace interpolation.

Paper-aligned implementation of Wu et al., Geophysics 2024,
DOI: 10.1190/geo2022-0262.1, cross-checked against the user-supplied
POCS-Net development code.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


class _SparsePriorPhase(nn.Module):
    """Apply one learned sparse-prior transform and POCS reconstruction phase."""

    def __init__(
        self,
        channels: int,
        features: int,
        kernel_size: int,
        threshold_init: float,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"SPNet kernel_size must be a positive odd integer, got {kernel_size}."
            )
        padding = kernel_size // 2

        self.analysis_conv1 = nn.Conv2d(
            channels,
            features,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.analysis_conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.synthesis_conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.synthesis_conv2 = nn.Conv2d(
            features,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.soft_threshold = nn.Parameter(
            torch.tensor([float(threshold_init)], dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize convolution kernels with Xavier-normal weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)

    def _synthesis(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Map sparse coefficients back to the seismic-data domain."""
        features = F.relu(self.synthesis_conv1(coefficients), inplace=False)
        return self.synthesis_conv2(features)

    def forward(
        self,
        current: torch.Tensor,
        observed_data: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the updated data, sparse prior, and transform-symmetry residual."""
        analysis_features = F.relu(
            self.analysis_conv1(current),
            inplace=False,
        )
        coefficients = self.analysis_conv2(analysis_features)
        thresholded = torch.sign(coefficients) * F.relu(
            coefficients.abs() - self.soft_threshold,
            inplace=False,
        )

        sparse_prior = self._synthesis(thresholded)
        symmetry_residual = self._synthesis(coefficients) - current
        updated = observed_data + missing_mask * sparse_prior
        return updated, sparse_prior, symmetry_residual


@register_model("spnet")
class SPNet(nn.Module):
    """Unroll learned sparse-prior estimation and POCS data consistency."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: int = 64,
        phases: int = 9,
        kernel_size: int = 3,
        threshold_init: float = 0.01,
        mask_eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("SPNet input and output channels must be positive.")
        if in_channels != out_channels:
            raise ValueError(
                "SPNet requires in_channels == out_channels for POCS data consistency."
            )
        if features < 1:
            raise ValueError(f"SPNet features must be positive, got {features}.")
        if phases < 1:
            raise ValueError(f"SPNet phases must be positive, got {phases}.")
        if threshold_init < 0:
            raise ValueError(
                f"SPNet threshold_init must be non-negative, got {threshold_init}."
            )
        if mask_eps <= 0:
            raise ValueError(f"SPNet mask_eps must be positive, got {mask_eps}.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.features = int(features)
        self.num_phases = int(phases)
        self.mask_eps = float(mask_eps)
        self.phases = nn.ModuleList(
            [
                _SparsePriorPhase(
                    channels=self.in_channels,
                    features=self.features,
                    kernel_size=int(kernel_size),
                    threshold_init=float(threshold_init),
                )
                for _ in range(self.num_phases)
            ]
        )
        self._symmetry_residuals: Tuple[torch.Tensor, ...] = ()

    def _build_observed_trace_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Infer observed traces from whole-trace zero filling."""
        observed = x.detach().abs().amax(dim=(1, 3), keepdim=True) > self.mask_eps
        return observed.expand(-1, x.shape[1], -1, x.shape[3]).to(dtype=x.dtype)

    def _prepare_observed_mask(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Normalize an observed-data mask to the input shape."""
        if mask is None:
            return self._build_observed_trace_mask(x)

        if mask.ndim == 2:
            if mask.shape != (x.shape[0], x.shape[2]):
                raise ValueError(
                    "A 2-D SPNet trace mask must have shape [B, H], got "
                    f"{tuple(mask.shape)} for input {tuple(x.shape)}."
                )
            mask = mask[:, None, :, None].expand(-1, 1, -1, x.shape[3])
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(
                "SPNet mask must have shape [B, H], [B, H, W], or [B, C, H, W], "
                f"got {tuple(mask.shape)}."
            )
        if mask.shape[0] != x.shape[0] or mask.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                "SPNet mask must match the input batch and spatial dimensions, got "
                f"input {tuple(x.shape)} and mask {tuple(mask.shape)}."
            )
        if mask.shape[1] == 1 and x.shape[1] > 1:
            mask = mask.expand(-1, x.shape[1], -1, -1)
        elif mask.shape[1] != x.shape[1]:
            raise ValueError(
                f"SPNet mask must have 1 or {x.shape[1]} channels, got {mask.shape[1]}."
            )
        return mask.to(device=x.device).gt(0.5).to(dtype=x.dtype)

    def forward_with_aux(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Return the final reconstruction, phase outputs, and symmetry residuals."""
        if x.ndim != 4:
            raise ValueError(f"SPNet expects [B, C, H, W], got {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"SPNet expects {self.in_channels} input channels, got {x.shape[1]}."
            )

        observed_mask = self._prepare_observed_mask(x, mask)
        missing_mask = 1.0 - observed_mask
        observed_data = x * observed_mask
        current = observed_data
        phase_outputs = []
        symmetry_residuals = []

        for phase in self.phases:
            current, _, symmetry_residual = phase(
                current,
                observed_data,
                missing_mask,
            )
            phase_outputs.append(current)
            symmetry_residuals.append(symmetry_residual)

        return current, tuple(phase_outputs), tuple(symmetry_residuals)

    def pop_loss_extras(self) -> Dict[str, Any]:
        """Return and clear auxiliary tensors consumed by the configured loss."""
        extras: Dict[str, Any] = {
            "symmetry_residuals": self._symmetry_residuals,
        }
        self._symmetry_residuals = ()
        return extras

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the final reconstruction expected by the benchmark."""
        self._symmetry_residuals = ()
        output, _, symmetry_residuals = self.forward_with_aux(x, mask)
        self._symmetry_residuals = symmetry_residuals
        return output
