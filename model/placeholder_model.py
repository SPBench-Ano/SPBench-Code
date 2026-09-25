"""Identity-forward placeholder that keeps the model registry non-empty."""

from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


@register_model("placeholder")
class PlaceholderModel(nn.Module):
    """Identity-forward placeholder; replace with a real backbone."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.dummy = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dummy(x)
