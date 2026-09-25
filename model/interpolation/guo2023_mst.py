"""Multi-Scale Transformer (MST) from Guo, Fu, and Li "Seismic Data
Interpolation Based on Multi-Scale Transformer" (IEEE GRSL, 2023).

The MST generates multi-scale seismic feature maps via strided convolutions,
applies independent self-attention at each scale to capture long-range
dependencies, then fuses multi-scale features back through upsampling,
concatenation, and convolutional fusion into a complete reconstruction.

Conservative reproduction profile:
- 3 scales, embed_dims=(32, 64, 128), depths=(1, 1, 1)
- num_heads=(4, 4, 8), mlp_ratio=4.0, Pre-LN Transformer
- GELU activation, no positional encoding, direct output
- constant zero-pad for size alignment
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_norm(norm_type: str, channels: int, num_groups: int = 8) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(channels)
    if norm_type == "group":
        g = min(num_groups, channels)
        return nn.GroupNorm(g, channels)
    if norm_type == "layer":
        return nn.GroupNorm(1, channels)  # LayerNorm over C only (2D compatible)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type: {norm_type}")


def _make_activation(activation: str, negative_slope: float = 0.01) -> nn.Module:
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    if activation == "gelu":
        return nn.GELU()
    if activation == "elu":
        return nn.ELU(inplace=True)
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {activation}")


# ---------------------------------------------------------------------------
# Conv Stem
# ---------------------------------------------------------------------------

class ConvStem(nn.Module):
    """Two 3x3 convolutions mapping input channels to the first embed dim.

    Parameters
    ----------
    in_channels : Input channels.
    out_channels : Output channels (first embed dim).
    kernel_size : Convolution kernel size (default 3).
    activation : Activation type (default ``"gelu"``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        self.act1 = _make_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=True)
        self.act2 = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        return out


# ---------------------------------------------------------------------------
# Multi-Scale Feature Pyramid
# ---------------------------------------------------------------------------

class MultiScaleFeaturePyramid(nn.Module):
    """Generate multi-scale feature maps via stride-2 convolutions.

    Parameters
    ----------
    embed_dims : Tuple of embedding dimensions per scale.
    kernel_size : Convolution kernel size for downsampling.
    downsample_mode : ``"stride_conv"`` (stride-2 conv) or ``"maxpool"``.
    activation : Activation type.
    """

    def __init__(
        self,
        embed_dims: Tuple[int, ...],
        kernel_size: int = 3,
        downsample_mode: str = "stride_conv",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if downsample_mode not in ("stride_conv", "maxpool"):
            raise ValueError(f"Unknown downsample_mode: {downsample_mode}")

        self.num_scales = len(embed_dims)
        self.embed_dims = embed_dims
        self.downsample_mode = downsample_mode

        self.downsamples = nn.ModuleList()
        for i in range(self.num_scales - 1):
            if downsample_mode == "stride_conv":
                self.downsamples.append(nn.Conv2d(
                    embed_dims[i], embed_dims[i + 1],
                    kernel_size=kernel_size, stride=2,
                    padding=kernel_size // 2, bias=True,
                ))
            else:
                self.downsamples.append(nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        embed_dims[i], embed_dims[i + 1],
                        kernel_size=1, bias=True,
                    ),
                ))

        self.act = _make_activation(activation)

    def forward(self, f0: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = [f0]
        for downsample in self.downsamples:
            prev = features[-1]
            features.append(self.act(downsample(prev)))
        return features


# ---------------------------------------------------------------------------
# MSE (Multi-head Self-Attention) wrapper
# ---------------------------------------------------------------------------

class _MHA(nn.Module):
    """Multi-head self-attention with optional attention dropout.

    Parameters
    ----------
    embed_dim : Total embedding dimension.
    num_heads : Number of attention heads.
    attention_dropout : Dropout on attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attention_dropout) if attention_dropout > 0 else nn.Identity()
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


# ---------------------------------------------------------------------------
# Transformer Block (Pre-Norm)
# ---------------------------------------------------------------------------

class MSTTransformerBlock(nn.Module):
    """Pre-Norm Transformer block: LN -> MHA -> +residual, LN -> MLP -> +residual.

    Parameters
    ----------
    embed_dim : Embedding dimension.
    num_heads : Number of attention heads.
    mlp_ratio : Expansion ratio for the MLP hidden layer.
    dropout : Dropout applied after the attention projection and MLP.
    attention_dropout : Dropout applied to attention weights.
    drop_path : Stochastic depth drop probability.
    activation : Activation type for the MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _MHA(embed_dim, num_heads, attention_dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            _make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.drop1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Per-scale transformer
# ---------------------------------------------------------------------------

class ScaleTransformer(nn.Module):
    """Wrap a feature map through a stack of Transformer blocks.

    Tokenizes (flatten -> transpose), runs the block stack, then restores
    to 2D.

    Parameters
    ----------
    embed_dim : Embedding / channel dimension.
    depth : Number of Transformer blocks.
    num_heads : Number of attention heads.
    mlp_ratio : MLP hidden expansion ratio.
    dropout : Dropout rate.
    attention_dropout : Attention dropout rate.
    drop_path : Stochastic depth drop probability.
    max_tokens : If set, raise an error when token count exceeds this limit.
    activation : Activation type for the MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
        max_tokens: Optional[int] = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens

        # Stochastic depth schedule (linear increase)
        drop_path_rates = [
            drop_path * (i / max(1, depth - 1)) if depth > 1 else drop_path
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList([
            MSTTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=drop_path_rates[i],
                activation=activation,
            )
            for i in range(depth)
        ])

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feature.shape
        N = H * W
        if self.max_tokens is not None and N > self.max_tokens:
            raise RuntimeError(
                f"Token count {N} (H={H}, W={W}) exceeds max_tokens={self.max_tokens}. "
                f"Use a smaller patch size or increase max_tokens_per_scale."
            )

        # Tokenize: [B, C, H, W] -> [B, N, C]
        tokens = feature.flatten(2).transpose(1, 2)

        for blk in self.blocks:
            tokens = blk(tokens)

        # Restore: [B, N, C] -> [B, C, H, W]
        out = tokens.transpose(1, 2).reshape(B, C, H, W)
        return out


# ---------------------------------------------------------------------------
# Multi-scale feature fusion
# ---------------------------------------------------------------------------

class MultiScaleFeatureFusion(nn.Module):
    """Upsample, concatenate, and fuse multi-scale transformer outputs.

    Parameters
    ----------
    embed_dims : Tuple of embedding dimensions per scale.
    out_channels : Output channel count.
    kernel_size : Convolution kernel size for fusion blocks.
    upsample_mode : Upsampling mode (``"bilinear"``, ``"nearest"``).
    activation : Activation type.
    """

    def __init__(
        self,
        embed_dims: Tuple[int, ...],
        out_channels: int = 1,
        kernel_size: int = 3,
        upsample_mode: str = "bilinear",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if upsample_mode not in ("bilinear", "nearest", "transpose"):
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")

        self.upsample_mode = upsample_mode
        self.embed_dims = embed_dims

        total_channels = sum(embed_dims)
        fusion_channels = embed_dims[0]  # fuse back to first-scale channels
        padding = kernel_size // 2

        self.fusion_conv1 = nn.Conv2d(total_channels, fusion_channels, kernel_size,
                                       padding=padding, bias=True)
        self.fusion_act1 = _make_activation(activation)
        self.fusion_conv2 = nn.Conv2d(fusion_channels, fusion_channels, kernel_size,
                                       padding=padding, bias=True)
        self.fusion_act2 = _make_activation(activation)

        self.output_head = nn.Conv2d(fusion_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_h, target_w = features[0].shape[-2:]

        # Upsample lower-resolution features to the first scale's size
        aligned: List[torch.Tensor] = [features[0]]
        for feat in features[1:]:
            if feat.shape[-2:] != (target_h, target_w):
                up = F.interpolate(
                    feat, size=(target_h, target_w),
                    mode=self.upsample_mode,
                    align_corners=False if self.upsample_mode != "nearest" else None,
                )
            else:
                up = feat
            aligned.append(up)

        fused = torch.cat(aligned, dim=1)
        out = self.fusion_act1(self.fusion_conv1(fused))
        out = self.fusion_act2(self.fusion_conv2(out))
        out = self.output_head(out)
        return out


# ---------------------------------------------------------------------------
# Full Multi-Scale Transformer Network
# ---------------------------------------------------------------------------

@register_model("guo2023_mst")
class Guo2023MST(nn.Module):
    """Multi-Scale Transformer for seismic data interpolation.

    Generates multi-scale feature maps, applies independent self-attention at
    each scale, then fuses the results through upsampling and convolution.

    Parameters
    ----------
    in_channels : Input channels (default 1).
    out_channels : Output channels (default 1).
    num_scales : Number of scales (default 3).  Must match the length of
        ``embed_dims``, ``depths``, and ``num_heads``.
    embed_dims : Embedding dimension per scale.
    depths : Number of Transformer blocks per scale.
    num_heads : Number of attention heads per scale.
    mlp_ratio : MLP hidden expansion ratio.
    kernel_size : Convolution kernel size.
    downsample_mode : Downsampling mode (``"stride_conv"`` or ``"maxpool"``).
    upsample_mode : Upsampling mode (``"bilinear"``, ``"nearest"``).
    norm_type : Normalization type (unused in conservative profile, kept for
        extension compatibility).
    activation : Activation type.
    positional_encoding : Positional encoding mode (``"none"``, ``"sincos_2d"``).
    dropout : Dropout rate.
    attention_dropout : Attention dropout rate.
    drop_path : Stochastic depth drop probability.
    output_mode : ``"direct"`` or ``"global_residual"``.
    max_tokens_per_scale : Token count limit per scale (None = unlimited).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_scales: int = 3,
        embed_dims: Tuple[int, ...] = (32, 64, 128),
        depths: Tuple[int, ...] = (1, 1, 1),
        num_heads: Tuple[int, ...] = (4, 4, 8),
        mlp_ratio: float = 4.0,
        kernel_size: int = 3,
        downsample_mode: str = "stride_conv",
        upsample_mode: str = "bilinear",
        norm_type: str = "layer",
        activation: str = "gelu",
        positional_encoding: str = "none",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
        output_mode: str = "direct",
        max_tokens_per_scale: Optional[int] = None,
    ) -> None:
        super().__init__()

        # ---- Validation ----
        if num_scales <= 0:
            raise ValueError(f"num_scales must be > 0, got {num_scales}.")
        if len(embed_dims) != num_scales:
            raise ValueError(
                f"len(embed_dims)={len(embed_dims)} must equal num_scales={num_scales}."
            )
        if len(depths) != num_scales:
            raise ValueError(
                f"len(depths)={len(depths)} must equal num_scales={num_scales}."
            )
        if len(num_heads) != num_scales:
            raise ValueError(
                f"len(num_heads)={len(num_heads)} must equal num_scales={num_scales}."
            )
        for i, (ed, nh) in enumerate(zip(embed_dims, num_heads)):
            if ed <= 0:
                raise ValueError(f"embed_dims[{i}] must be > 0, got {ed}.")
            if nh <= 0:
                raise ValueError(f"num_heads[{i}] must be > 0, got {nh}.")
            if ed % nh != 0:
                raise ValueError(
                    f"embed_dims[{i}]={ed} must be divisible by num_heads[{i}]={nh}."
                )
        for i, d in enumerate(depths):
            if d <= 0:
                raise ValueError(f"depths[{i}] must be > 0, got {d}.")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be > 0, got {mlp_ratio}.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if not (0.0 <= attention_dropout < 1.0):
            raise ValueError(f"attention_dropout must be in [0, 1), got {attention_dropout}.")
        if not (0.0 <= drop_path < 1.0):
            raise ValueError(f"drop_path must be in [0, 1), got {drop_path}.")
        if downsample_mode not in ("stride_conv", "maxpool"):
            raise ValueError(f"Unknown downsample_mode: {downsample_mode}")
        if upsample_mode not in ("bilinear", "nearest", "transpose"):
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")
        if positional_encoding not in ("none", "sincos_2d"):
            raise ValueError(f"Unknown positional_encoding: {positional_encoding}")
        if output_mode not in ("direct", "global_residual"):
            raise ValueError(f"Unknown output_mode: {output_mode}")
        if output_mode == "global_residual" and in_channels != out_channels:
            raise ValueError(
                f"global_residual requires in_channels == out_channels, "
                f"got {in_channels} != {out_channels}."
            )
        _valid_norms = {"none", "batch", "instance", "group", "layer"}
        if norm_type not in _valid_norms:
            raise ValueError(f"Unknown norm_type: {norm_type}")
        _valid_activations = {"relu", "leaky_relu", "gelu", "elu", "none"}
        if activation not in _valid_activations:
            raise ValueError(f"Unknown activation: {activation}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales
        self.embed_dims = embed_dims
        self.output_mode = output_mode

        self._stride = 2 ** (num_scales - 1)

        # Stem
        self.stem = ConvStem(in_channels, embed_dims[0], kernel_size=kernel_size,
                             activation=activation)

        # Feature pyramid
        self.pyramid = MultiScaleFeaturePyramid(
            embed_dims=embed_dims, kernel_size=kernel_size,
            downsample_mode=downsample_mode, activation=activation,
        )

        # Scale transformers
        self.scale_transformers = nn.ModuleList([
            ScaleTransformer(
                embed_dim=embed_dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=drop_path,
                max_tokens=max_tokens_per_scale,
                activation=activation,
            )
            for i in range(num_scales)
        ])

        # Fusion
        self.fusion = MultiScaleFeatureFusion(
            embed_dims=embed_dims, out_channels=out_channels,
            kernel_size=kernel_size, upsample_mode=upsample_mode,
            activation=activation,
        )

        # Positional encoding (optional extension)
        self.pos_encoding_mode = positional_encoding

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape [B, C, T, X], got {tuple(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s), got {x.shape[1]}."
            )
        if x.shape[-2] < 2:
            raise ValueError(f"Spatial height must be >= 2, got {x.shape[-2]}.")
        if x.shape[-1] < 2:
            raise ValueError(f"Spatial width must be >= 2, got {x.shape[-1]}.")

    def _pad_to_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Pad spatial dims to multiples of ``2**(num_scales-1)`` using constant zeros."""
        h, w = x.shape[-2:]
        target_h = ((h + self._stride - 1) // self._stride) * self._stride
        target_w = ((w + self._stride - 1) // self._stride) * self._stride
        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h == 0 and pad_w == 0:
            return x, h, w
        out = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        return out, h, w

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Input tensor of shape ``(B, C_in, T, X)``.
        mask : Ignored (trainer compatibility).
        positions : Ignored.
        **kwargs : Ignored.

        Returns
        -------
        Output tensor of shape ``(B, C_out, T, X)``.
        """
        self._validate_input(x)

        orig_h, orig_w = x.shape[-2:]
        x_padded, crop_h, crop_w = self._pad_to_multiple(x)

        # Stem
        f0 = self.stem(x_padded)

        # Multi-scale features
        features = self.pyramid(f0)

        # Per-scale transformer
        transformed: List[torch.Tensor] = []
        for i, feature in enumerate(features):
            t = self.scale_transformers[i](feature)
            transformed.append(t)

        # Fusion
        output = self.fusion(transformed)

        # Crop to original size
        if (crop_h, crop_w) != (output.shape[-2], output.shape[-1]):
            output = output[..., :crop_h, :crop_w]

        if self.output_mode == "global_residual":
            output = output + x

        return output
