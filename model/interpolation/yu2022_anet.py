"""ANet from Yu and Wu, "Attention and Hybrid Loss Guided Deep Learning for
Consecutively Missing Seismic Data Reconstruction" (IEEE TGRS, vol. 60, 2022).
DOI: 10.1109/TGRS.2021.3068279

Paper-explicit topology: two stride-2 downsampling convolutions (64 -> 128
channels), six consecutive residual blocks (two convs + identity shortcut
each), one non-local attention module (1x1 Q/K/V convolutions, full spatial
correlation matrix, residual output), two upsampling + convolution groups,
and a final convolution.  BatchNorm + ReLU follow every convolution except
the upsampling layers and the final layer.

Reproduction assumptions (paper figure not machine-verifiable): 3x3 kernels,
decoder channels (64, 32), nearest-neighbor resize + convolution upsampling,
Conv->BN->ReLU ordering, post-add ReLU in residual blocks, full-channel Q/K,
no attention logit scaling.  Arbitrary-size zero padding and exact attention
query chunking are repository adaptations that do not change the mathematics.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


def _make_norm(norm: str, channels: int) -> nn.Module:
    """Return a normalization layer for the given type and channel count."""
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm type: {norm!r}.")


def _make_activation(activation: str) -> nn.Module:
    """Return an activation module for the given name."""
    if activation == "relu":
        return nn.ReLU(inplace=False)
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=False)
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {activation!r}.")


class ANetResidualBlock(nn.Module):
    """Residual block with two convolutions and an identity shortcut.

    Parameters
    ----------
    channels    : feature channels (kept constant through the block).
    kernel_size : odd convolution kernel size.
    norm        : ``"batch"`` / ``"instance"`` / ``"none"``.
    activation  : ``"relu"`` / ``"leaky_relu"`` / ``"none"``.
    conv_bias   : whether convolutions carry a bias term.
    post_add_activation : apply the activation after the residual addition.

    Returns
    -------
    Tensor of the same shape as the input.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm: str = "batch",
        activation: str = "relu",
        conv_bias: bool = True,
        post_add_activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=conv_bias)
        self.norm1 = _make_norm(norm, channels)
        self.act1 = _make_activation(activation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=conv_bias)
        self.norm2 = _make_norm(norm, channels)
        self.post_add_activation = bool(post_add_activation)
        self.act2 = _make_activation(activation) if post_add_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + x
        return self.act2(out)


class ANetNonLocalAttention2D(nn.Module):
    """Exact non-local attention over all spatial positions with residual output.

    Parameters
    ----------
    channels           : input/output feature channels.
    attention_channels : Q/K projection channels; ``None`` keeps ``channels``.
    scaling            : ``"none"`` (paper-inferred) or ``"sqrt_channel"``.
    query_chunk_size   : optional query-dim chunking; mathematically exact.

    Returns
    -------
    Tensor of shape ``(B, C, H, W)`` equal to ``x + W_o(softmax(QK) V)``.
    """

    def __init__(
        self,
        channels: int,
        attention_channels: Optional[int] = None,
        scaling: str = "none",
        query_chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if scaling not in ("none", "sqrt_channel"):
            raise ValueError(f"scaling must be 'none' or 'sqrt_channel', got {scaling!r}.")
        qk_channels = int(attention_channels) if attention_channels is not None else int(channels)
        if qk_channels <= 0:
            raise ValueError(f"attention_channels must be positive, got {qk_channels}.")
        if query_chunk_size is not None and query_chunk_size <= 0:
            raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}.")
        self.channels = int(channels)
        self.qk_channels = qk_channels
        self.scaling = scaling
        self.query_chunk_size = query_chunk_size
        self.query = nn.Conv2d(channels, qk_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, qk_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute ``softmax(q @ k) @ v`` for flattened Q ``(B, N, Cq)``."""
        logits = torch.bmm(q, k)
        if self.scaling == "sqrt_channel":
            logits = logits / float(self.qk_channels) ** 0.5
        attn = torch.softmax(logits, dim=-1)
        return torch.bmm(attn, v)

    def compute_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Return the full attention matrix ``(B, H*W, H*W)`` (diagnostics only)."""
        q = self.query(x).flatten(2).transpose(1, 2)
        k = self.key(x).flatten(2)
        logits = torch.bmm(q, k)
        if self.scaling == "sqrt_channel":
            logits = logits / float(self.qk_channels) ** 0.5
        return torch.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query(x).flatten(2).transpose(1, 2)   # (B, N, Cq)
        k = self.key(x).flatten(2)                     # (B, Cq, N)
        v = self.value(x).flatten(2).transpose(1, 2)   # (B, N, C)

        chunk = self.query_chunk_size
        if chunk is None or chunk >= q.shape[1]:
            y = self._attend(q, k, v)
        else:
            parts = [
                self._attend(q[:, i : i + chunk], k, v)
                for i in range(0, q.shape[1], chunk)
            ]
            y = torch.cat(parts, dim=1)

        y = y.transpose(1, 2).reshape(b, c, h, w)
        return x + self.out_proj(y)


class ANetUpsampleStage(nn.Module):
    """One x2 upsampling step followed by a convolution with norm + activation.

    Parameters
    ----------
    in_channels / out_channels : channels before / after the stage convolution.
    kernel_size : odd convolution kernel size.
    mode        : ``"nearest_conv"`` (resize convolution) or ``"transposed"``.
    norm, activation, conv_bias : applied to the stage convolution only; the
        upsampling layer itself carries no norm or activation (paper-explicit).

    Returns
    -------
    Tensor with doubled spatial size and ``out_channels`` channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        mode: str = "nearest_conv",
        norm: str = "batch",
        activation: str = "relu",
        conv_bias: bool = True,
    ) -> None:
        super().__init__()
        if mode == "nearest_conv":
            self.upsample: nn.Module = nn.Upsample(scale_factor=2, mode="nearest")
        elif mode == "transposed":
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            raise ValueError(f"mode must be 'nearest_conv' or 'transposed', got {mode!r}.")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=conv_bias)
        self.norm = _make_norm(norm, out_channels)
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.act(self.norm(self.conv(x)))


@register_model("yu2022_anet")
class Yu2022ANet(nn.Module):
    """ANet: downsampling convs -> residual blocks -> non-local attention -> upsampling.

    Parameters
    ----------
    in_channels  : input channels (paper: 1, no mask channel).
    out_channels : output channels (paper: 1).
    base_channels : first convolution channels (paper: 64).
    scaling_factor : number of stride-2 downsampling stages (paper: 2).
    num_residual_blocks : consecutive residual blocks (paper: 6).
    kernel_size  : convolution kernel size (reproduction-assumption: 3).
    attention_channels : Q/K channels; ``None`` keeps the body channels.
    attention_scaling  : ``"none"`` (paper-inferred) or ``"sqrt_channel"``.
    upsample_mode : ``"nearest_conv"`` (reproduction-assumption) or ``"transposed"``.
    activation / norm / conv_bias : layer construction options.
    post_add_activation : activation after the residual addition.
    attention_query_chunk_size : exact attention chunking for large inputs.
    decoder_channels : channels after each upsampling stage; ``None`` halves
        the body channels per stage (reproduction-assumption: (64, 32)).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        scaling_factor: int = 2,
        num_residual_blocks: int = 6,
        kernel_size: int = 3,
        attention_channels: Optional[int] = None,
        attention_scaling: str = "none",
        upsample_mode: str = "nearest_conv",
        activation: str = "relu",
        norm: str = "batch",
        conv_bias: bool = True,
        post_add_activation: bool = True,
        attention_query_chunk_size: Optional[int] = None,
        decoder_channels: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}.")
        if scaling_factor <= 0:
            raise ValueError(f"scaling_factor must be positive, got {scaling_factor}.")
        if num_residual_blocks <= 0:
            raise ValueError(f"num_residual_blocks must be positive, got {num_residual_blocks}.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_down_stages = int(scaling_factor)
        self.size_multiple = 2 ** self.num_down_stages

        padding = kernel_size // 2

        # ---- Downsampling convolutions (stride 2, channel doubling) ----
        down_layers = []
        ch = int(in_channels)
        stage_ch = int(base_channels)
        for _ in range(self.num_down_stages):
            down_layers.append(
                nn.Sequential(
                    nn.Conv2d(ch, stage_ch, kernel_size, stride=2, padding=padding, bias=conv_bias),
                    _make_norm(norm, stage_ch),
                    _make_activation(activation),
                )
            )
            ch = stage_ch
            stage_ch *= 2
        self.down_stages = nn.ModuleList(down_layers)
        body_channels = ch

        # ---- Residual body ----
        self.residual_blocks = nn.ModuleList(
            ANetResidualBlock(
                body_channels,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                conv_bias=conv_bias,
                post_add_activation=post_add_activation,
            )
            for _ in range(num_residual_blocks)
        )

        # ---- Global non-local attention ----
        self.attention = ANetNonLocalAttention2D(
            body_channels,
            attention_channels=attention_channels,
            scaling=attention_scaling,
            query_chunk_size=attention_query_chunk_size,
        )

        # ---- Upsampling reconstruction ----
        if decoder_channels is None:
            dec_channels = tuple(
                max(1, body_channels // (2 ** (i + 1))) for i in range(self.num_down_stages)
            )
        else:
            dec_channels = tuple(int(c) for c in decoder_channels)
            if len(dec_channels) != self.num_down_stages:
                raise ValueError(
                    f"decoder_channels must have {self.num_down_stages} entries, "
                    f"got {len(dec_channels)}."
                )
            if any(c <= 0 for c in dec_channels):
                raise ValueError(f"decoder_channels must be positive, got {dec_channels}.")

        up_layers = []
        ch = body_channels
        for dec_ch in dec_channels:
            up_layers.append(
                ANetUpsampleStage(
                    ch,
                    dec_ch,
                    kernel_size=kernel_size,
                    mode=upsample_mode,
                    norm=norm,
                    activation=activation,
                    conv_bias=conv_bias,
                )
            )
            ch = dec_ch
        self.upsample_stages = nn.ModuleList(up_layers)

        # ---- Final layer: no norm, no activation (paper-explicit) ----
        self.final = nn.Conv2d(ch, out_channels, kernel_size, padding=padding, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x    : input tensor of shape ``(B, in_channels, H, W)``.
        mask : ignored (accepted for trainer compatibility).

        Returns
        -------
        Output tensor of shape ``(B, out_channels, H, W)``.
        """
        if x.dim() != 4:
            raise ValueError(f"Yu2022ANet expects a 4D input (B, C, H, W), got ndim={x.dim()}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Yu2022ANet expects {self.in_channels} input channels, got {x.shape[1]}."
            )

        h_in, w_in = x.shape[2], x.shape[3]
        m = self.size_multiple
        pad_h = (m - h_in % m) % m
        pad_w = (m - w_in % m) % m
        if pad_h or pad_w:
            # Constant zero padding for arbitrary sizes (repository adaptation).
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        h = x
        for stage in self.down_stages:
            h = stage(h)
        for block in self.residual_blocks:
            h = block(h)
        h = self.attention(h)
        for stage in self.upsample_stages:
            h = stage(h)
        out = self.final(h)

        if pad_h or pad_w:
            out = out[:, :, :h_in, :w_in]
        return out
