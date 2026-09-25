"""Pan et al. 2020 Partial Convolution U-Net for seismic data regularization.

Paper: "A Partial Convolution-Based Deep-Learning Network for Seismic Data
Regularization", Computers & Geosciences, vol. 145, 2020.
DOI: 10.1016/j.cageo.2020.104609
Author code: https://github.com/shulinpan/seismic-data-regularization

This implementation follows the **author-code profile**: global mask-mean
normalization (not the local per-window normalization in paper Eq. 6),
six encoder PConv stages (kernels 7,5,5,3,3,3, channels 32-512), six
decoder PConv stages (kernels all 3, channels 512-1), skip connections
with both data and mask, final 1x1 Conv + Sigmoid, and LeakyReLU(0.2)
in all decoder stages.

Key difference from standard image inpainting PConv: normalization divides by
the per-channel *global* spatial mean of the mask, not the per-window local
valid count.  See ``Pan2020PartialConv2d`` for details.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


# ---------------------------------------------------------------------------
# Same-padding helper (Keras "same" compatibility)
# ---------------------------------------------------------------------------

def _same_pad_size(
    input_size: int,
    kernel: int,
    stride: int,
    dilation: int = 1,
) -> Tuple[int, int]:
    """Return (pad_left, pad_right) so that ``ceil(input_size / stride)``
    output positions are produced, matching Keras ``padding='same'``."""
    output_size = (input_size + stride - 1) // stride
    total_pad = max(
        0,
        (output_size - 1) * stride + dilation * (kernel - 1) + 1 - input_size,
    )
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    return pad_left, pad_right


def _pad_same(
    x: torch.Tensor,
    kernel: int,
    stride: int,
    dilation: int = 1,
) -> torch.Tensor:
    """Apply same-padding to the last two spatial dimensions of ``x``."""
    h, w = x.shape[-2], x.shape[-1]
    if isinstance(kernel, tuple):
        kh, kw = kernel[0], kernel[-1]
    else:
        kh = kw = int(kernel)
    if isinstance(stride, tuple):
        sh, sw = stride[0], stride[-1]
    else:
        sh = sw = int(stride)
    pad_h0, pad_h1 = _same_pad_size(h, kh, sh, dilation)
    pad_w0, pad_w1 = _same_pad_size(w, kw, sw, dilation)
    if pad_h0 or pad_h1 or pad_w0 or pad_w1:
        x = F.pad(x, (pad_w0, pad_w1, pad_h0, pad_h1))
    return x


# ---------------------------------------------------------------------------
# Pan2020 Partial Convolution 2D
# ---------------------------------------------------------------------------

class Pan2020PartialConv2d(nn.Module):
    """Partial convolution with author-code global mask-mean normalization.

    For input data ``X`` (B, C_in, H, W) and binary mask ``M`` (B, C_in, H, W)::

        rho = mean(M, dim=(-2, -1))                       # per-channel global mean
        Y   = Conv2d((X * M) / rho, weight, bias, ...)     # data output
        S   = Conv2d(M, ones_kernel, ...)                  # mask count
        M'  = (S > 0).float()                              # updated binary mask

    When ``M`` is all-ones, ``rho = 1`` and the layer degenerates to ordinary
    ``Conv2d``.  The mask kernel is a fixed (non-trainable) all-ones buffer.

    Parameters
    ----------
    in_channels / out_channels : channel dimensions.
    kernel_size  : square convolution kernel (int or 2-tuple).
    stride       : convolution stride.
    dilation     : convolution dilation.
    bias         : whether the data convolution carries a bias term.
    normalization_mode : ``"author_global_mask_mean"`` (Pan2020 author code)
                         or ``"standard_local_valid_ratio"`` (inpainting PConv).
    eps          : epsilon for local-ratio mode only.
    zero_valid_policy : ``"error"`` (raise) or ``"clamp"`` (use eps).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int | Tuple[int, int] = 1,
        dilation: int | Tuple[int, int] = 1,
        bias: bool = True,
        normalization_mode: str = "author_global_mask_mean",
        eps: float = 1.0e-8,
        zero_valid_policy: str = "error",
    ) -> None:
        super().__init__()
        if normalization_mode not in (
            "author_global_mask_mean",
            "standard_local_valid_ratio",
        ):
            raise ValueError(
                f"normalization_mode must be 'author_global_mask_mean' or "
                f"'standard_local_valid_ratio', got {normalization_mode!r}."
            )
        if zero_valid_policy not in ("error", "clamp"):
            raise ValueError(
                f"zero_valid_policy must be 'error' or 'clamp', got {zero_valid_policy!r}."
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.normalization_mode = normalization_mode
        self.eps = float(eps)
        self.zero_valid_policy = zero_valid_policy

        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size[0], kernel_size[-1]
        if isinstance(stride, int):
            sh = sw = stride
        else:
            sh, sw = stride[0], stride[-1]
        if isinstance(dilation, int):
            dh = dw = dilation
        else:
            dh, dw = dilation[0], dilation[-1]

        self.kernel_size = (kh, kw)
        self.stride = (sh, sw)
        self.dilation = (dh, dw)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kh, kw)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Fixed all-ones mask kernel — not trainable.
        self.register_buffer(
            "mask_kernel",
            torch.ones(out_channels, in_channels, kh, kw),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.weight.numel() // self.out_channels
            bound = 1.0 / (fan_in ** 0.5) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        data : ``(B, C_in, H, W)`` seismic feature tensor.
        mask : ``(B, C_in, H, W)`` binary mask (1 = valid, 0 = missing).

        Returns
        -------
        (data_out, mask_out) each ``(B, C_out, H_out, W_out)``.
        """
        if data.shape[1] != self.in_channels:
            raise ValueError(
                f"Pan2020PartialConv2d expects {self.in_channels} data channels, "
                f"got {data.shape[1]}."
            )
        if mask.shape[1] != self.in_channels:
            raise ValueError(
                f"Pan2020PartialConv2d expects {self.in_channels} mask channels, "
                f"got {mask.shape[1]}."
            )

        mask = mask.detach()

        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation

        # ---- Same-padding ----
        data = _pad_same(data, (kh, kw), (sh, sw), dh)
        mask = _pad_same(mask, (kh, kw), (sh, sw), dh)

        # ---- Data convolution ----
        if self.normalization_mode == "author_global_mask_mean":
            valid_fraction = mask.mean(dim=(-2, -1), keepdim=True)  # (B,C,1,1)
            if self.zero_valid_policy == "error":
                if (valid_fraction <= 0).any():
                    raise RuntimeError(
                        "Pan2020PartialConv2d received a feature channel with "
                        "zero valid fraction.  Input may contain an all-missing "
                        "spatial region propagating through the network."
                    )
            normalized = (data * mask) / valid_fraction
        else:  # standard_local_valid_ratio
            mask_count = F.conv2d(
                mask, self.mask_kernel, bias=None,
                stride=(sh, sw), dilation=(dh, dw),
            )
            if self.zero_valid_policy == "error":
                if (mask_count <= 0).any():
                    raise RuntimeError(
                        "Pan2020PartialConv2d: local valid-pixel count is zero "
                        "for at least one output position."
                    )
            conv_masked = F.conv2d(
                data * mask, self.weight, bias=None,
                stride=(sh, sw), dilation=(dh, dw),
            )
            data_out = conv_masked * (kh * kw * self.in_channels) / mask_count.clamp_min(self.eps)
            if self.bias is not None:
                data_out = data_out + self.bias.view(1, -1, 1, 1)
            # ---- Mask update ----
            mask_out = (mask_count > 0).to(dtype=data.dtype).detach()
            return data_out, mask_out

        data_out = F.conv2d(
            normalized, self.weight, self.bias,
            stride=(sh, sw), dilation=(dh, dw),
        )

        # ---- Mask update ----
        mask_count = F.conv2d(
            mask, self.mask_kernel, bias=None,
            stride=(sh, sw), dilation=(dh, dw),
        )
        mask_out = (mask_count > 0).to(dtype=data.dtype).detach()

        return data_out, mask_out


# ---------------------------------------------------------------------------
# Encoder / Decoder building blocks
# ---------------------------------------------------------------------------

class Pan2020EncoderStage(nn.Module):
    """One encoder stage: stride-2 PConv + optional BatchNorm + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_bn: bool = True,
        normalization_mode: str = "author_global_mask_mean",
        zero_valid_policy: str = "error",
    ) -> None:
        super().__init__()
        self.pconv = Pan2020PartialConv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=2,
            normalization_mode=normalization_mode,
            zero_valid_policy=zero_valid_policy,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data, mask = self.pconv(data, mask)
        data = self.act(self.bn(data))
        return data, mask


class Pan2020DecoderStage(nn.Module):
    """One decoder stage: upsample + concat(data+mask) + stride-1 PConv +
    optional BatchNorm + LeakyReLU(0.2)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_bn: bool = True,
        leaky_relu_slope: float = 0.2,
        normalization_mode: str = "author_global_mask_mean",
        zero_valid_policy: str = "error",
    ) -> None:
        super().__init__()
        self.pconv = Pan2020PartialConv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=1,
            normalization_mode=normalization_mode,
            zero_valid_policy=zero_valid_policy,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)

    def forward(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data, mask = self.pconv(data, mask)
        data = self.act(self.bn(data))
        return data, mask


# ---------------------------------------------------------------------------
# Pan2020 PConv U-Net
# ---------------------------------------------------------------------------

@register_model("pan2020_pconv_unet")
class Pan2020PConvUNet(nn.Module):
    """Partial Convolution U-Net for seismic data regularization.

    Parameters
    ----------
    in_channels : input seismic channels (paper: 1).
    out_channels : output channels (paper: 1).
    encoder_channels : output channels for each of the 6 encoder stages.
    encoder_kernels : kernel sizes for each encoder stage.
    decoder_channels : output channels for each of the 6 decoder stages.
    normalization_mode : PConv normalization (``"author_global_mask_mean"``
        for the author-code profile, ``"standard_local_valid_ratio"`` for
        ablation).
    upsample_mode : ``"nearest"`` or ``"bilinear"``.
    output_activation : ``"sigmoid"`` (author-code) or ``"none"``.
    leaky_relu_slope : negative slope for decoder LeakyReLU.
    input_mode : ``"separate"`` (model(data, mask)) or ``"packed"``
        (model(packed_x) with data in channel 0, mask in channel 1).
    zero_valid_policy : ``"error"`` or ``"clamp"`` for all-zero mask channels.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: Tuple[int, ...] = (32, 64, 128, 256, 512, 512),
        encoder_kernels: Tuple[int, ...] = (7, 5, 5, 3, 3, 3),
        decoder_channels: Tuple[int, ...] = (512, 256, 128, 64, 32, 1),
        normalization_mode: str = "author_global_mask_mean",
        upsample_mode: str = "nearest",
        output_activation: str = "sigmoid",
        leaky_relu_slope: float = 0.2,
        input_mode: str = "separate",
        zero_valid_policy: str = "error",
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 6:
            raise ValueError(f"encoder_channels must have 6 entries, got {len(encoder_channels)}.")
        if len(encoder_kernels) != 6:
            raise ValueError(f"encoder_kernels must have 6 entries, got {len(encoder_kernels)}.")
        if len(decoder_channels) != 6:
            raise ValueError(f"decoder_channels must have 6 entries, got {len(decoder_channels)}.")
        if upsample_mode not in ("nearest", "bilinear"):
            raise ValueError(f"upsample_mode must be 'nearest' or 'bilinear', got {upsample_mode!r}.")
        if output_activation not in ("sigmoid", "none"):
            raise ValueError(f"output_activation must be 'sigmoid' or 'none', got {output_activation!r}.")
        if input_mode not in ("separate", "packed"):
            raise ValueError(f"input_mode must be 'separate' or 'packed', got {input_mode!r}.")
        if zero_valid_policy not in ("error", "clamp"):
            raise ValueError(f"zero_valid_policy must be 'error' or 'clamp', got {zero_valid_policy!r}.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.normalization_mode = normalization_mode
        self.upsample_mode = upsample_mode
        self.output_activation = output_activation
        self.input_mode = input_mode

        # ---- Encoder ----
        enc_kwargs = dict(
            normalization_mode=normalization_mode,
            zero_valid_policy=zero_valid_policy,
        )
        self.encoder_stages = nn.ModuleList()
        prev_ch = int(in_channels)
        for i, (ch, k) in enumerate(zip(encoder_channels, encoder_kernels)):
            use_bn = (i >= 1)  # encoder 1 has no BN
            self.encoder_stages.append(
                Pan2020EncoderStage(prev_ch, ch, kernel_size=k, use_bn=use_bn, **enc_kwargs)
            )
            prev_ch = ch

        # ---- Decoder ----
        self.decoder_stages = nn.ModuleList()
        # Decoder input channels: upsampled previous + encoder skip
        # Decoder 7:  512(upsampled) + 512(enc5) = 1024
        # Decoder 8:  256(upsampled) + 256(enc4) = 512  -> wait, let me recalculate
        # Decoder channels: [512, 256, 128, 64, 32, 1]
        # Up from encoder6: (B,512,2,2), up to (B,512,4,4), concat enc5 (B,512,4,4) → 1024
        # Dec7 out: 512
        # Up to (B,512,8,8), concat enc4 (B,256,8,8) → 768
        # Dec8 out: 256
        # Up to (B,256,16,16), concat enc3 (B,128,16,16) → 384
        # Dec9 out: 128
        # Up to (B,128,32,32), concat enc2 (B,64,32,32) → 192
        # Dec10 out: 64
        # Up to (B,64,64,64), concat enc1 (B,32,64,64) → 96
        # Dec11 out: 32
        # Up to (B,32,128,128), concat input (B,1,128,128) → 33
        # Dec12 out: 1
        dec_in_channels: List[int] = []
        prev_dec_ch = encoder_channels[-1]  # 512 from enc6
        for i in range(6):
            enc_skip_idx = 4 - i  # enc5, enc4, enc3, enc2, enc1, input
            if enc_skip_idx >= 0:
                skip_ch = encoder_channels[enc_skip_idx]
            else:
                skip_ch = int(in_channels)  # original input
            dec_in_channels.append(prev_dec_ch + skip_ch)
            prev_dec_ch = decoder_channels[i]

        for i, (in_ch, out_ch) in enumerate(zip(dec_in_channels, decoder_channels)):
            is_last = (i == 5)  # decoder 12 has no BN
            self.decoder_stages.append(
                Pan2020DecoderStage(
                    in_ch, out_ch,
                    kernel_size=3,
                    use_bn=not is_last,
                    leaky_relu_slope=leaky_relu_slope,
                    **enc_kwargs,
                )
            )

        # ---- Output head ----
        self.output_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)
        self.output_act: nn.Module
        if output_activation == "sigmoid":
            self.output_act = nn.Sigmoid()
        else:
            self.output_act = nn.Identity()

    def _extract_inputs(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve data and mask from the configured input_mode."""
        if self.input_mode == "separate":
            if mask is not None:
                return x, mask
            # Fallback: assume packed 2-channel input.
            if x.shape[1] >= 2:
                return x[:, :1], x[:, 1:2]
            raise ValueError(
                "Pan2020PConvUNet in 'separate' mode requires `mask` argument "
                "or a packed input with >= 2 channels."
            )
        else:  # packed
            if x.shape[1] < 2:
                raise ValueError(
                    "Pan2020PConvUNet in 'packed' mode requires >= 2 input channels "
                    "(data + mask)."
                )
            return x[:, :1], x[:, 1:2]

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x    : seismic data tensor ``(B, C, H, W)``.  In ``"separate"`` mode
               this is the masked input only; in ``"packed"`` mode channels
               0 and 1 carry data and mask respectively.
        mask : optional observed mask ``(B, 1, H, W)`` (1 = observed, 0 =
               missing).  Required in ``"separate"`` mode; ignored in
               ``"packed"`` mode.
        **kwargs : absorbed for trainer compatibility.

        Returns
        -------
        Output tensor ``(B, out_channels, H, W)`` in [0, 1] (sigmoid) or
        unbounded (output_activation='none').
        """
        data, current_mask = self._extract_inputs(x, mask)

        # ---- Encoder ----
        enc_data_list: List[torch.Tensor] = []
        enc_mask_list: List[torch.Tensor] = []
        for stage in self.encoder_stages:
            data, current_mask = stage(data, current_mask)
            enc_data_list.append(data)
            enc_mask_list.append(current_mask)

        # ---- Decoder ----
        # enc_data_list indices: [0]=enc1, [1]=enc2, ..., [5]=enc6
        # Decoder skips: dec7←enc5(idx4), dec8←enc4(idx3), ..., dec11←enc1(idx0), dec12←input
        for i, stage in enumerate(self.decoder_stages):
            # Upsample data and mask
            data = F.interpolate(
                data, scale_factor=2, mode=self.upsample_mode,
            )
            current_mask = F.interpolate(
                current_mask, scale_factor=2, mode="nearest",
            )
            # Nearest upsampling may produce fractional values; re-binarize.
            current_mask = (current_mask > 0.5).to(dtype=data.dtype)

            if i < 5:
                skip_idx = 4 - i  # enc5 → idx4, enc4 → idx3, ..., enc1 → idx0
                skip_data = enc_data_list[skip_idx]
                skip_mask = enc_mask_list[skip_idx]
            else:
                # Last decoder: skip = original input
                skip_data, skip_mask = self._extract_inputs(x, mask)
                # Expand skip_mask to match data channels for concatenation
                if skip_mask.shape[1] == 1 and skip_data.shape[1] == 1:
                    skip_mask = skip_mask.expand_as(skip_data)
                elif skip_mask.shape[1] != skip_data.shape[1]:
                    skip_mask = skip_mask.expand(
                        -1, skip_data.shape[1], -1, -1
                    ).contiguous()

            # Align spatial sizes
            if data.shape[-2:] != skip_data.shape[-2:]:
                data = F.interpolate(
                    data, size=skip_data.shape[-2:],
                    mode=self.upsample_mode,
                )
                current_mask = F.interpolate(
                    current_mask, size=skip_data.shape[-2:],
                    mode="nearest",
                )
                current_mask = (current_mask > 0.5).to(dtype=data.dtype)

            data = torch.cat([skip_data, data], dim=1)
            current_mask = torch.cat([skip_mask, current_mask], dim=1)
            data, current_mask = stage(data, current_mask)

        # ---- Output ----
        out = self.output_conv(data)
        return self.output_act(out)
