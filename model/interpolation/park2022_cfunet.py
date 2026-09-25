"""Park et al. 2022 Coarse-Refine Network (CFunet) for missing-trace interpolation.

Paper: "Coarse-Refine Network With Upsampling Techniques and Fourier Loss for
the Reconstruction of Missing Seismic Data", IEEE TGRS, vol. 60, 2022,
art. 5917615.  DOI: 10.1109/TGRS.2022.3190292.

Pipeline (paper Eqs. 2-5): coarse U-Net Gc -> Fourier zero-padding
upsampling x2 in both axes -> refine U-Net Gf -> final 3x3 stride-2 conv
projecting back to the label size.  Observed traces are copied into the
coarse output (Eq. 2: ``Oc = Gc(I) * (1 - M) + I * M`` with ``M = 1`` at
observed positions).  Fourier upsampling (Eqs. 3-4) centers the 2D spectrum
with ``torch.fft.fftshift``, zero-pads it symmetrically so that
``|i| <= Ni/2`` and ``|j| <= Nj/2`` are preserved, ``ifftshift``-es back and
takes the real part; the amplitude correction keeps aligned samples exact
for the chosen ``torch.fft`` normalization.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .unet import UNet


class FourierZeroPaddingUpsample2D(nn.Module):
    """2D Fourier-domain zero-padding upsampler (paper Eqs. 3-4).

    Parameters
    ----------
    scale_factor : integer upsampling factor applied to both spatial axes.
    use_fftshift : center the spectrum before zero-padding (paper Eqs. 3-4).
    fft_norm : ``"backward"``, ``"ortho"`` or ``"forward"`` passed to
        ``torch.fft.fft2`` / ``torch.fft.ifft2``.
    amplitude_correction : ``"scale"`` multiplies the output by the exact
        aligned-sample correction factor (s_h*s_w for ``"backward"``,
        sqrt(s_h*s_w) for ``"ortho"``, 1.0 for ``"forward"``); ``"none"``
        returns the raw inverse-transform output.
    """

    def __init__(
        self,
        scale_factor: int = 2,
        use_fftshift: bool = True,
        fft_norm: str = "backward",
        amplitude_correction: str = "scale",
    ) -> None:
        super().__init__()
        if not isinstance(scale_factor, int) or scale_factor < 2:
            raise ValueError(
                f"scale_factor must be an integer >= 2, got {scale_factor!r}."
            )
        if fft_norm not in ("backward", "ortho", "forward"):
            raise ValueError(
                f"fft_norm must be 'backward', 'ortho' or 'forward', got {fft_norm!r}."
            )
        if amplitude_correction not in ("scale", "none"):
            raise ValueError(
                f"amplitude_correction must be 'scale' or 'none', got {amplitude_correction!r}."
            )
        self.scale_factor = int(scale_factor)
        self.use_fftshift = bool(use_fftshift)
        self.fft_norm = fft_norm
        self.amplitude_correction = amplitude_correction

    def _correction_factor(self) -> float:
        """Aligned-sample amplitude correction for the configured norm."""
        s = self.scale_factor
        if self.fft_norm == "forward":
            return 1.0
        if self.fft_norm == "ortho":
            return float(s)
        return float(s * s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample ``(B, C, H, W)`` by ``scale_factor`` along both axes.

        Parameters
        ----------
        x : input tensor; both spatial sizes must be even.

        Returns
        -------
        Upsampled tensor ``(B, C, s*H, s*W)``.  Aligned samples equal the
        input up to the correction factor; interpolation points follow the
        sinc kernel implied by the zero-padding.
        """
        h, w = x.shape[-2:]
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(
                f"FourierZeroPaddingUpsample2D requires even spatial sizes, "
                f"got (H, W) = ({h}, {w})."
            )
        s = self.scale_factor
        pad_h = (s - 1) * h // 2
        pad_w = (s - 1) * w // 2
        spec = torch.fft.fft2(x, norm=self.fft_norm)
        if self.use_fftshift:
            spec = torch.fft.fftshift(spec, dim=(-2, -1))
        spec = F.pad(spec, (pad_w, pad_w, pad_h, pad_h))
        if self.use_fftshift:
            spec = torch.fft.ifftshift(spec, dim=(-2, -1))
        out = torch.fft.ifft2(spec, norm=self.fft_norm).real
        if self.amplitude_correction == "scale":
            out = out * self._correction_factor()
        return out


@register_model("park2022_cfunet")
class Park2022CFUNet(nn.Module):
    """Coarse-Refine Network for missing-trace reconstruction (paper Fig. 1).

    Parameters
    ----------
    in_channels : input channels (paper: 1).
    out_channels : output channels (paper: 1).
    base_channels : width of the first encoder stage of both U-Nets
        (paper: 22; channels double per level: 22, 44, 88, 176).
    num_levels : encoder levels per U-Net (paper: 4; the bottleneck width is
        ``base_channels * 2**num_levels`` = 352).
    upsample_mode : ``"fourier_zero_padding"`` (paper Eqs. 3-4),
        ``"bilinear"`` or ``"nearest"`` (ablation).
    upsampler_scale_factor : integer upsampling factor (paper: 2).
    use_fftshift : center the spectrum before zero-padding.
    fft_norm : FFT normalization for the Fourier upsampler.
    amplitude_correction : ``"scale"`` or ``"none"`` for the Fourier
        upsampler.

    Attributes
    ----------
    _intermediates : latest forward's ``{"coarse", "upsampled", "final"}``
        tensors, consumed by ``cfunet_mse_fourier`` via ``attach_model``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 22,
        num_levels: int = 4,
        upsample_mode: str = "fourier_zero_padding",
        upsampler_scale_factor: int = 2,
        use_fftshift: bool = True,
        fft_norm: str = "backward",
        amplitude_correction: str = "scale",
    ) -> None:
        super().__init__()
        if upsample_mode not in ("fourier_zero_padding", "bilinear", "nearest"):
            raise ValueError(
                f"upsample_mode must be 'fourier_zero_padding', 'bilinear' or "
                f"'nearest', got {upsample_mode!r}."
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.upsample_mode = upsample_mode
        self.upsampler_scale_factor = int(upsampler_scale_factor)

        # Gc: coarse U-Net producing the label-size reconstruction.
        self.coarse_net = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=num_levels,
        )
        # Gf: refine U-Net operating at the upsampled resolution.
        self.refine_net = UNet(
            in_channels=out_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=num_levels,
        )
        if upsample_mode == "fourier_zero_padding":
            self.upsampler = FourierZeroPaddingUpsample2D(
                scale_factor=upsampler_scale_factor,
                use_fftshift=use_fftshift,
                fft_norm=fft_norm,
                amplitude_correction=amplitude_correction,
            )
        else:
            self.upsampler = nn.Identity()
        # Mandatory final projection (paper Eq. 5): 3x3 stride-2 conv.
        self.refine_down = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self._intermediates: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        **kwargs: object,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        x : masked seismic data ``(B, C, H, W)`` with zeros at missing
            traces.
        mask : observed mask ``(B, 1, H, W)`` with 1 = observed, 0 =
            missing; inferred as ``x.abs() > 1e-8`` when not given.
        return_intermediates : when ``True`` return the
            ``{"coarse", "upsampled", "final"}`` dict instead of the output.
        **kwargs : absorbed for trainer compatibility.

        Returns
        -------
        Final reconstruction ``(B, C, H, W)`` (or the intermediate dict).
        """
        if mask is None:
            mask = (x.abs() > 1.0e-8).to(dtype=x.dtype)
        mask = mask.to(dtype=x.dtype)
        if mask.shape[1] == 1 and x.shape[1] > 1:
            mask = mask.expand(-1, x.shape[1], -1, -1)
        if mask.shape != x.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match input "
                f"{tuple(x.shape)}."
            )

        oc = self.coarse_net(x)
        # Eq. 2: observed traces are copied from the input, missing traces
        # from the coarse network.
        oc = oc * (1.0 - mask) + x * mask

        if self.upsample_mode == "fourier_zero_padding":
            iu = self.upsampler(oc)
        else:
            iu = F.interpolate(
                oc,
                scale_factor=self.upsampler_scale_factor,
                mode=self.upsample_mode,
            )
        of = self.refine_down(self.refine_net(iu))
        self._intermediates = {"coarse": oc, "upsampled": iu, "final": of}
        if return_intermediates:
            return self._intermediates
        return of
