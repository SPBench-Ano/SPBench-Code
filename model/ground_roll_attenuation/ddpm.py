"""Conditional DDPM (cDDPM-2c) U-Net backbone + noise scheduler for ground-roll attenuation.

The model jointly predicts the noise added to both signal and ground-roll components.
Input : concat(y, x_t, z_t) → 3 channels (condition, noised signal, noised ground-roll)
Output: (eps_sig, eps_gr) → 2 channels (predicted noise for signal and ground-roll)

Reference
---------
Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


# ---------------------------------------------------------------------------
# time embedding
# ---------------------------------------------------------------------------

def _sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal position encoding (Vaswani et al., 2017)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _TimeEmbedding(nn.Module):
    """Sinusoidal → Linear → SiLU → Linear."""

    def __init__(self, out_dim: int, base_dim: int = 128) -> None:
        super().__init__()
        self.base_dim = base_dim
        self.net = nn.Sequential(
            nn.Linear(base_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = _sinusoidal_embedding(t, self.base_dim)
        return self.net(emb)


# ---------------------------------------------------------------------------
# ResNet block with time embedding injection
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → Conv3×3 + time proj + residual."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, num_groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_ch), in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_ch)

        self.skip: Optional[nn.Conv2d] = None
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.norm1(x))
        h = self.conv1(h)

        t_out = self.time_proj(t_emb)
        while t_out.dim() < h.dim():
            t_out = t_out.unsqueeze(-1)
        h = h + t_out

        h = self.act2(self.norm2(h))
        h = self.conv2(h)

        skip_x = self.skip(x) if self.skip is not None else x
        return h + skip_x


# ---------------------------------------------------------------------------
# self-attention (bottleneck)
# ---------------------------------------------------------------------------

class _SelfAttention(nn.Module):
    """Multi-head self-attention with residual connection."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)                     # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)         # each (B, C, H, W)

        head_dim = C // self.num_heads
        scale = head_dim ** -0.5

        q = q.reshape(B, self.num_heads, head_dim, H * W).permute(0, 1, 3, 2)  # (B, nh, HW, hd)
        k = k.reshape(B, self.num_heads, head_dim, H * W)                        # (B, nh, hd, HW)
        v = v.reshape(B, self.num_heads, head_dim, H * W).permute(0, 1, 3, 2)  # (B, nh, HW, hd)

        attn = (q @ k) * scale                                                  # (B, nh, HW, HW)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v)                                                         # (B, nh, HW, hd)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return x + self.proj(out)


# ---------------------------------------------------------------------------
# downsampling / upsampling helpers
# ---------------------------------------------------------------------------

class _Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(h)


# ---------------------------------------------------------------------------
# DDPM U-Net
# ---------------------------------------------------------------------------

@register_model("ddpm_unet")
class DDPMUNet(nn.Module):
    """Modified U-Net for cDDPM-2c ground-roll attenuation.

    Parameters
    ----------
    in_channels : input channels (3 = condition + noised_signal + noised_gr).
    out_channels : output channels (2 = eps_signal, eps_groundroll).
    base_channels : first-level channel count.
    channel_mults : multipliers per resolution level (len = num levels).
    time_emb_dim : internal dimension for time embedding.
    num_res_blocks : ResNet blocks per level (default 1, matching paper's 5 total).
    attn_resolutions : resolutions at which to insert self-attention (list of ints).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8, 8),
        time_emb_dim: int = 256,
        num_res_blocks: int = 1,
        attn_resolutions: Tuple[int, ...] = (16,),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        chs = [base_channels * m for m in channel_mults]  # [64, 128, 256, 512, 512]
        num_levels = len(chs)

        # time embedding
        self.time_emb = _TimeEmbedding(time_emb_dim)

        # ---- input projection -----------------------------------------------
        self.input_conv = nn.Conv2d(in_channels, chs[0], kernel_size=3, padding=1)

        # ---- encoder --------------------------------------------------------
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        enc_chs: List[int] = []

        prev = chs[0]
        for level in range(num_levels):
            c = chs[level]
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(_ResBlock(prev, c, time_emb_dim))
                prev = c
            enc_chs.append(prev)
            self.enc_blocks.append(level_blocks)
            # downsample except at last level
            if level < num_levels - 1:
                self.downsamples.append(_Downsample(prev))

        # ---- bottleneck (mid block with attention) --------------------------
        self.mid_block1 = _ResBlock(prev, prev, time_emb_dim)
        self.mid_attn = _SelfAttention(prev)
        self.mid_block2 = _ResBlock(prev, prev, time_emb_dim)

        # ---- decoder --------------------------------------------------------
        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in range(num_levels - 1, -1, -1):
            c = chs[level]
            # after skip concatenation, input channels = prev + enc_chs[level]
            in_c = prev + enc_chs[level]
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(_ResBlock(in_c, c, time_emb_dim))
                in_c = c  # subsequent blocks in same level use c→c
            # upsample receives input from the previous (deeper) decoder level,
            # so its channel count is the pre-update `prev`, not the new `c`
            if level > 0:
                self.upsamples.append(_Upsample(prev))
            prev = c
            self.dec_blocks.append(level_blocks)

        # ---- output ---------------------------------------------------------
        self.out_norm = nn.GroupNorm(min(8, prev), prev)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(prev, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, 3, H, W) — concatenated [y, x_t, z_t].
        t : (B,) or (B, 1) — diffusion timestep integers.

        Returns
        -------
        (B, 2, H, W) — predicted noise (eps_signal, eps_groundroll).
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_emb = self.time_emb(t)  # (B, time_emb_dim)

        h = self.input_conv(x)

        # encoder
        skips: List[torch.Tensor] = []
        for level_blocks, downsample in zip(
            self.enc_blocks,
            list(self.downsamples) + [None],
        ):
            for blk in level_blocks:
                h = blk(h, t_emb)
            skips.append(h)
            if downsample is not None:
                h = downsample(h)

        # bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # decoder
        for level_blocks, upsample, skip in zip(
            self.dec_blocks,
            list(self.upsamples) + [None],
            reversed(skips),
        ):
            if upsample is not None:
                h = upsample(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([skip, h], dim=1)
            for blk in level_blocks:
                h = blk(h, t_emb)

        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)


# ---------------------------------------------------------------------------
# DDPM noise scheduler
# ---------------------------------------------------------------------------

class DDPMNoiseScheduler:
    """Linear β schedule DDPM scheduler for the two-component model.

    Parameters
    ----------
    num_timesteps : T (default 1000).
    beta_start, beta_end : linear schedule endpoints.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        self.num_timesteps = int(num_timesteps)

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self._betas = betas.float()
        self._alphas = alphas.float()
        self._alphas_cumprod = alphas_cumprod.float()
        self._sqrt_alphas_cumprod = alphas_cumprod.sqrt().float()
        self._sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt().float()
        self._sqrt_recip_alphas = alphas.rsqrt().float()
        self._sqrt_recip_alphas_cumprod = alphas_cumprod.rsqrt().float()
        self._posterior_variance = (
            betas * (1.0 - self._alphas_cumprod_prev()) / (1.0 - alphas_cumprod)
        ).float()

    def _alphas_cumprod_prev(self) -> torch.Tensor:
        return F.pad(self._alphas_cumprod[:-1], (1, 0), value=1.0)

    def to(self, device: torch.device) -> "DDPMNoiseScheduler":
        """Move all precomputed tensors to *device* (in-place, returns self)."""
        for name in list(self.__dict__.keys()):
            val = getattr(self, name)
            if isinstance(val, torch.Tensor):
                setattr(self, name, val.to(device))
        return self

    def add_noise(
        self,
        x_0: torch.Tensor,
        z_0: torch.Tensor,
        t: torch.Tensor,
        eps_sig: Optional[torch.Tensor] = None,
        eps_gr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: return (x_t, z_t).

        Parameters
        ----------
        x_0 : (B, C, H, W) clean signal.
        z_0 : (B, C, H, W) ground-roll noise.
        t   : (B,) integer timesteps.
        eps_sig, eps_gr : optional pre-sampled noise; sampled if None.
        """
        device = x_0.device
        B = x_0.shape[0]
        if eps_sig is None:
            eps_sig = torch.randn_like(x_0)
        if eps_gr is None:
            eps_gr = torch.randn_like(z_0)

        a_bar = self._alphas_cumprod[t].to(device)
        while a_bar.dim() < x_0.dim():
            a_bar = a_bar.unsqueeze(-1)

        x_t = a_bar.sqrt() * x_0 + (1.0 - a_bar).sqrt() * eps_sig
        z_t = a_bar.sqrt() * z_0 + (1.0 - a_bar).sqrt() * eps_gr
        return x_t, z_t

    @torch.inference_mode()
    def sample_prev_step(
        self,
        model: nn.Module,
        y: torch.Tensor,
        x_t: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single DDPM reverse step: (x_t, z_t) → (x_{t-1}, z_{t-1})."""
        device = x_t.device
        B = x_t.shape[0]

        inp = torch.cat([y, x_t, z_t], dim=1)   # (B, 3, H, W)
        eps_pred = model(inp, t)                 # (B, 2, H, W)
        eps_sig_pred = eps_pred[:, :1, :, :]    # (B, 1, H, W)
        eps_gr_pred = eps_pred[:, 1:, :, :]     # (B, 1, H, W)

        a = self._alphas[t].to(device)
        a_bar = self._alphas_cumprod[t].to(device)
        sqrt_recip_a = self._sqrt_recip_alphas[t].to(device)
        while a.dim() < x_t.dim():
            a = a.unsqueeze(-1)
            a_bar = a_bar.unsqueeze(-1)
            sqrt_recip_a = sqrt_recip_a.unsqueeze(-1)

        # posterior variance σ_t
        beta = self._betas[t].to(device)
        sigma = beta.sqrt()
        # for t=0 we use σ=0 (handled by the noise term below)
        while sigma.dim() < x_t.dim():
            sigma = sigma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        coef = (1.0 - a) / (1.0 - a_bar).sqrt()

        x_prev = sqrt_recip_a * (x_t - coef * eps_sig_pred)
        z_prev = sqrt_recip_a * (z_t - coef * eps_gr_pred)

        # add noise for t > 0
        is_not_zero = (t > 0).float()
        while is_not_zero.dim() < x_t.dim():
            is_not_zero = is_not_zero.unsqueeze(-1)
        x_prev = x_prev + sigma * is_not_zero * torch.randn_like(x_t)
        z_prev = z_prev + sigma * is_not_zero * torch.randn_like(z_t)

        return x_prev, z_prev

    @torch.inference_mode()
    def sample_full(
        self,
        model: nn.Module,
        y: torch.Tensor,
        num_steps: Optional[int] = None,
        use_ddim: bool = True,
        ddim_eta: float = 0.0,
        progress: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full DDPM/DDIM reverse chain: pure noise → (x_0, z_0).

        Parameters
        ----------
        model     : trained DDPMUNet.
        y         : (B, 1, H, W) condition (noisy input).
        num_steps : number of steps for accelerated sampling (None = full num_timesteps).
                    Steps are evenly spaced across the full schedule.
        use_ddim  : use DDIM sampling when num_steps is set (default True).
        ddim_eta  : DDIM stochasticity (0 = deterministic, 1 = DDPM-like).
        progress  : print progress every 100 steps.

        Returns
        -------
        x_0, z_0 : (B, 1, H, W) each — denoised signal and ground-roll.
        """
        device = next(model.parameters()).device
        B, _, H, W = y.shape

        x_t = torch.randn(B, 1, H, W, device=device)
        z_t = torch.randn(B, 1, H, W, device=device)

        if num_steps is None:
            ts = list(range(self.num_timesteps - 1, -1, -1))
        else:
            step = max(1, self.num_timesteps // int(num_steps))
            ts = list(range(self.num_timesteps - 1, -1, -step))

        total_steps = len(ts)
        for i in range(len(ts)):
            t_curr = ts[i]
            t_prev = ts[i + 1] if i + 1 < len(ts) else 0

            if use_ddim and num_steps is not None:
                x_t, z_t = self._sample_ddim_step(model, y, x_t, z_t, t_curr, t_prev, ddim_eta)
            else:
                t = torch.full((B,), t_curr, device=device, dtype=torch.long)
                x_t, z_t = self.sample_prev_step(model, y, x_t, z_t, t)

            if progress and (i + 1) % 100 == 0:
                print(f"  DDPM sampling step {i + 1}/{total_steps}")

        return x_t, z_t

    @torch.inference_mode()
    def _sample_ddim_step(
        self,
        model: nn.Module,
        y: torch.Tensor,
        x_t: torch.Tensor,
        z_t: torch.Tensor,
        t_curr: int,
        t_prev: int,
        eta: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single DDIM step: (x_t, z_t) at t_curr → (x_{t_prev}, z_{t_prev}).

        Uses the non-Markovian DDIM update (Song et al., 2021) for
        accelerated sampling with strided timesteps.
        """
        device = x_t.device
        B = x_t.shape[0]

        # model forward
        t = torch.full((B,), t_curr, device=device, dtype=torch.long)
        inp = torch.cat([y, x_t, z_t], dim=1)
        eps_pred = model(inp, t)
        eps_sig_pred = eps_pred[:, :1, :, :]
        eps_gr_pred = eps_pred[:, 1:, :, :]

        # cumulative alphas
        a_curr = self._alphas_cumprod[t_curr].to(device)
        a_prev = self._alphas_cumprod[t_prev].to(device)
        while a_curr.dim() < x_t.dim():
            a_curr = a_curr.unsqueeze(-1)
            a_prev = a_prev.unsqueeze(-1)

        # predicted x_0 and z_0
        x_0_pred = (x_t - (1.0 - a_curr).sqrt() * eps_sig_pred) / a_curr.sqrt().clamp(min=1e-8)
        z_0_pred = (z_t - (1.0 - a_curr).sqrt() * eps_gr_pred) / a_curr.sqrt().clamp(min=1e-8)

        # sigma_t (DDIM stochasticity)
        if eta > 0:
            sigma = eta * ((1.0 - a_prev) / (1.0 - a_curr).clamp(min=1e-8)).sqrt() * (
                1.0 - a_curr / a_prev.clamp(min=1e-8)
            ).sqrt().clamp(min=0.0)
        else:
            sigma = torch.zeros_like(a_curr)

        # direction pointing to x_t: sqrt(1 - a_prev - sigma^2)
        coeff_dir = (1.0 - a_prev - sigma ** 2).clamp(min=0.0).sqrt()

        # DDIM update
        x_prev = a_prev.sqrt() * x_0_pred + coeff_dir * eps_sig_pred
        z_prev = a_prev.sqrt() * z_0_pred + coeff_dir * eps_gr_pred

        if eta > 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)
            z_prev = z_prev + sigma * torch.randn_like(z_t)

        return x_prev, z_prev

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Serialize precomputed tensors for checkpointing."""
        return {
            "betas": self._betas,
            "alphas": self._alphas,
            "alphas_cumprod": self._alphas_cumprod,
        }

    def load_state_dict(self, d: Dict[str, torch.Tensor]) -> None:
        """Restore from a state dict (tensors are placed on the stored device)."""
        self._betas = d["betas"]
        self._alphas = d["alphas"]
        self._alphas_cumprod = d["alphas_cumprod"]
        self._sqrt_alphas_cumprod = self._alphas_cumprod.sqrt()
        self._sqrt_one_minus_alphas_cumprod = (1.0 - self._alphas_cumprod).sqrt()
        self._sqrt_recip_alphas = self._alphas.rsqrt()
        self._sqrt_recip_alphas_cumprod = self._alphas_cumprod.rsqrt()
        self._posterior_variance = (
            self._betas * (1.0 - self._alphas_cumprod_prev()) / (1.0 - self._alphas_cumprod)
        )
        self.num_timesteps = len(self._betas)
