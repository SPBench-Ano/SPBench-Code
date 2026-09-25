# Park et al. 2022 — CFunet (Coarse-Refine Network with Fourier Loss)

## Paper

- **Title:** Coarse-Refine Network With Upsampling Techniques and Fourier Loss
  for the Reconstruction of Missing Seismic Data
- **Authors:** Junhwan Park, Jungkyun Shin, Soon Jee Seol, Joongmoo Byun
- **Venue:** IEEE Transactions on Geoscience and Remote Sensing, vol. 60, 2022,
  article 5917615
- **DOI:** 10.1109/TGRS.2022.3190292
- **Repository:** `model/interpolation/park2022_cfunet.py`
  (registered as `"park2022_cfunet"`)

## Method Overview

CFunet is NOT a simple double U-Net, nor a U-Net with an FFT loss bolted on.
It chains three stages that each play a distinct role:

1. **Coarse network Gc** (UNet, Eq. 1) produces a first reconstruction at the
   input (label) resolution. Its output is masked-overwritten with the
   observed data via Eq. 2, so the observed traces are copied through exactly
   and the network only has to fill missing traces.
2. **Fourier zero-padding upsampling (Eqs. 3-4)** resamples the coarse output
   by 2× in *both* axes. The 2D spectrum is centered (`fftshift`), padded with
   zeros so that all bins with `|i| ≤ Ni/2` and `|j| ≤ Nj/2` are preserved,
   then inverse-transformed. This is the paper's key "upsampling technique" —
   it yields sharper high-frequency content than bilinear interpolation.
3. **Refine network Gf** (UNet, Eq. 5) processes the upsampled representation
   and ends with a mandatory 3×3 stride-2 convolution that projects it back to
   the label size. The stride-2 conv is what makes the 2× upsampling factor in
   stage 2 exact (Eqs. 5-6 only close if `Of` has the label size).

The loss (Eqs. 6-8) is `lossMSE + α·lossF`:

```
lossMSE = (‖Oc − L‖²_F + ‖Of − L‖²_F) / 2        (Eq. 6)
lossF   = ‖F(Of) − F(L)‖₁                         (Eq. 8)
total   = lossMSE + α · lossF                     (Eq. 7)
```

The Fourier term is an L1 norm of the *complex difference* of the spectra —
not a magnitude-only comparison — which directly penalises phase errors.
The paper chooses α = 1.0 (synthetic, Table I) and α = 0.1 (field, Table II).

## Architecture

```
Input I (B, 1, T, X) — missing traces zero-filled; mask M (1 = observed)
    |
  Gc: UNet 22→44→88→176→352, 3×3 convs, BN+ReLU      (paper Fig. 3)
    |
  Oc = Gc(I)·(1−M) + I·M                              (Eq. 2)
    |
  Fourier zero-padding upsampling ×2 both axes        (Eqs. 3-4)
    |
  IU = F⁻¹([F(Oc)]_ZP)  — (B, 1, 2T, 2X)
    |
  Gf: UNet 22→44→88→176→352, 3×3 convs, BN+ReLU      (paper Fig. 4)
    |
  Of = Gf(IU) followed by 3×3 stride-2 conv           (Eq. 5)
    |
  Output (B, 1, T, X)
```

Both U-Nets reuse the repository's `UNet` (`model/interpolation/unet.py`):
4 encoder levels with channels `[22, 44, 88, 176]` and a bottleneck of 352
(`base_channels=22, num_levels=4`). Each double-conv block is
Conv3×3 → BN → ReLU ×2.

## Fourier Zero-Padding Upsampling (Eqs. 3-4)

The paper defines the upsampling as:

```
IU = F⁻¹([F(Oc)]_ZP)
(A_ZP)_{i,j} = A_{i,j}  for |i| ≤ Ni/2 and |j| ≤ Nj/2; 0 otherwise
```

Implementation (`FourierZeroPaddingUpsample2D`):

1. `spec = torch.fft.fft2(x, norm=fft_norm)`
2. `spec = torch.fft.fftshift(spec, dim=(-2,-1))` — DC to the center, Nyquist
   bin lands at index 0 (the `|i| ≤ Ni/2` convention of Eq. 4).
3. Symmetric zero-pad: `pad = (s−1)·N/2` on each side. Requires even input
   sizes (odd sizes raise `ValueError`).
4. `spec = torch.fft.ifftshift(spec, dim=(-2,-1))`, then
   `out = torch.fft.ifft2(spec, norm=fft_norm).real`.

**Amplitude correction.** Eq. 3 carries no normalisation factor. With torch's
`"backward"` norm (numpy default), aligned samples of a constant/sine come out
scaled by `1/s²`; `"ortho"` by `1/s`; `"forward"` by 1 (exact). The
`amplitude_correction="scale"` option multiplies the output by the exact
aligned-sample factor (s² / s / 1) so that the original samples are preserved
bit-exactly on the aligned grid. Interpolation points follow the sinc kernel
implied by the zero-padding (a bandlimited sine is reproduced exactly at every
point; the Nyquist bin stays at the spectrum edge, so the alternating pattern
survives). These choices are *reproduction-assumptions* — the paper is silent
on both the FFT normalisation and any correction factor.

## Mask Semantics (Eq. 2)

- `M = 1` means the trace is **observed**; `M = 0` means **missing**.
- `Oc = Gc(I)·(1−M) + I·M`: observed positions copy the input exactly, so they
  contribute zero to `‖Oc − L‖²` (the input equals the ground truth there) and
  the coarse term supervises the missing positions only — no separate loss
  mask is needed.
- The trainer passes the observed mask from the dataset as the `mask` kwarg
  (signature inspection in `utils/train_utils.py`). When no mask is given the
  model infers it as `x.abs() > 1e-8` (valid under zero-filled missing traces).

## Loss Implementation

- `CFunetMSEFourierLoss` (`utils/losses.py`, registered `"cfunet_mse_fourier"`):
  - `loss_mse = (‖Oc−L‖²_F + ‖Of−L‖²_F)/2` with per-sample spatial sum then
    batch mean (Pan2020 convention).
  - `loss_fourier = Σ|F(Of)−F(L)|` over the complex difference, same reduction.
  - `total = loss_mse + alpha·loss_fourier`.
  - The coarse output is read from `model._intermediates["coarse"]`; the model
    must be attached first via `loss.attach_model(unwrap_ddp(model))`. The
    train script wires this guarded by `hasattr(loss_fn, "attach_model")`.
- `FourierL1Loss` (`"fourier_l1"`): standalone Eq. 8 term, for ablations/tests.

## Training Protocol (paper Sec. IV)

| Setting | Value | Classification |
|---|---|---|
| Mask (train) | random trace removal, 50%–87.5% per patch | paper-explicit |
| Mask (validation) | fixed 75% missing | paper-explicit |
| Patch size | 128×128 (synthetic), 128×120 (field) | paper-explicit |
| Optimizer | Adam | paper-explicit |
| Learning rate | 5e-5 (synthetic), 1e-5 (field) | paper-explicit |
| Batch size | 32 | paper-explicit |
| Epochs | ≤ 20 | paper-explicit |
| α (Eq. 7) | 1.0 (synthetic, Table I), 0.1 (field, Table II) | paper-explicit |
| FFT norm / amplitude correction | backward / scale | reproduction-assumption |
| BN in UNet blocks | present (repo UNet) | repository-adaptation |
| Output head | 1×1 conv (repo UNet) | reproduction-assumption |
| Data normalisation | global max_abs + patch normalise | repository-adaptation |
| Scheduler / weight decay | none / 0 | reproduction-assumption |

## Input/Output Convention

- Input `(B, 1, T, X)` float32, missing traces exactly zero, values in
  approximately [-1, 1] under the repository normalisation.
- Output `(B, 1, T, X)` — the full reconstruction (observed + missing).
- The Fourier upsampler requires even `T` and `X`; the paper only uses even
  sizes (128×128, 128×120).

## Parameter Classification

| Item | Classification |
|---|---|
| CFunet two-stage design (Gc → upsample → Gf) | paper-explicit (Fig. 1-2) |
| Channels 22→44→88→176, bottleneck 352 | paper-explicit (Fig. 3-4) |
| 3×3 kernels, 4 levels + bottleneck | paper-explicit (Fig. 3-4) |
| Fourier zero-padding upsampling ×2 both axes | paper-explicit (Eqs. 3-4, Fig. 5) |
| Final 3×3 stride-2 conv in Gf | paper-explicit (Eq. 5, Fig. 4) |
| Eq. 2 observed-trace overwrite | paper-explicit |
| Eq. 6 MSE on coarse + final | paper-explicit |
| Eq. 8 Fourier L1 (complex difference) | paper-explicit |
| α values and α-sweep {10, 1, 0.1, 0.01, 0.001, 0} | paper-explicit (Tables I-II) |
| 50%–87.5% random / 75% fixed masking | paper-explicit (Sec. IV-B) |
| Adam, lr 5e-5/1e-5, batch 32, ≤20 epochs | paper-explicit (Sec. IV-B) |
| FFT normalisation | reproduction-assumption (backward) |
| Amplitude correction | reproduction-assumption (scale) |
| BN presence | repository-adaptation (figure unclear) |
| Output head type | reproduction-assumption (1×1 conv) |
| Data normalisation | repository-adaptation |
| Scheduler / weight decay | reproduction-assumption (none) |
| Batch reduction | reproduction-assumption (per-sample sum + mean) |
| Validation masking on the same range | repository-adaptation (paper fixes 75%) |
| Field low-pass filter (62.5 Hz) | not reproduced (repository has no filter stage) |

## Implementation Notes

- `model/interpolation/park2022_cfunet.py`: `FourierZeroPaddingUpsample2D` +
  `Park2022CFUNet`. The U-Net bodies are the repository's `UNet` (no
  duplication); the only paper-specific blocks are the Fourier upsampler and
  the stride-2 final conv.
- `utils/losses.py`: `FourierL1Loss`, `CFunetMSEFourierLoss` (with the
  `attach_model` protocol; other losses are unaffected).
- `scripts/interpolation/train_interpolation_unet.py`:
  - New `cfunet_random` mask mode in `_patchify_pairs` — per-patch random
    trace removal with `mask_ratio_range`, seeded by the experiment seed, and
    an observed mask (1 = observed) broadcast to the patch shape. Falls
    through to patch normalisation like the `continuous` branch.
  - `_preprocess_shots` skips shot-level masking for `cfunet_random`.
  - Loss wiring: `if hasattr(loss_fn, "attach_model"): loss_fn.attach_model(unwrap_ddp(model))`.
  - `--mask-mode` accepts `cfunet_random`.
- The `cfunet_random` mask follows the repository's mask semantics
  (1 = observed). Note: the pre-existing `continuous` branch returns a
  missing-mask (True = missing) in its obs_mask slot — a known quirk that is
  NOT changed; `cfunet_random` uses the correct convention.
- Validation ratio: `_patchify_pairs` runs once for the whole volume, so the
  train and validation splits share `mask_ratio_range`. The paper fixes
  validation at 75%; `park2022_cfunet_eval75.yaml` pins
  `mask_ratio_range: [0.75, 0.75]` for paper-style evaluation runs.

## Configs

- `configs/interpolation/park2022_cfunet_paper.yaml` — synthetic profile
  (SEGC3 stand-in), α=1, lr 5e-5, 128×128.
- `configs/interpolation/park2022_cfunet_field_paper.yaml` — field profile
  (Mobil AVO Viking Graben), α=0.1, lr 1e-5, 128×120.
- `configs/interpolation/park2022_cfunet_eval75.yaml` — fixed-75% validation.
- `configs/interpolation/park2022_cfunet_mse.yaml` — α=0 ablation (Eq. 6 only).
- `configs/interpolation/park2022_cfunet_upsampling_ablation.yaml` — bilinear
  upsampling ablation.
- `configs/interpolation/park2022_cfunet_smoke.yaml` — 2-epoch smoke test.
- `configs/interpolation/park2022_baseline_unet.yaml` — plain UNet + MSE
  baseline (paper's "UNet with the MSE" comparison).

## Tests

- `tests/test_park2022_fourier_upsample.py` — constant/sine/Nyquist exactness
  under all three FFT norms, amplitude correction, shapes, odd-size error,
  gradient flow.
- `tests/test_park2022_cfunet_architecture.py` — registration, channel layout
  (22/44/88/176/352), stride-2 final conv, Eq. 2 overwrite, mask inference,
  intermediate shapes.
- `tests/test_park2022_cfunet_loss.py` — Eq. 6/8 arithmetic, complex-vs-
  magnitude distinction, α scaling, attach protocol, gradient flow.
- `tests/test_park2022_cfunet_training.py` — fwd/bwd, single-patch overfit
  with 75% missing, state-dict roundtrip, trainer mask protocol.
- `tests/test_park2022_cfunet_configs.py` — all 7 configs load/build; paper
  hyperparameter values asserted.

## References

- J. Park, J. Shin, S. J. Seol, and J. Byun, "Coarse-Refine Network With
  Upsampling Techniques and Fourier Loss for the Reconstruction of Missing
  Seismic Data," IEEE TGRS, vol. 60, 2022, art. 5917615,
  DOI 10.1109/TGRS.2022.3190292.
