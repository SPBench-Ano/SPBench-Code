---
tags:
- seismic
- random-noise
- denoising
- unet
- resunet
- dncnn
- attention-unet
- scrn
- pytorch
library_name: pytorch
---

# Random Noise Suppression Benchmark

Deep-learning-based random-noise attenuation on pre-stack seismic shot gathers, using SEG C3 synthetic data and synthetic Gaussian / Poisson noise injection.

## Task

Given a clean shot gather, the benchmark first injects synthetic random noise at a specified SNR, then trains a model to directly reconstruct the clean signal:

```python
denoised = model(noisy_input)
```

This is a **paired regression** task with clean-target supervision. The loss is MSE between the predicted clean gather and the clean reference.

## Dataset

- **Source**: SEG C3 pre-stack synthetic data
- **Current training volume**: `SEG_45Shot_shots1-9.sgy`
- **Geometry**: 201 traces per shot, time sampling interval `dt = 2 ms`
- **Split**: Shot-level sequential `7:1:1`
  - 7 training shots
  - 1 validation shot
  - 1 held-out test shot

### Synthetic Noise Settings

- **Noise kinds**: `gaussian`, `poisson`
- **Default sweep in training / inference scripts**: `SNR = -5, 0, 5 dB`
- **Noise injection**: per-shot variance-controlled synthetic corruption
- **Reproducibility**: noise generation is seeded from `experiment.seed`

## Model Architectures

- **UNet** (`unet`): classic encoder-decoder with skip connections. Base channels: 32, depth: 4.
- **ResUNet** (`res_unet`): U-Net with residual blocks. Base channels: 32, depth: 4.
- **DnCNN** (`dncnn`): residual denoising CNN with 17 layers and 64 feature channels.
- **Attention UNet** (`atten_unet`): U-Net with attention gates. Base channels: 32, depth: 4.
- **SCRN** (`SCRN`): Swin Transformer convolutional residual network adapted to the same random-noise benchmark pipeline.

## Preprocessing

- **Amplitude correction**: spherical divergence correction is skipped by default
- **Normalization**: `max_abs`, per-shot
- **Patching**: overlapping 2D patches of size `128 x 256` (`trace x time`)
- **Patch overlap**: `50%`

Training uses patched shot gathers. Inference reloads the raw volume, applies `inference.shot_split`, injects synthetic noise, runs patch-based reconstruction, and inverse-normalizes outputs for visualization.

## Repository Structure

```text
scripts/random_noise_suppression/
├── train_denoise_unet.sh
├── train_denoise_res_unet.sh
├── train_denoise_dncnn.sh
├── train_denoise_atten_unet.sh
├── train_denoise_SCRN.sh
├── inference_denoise_unet.sh
├── inference_denoise_res_unet.sh
├── inference_denoise_dncnn.sh
├── inference_denoise_atten_unet.sh
├── inference_denoise_SCRN.sh
└── run_all_random_noise_models.sh

configs/random_noise_suppression/
├── denoise_unet.yaml
├── denoise_res_unet.yaml
├── denoise_dncnn.yaml
├── denoise_atten_unet.yaml
└── denoise_SCRN.yaml
```

Each experiment directory is named by model, noise kind, SNR, and seed, for example:

```text
random_noise_unet_base_gaussian_snr5_seed42/
random_noise_dncnn_base_poisson_snr0_seed43/
random_noise_SCRN_base_gaussian_snrneg5_seed44/
```

## Training Details

Shared benchmark defaults:

| Hyperparameter | Value |
|---|---|
| Loss | MSE |
| Optimizer | AdamW (`lr=1e-4`, `weight_decay=1e-5`) |
| Scheduler | Cosine annealing (`min_lr=1e-6`) |
| Epochs | 200 |
| Gradient clipping | 1.0 |
| Seeds | 42, 43, 44 by default in shell sweeps |
| Batch size | 192 in current YAML defaults |

## Usage

### Train One Model Family

```bash
bash scripts/random_noise_suppression/train_denoise_unet.sh
```

or

```bash
bash scripts/random_noise_suppression/train_denoise_SCRN.sh
```

Each training shell script sweeps:

- noise kind
- SNR
- seed

by rewriting a temporary YAML config before calling `torchrun`.

### Run Inference

```bash
bash scripts/random_noise_suppression/inference_denoise_unet.sh
```

Inference outputs:

- per-shot metrics CSV
- summary JSON
- visualizations
- optional `.npy` files
- multi-seed mean/std aggregation JSON

### Run All Model Families

```bash
bash scripts/random_noise_suppression/run_all_random_noise_models.sh
```

Current total-run script executes:

1. `unet`
2. `dncnn`
3. `res_unet`
4. `atten_unet`

Each model is trained first, then its inference sweep is launched immediately after training finishes.

## Inference Outputs

For each experiment, the inference directory typically contains:

```text
inference/
├── inference.log
├── metrics_per_shot.csv
├── metrics_summary.json
├── visualizations/
└── npy/                    # only when save_npy=true
```

### Metrics

The benchmark reports:

- `snr`
- `psnr`
- `ssim`
- `mae`
- `mse`
- `rmse`

for three groups:

- **noisy**: noisy input vs clean target
- **denoised**: model prediction vs clean target
- **delta**: `denoised - noisy`

Metrics are computed in the **normalized domain**. Saved visualization outputs are inverse-normalized back to the original amplitude domain.

## Notes

- This benchmark injects **synthetic random noise once per experiment** before patch extraction; it is not an epoch-wise dynamic noise augmentation setup.
- Shot-level inference uses the held-out test shot defined by `inference.shot_split`.
- Batch size and patch size may need adjustment for memory-heavy models such as DnCNN.

## References

- Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015
- He et al., Deep Residual Learning for Image Recognition, CVPR 2016
- Zhang et al., Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising, IEEE TIP 2017
- Oktay et al., Attention U-Net: Learning Where to Look for the Pancreas, MIDL 2018
- Gao et al., Swin Transformer for simultaneous denoising and interpolation of seismic data, Computers and Geosciences 2024
- SEG C3 Velocity Model: https://wiki.seg.org/wiki/C3
