# Ground-Roll Attenuation Benchmark

## Task Overview

Suppress coherent ground-roll noise from pre-stack seismic shot gathers using four deep-learning architectures: **UNet**, **ResUNet**, **DnCNN**, and **Attention UNet**. The 9-shot SEG-C3 dataset is synthetic, generated via forward modeling on the official SEG C3 velocity model — reflection signals are modeled with the acoustic wave equation, while ground-roll noise is modeled with the elastic wave equation to capture its dispersive, low-velocity character. Given a noisy shot gather, the model predicts the additive noise component; the denoised signal is obtained by subtracting the prediction from the input. This is a **paired regression** task trained with a noise-label objective (the ground-truth noise component).

Supervision signal: `denoised = noisy_input - predicted_noise`, evaluated against `clean_reference = noisy_input - label_noise`.

## Dataset

### Source

SEG-C3 pre-stack data, 9 regular shot gathers (`shots1-9`), stored in SEG-Y format.

### Geometry

| Property | Value |
|----------|-------|
| Traces per shot | 201 |
| Time samples | 625 |
| Sampling interval (dt) | 8 ms |
| Shot shape | `(n_shots=9, n_traces=201, n_time=625)` |

### Noise Intensity Levels

Ground-roll noise at five intensity levels injected into the clean signal, producing paired `(noisy, noise_label)` shot gathers:

| Noise level | Noisy file | Noise-label file | SNR (dB) | PSNR (dB) | SSIM | MAE | MSE | RMSE |
|-------------|-----------|-----------------|----------|-----------|------|-----|-----|------|
| 1.0 | `SEGC3_shots1_9_noisy_1.0.sgy` | `SEGC3_shots1_9_noise_1.0.sgy` | 2.71 | 15.81 | 0.9480 | 0.030624 | 0.026232 | 0.161964 |
| 3.0 | `SEGC3_shots1_9_noisy_3.0.sgy` | `SEGC3_shots1_9_noise_3.0.sgy` | -6.83 | 6.27 | 0.9418 | 0.091871 | 0.236091 | 0.485892 |
| 5.0 | `SEGC3_shots1_9_noisy_5.0.sgy` | `SEGC3_shots1_9_noise_5.0.sgy` | -11.27 | 1.83 | 0.9402 | 0.153118 | 0.655809 | 0.809820 |
| 7.0 | `SEGC3_shots1_9_noisy_7.0.sgy` | `SEGC3_shots1_9_noise_7.0.sgy` | -14.19 | -1.09 | 0.9395 | 0.214366 | 1.285386 | 1.133749 |
| 9.0 | `SEGC3_shots1_9_noisy_9.0.sgy` | `SEGC3_shots1_9_noise_9.0.sgy` | -16.37 | -3.27 | 0.9390 | 0.275613 | 2.124822 | 1.457677 |

Initial metrics computed on the full 9-shot dataset in the original amplitude domain.

## Data Preprocessing

### Normalization

- **Mode**: `max_abs` — each dataset scaled to `[-1, 1]` by dividing by `max(|x|)`.
- **Scope**: `global` — statistics are computed once over the full noisy dataset; the same scalars are applied to the noise-label target so both stay in a consistent amplitude range.
- **Clipping**: None (optional percentile clipping available but not active).

### Patchify

Overlapping 2D patches of size `(trace=128, time=256)` extracted with a uniform sliding grid, 50% overlap. Each shot gather produces many patches; all 9 shots are patchified and pooled.

Output shape per patch: `(1, 128, 256)` — channel-last for 2D ConvNet consumption.

### Train / Validation / Test Split

**Shot-level (FFID) sequential split** — 7 : 1 : 1.

| Split | FFIDs | Shots | Purpose |
|-------|-------|-------|---------|
| Train | 1–7 | 7 | Model training |
| Validation | 8 | 1 | Early stopping, best-checkpoint selection |
| Test | 9 | 1 | Held-out final evaluation |

The split boundary is at the shot (FFID) level, so no patches from the same shot appear in multiple splits. The raw test-set shots are saved as `.npy` alongside the checkpoint for downstream inference.

## Models

Four architectures are evaluated. All models operate on `(1, H, W)` single-channel 2D patches. The output is the predicted noise residual of the same shape.

| Model | Registry Key | Base Channels | Depth | Key Parameters | Reference |
|-------|-------------|--------------|-------|----------------|-----------|
| **UNet** | `unet` | 32 | 4 | DoubleConv encoder/decoder, MaxPool ×2 down, ConvTranspose ×2 up, skip connections | Ronneberger et al., 2015 |
| **ResUNet** | `res_unet` | 32 | 4 | Same U-Net topology; residual blocks (`Conv→BN→ReLU→Conv→BN + shortcut`) replace plain double-conv blocks | He et al., 2016 (residual block); Zhang et al., 2018 (ResUNet) |
| **DnCNN** | `dncnn` | 64 | 17 | Flat Conv→BN→ReLU stack, residual output `x - F(x)`, orthogonal weight init | Zhang et al., 2017 (IEEE TIP) |
| **Attention UNet** | `atten_unet` | 32 | 4 | U-Net + additive attention gates (`W_g·g + W_x·x → sigmoid → gate`) on skip connections | Oktay et al., 2018 (MIDL) |

### Architecture Details

**UNet** — Classic encoder-decoder. Each stage is two `Conv2d(3×3)→BN→ReLU` blocks. Downsampling: `MaxPool2d(2×2)`. Upsampling: `ConvTranspose2d(2×2)`. Skip connections concatenate encoder features onto decoder inputs. Final layer: `Conv2d(1×1)`.

**ResUNet** — Identical U-Net skeleton with `_DoubleConv` blocks replaced by residual blocks that add an identity (or 1×1-projected) shortcut before the final ReLU activation. Improves gradient flow through deeper encoder/decoder stages.

**DnCNN** — No encoder-decoder; a flat stack of 17 convolutional layers with BatchNorm and ReLU. The first layer uses bias, subsequent layers are bias-free following the original paper. The model learns the noise residual: `output = x - net(x)`. Deeper than the U-Nets but has no spatial downsampling, preserving full resolution throughout.

**Attention UNet** — Same encoder-decoder as UNet, with additive attention gates inserted between each upsampled feature map and its corresponding skip connection. The gate computes: `α = σ(ψ(ReLU(W_g·g + W_x·x)))` and gates the skip: `skip' = skip ⊙ α`. The gating bottleneck uses `F_int = max(c // 2, 8)` channels.

## Training Configuration

### Loss Function

Mean Squared Error (MSE) between predicted noise and label noise:

`L = (1/N) Σ (pred_noise - label_noise)²`

### Optimizer & Scheduler

| Component | Type | Parameters |
|-----------|------|------------|
| Optimizer | AdamW | `lr=1e-3`, `weight_decay=1e-4` |
| Scheduler | Cosine annealing | `min_lr=1e-6`, over full epoch budget |

### Per-Model Batch Size

| Model | Batch Size | Notes |
|-------|-----------|-------|
| UNet | 196 | Lightweight; fits more patches per GPU step |
| ResUNet | 196 | Residual blocks add memory overhead |
| DnCNN | 196 | 17-layer deep stack; largest memory footprint |
| Attention UNet | 196 | Similar overhead to baseline UNet despite attention gates |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Gradient clipping | 1.0 (max norm) |
| Evaluation interval | Every epoch |
| Checkpoint interval | Every 20 epochs |
| Visualization interval | Every 5 epochs |
| Best checkpoint | Minimum validation loss |

### Reproducibility

- **Seeds**: 3 independent runs per model per noise level (seed = 42, 43, 44).
- **RNG**: Python `random`, NumPy, and PyTorch RNGs are all seeded from `experiment.seed`.
- **Data split**: Deterministic by FFID ordering (same split for all seeds and noise levels).
- **Distributed training**: 2 × NVIDIA RTX 4090 (24 GB), `torchrun` DDP, one process per GPU.
- **Output directory**: `<output_dir>/<experiment_name>_level<L>_seed<S>/` — isolated per run.

## Evaluation Metrics

All metrics are computed in the **normalized domain** on the denoised signal estimate (`input - pred_noise`) vs. the reference signal (`input - label_noise`):

| Metric | Reduction | Notes |
|--------|-----------|-------|
| **SNR** | per-sample mean | Signal-to-Noise Ratio; `+∞` when noise-free |
| **PSNR** | per-sample mean | Peak `data_range = 1.0` (max-abs normalized, peak amplitude = 1) |
| **SSIM** | N/A (uniform weight) | `data_range = 2.0` (signal spans `[-1, 1]`), `window_size = 11`, `σ = 1.5` |
| **MAE** | mean | Mean Absolute Error |
| **MSE** | mean | Mean Squared Error |
| **RMSE** | per-sample mean | Root Mean Squared Error |

For SNR and PSNR, `per_sample` reduction computes the metric independently for each patch (or shot) then averages them, matching the per-shot evaluation convention in seismic processing.

## Results

### Summary

**SNR (dB)** across noise levels (mean ± std over 3 seeds; higher is better):

| Model | Params (M) | Level 1.0 | Level 3.0 | Level 5.0 | Level 7.0 | Level 9.0 |
|-------|:----------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Raw (noisy) | — | 2.71±0.00 | −6.83±0.00 | −11.27±0.00 | −14.19±0.00 | −16.37±0.00 |
| UNet | 7.76 | 28.39±1.09 | 23.07±0.10 | 20.22±0.32 | 17.90±0.28 | 16.60±0.46 |
| ResUNet | 8.11 | 31.62±0.56 | 22.65±0.93 | 18.11±2.12 | 16.60±1.44 | 14.61±0.01 |
| DnCNN | 0.56 | 31.67±0.11 | 30.02±0.43 | 22.91±3.74 | — | — |
| Attention UNet | 7.85 | 28.79±0.46 | 23.10±1.00 | 19.66±0.22 | 17.57±0.11 | 16.61±0.19 |

### Detailed Results by Noise Level

All metrics reported as mean ± std over 3 independent runs (seeds 42, 43, 44). DnCNN at levels 7.0 and 9.0 has no available data.

#### Noise Level 1.0

| Method | Params (M) | SNR (dB) | PSNR (dB) | SSIM | MAE | MSE | RMSE |
|--------|:----------:|:--------:|:---------:|:----:|:---:|:---:|:----:|
| Raw (noisy) | — | 2.71±0.00 | 21.83±0.00 | 0.95±0.00 | 0.020000±0.00 | 0.006558±0.00 | 0.080000±0.00 |
| UNet | 7.76 | 28.39±1.09 | 47.51±1.09 | 1.00±0.00 | 0.000000±0.00 | 0.000018±0.00 | 0.000000±0.00 |
| ResUNet | 8.11 | 31.62±0.56 | 50.74±0.57 | 1.00±0.00 | 0.000000±0.00 | 0.000008±0.00 | 0.000000±0.00 |
| DnCNN | 0.56 | 31.67±0.11 | 50.79±0.11 | 1.00±0.00 | 0.000000±0.00 | 0.000008±0.00 | 0.000000±0.00 |
| Attention UNet | 7.85 | 28.79±0.46 | 47.91±0.46 | 1.00±0.00 | 0.000000±0.00 | 0.000016±0.00 | 0.000000±0.00 |

#### Noise Level 3.0

| Method | Params (M) | SNR (dB) | PSNR (dB) | SSIM | MAE | MSE | RMSE |
|--------|:----------:|:--------:|:---------:|:----:|:---:|:---:|:----:|
| Raw (noisy) | — | −6.83±0.00 | 18.31±0.00 | 0.95±0.00 | 0.020000±0.00 | 0.014756±0.00 | 0.120000±0.00 |
| UNet | 7.76 | 23.07±0.10 | 48.21±0.10 | 1.00±0.00 | 0.000000±0.00 | 0.000015±0.00 | 0.000000±0.00 |
| ResUNet | 8.11 | 22.65±0.93 | 47.79±0.93 | 0.99±0.00 | 0.000000±0.00 | 0.000017±0.00 | 0.000000±0.00 |
| DnCNN | 0.56 | 30.02±0.43 | 55.16±0.43 | 1.00±0.00 | 0.000000±0.00 | 0.000003±0.00 | 0.000000±0.00 |
| Attention UNet | 7.85 | 23.10±1.00 | 48.24±1.00 | 1.00±0.00 | 0.000000±0.00 | 0.000015±0.00 | 0.000000±0.00 |

#### Noise Level 5.0

| Method | Params (M) | SNR (dB) | PSNR (dB) | SSIM | MAE | MSE | RMSE |
|--------|:----------:|:--------:|:---------:|:----:|:---:|:---:|:----:|
| Raw (noisy) | — | −11.27±0.00 | 17.40±0.00 | 0.95±0.00 | 0.030000±0.00 | 0.018217±0.00 | 0.130000±0.00 |
| UNet | 7.76 | 20.22±0.32 | 48.88±0.32 | 1.00±0.00 | 0.000000±0.00 | 0.000013±0.00 | 0.000000±0.00 |
| ResUNet | 8.11 | 18.11±2.12 | 46.77±2.12 | 0.99±0.00 | 0.000000±0.00 | 0.000023±0.00 | 0.003333±0.01 |
| DnCNN | 0.56 | 22.91±3.74 | 51.58±3.74 | 0.99±0.01 | 0.000000±0.00 | 0.000008±0.00 | 0.000000±0.00 |
| Attention UNet | 7.85 | 19.66±0.22 | 48.32±0.22 | 1.00±0.00 | 0.000000±0.00 | 0.000015±0.00 | 0.000000±0.00 |

#### Noise Level 7.0

| Method | Params (M) | SNR (dB) | PSNR (dB) | SSIM | MAE | MSE | RMSE |
|--------|:----------:|:--------:|:---------:|:----:|:---:|:---:|:----:|
| Raw (noisy) | — | −14.19±0.00 | 16.97±0.00 | 0.95±0.00 | 0.030000±0.00 | 0.020084±0.00 | 0.140000±0.00 |
| UNet | 7.76 | 17.90±0.28 | 49.06±0.28 | 1.00±0.00 | 0.000000±0.00 | 0.000012±0.00 | 0.000000±0.00 |
| ResUNet | 8.11 | 16.60±1.44 | 47.76±1.44 | 0.99±0.00 | 0.000000±0.00 | 0.000017±0.00 | 0.000000±0.00 |
| DnCNN | 0.56 | — | — | — | — | — | — |
| Attention UNet | 7.85 | 17.57±0.11 | 48.73±0.11 | 1.00±0.00 | 0.000000±0.00 | 0.000013±0.00 | 0.000000±0.00 |

#### Noise Level 9.0

| Method | Params (M) | SNR (dB) | PSNR (dB) | SSIM | MAE | MSE | RMSE |
|--------|:----------:|:--------:|:---------:|:----:|:---:|:---:|:----:|
| Raw (noisy) | — | −16.37±0.00 | 16.73±0.00 | 0.95±0.00 | 0.030000±0.00 | 0.021248±0.00 | 0.150000±0.00 |
| UNet | 7.76 | 16.60±0.46 | 49.70±0.46 | 1.00±0.00 | 0.000000±0.00 | 0.000011±0.00 | 0.000000±0.00 |
| ResUNet | 8.11 | 14.61±0.01 | 47.72±0.01 | 0.99±0.00 | 0.000000±0.00 | 0.000017±0.00 | 0.000000±0.00 |
| DnCNN | 0.56 | — | — | — | — | — | — |
| Attention UNet | 7.85 | 16.61±0.19 | 49.71±0.19 | 1.00±0.00 | 0.000000±0.00 | 0.000011±0.00 | 0.000000±0.00 |

### Key Observations

- All four deep-learning architectures achieve substantial SNR improvement over the raw noisy input across all noise levels (15–31 dB gain).
- **DnCNN** delivers the best overall performance at low-to-mid noise levels (1.0–5.0) with the smallest model footprint (0.56 M parameters), but lacks results at the highest noise levels (7.0, 9.0).
- **ResUNet** and **Attention UNet** show comparable performance to the baseline UNet, with ResUNet slightly ahead at level 1.0 and the baseline UNet marginally better at higher noise levels.
- SSIM saturates near 1.0 for all methods, indicating strong structural preservation in the denoised output.

## Quick Start

All four models use the same nested-loop launcher pattern: edit the config block at the top of each `.sh` to select noise levels, seeds, GPUs, and master port, then run a single command.

```bash
# UNet — full grid sweep (5 noise levels × 3 seeds)
bash scripts/ground_roll_attenuation/train_denoise_unet.sh

# ResUNet — full grid sweep
bash scripts/ground_roll_attenuation/train_denoise_res_unet.sh

# DnCNN — full grid sweep
bash scripts/ground_roll_attenuation/train_denoise_dncnn.sh

# Attention UNet — full grid sweep
bash scripts/ground_roll_attenuation/train_denoise_atten_unet.sh
```

Or launch a single model / noise level / seed directly via `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
    scripts/ground_roll_attenuation/train_denoise_unet.py \
    --config configs/ground_roll_attenuation/denoise_unet.yaml
```

Edit the configuration block at the top of each `.sh` script to select noise levels (`NOISE_LEVELS`), seed count (`N_SEEDS`), starting seed (`START_SEED`), GPUs (`CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`), and base port (`MASTER_PORT`).
