# Ground-Roll Attenuation

Supervised denoising for seismic shot gathers using paired noisy / noise-label SEG-Y volumes.

## Directory layout

```
scripts/ground_roll_attenuation/
    train_denoise_<model>.py          Training entry point per model
    train_denoise_<model>.sh          Launch script with noise-level × seed grid search
    batch_evaluate.py                 Batch inference + metrics workbook (.xlsx)

configs/ground_roll_attenuation/
    denoise_<model>.yaml              One YAML config per model

model/ground_roll_attenuation/
    <model>.py                        Model definition (registered via @register_model)

results/ground_roll_attenuation/
    <exp_name>/                       Output dir per experiment
        checkpoints/                  best.pt + epoch_*.pt
        logs/                         train_log.txt, loss_history.csv, metrics_history.csv, *.png
        visualizations/               input/prediction/target/residual panels
        test_set/                     Unpatchified test shots (.npy)
        config.yaml                   Experiment snapshot
```

## Models

| Script | Config | Model description |
|--------|--------|-------------------|
| `train_denoise_unet.py` | `denoise_unet.yaml` | Standard U-Net (baseline) |
| `train_denoise_atten_unet.py` | `denoise_atten_unet.yaml` | U-Net with CBAM attention gates |
| `train_denoise_res_unet.py` | `denoise_res_unet.yaml` | Residual U-Net |
| `train_denoise_dncnn.py` | `denoise_dncnn.yaml` | DnCNN encoder-decoder |
| `train_denoise_enhanced_unet.py` | `denoise_enhanced_unet.yaml` | Enhanced U-Net with multi-scale blocks |
| `train_denoise_ddpm.py` | `denoise_ddpm.yaml` | Denoising diffusion probabilistic model |
| `train_denoise_physics.py` | `denoise_physics_unet.yaml` | Physics-constrained U-Net |
| `train_denoise_pix2pix.py` | `denoise_pix2pix.yaml` | pix2pix cGAN (PatchGAN discriminator) |
| `train_denoise_sanet.py` | `denoise_sanet.yaml` | Self-attention U-Net |

## Data preparation

### Required format

Two paired SEG-Y files (must have identical geometry — same trace count, sample count, sample interval, and FFID headers):

- **Noisy input** — raw shot gathers containing signal + coherent noise (ground roll)
- **Noise label** — the additive noise component (noisy − clean). This is the supervised target.

The model predicts the noise map; the denoised output is `noisy_input − predicted_noise`.

### File naming convention (used by shell launch scripts)

```
/path/to/noisy/SEGC3_shots1_9_noisy_<noise_level>.sgy
/path/to/noise/SEGC3_shots1_9_noise_<noise_level>.sgy
```

Where `<noise_level>` is a float (1.0, 3.0, 5.0, 7.0, 9.0).

Alternative: NPY or MAT volume pairs. Replace `data.segy_pair` with `data.npy_pair` or `data.mat_pair` in the config.

### Volume shape

`(n_shots, n_traces, n_time)` — e.g. `(9, 201, 1251)` for 9 shots × 201 traces × 1251 time samples.

## Configuration reference

Example: `configs/ground_roll_attenuation/denoise_unet.yaml`

### experiment

```yaml
experiment:
  name: denoise_unet_base          # experiment name (auto-suffixed by .sh scripts)
  output_dir: results/ground_roll_attenuation
  seed: 42
  device: cuda:0                   # single-GPU; ignored under torchrun
```

### data

```yaml
data:
  segy_pair:                       # or npy_pair / mat_pair
    input_path: /path/to/noisy.sgy
    target_path: /path/to/noise_label.sgy
    traces_per_shot: 201
    time_downsample: 1

  shot_split:                      # FFID-level split (no inter-shot leakage)
    train: 7                       # first 7 unique FFIDs
    val: 1                         # next 1
    test: 1                        # last 1

  loader:
    batch_size: 196                # patches per step (not shots)
    num_workers: 4
    pin_memory: true
```

When `shot_split` is omitted, a random train/test split on patches is used instead.

### preprocess

```yaml
preprocess:
  normalize_mode: max_abs          # max_abs | minmax | mean_std
  normalize_scope: global          # global | shot | trace
  clip_percentile: null            # optional float; clips abs value before normalization
  patch_time: 256
  patch_trace: 128
  patch_overlap: 0.5               # 0.0 = no overlap; 0.5 = 50 %
  max_shots: null                  # optional int; limit number of shots
```

Normalization stats are computed on the noisy input; the same scalars are applied to the noise label so the residual relationship is preserved.

### model / loss / metrics / optim / scheduler

Standard registry blocks. See `configs/` for the full schema:

```yaml
model:
  type: unet
  params: { in_channels: 1, out_channels: 1, base_channels: 32, depth: 4 }

loss:
  type: mse                        # mse | l1 | weighted_mse
  params: { reduction: mean }

metrics:
  - name: snr                      # computed on denoised vs reference signal
    params: { reduction: per_sample }
  # ... psnr, ssim, mae, mse, rmse

optim:
  type: adamw
  params: { lr: 1.0e-4, weight_decay: 1.0e-5 }

scheduler:
  type: cosine
  params: { min_lr: 1.0e-6 }

train:
  epochs: 200
  grad_clip: 1.0
  eval_interval: 1
  ckpt_interval: 20
  vis_interval: 5
  resume: null
```

## Running training

### Single run (manual)

```bash
python scripts/ground_roll_attenuation/train_denoise_unet.py \
    --config configs/ground_roll_attenuation/denoise_unet.yaml
```

### Multi-GPU via torchrun

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/ground_roll_attenuation/train_denoise_unet.py \
    --config configs/ground_roll_attenuation/denoise_unet.yaml
```

### Grid search (shell scripts)

Each `.sh` script iterates over multiple noise levels × seeds, auto-generates per-run configs, and runs `torchrun` for each:

```bash
bash scripts/ground_roll_attenuation/train_denoise_unet.sh
```

Edit these variables at the top of the script:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `"0,1,2,3"` | Visible GPUs |
| `NPROC_PER_NODE` | `4` | Must match GPU count |
| `NOISE_LEVELS` | `(1.0 3.0 5.0 7.0 9.0)` | Noise intensities |
| `N_SEEDS` | `3` | Seeds per noise level |
| `START_SEED` | `42` | First seed (others: +1, +2, …) |
| `MASTER_PORT` | `28500` | Base port (incremented per run) |

The script generates experiment names as `<config_name>_level<X.X>_seed<YY>`.

## Batch evaluation

After training, run `batch_evaluate.py` to compute shot-level metrics for all experiments:

```bash
python scripts/ground_roll_attenuation/batch_evaluate.py \
    --root_dir results/ground_roll_attenuation \
    --output results/ground_roll_attenuation/batch_evaluation.xlsx
```

This iterates every experiment directory under `--root_dir`, loads the best checkpoint, runs inference on the held-out test shots, and produces an Excel workbook with per-noise-level sheets comparing:

- **Noisy** metrics (raw input vs reference signal)
- **Denoised** metrics (model output vs reference signal)

Metrics: SNR (dB), PSNR (dB), SSIM, MAE, MSE, RMSE.

## Adding a new model

1. Create `model/ground_roll_attenuation/<name>.py` with a `@register_model("<name>")` class.
2. Add `from . import <name>  # noqa: F401` to `model/ground_roll_attenuation/__init__.py`.
3. Copy `scripts/ground_roll_attenuation/train_denoise_unet.py` to `train_denoise_<name>.py`.
4. Copy `configs/ground_roll_attenuation/denoise_unet.yaml` to `denoise_<name>.yaml` and update `model.type` + `model.params`.
5. (Optional) Copy a `.sh` launch script and update `BASE_CONFIG` / `PY_SCRIPT`.

No edits to `utils/` or `scripts/train.py` are required.
