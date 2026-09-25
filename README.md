# Benchmark Code - Exploration Geophysics Data Processing Benchmark Template

A PyTorch benchmark template for seismic data processing, supporting interpolation, denoising, and supervised restoration on SEG-Y / NPY / MAT volumes.

Features:
- **Registry + factory pattern** for models, losses, metrics, and datasets — add new components via YAML without editing training scripts.
- **Preprocessing** — spherical divergence correction, normalization (minmax / max_abs / mean_std, per shot / trace / global), trace masking, noise injection.
- **Patchify / unpatchify** for 2D conv models with overlapping grids.
- **Training** — single-GPU and DDP multi-GPU via `torchrun`.
- **Shot-level splitting** — split train/val/test by unique FFID (sequential order) instead of random patch-level split, preventing data leakage.
- **Best-validation checkpoint** — automatically retains `best.pt` (lowest val loss) independently of the periodic `epoch_*.pt` snapshots.
- **Unified visualization** — consistent symmetric color scale across input, prediction, target, and residual panels.
- **Inference** — full-shot reconstruction, per-shot metrics (MSE, RMSE, MAE, SNR, PSNR, SSIM), visualization, and optional `.npy` export.

---

## Directory Overview

- **tools/** - Data utilities: I/O (`array_io.py`, `segy_read.py`), preprocessing (`preprocessing.py`), patching (`patching.py`).
- **model/** - `nn.Module` definitions and `MODEL_REGISTRY`. Task subpackages register their own model implementations.
- **utils/** - Training infrastructure: datasets, losses, metrics, visualization, logging, optimizer/scheduler builders, train/eval loops, DDP helpers, checkpoint I/O.
- **configs/** - One YAML file per experiment; hyper-parameters are never hard-coded in source.
- **scripts/** - CLI entry points for training (`train_*.py`) and inference (`inference_*.py`), plus bash launchers (`*.sh`).
- **results/** - Experiment outputs (checkpoints, logs, CSVs, PNGs). Not tracked by Git.
- **.cursor/rules/** - Agent rule files (`*.mdc`).

---

## Quick Start

### Training

```bash
# Interpolation (single volume + trace masking)
python scripts/interpolation/train_interpolation_unet.py --config configs/interpolation/interpolation_unet.yaml

# Paired supervised training (input + target volumes)
python scripts/interpolation/train_paired_unet.py --config configs/interpolation/paired_unet.yaml

# Ground-roll attenuation (paired noisy / noise-label volumes)
python scripts/ground_roll_attenuation/train_denoise_res_unet.py --config configs/ground_roll_attenuation/denoise_res_unet.yaml

# Multiples attenuation (paired noisy / noise-label volumes)
python scripts/multiples_attenuation/train_denoise_res_unet.py --config configs/multiples_attenuation/denoise_res_unet.yaml

# DDP multi-GPU
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/interpolation/train_interpolation_unet.py --config configs/interpolation/interpolation_unet.yaml
```

### Inference

```bash
python scripts/interpolation/inference_interpolation.py \
  --config configs/interpolation/interpolation_unet.yaml \
  --checkpoint results/<exp_name>/checkpoints/epoch_0049.pt
```

See `使用说明.md` (Chinese) for detailed usage, config schema, and FAQ.

---

## Rules (see `.cursor/rules/`)

2. **no-duplication** — No duplicated logic; consolidate into `utils/` or `tools/`; check existing code before adding.
3. **efficiency-first** — Avoid redundant loops, I/O, copies, and needless CPU↔GPU / dtype moves.
4. **research-first** — Prefer mature open-source implementations over from-scratch code; cite sources in the relevant README.
5. **clarify-before-execute** — Ask when uncertain; present a plan and a diff summary, and wait for confirmation before editing.
6. **no-auto-run** — The agent never runs scripts or training commands; only tells the user how to run them.
7. **english-only** — All project docs, comments, log strings, and commit messages are written in English. Agent↔user chat is not restricted.
8. **concise-docs** — Docstrings stay at one-sentence functionality + `Parameters` + `Returns`; design rationale and "how to add" tutorials live in READMEs.

---

## Contribution Guide (minimal)

- **No duplicated code.** Before adding a feature, search `utils/` and `tools/` for an equivalent implementation.
- **Reuse first.** Shared logic goes into `utils/` or `tools/`; entry points into `scripts/`; model definitions into `model/`; configs into `configs/`.
- **Open source first.** When a mature open-source implementation exists (timm, HuggingFace, Meta official repos, etc.), use it as the baseline and cite the source.
- **Do not auto-run.** The agent must not execute `python` / `pip` / `torchrun`; the user runs scripts manually.
- **English only.** All project docs, comments, and commit messages must be written in English.
