# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch benchmark template for exploration geophysics (seismic) data processing. The codebase uses a **registry + factory** pattern so that models, datasets, losses, and metrics can be added as plugins without touching the training script.

## Common Commands

Training entry points live in `scripts/`:

```bash
# Generic component-agnostic training (uses registry-based YAML config)
python scripts/train.py --config configs/default.yaml

# SEG-Y interpolation baseline (UNet, uniform trace-masking, patchify)
python scripts/train_interpolation_unet.py --config configs/interpolation_unet.yaml

# Resume from checkpoint
python scripts/train.py --config configs/<exp>.yaml --resume results/<exp>/checkpoints/epoch_0010.pt
```

**Do not auto-run scripts.** Per project rules, only provide the command in plain text; the user executes manually.

There is no centralized dependency file (requirements.txt / pyproject.toml) yet; required packages include `torch`, `numpy`, `matplotlib`, `pyyaml`, `segyio`, and `scipy`.

## High-Level Architecture

### Registry + Factory Pattern

Every pluggable component is declared in YAML as `{ type: <registered_name>, params: {...} }` and instantiated by a generic `build_*` factory:

- **Models** — `model/registry.py`: `MODEL_REGISTRY`, `@register_model`, `build_model`
- **Datasets** — `utils/datasets.py`: `DATASET_REGISTRY`, `@register_dataset`, `build_dataset`, `build_dataloader`
- **Losses** — `utils/losses.py`: `LOSS_REGISTRY`, `@register_loss`, `build_loss`
- **Metrics** — `utils/metrics.py`: `METRIC_REGISTRY`, `@register_metric`, `build_metrics`

To add a new component:

1. Subclass the corresponding base class (`BaseArrayDataset`, `BaseLoss`, `BaseMetric`, or `nn.Module`).
2. Decorate it with `@register_<kind>("name")`.
3. For models, add `from . import <file>  # noqa: F401` to `model/__init__.py` so the decorator runs at import time.
4. Reference it from YAML. No edits to `scripts/train.py` or `utils/__init__.py` are needed.

`scripts/train.py` is intentionally **component-agnostic** — it only parses CLI args, loads the YAML config, and wires up factories. It must never import a concrete model, dataset, loss, or metric directly.

### Module Boundaries

- `model/` — `nn.Module` subclasses and factory functions only; no training loops.
- `utils/` — Training infrastructure: datasets, losses, metrics, visualization, logging, optimizer/scheduler builders, train/eval loops, checkpoint I/O.
- `tools/` — Generic data helpers (SEG-Y I/O, preprocessing, patchify/unpatchify). No training pipelines or model definitions.
- `scripts/` — CLI entry points that parse args and wire up `model/` + `utils/`.
- `configs/` — One YAML file per experiment; hyper-parameters must never be hard-coded in source.
- `results/` — Experiment outputs (checkpoints, logs, CSVs, PNGs). Gitignored except for `results/README.md`.

### Key Data Flows

**SEG-Y reader:** `tools/segy_read.py` uses `segyio` to read pre-stack SEG-Y files into shot gathers of shape `(n_shots, n_traces, n_time)`. `read_regular_shots(path, traces_per_shot, time_downsample=1)` handles regular shot gathers; `read_irregular_shots_by_header(...)` is a placeholder for header-based splitting.

**Preprocessing:** `tools/preprocessing.py` provides pure-numpy, vectorized primitives on `(n_shots, n_traces, n_time)`:
- `add_noise(..., kind="gaussian"|"poisson", snr_db, rng)`
- `mask_traces(..., mode="uniform"|"random"|"continuous", ratio, uniform_stride=None, rng)` — `"uniform"` keeps every `uniform_stride`-th trace.
- `spherical_divergence_correction(shots, dt, t0=0.0, power=2.0)`
- `normalize(shots, mode="minmax"|"max_abs"|"mean_std", per="shot"|"trace"|"global", override_stats=None)` — `override_stats` applies pre-computed scalars at inference.

**Patching:** `tools/patching.py` operates on `(trace, time)`:
- `patchify_uniform(data, patch_size=(trace, time), overlap=0.0, output_ndim=3|4)` — overlapping grid; returns `(P, h, w)` or `(P, 1, h, w)`.
- `patchify_random(...)` — per-shot random sampling; not invertible.
- `unpatchify_uniform(patches, info)` — inverse of `patchify_uniform`; averages overlaps (`sum / count`).

**Metrics:** `utils/metrics.py` implements `mse`, `rmse`, `mae`, `snr`, `psnr`, `ssim`. `rmse` / `snr` / `psnr` accept `reduction="per_sample"` (default; mean of per-sample scores) or `"global"` (preserves textbook identities like `RMSE == sqrt(MSE)`).

**Logging:** `utils/logger.py` (`TrainingLogger`) writes:
- `train_log.txt` — timestamped human-readable lines.
- `loss_history.csv` — columns `epoch, lr, <loss_keys>`.
- `metrics_history.csv` — columns `epoch, <metric_keys>`.
- `loss_curve.png` / `metrics_curve.png` — auto-refreshed every `plot_interval` epochs (0 disables). Resumed runs rehydrate history from existing CSVs so curves stay continuous.

### Config Schema

Every pluggable block follows the same shape:

```yaml
model:
  type: unet
  params: { in_channels: 1, out_channels: 1, base_channels: 32, depth: 4 }

loss:
  type: mse
  params: { reduction: mean }

metrics:
  - name: snr
    params: { reduction: per_sample }

optim:
  type: adamw
  params: { lr: 1.0e-3, weight_decay: 1.0e-4 }

scheduler:
  type: cosine
  params: { min_lr: 1.0e-6 }
```

New optional config fields must be read in the corresponding builder with a sensible default so older configs still work.

## Project Rules (from `.cursor/rules/`)

These rules are enforced for all edits:

1. **memory-first** — Read all files under `memory/` (`code_design.md`, `techniques.md`, `updates.md`, `research_first.md`) before any code change. Append an entry to `memory/updates.md` after important changes.
2. **no-duplication** — Search `utils/` and `tools/` before adding new logic; consolidate reusable code rather than copy-pasting.
3. **efficiency-first** — Avoid Python-level loops when a vectorized `torch`/`numpy` op works; avoid redundant I/O, implicit tensor copies, and unnecessary CPU↔GPU or dtype moves on the hot path.
4. **research-first** — Prefer mature open-source implementations (timm, HuggingFace, Meta official repos) over from-scratch code; cite the source in file headers and `memory/updates.md`.
5. **clarify-before-execute** — If a requirement is ambiguous, ask the user first. Present a plan (files touched, add/modify/remove, key design points) and a diff summary, then wait for explicit confirmation before editing.
6. **no-auto-run** — Never execute `python`, `pip`, `torchrun`, `pytest`, or training scripts autonomously. Only provide the command, expected output, and prerequisites in plain text.
7. **english-only** — All project docs, READMEs, source comments, log strings, CLI help, exception messages, config comments, and commit messages must be written in English.
8. **concise-docs** — Docstrings are one sentence + `Parameters` + `Returns` only. No design rationale, history, or "how to add" tutorials in docstrings (those belong in READMEs or `memory/`).

## Memory Files

Always read `memory/` before editing:

- `memory/code_design.md` — Architectural invariants, module boundaries, and the maintenance guide (checklist before merging).
- `memory/techniques.md` — Landed algorithms and training tricks.
- `memory/updates.md` — Chronological log of important changes (files added/removed, API changes, dependency upgrades, open-source references).
- `memory/research_first.md` — Open-source survey notes and alignment decisions.
