# scripts/

## Purpose

Entry-point scripts for training, evaluation, and data preparation. Scripts only parse CLI, load configs, and wire up components from `model/` and `utils/`; they do not implement algorithms.

## Planned contents

- `train.py` — Training entry, supports single-GPU and `torchrun` DDP.
- `eval.py` — Evaluation entry, loads a checkpoint and runs the validation set.
- `prepare_data.py` — Preprocessing entry (raw data → benchmark HDF5 format).
- Optional: `sweep.py`, `profile.py`, etc.

## Constraints

- Scripts must **not** define model structures or loss functions; everything is imported from `model/` and `utils/`.
- Each script must accept `--config <yaml>` and dump the effective final config into the experiment output directory.
- The agent must not run these scripts; the user executes them manually.

## How to run the training script

Once data paths are filled into the config and the placeholder function bodies are implemented, run:

```bash
# single GPU
python scripts/train.py --config configs/default.yaml

# resume from a specific checkpoint
python scripts/train.py --config configs/default.yaml --resume results/<exp>/checkpoints/epoch_0010.pt
```

Expected outputs under `results/<experiment.name>/`:

- `checkpoints/epoch_*.pt`
- `logs/train_log.txt`, `logs/loss_history.csv`, `logs/metrics_history.csv`
- `logs/loss_curve.png`, `logs/metrics_curve.png`
- `visualizations/epoch_*.png`
- `config_used.yaml`

## How to add a new entry script

1. Create `scripts/<new_entry>.py` that:
   - Parses CLI args (at minimum `--config`).
   - Calls `utils.load_config` and the appropriate `build_*` factories.
   - Contains **no** algorithmic logic of its own.
2. Document the command and expected outputs in this README.
3. Keep the script idempotent: safe to re-run against the same `--config`.
