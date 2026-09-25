# configs/

## Purpose

Training / evaluation parameter configs. One file per experiment. Hyper-parameters must live here, **never** hard-coded in source.

## Planned contents

- `default.yaml` — Default template; copy and edit it to create a new experiment.
- Task- or model-specific configs, e.g. `mae_pretrain.yaml`, `velocity_inversion.yaml`, `denoise_unet.yaml`.
- Suggested sections per file: `data` / `model` / `optim` / `scheduler` / `train` / `eval` / `log`.

## Constraints

- Keep configs decoupled from code: any new field must be read explicitly in `scripts/` with a sensible default.
- Each hyper-parameter is defined in exactly one place; no duplicated overrides across files.
- Every new field must be noted briefly in `memory/updates.md`.

## How to add a new experiment

1. Copy `configs/default.yaml` to `configs/<exp_name>.yaml`.
2. Override only the fields that differ from the default; keep the rest unchanged.
3. Use the uniform `{ type: <registered_name>, params: {...} }` pattern for pluggable components (model, loss, dataset, optimizer, scheduler).
4. Run training with:

   ```bash
   python scripts/train.py --config configs/<exp_name>.yaml
   ```

5. If the experiment introduces a new config field, also:
   - Read and validate it in `scripts/train.py`.
   - Note the addition in `memory/updates.md`.
