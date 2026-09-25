# utils/

## Purpose

Training infrastructure shared by models and scripts: datasets, losses, metrics, visualization, logging, optimizer / scheduler helpers, position encoding, masking, etc.

## Planned contents

- `datasets.py` — `torch.utils.data.Dataset` subclasses (H5 / SEGY / custom) and DataLoader factories.
- `losses.py` — Composable loss functions (MSE / L1 / energy consistency / perceptual, etc.).
- `metrics.py` — Validation metrics. Implemented: `mse`, `rmse`, `mae`, `snr`, `psnr`, `ssim` (Wang et al. 2004, self-implemented; accepts `(B, C, H, W)` or `(B, H, W)`). `rmse` / `snr` / `psnr` accept `reduction="per_sample"` (default; mean over per-sample scores) or `reduction="global"` (preserves `RMSE == sqrt(MSE)` and `PSNR == 10·log10(data_range²/MSE)`). All return a python `float`.
- `visualization.py` — Diagnostic plots: `plot_sample` (4 panels: input | pred | target | residual), `plot_loss_curve`, `plot_metrics_curve`, plus `visualize_random_sample(model, loader, ..., seed=None)` that draws a random validation sample (set `seed` to lock the same one across epochs).
- `logger.py` — `TrainingLogger` writes a timestamped `train_log.txt` plus append-only `loss_history.csv` (cols `epoch, lr, *loss_keys`) and `metrics_history.csv` (cols `epoch, *metric_keys`); also auto-refreshes `loss_curve.png` / `metrics_curve.png` every `plot_interval` epochs (default 5; `0` disables) and re-draws once on `close()`. Resumed runs rehydrate history from the existing CSVs so curves stay continuous.
- `train_utils.py` — `train_epoch` / `evaluate` / checkpointing / seeding / DDP reduction helpers.
- `position_encoding.py` — Position encodings (RoPE, sinusoidal, learned).
- `masking.py` — Masking strategies (random, structured).

## Constraints

- Host **reusable capabilities** only; business pipelines belong in `scripts/`, model definitions in `model/`.
- One responsibility per file; do not mix datasets, losses, and visualization in a single module.
- Before adding a new utility, check for an existing one; when a mature open-source implementation exists, use it and cite the source.

## How to add a new dataset / loss / metric

Each of these modules ships a registry; follow the same three-step pattern:

1. **Subclass** the corresponding base class.
2. **Decorate** with `@register_<kind>("your_name")`.
3. **Reference** the name from a YAML config.

### Dataset

```python
from utils.datasets import BaseArrayDataset, register_dataset

@register_dataset("my_format")
class MyDataset(BaseArrayDataset):
    def _build_index(self): ...
    def _load_sample(self, path): ...
```

```yaml
data:
  train:
    type: my_format
    params: { root: /path/to/data, input_key: data, target_key: label }
    loader: { batch_size: 16, num_workers: 4, shuffle: true }
```

### Loss

```python
from utils.losses import BaseLoss, register_loss

@register_loss("my_loss")
class MyLoss(BaseLoss):
    def forward(self, pred, target=None, **extras): ...
```

```yaml
loss:
  type: my_loss
  params: {}
```

### Metric

```python
from utils.metrics import BaseMetric, register_metric

@register_metric("my_metric")
class MyMetric(BaseMetric):
    higher_is_better = True
    def __call__(self, pred, target): ...
```

```yaml
metrics:
  - name: my_metric
    params: {}
```

No edits to `scripts/train.py` are needed.

## How to add a new visualization / logger hook

- Visualization: add a function in `utils/visualization.py` and call it from `scripts/train.py` where appropriate (e.g. every `vis_interval` epochs).
- Logger backend: extend `TrainingLogger` in `utils/logger.py` to push to TensorBoard / W&B, keeping the public methods (`info`, `log_epoch`, `flush`, `close`) unchanged so callers stay the same.
