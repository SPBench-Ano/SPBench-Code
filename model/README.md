# model/

## Purpose

Deep-learning model definitions (`nn.Module` subclasses and factory functions). Only network assembly; no training loops, no data loading.

## Planned contents

- Backbones: ViT, Swin, UNet, ConvNeXt, MAE, etc.
- Task heads: classification, regression, segmentation, reconstruction, etc.
- Factory functions: `build_model(name, **kwargs)` or a `model_registry`.
- Domain-specific structures for exploration geophysics: trace-token Transformers, patch-based seismic MAE, velocity-inversion networks, etc.

## Constraints

- Prefer mature open-source implementations (timm, HuggingFace, Meta official repos); record the reference in `memory/updates.md`.
- Shared operators (attention, RoPE, MLP, DropPath, etc.) live in `utils/`; this directory only assembles them.
- Each model file exposes only its **factory function(s)** and **model class(es)**; avoid import-time side effects.

## How to add a new model

1. Create `model/<my_net>.py` and implement your `nn.Module` subclass.
2. Decorate it with `@register_model("my_name")` from `model.registry`:

   ```python
   from model.registry import register_model
   import torch.nn as nn

   @register_model("my_name")
   class MyNet(nn.Module):
       def __init__(self, in_channels: int = 1, hidden_dim: int = 64):
           super().__init__()
           ...

       def forward(self, x):
           ...
   ```

3. Add `from . import my_net  # noqa: F401` to `model/__init__.py` so the decorator runs on import.
4. Reference it from YAML:

   ```yaml
   model:
     type: my_name
     params: { in_channels: 1, hidden_dim: 64 }
   ```

5. Training scripts do not need any change.
