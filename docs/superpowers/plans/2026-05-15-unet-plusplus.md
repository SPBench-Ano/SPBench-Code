# UNet++ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a UNet++ model file (`model/interpolation/unet_plusplus.py`) and wire it into the registry, following the existing UNet style.

**Architecture:** Plain encoder (same as existing UNet) + nested dense skip-connection decoder (based on the UNet++ paper and verified against `segmentation_models.pytorch`). Single output head, no deep supervision.

**Tech Stack:** PyTorch, existing project registry pattern.

---

### Task 1: Create `model/interpolation/unet_plusplus.py`

**Files:**
- Create: `model/interpolation/unet_plusplus.py`

- [ ] **Step 1: Write the model implementation**

```python
"""UNet++ for seismic interpolation / denoising.

Reference: Zhou et al., "UNet++: Redesigning Skip Connections to Exploit
Multiscale Features in Image Segmentation", IEEE TMI 2019.
Decoder logic adapted from segmentation_models.pytorch (MIT licence).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..registry import register_model


class _DoubleConv(nn.Module):
    """(Conv->BN->ReLU) x 2 with same spatial size."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpConv(nn.Module):
    """Upsample by 2x and halve channels."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


@register_model("unet_plusplus")
class UNetPlusPlus(nn.Module):
    """UNet++ with nested dense skip connections.

    Parameters match the existing ``unet`` model for drop-in replacement.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"UNetPlusPlus depth must be >= 2, got {depth}.")

        chans: List[int] = [base_channels * (2**i) for i in range(depth)]

        # ----- Encoder -----
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for c in chans:
            self.encoders.append(_DoubleConv(prev, c))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = c

        # ----- Bottleneck -----
        self.bottleneck = _DoubleConv(chans[-1], chans[-1] * 2)

        # ----- Decoder nodes -----
        # Node X^{i,j}: i = resolution level, j = layer index (0 = encoder)
        # For j > 0: upconv from X^{i+1, j-1} then concat all X^{i, k} (k < j)
        self.upconvs: nn.ModuleDict = nn.ModuleDict()
        self.decoders: nn.ModuleDict = nn.ModuleDict()

        for i in range(depth):  # resolution level
            lower_ch = chans[-1] * 2 if i == depth - 1 else chans[i + 1]
            for j in range(1, depth - i + 1):  # layer index
                up_name = f"up_{i}_{j}"
                dec_name = f"dec_{i}_{j}"
                self.upconvs[up_name] = _UpConv(lower_ch, chans[i])
                # concat: upconv output (chans[i]) + j previous same-res features (j * chans[i])
                self.decoders[dec_name] = _DoubleConv((j + 1) * chans[i], chans[i])
                # next lower node for this resolution uses chans[i] as its lower_ch
                lower_ch = chans[i]

        self.head = nn.Conv2d(chans[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store encoder features and decoder outputs
        # enc[i] = X^{i,0}
        enc: List[torch.Tensor] = []
        h = x
        for encoder, pool in zip(self.encoders, self.pools):
            h = encoder(h)
            enc.append(h)
            h = pool(h)

        # bottleneck = X^{depth,0}
        bottleneck = self.bottleneck(h)

        # dense[i][j] stores X^{i,j}; dense[i][0] is enc[i]
        dense: List[List[torch.Tensor]] = [[f] for f in enc]
        # Append bottleneck as its own level
        dense.append([bottleneck])

        depth = len(enc)
        for i in range(depth - 1, -1, -1):  # from deepest to shallowest
            for j in range(1, depth - i + 1):
                up = self.upconvs[f"up_{i}_{j}"]
                dec = self.decoders[f"dec_{i}_{j}"]
                # upsample from lower level
                lower = dense[i + 1][j - 1]
                h_up = up(lower)
                # align spatial size (safeguard)
                target_size = dense[i][0].shape[-2:]
                if h_up.shape[-2:] != target_size:
                    h_up = nn.functional.interpolate(
                        h_up, size=target_size, mode="bilinear", align_corners=False
                    )
                # concatenate all same-resolution features
                feats = [h_up] + dense[i]
                h = dec(torch.cat(feats, dim=1))
                dense[i].append(h)

        return self.head(dense[0][-1])
```

- [ ] **Step 2: Commit the new file**

```bash
git add model/interpolation/unet_plusplus.py
git commit -m "feat: add UNet++ model implementation"
```

### Task 2: Register the model in `model/interpolation/__init__.py`

**Files:**
- Modify: `model/interpolation/__init__.py`

- [ ] **Step 1: Add import line**

Insert after the last import:
```python
from . import unet_plusplus  # noqa: F401
```

The `__init__.py` should now read:
```python
"""Models registered for the interpolation task."""

from ..registry import MODEL_REGISTRY, build_model, register_model

from . import atten_unet  # noqa: F401
from . import dncnn  # noqa: F401
from . import res_unet  # noqa: F401
from . import unet  # noqa: F401
from . import unet_plusplus  # noqa: F401

__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "register_model",
]
```

- [ ] **Step 2: Verify import succeeds**

Run (do not auto-run, provide command only):
```bash
python -c "from model.interpolation import unet_plusplus; print('OK')"
```

Expected: prints `OK` with no errors.

- [ ] **Step 3: Commit**

```bash
git add model/interpolation/__init__.py
git commit -m "chore: register unet_plusplus model"
```

### Task 3: Smoke test forward pass

**Files:**
- Test: manual verification (no new test file required)

- [ ] **Step 1: Run a one-off shape check**

Provide the command:
```bash
python -c "
import torch
from model.interpolation.unet_plusplus import UNetPlusPlus

model = UNetPlusPlus(in_channels=1, out_channels=1, base_channels=32, depth=4)
x = torch.randn(1, 1, 256, 256)
y = model(x)
assert y.shape == (1, 1, 256, 256), f'Expected (1,1,256,256), got {y.shape}'
print('Forward pass OK:', y.shape)
"
```

Expected: `Forward pass OK: torch.Size([1, 1, 256, 256])`

- [ ] **Step 2: Test with varying depth**

Provide the command:
```bash
python -c "
import torch
from model.interpolation.unet_plusplus import UNetPlusPlus

for depth in [2, 3, 4, 5]:
    model = UNetPlusPlus(depth=depth)
    x = torch.randn(1, 1, 128, 128)
    y = model(x)
    assert y.shape == x.shape
    print(f'depth={depth} OK')
"
```

Expected: prints `depth=2 OK` through `depth=5 OK`.

---

## Self-Review Checklist

1. **Spec coverage:**
   - UNet++ decoder with nested skip connections? Yes (Task 1).
   - Single output head? Yes (no deep supervision).
   - Plain encoder matching existing UNet? Yes (same `_DoubleConv` + `MaxPool2d`).
   - Registry integration? Yes (Task 2).
   - Config interface matching existing UNet? Yes (`in_channels`, `out_channels`, `base_channels`, `depth`).

2. **Placeholder scan:**
   - No TBD/TODO/fill-in-details found.
   - All code is complete and copy-paste ready.

3. **Type consistency:**
   - `depth` validated as `>= 2` to match existing UNet guard.
   - Channel progression `base_channels * 2**i` matches existing UNet.
   - Output head uses `chans[0]` -> `out_channels`, same as existing UNet.
