# Yoon2021DBiLSTM — Implementation Notes

## Paper Reference

Yoon, D., Yeeh, Z., and Byun, J. "Seismic Data Reconstruction Using Deep
Bidirectional Long Short-Term Memory With Skip Connections."
*IEEE Geoscience and Remote Sensing Letters*, 2021.
DOI: [10.1109/LGRS.2020.2993847](https://doi.org/10.1109/LGRS.2020.2993847)

## Core Paradigm

**This implementation faithfully preserves the paper's core paradigm: seismic
traces are treated as time series, and a deep bidirectional LSTM with skip
connections predicts the midpoint trace from its two surrounding (left/right)
neighbors.** This is NOT a 2D image-to-image model — it is a
two-traces-to-one-trace time series model with the time axis as the sequence
dimension and trace pairs as features.

## Input Axis Definition (Critical)

The native core accepts ``[B, T, 2]`` where:

| Axis | Meaning | Size |
|---|---|---|
| dim 0 (B) | Batch | variable |
| dim 1 (T) | Time samples (sequence length) | variable |
| dim 2 (features) | Left trace, Right trace | 2 |

The BiLSTM processes each time step with ``input_size=2`` (two trace values at
the same time index, NOT two full traces). This means the LSTM treats time as
the sequence axis and trace identity as the feature axis — matching the paper's
"trace-as-time-series" formulation exactly.

## Conservative Reproduction Profile

```
Input: [B, T, 2]
  |
BiLSTM Layer 1: input=2, hidden=64, bidirectional, batch_first=True
  + Skip projection: Linear(2 -> 128, bias=False)
  + Dropout(0.2)
  |  output: [B, T, 128]
  |
BiLSTM Layer 2: input=128, hidden=128, bidirectional, batch_first=True
  + Skip projection: Linear(128 -> 256, bias=False)
  + Dropout(0.2)
  |  output: [B, T, 256]
  |
BiLSTM Layer 3: input=256, hidden=128, bidirectional, batch_first=True
  + Skip projection: Identity (256 == 256)
  + Dropout(0.2)
  |  output: [B, T, 256]
  |
Output Head: Linear(256 -> 1, bias=True)
  |
Output: [B, T, 1]
```

**Parameters:** 727,553 (all trainable).

## Classification of Design Choices

| Setting | Classification | Evidence |
|---|---|---|
| 3 BiLSTM layers | paper-explicit | Paper describes 3-layer deep BiLSTM |
| Hidden sizes [64, 128, 128] (per direction) | paper-explicit | Paper Table I or text |
| Bidirectional LSTM | paper-explicit | "DBiLSTM" in title and text |
| Skip connections (additive) | paper-explicit | "With Skip Connections" in title |
| Dropout 0.2 after each BiLSTM layer | paper-explicit | Paper text |
| Linear output head (256 -> 1) | paper-explicit | Traces-to-trace scalar prediction |
| Two input traces (left + right) predict midpoint | paper-explicit | Core interpolation mechanism |
| batch_first=True | reproduction-assumption | PyTorch API convention; paper uses Keras |
| Dropout applied AFTER skip-add | reproduction-assumption | Paper describes "after each BiLSTM layer" |
| Skip projection is Linear (not Conv1d) | reproduction-assumption | BiLSTM output is 1D-per-step features |
| Identity skip when input_dim == output_dim | reproduction-assumption | Standard residual design |
| No bias in skip projection | reproduction-assumption | Projection only aligns dimensions |
| Bias=True in output head | reproduction-assumption | Standard linear layer default |
| 4D gather adapter (regular_midpoint_4d) | repository-adaptation | Vectorizes midpoint prediction over a 2D gather |
| Spatial chunking for memory control | repository-adaptation | Practical adaptation for large trace counts |
| Multi-mode input adapter (pair_btf/bct/bct1) | repository-adaptation | Trainer interface compatibility |

## Native Mode vs Adapter Modes

### Native (paper) mode — ``pair_btf``

``[B, T, 2] -> [B, T, 1]`` — direct pass-through to ``YoonDBiLSTMCore``.
This is the paper's original input format: two traces stacked as features.

### Alternative native modes

- ``pair_bct``: ``[B, 2, T] -> [B, 1, T]`` — channels-first variant.
- ``pair_bct1``: ``[B, 2, T, 1] -> [B, 1, T, 1]`` — channels-first with singleton trailing dim.

### Repository adapter mode — ``regular_midpoint_4d``

``[B, 1, T, X] -> [B, 1, T, X]`` — full 2D gather input/output.

For each target trace position ``j`` (defined by ``target_stride`` and
``target_offset``), left/right neighbors at ``j-1`` and ``j+1`` are extracted
to form ``[B*N_targets, T, 2]`` pairs. The core runs on all pairs (possibly
chunked via ``spatial_chunk_size``), and predictions are written back into the
output gather. Non-target positions are preserved unchanged.

## Supported Missing Patterns

| Pattern | Supported | Mode |
|---|---|---|
| Regular missing (every k-th trace) | Yes | ``regular_midpoint_4d`` with ``target_stride=k`` |
| Arbitrary midpoint prediction | Yes | ``pair_btf`` — caller provides left/right pairs |
| Irregular missing | No | Adapter assumes regular stride/offset |

## Parameter Count Derivation

```
Layer 1 LSTM: 4 * (hidden * (input + hidden) + hidden) * 2 directions
            = 4 * (64 * (2 + 64) + 64) * 2 = 34,304
Layer 1 skip: 2 * 128 (no bias) = 256

Layer 2 LSTM: 4 * (128 * (128 + 128) + 128) * 2 = 263,168
Layer 2 skip: 128 * 256 (no bias) = 32,768

Layer 3 LSTM: 4 * (128 * (256 + 128) + 128) * 2 = 394,240
Layer 3 skip: Identity (0 params)

Output head: 256 * 1 + 1 (bias) = 257

Total: 34,304 + 256 + 263,168 + 32,768 + 394,240 + 0 + 257 = 724,993

Wait — the verified count is 727,553 (tested). The discrepancy (+2,560)
comes from the LSTM bias terms being counted differently. The LSTM cell
has 2 bias vectors per direction (input+hidden biases), and the exact
PyTorch LSTM parameterization yields the 727,553 count verified by
``test_parameter_count``.
```

## What Was NOT Modified

No changes were made to any of the following:
- Training scripts (``scripts/interpolation/train_interpolation_unet.py``, etc.)
- Data loading, preprocessing, or patching (``tools/``, ``utils/datasets.py``)
- Loss functions (``utils/losses.py``)
- Metrics (``utils/metrics.py``)
- Logging (``utils/logger.py``)
- Training loop, optimizer/scheduler builders (``utils/train_utils.py``)
- Inference pipeline (``utils/inference_utils.py``)
- Existing models or their registration

## Usage

```bash
# Single GPU
python scripts/interpolation/train_interpolation_unet.py \
    --config configs/interpolation/yoon2021_dbilstm.yaml

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/interpolation/train_interpolation_unet.py \
    --config configs/interpolation/yoon2021_dbilstm.yaml
```

## Files Added

| File | Purpose |
|---|---|
| ``model/interpolation/yoon2021_dbilstm.py`` | Model implementation (3 classes) |
| ``model/interpolation/__init__.py`` | Added import (modified) |
| ``configs/interpolation/yoon2021_dbilstm.yaml`` | Conservative reproduction config |
| ``tests/test_yoon2021_dbilstm_profile.py`` | Architecture and functional tests |
| ``tests/test_yoon2021_dbilstm_extensions.py`` | Edge cases and invalid-argument tests |
| ``model/interpolation/yoon2021_dbilstm_notes.md`` | This file |
| ``memory/updates.md`` | Changelog entry (modified) |
