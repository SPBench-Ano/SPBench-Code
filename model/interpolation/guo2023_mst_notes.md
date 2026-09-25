# Guo2023MST — Implementation Notes

## Paper Reference

Guo, Y., Fu, Z., and Li, B. "Seismic Data Interpolation Based on Multi-Scale
Transformer." *IEEE Geoscience and Remote Sensing Letters*, vol. 20, art. 7505905,
2023. DOI: [10.1109/LGRS.2023.3298101](https://doi.org/10.1109/LGRS.2023.3298101)

## Paper Architecture Audit

The paper (IEEE GRSL, 2023) is behind the IEEE paywall. No public implementation
or detailed architecture description was found via web search (GitHub, arXiv,
Semantic Scholar abstract). The implementation uses a **Conservative Reproduction
Profile** as specified below.

## Core Methodology (Preserved)

**该实现保留了论文"对多尺度地震特征图分别计算多组自注意力，以捕获长距离依赖并完成二维地震插值"的核心方法。**

The MST generates multi-scale seismic feature maps via strided convolutions,
applies independent self-attention at each scale to capture long-range dependencies,
then fuses multi-scale features through upsampling, concatenation, and
convolutional fusion into a complete seismic reconstruction.

## Architecture

```
Input [B, C_in, T, X]
    |
Conv Stem (2x Conv3x3 + GELU)
    |
Feature F0 [B, E0, T, X]
    |                           |                        |
Scale 0 Transformer         Downsample conv -> F1     Downsample conv -> F2
(Pre-Norm, MHA, MLP)        [B, E1, T/2, X/2]        [B, E2, T/4, X/4]
    |                        Scale 1 Transformer       Scale 2 Transformer
Y0 [B, E0, T, X]             (Pre-Norm, MHA, MLP)      (Pre-Norm, MHA, MLP)
    |                             |                         |
    |                        Upsample to [T,X]         Upsample to [T,X]
    |                             |                         |
    +---------- Concat([Y0, Y1_up, Y2_up]) -------------+
                           |
                  Fusion Conv3x3 + GELU
                  Fusion Conv3x3 + GELU
                           |
                  Output Conv1x1
                           |
                  Output [B, C_out, T, X]
```

### Component Details

**Conv Stem**: Conv3x3(in_ch, E0) + GELU + Conv3x3(E0, E0) + GELU. No spatial
size change, no BatchNorm.

**Multi-Scale Feature Pyramid**: For each scale transition: stride-2 Conv3x3
(or MaxPool2d + Conv1x1) to the next embed_dim, followed by GELU.

**Scale Transformer**: Tokenize (flatten H×W → transpose to [B, N, C]),
apply Pre-Norm Transformer blocks (LayerNorm → MHA → +residual,
LayerNorm → MLP(GELU) → +residual), restore to [B, C, H, W].

**Multi-Scale Feature Fusion**: Upsample all scales to the first scale's
spatial size (bilinear/nearest), concat along channel dim, then:
Conv3x3(sum(E_i), E0) + GELU + Conv3x3(E0, E0) + GELU + Conv1x1(E0, C_out).

### Attention Complexity

Self-attention computes O(N²) where N = H × W. For a 512×544 patch at scale 0,
N ≈ 278K tokens. The `max_tokens_per_scale` guard prevents silent OOM by
raising a clear RuntimeError. Users must use appropriate patch sizes or set
`max_tokens_per_scale` appropriately.

### Tokenization

```
[B, C, H, W]  →  flatten(2)  →  [B, C, N] where N = H*W
              →  transpose(1,2) → [B, N, C]  (token format)

Reverse: transpose(1,2) → [B, C, N] → reshape → [B, C, H, W]
```

Each spatial position becomes one token; the channel dimension becomes the
token embedding. Attention is global (N×N) — every token attends to every
other token at the same scale. The paper's "multi-scale" attention means
attention is computed independently at each scale, not across scales.

## Conservative Reproduction Profile

```
num_scales = 3
embed_dims = (32, 64, 128)
depths = (1, 1, 1)
num_heads = (4, 4, 8)
mlp_ratio = 4.0
kernel_size = 3
downsample_mode = "stride_conv"
upsample_mode = "bilinear"
norm_type = "layer"
activation = "gelu"
positional_encoding = "none"
dropout = 0.0
attention_dropout = 0.0
drop_path = 0.0
output_mode = "direct"
max_tokens_per_scale = None
```

## Classification of Design Choices

| Setting | Classification | Evidence |
|---|---|---|
| Multi-scale self-attention over feature maps | paper-explicit | Title and abstract |
| Long-range dependency modeling via attention | paper-explicit | Core claimed contribution |
| Supervised interpolation (masked→complete) | paper-explicit | Task definition |
| Multi-scale feature extraction (strided conv) | paper-inferred | Required for multi-scale attention |
| Per-scale independent attention | paper-inferred | "分别计算多组自注意力" |
| Upsampling + fusion for reconstruction | paper-inferred | Required to combine multi-scale outputs |
| 3 scales | reproduction-assumption | Not verified from paper |
| Embed dims (32, 64, 128) | reproduction-assumption | Not verified from paper |
| Depths (1, 1, 1) per scale | reproduction-assumption | Not verified from paper |
| Heads (4, 4, 8) per scale | reproduction-assumption | Not verified from paper |
| Pre-LN Transformer blocks | reproduction-assumption | Not verified from paper |
| GELU activation | reproduction-assumption | Not verified from paper |
| MLP ratio 4.0 | reproduction-assumption | Not verified from paper |
| No positional encoding | reproduction-assumption | Not verified from paper |
| Stride-conv downsampling | reproduction-assumption | Not verified from paper |
| Bilinear upsampling | reproduction-assumption | Not verified from paper |
| Direct output (no global residual) | reproduction-assumption | Not verified from paper |
| Concat-based multi-scale fusion | reproduction-assumption | Not verified from paper |
| Conv stem (no BatchNorm) | reproduction-assumption | Not verified from paper |
| Constant zero-pad alignment | repository-adaptation | Integration with arbitrary input sizes |
| Token-count memory guard | repository-adaptation | Practical protection against OOM |
| Registry / YAML / tests | repository-adaptation | Benchmark framework integration |

## Paper-Verified vs Unverified Items

**Verified from available sources (title/abstract):**
- Multi-scale Transformer architecture for seismic interpolation
- Self-attention for long-range dependency capture

**Not verified (paywall — requires full paper access):**
- Exact number of scales (assumed 3)
- Channel dimensions per scale (assumed 32, 64, 128)
- Transformer depth per scale (assumed 1 each)
- Number of attention heads per scale (assumed 4, 4, 8)
- Normalization scheme (assumed Pre-LN)
- Activation function (assumed GELU)
- Positional encoding scheme (assumed none)
- Multi-scale fusion method (assumed concat + convolution)
- Downsampling/upsampling methods (assumed stride-conv / bilinear)
- Whether global residual learning is used (assumed no)
- Whether drop path / stochastic depth is used (assumed no)

## Future Verification Checklist

When the full paper PDF becomes available, verify and update:

1. `num_scales` — is it 3, or another number?
2. `embed_dims` — what are the exact channel counts?
3. `depths` — how many Transformer blocks per scale?
4. `num_heads` — how many attention heads per scale?
5. Norm position — Pre-LN or Post-LN?
6. Activation — GELU, ReLU, or other?
7. Positional encoding — is one used? What type?
8. Fusion method — concat, sum, or attention-based?
9. Upsampling — bilinear, transpose conv, or other?
10. Global residual — is input added to output?
11. Update parameter count test when architecture is confirmed.

## What Was NOT Modified

No changes were made to any of the following:
- Training scripts (`scripts/interpolation/train_interpolation_unet.py`, etc.)
- Data loading, preprocessing, or patching (`tools/`, `utils/datasets.py`)
- Loss functions (`utils/losses.py`)
- Metrics (`utils/metrics.py`)
- Logging (`utils/logger.py`)
- Training loop, optimizer/scheduler builders (`utils/train_utils.py`)
- Inference pipeline (`utils/inference_utils.py`)
- Existing models or their registration

## Usage

```bash
# Single GPU
python scripts/interpolation/train_interpolation_unet.py \
    --config configs/interpolation/guo2023_mst.yaml

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/interpolation/train_interpolation_unet.py \
    --config configs/interpolation/guo2023_mst.yaml
```

## Files Added

| File | Purpose |
|---|---|
| `model/interpolation/guo2023_mst.py` | Model implementation (6 classes + 2 helpers) |
| `model/interpolation/__init__.py` | Added import (modified) |
| `configs/interpolation/guo2023_mst.yaml` | Conservative reproduction config |
| `tests/test_guo2023_mst_profile.py` | Architecture, attention, fusion, functional tests |
| `tests/test_guo2023_mst_extensions.py` | Non-profile option and invalid-argument tests |
| `model/interpolation/guo2023_mst_notes.md` | This file |
| `memory/updates.md` | Changelog entry (modified) |
