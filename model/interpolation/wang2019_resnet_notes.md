# Wang2019ResNet — Implementation Notes

## Paper Reference

Wang, B., Zhang, N., Lu, W., and Wang, J. "Deep-learning-based seismic data
interpolation: A preliminary result." *Geophysics*, vol. 84, no. 1, pp. V11–V20,
2019. DOI: [10.1190/geo2017-0495.1](https://doi.org/10.1190/geo2017-0495.1)

## Paper Architecture Audit

The paper (Geophysics, 2019) describes an 8-layer 2D residual CNN for regularly
missing seismic trace interpolation. The paper-explicit fact is an 8-layer
convolutional architecture that learns a direct mapping from regularly decimated
to complete shot gathers. Exact kernel size, channel count, normalization, and
output residual mode could not be verified from available sources and use a
**Conservative Reproduction Profile**.

## Core Methodology (Preserved)

**该实现保留了论文"使用八层残差卷积神经网络进行规则缺失地震道插值"的核心方法。**
The network uses 1 stem convolution + 3 residual blocks (each with 2 convolutions) +
1 output convolution = 8 Conv2d layers total. Residual blocks use identity shortcuts
to ease optimization. No spatial downsampling or upsampling is performed — the
network operates at the input resolution throughout.

## Architecture

```
Input [B, C_in, H, W]
    |
Stem: Conv3x3(in_ch -> features) + [Norm] + Activation
    |
Residual Block 1: Conv3x3 + [Norm] + Act + [Dropout] + Conv3x3 + [Norm] + +x + Act
    |
Residual Block 2: Conv3x3 + [Norm] + Act + [Dropout] + Conv3x3 + [Norm] + +x + Act
    |
Residual Block 3: Conv3x3 + [Norm] + Act + [Dropout] + Conv3x3 + [Norm] + +x + Act
    |
Output Head: Conv3x3(features -> out_ch) + [optional global residual]
    |
Output [B, C_out, H, W]
```

## Conservative Reproduction Profile

```
features = 64
num_blocks = 3           # 2*3 + 2 = 8 Conv2d total
kernel_size = 3
activation = "relu"
norm = "none"
output_mode = "direct"
conv_bias = True
dropout = 0.0
```

**Parameters (default):** 222,785

## Input Axis Definition

| Axis | Meaning |
|---|---|
| dim 0 (B) | Batch |
| dim 1 (C) | Channels |
| dim 2 (H) | Time samples (height) |
| dim 3 (W) | Traces (width, spatial direction) |

## Classification of Design Choices

| Setting | Classification | Evidence |
|---|---|---|
| 8-layer 2D residual CNN | paper-explicit | Paper title and description |
| Regular-missing interpolation task | paper-explicit | Paper problem statement |
| 3 residual blocks (2 convs each) | paper-explicit | "8-layer" = 2 stem+output + 3×2 block convs |
| features = 64 | reproduction-assumption | Not verified from paper |
| kernel_size = 3 | reproduction-assumption | Not verified from paper |
| No BatchNorm (norm=none) | reproduction-assumption | Paper predates widespread BN in this domain |
| ReLU activation | reproduction-assumption | Not verified from paper |
| Direct output (no global residual) | reproduction-assumption | Not verified from paper |
| Conv bias=True | reproduction-assumption | Default when no BN |
| No dropout | reproduction-assumption | Not verified from paper |
| No spatial downsampling/upsampling | paper-inferred | 2D interpolation preserves input size |
| Identity skip (no 1×1 projection) | reproduction-assumption | Channels constant throughout |
| Registry / YAML / tests | repository-adaptation | Benchmark framework integration |

## Paper-Verified vs Unverified Items

**Verified from paper description:**
- 8-layer residual CNN architecture
- 2D convolutional processing
- Regular-missing trace interpolation

**Not verified (requires full paper access):**
- Exact kernel size (assumed 3)
- Feature channel count (assumed 64)
- Normalization type (assumed none)
- Activation function (assumed ReLU)
- Whether global residual learning is used (assumed direct)
- Whether bias is used in convolutions (assumed True)
- Whether dropout is used (assumed no)

## Future Verification Checklist

When the full paper PDF becomes available, verify and update:

1. `kernel_size` — is it 3, 5, or another value?
2. `features` — what is the exact channel count?
3. `norm` — is BatchNorm, InstanceNorm, or no norm used?
4. `activation` — ReLU, LeakyReLU, or other?
5. `output_mode` — direct or global_residual?
6. Update parameter count test when architecture is confirmed.

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
    --config configs/interpolation/wang2019_resnet.yaml

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/interpolation/train_interpolation_unet.py \
    --config configs/interpolation/wang2019_resnet.yaml
```

## Files

| File | Purpose |
|---|---|
| `model/interpolation/wang2019_resnet.py` | Model implementation |
| `model/interpolation/__init__.py` | Added import (modified) |
| `configs/interpolation/wang2019_resnet.yaml` | Conservative reproduction config |
| `tests/test_wang2019_resnet_profile.py` | Architecture and functional tests |
| `tests/test_wang2019_resnet_extensions.py` | Non-profile option and invalid-arg tests |
| `model/interpolation/wang2019_resnet_notes.md` | This file |
| `memory/updates.md` | Changelog entry (modified) |
