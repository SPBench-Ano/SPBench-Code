# Chai2020UNet — Implementation Notes

## Paper Reference

Chai, X., et al. "Deep Learning for Regularly Missing Data Reconstruction."
*IEEE Transactions on Geoscience and Remote Sensing (TGRS)*, vol. 58, no. 6,
pp. 4409–4423, 2020.  DOI: [10.1109/TGRS.2019.2961015](https://doi.org/10.1109/TGRS.2019.2961015)

## Architecture Summary

The paper describes a 2D U-Net encoder-decoder for mapping regularly-missing
seismic shot gathers to complete data:

```
Input: (B, 1, H, W)
    |
Encoder 1: Conv5x5 1->64 + ReLU, Conv5x5 64->64 + ReLU, MaxPool2x2
Encoder 2: Conv5x5 64->128 + ReLU, Conv5x5 128->128 + ReLU, MaxPool2x2
Encoder 3: Conv5x5 128->256 + ReLU, Conv5x5 256->256 + ReLU, MaxPool2x2
Encoder 4: Conv5x5 256->512 + ReLU, Conv5x5 512->512 + ReLU, MaxPool2x2
    |
Bottleneck: Conv5x5 512->1024 + ReLU, Conv5x5 1024->1024 + ReLU
    |
Decoder 4: Upsample2x -> Concat(1024+512=1536) -> Conv5x5 1536->512 + ReLU, Conv5x5 512->512 + ReLU
Decoder 3: Upsample2x -> Concat(512+256=768)   -> Conv5x5 768->256  + ReLU, Conv5x5 256->256 + ReLU
Decoder 2: Upsample2x -> Concat(256+128=384)   -> Conv5x5 384->128  + ReLU, Conv5x5 128->128 + ReLU
Decoder 1: Upsample2x -> Concat(128+64=192)    -> Conv5x5 192->64   + ReLU, Conv5x5 64->64   + ReLU
    |
Output: Conv1x1 64->1 (no activation)
```

**Layer count (paper-verified):** 50 total = 1 input + 19 Conv + 18 ReLU + 4 MaxPool + 4 Upsample + 4 Concatenate.

**Conv layer breakdown:** 18 hidden convolutions use 5x5 kernels with same padding; the final output convolution uses a 1x1 kernel.

**Parameters:** 87,149,953 (all trainable).

## Classification of Design Choices

| Setting | Classification | Evidence |
|---|---|---|
| 5x5 convolution kernels | paper-explicit | IEEE Xplore: Kh=Kw=5 |
| Channels [64, 128, 256, 512, 1024] | paper-explicit | IEEE Xplore: F1=64, F2=128, F3=256, F4=512, F5=1024 |
| 4 encoder/decoder levels | paper-explicit | IEEE Xplore: "network 4" |
| ReLU activation | paper-explicit | Paper layer statistics include 18 ReLU layers |
| MaxPool2d(2,2) downsampling | paper-explicit | Paper explicitly studies pooling variants, settles on max pooling |
| Skip connections via concatenation | paper-explicit | Paper layer statistics include 4 "concatenate" layers |
| 19 Conv2d layers total | paper-explicit | Paper layer statistics |
| 1x1 output convolution | paper-explicit | Standard U-Net design |
| No BatchNorm | paper-inferred | 50-layer count leaves no room for 18 BN layers |
| No Dropout | paper-inferred | No dropout in layer statistics |
| Non-learned upsampling (not ConvTranspose2d) | paper-inferred | 19 conv count precludes additional transposed conv layers |
| Conv bias=True | paper-inferred | No norm to provide affine shift; same-author model docs include bias terms |
| nearest-neighbor upsampling (PyTorch API) | reproduction-assumption | Paper uses non-learned Keras upsampling; exact API call unclear |
| Reflect padding for size alignment | repository-adaptation | Paper uses 112x112 patches (already divisible by 16) |
| Support for arbitrary input sizes | repository-adaptation | Paper only tests 112x112 |
| Configurable norm/activation/upsample/kernel_size | repository-adaptation | Paper uses fixed architecture |

## Paper-Faithful Configuration

```yaml
model:
  type: chai2020_unet
  params:
    in_channels: 1
    out_channels: 1
    base_channels: 64
    num_levels: 4
    kernel_size: 5
    norm: none
    activation: relu
    upsample: nearest
    dropout: 0.0
    residual_learning: false
    conv_bias: true
```

## Extension Configuration (for ablation / lightweight experiments)

```yaml
model:
  type: chai2020_unet
  params:
    in_channels: 1
    out_channels: 1
    base_channels: 32       # smaller model
    num_levels: 4
    kernel_size: 3          # standard U-Net kernel
    norm: batch             # with BatchNorm
    activation: relu
    upsample: transpose     # learned upsampling
    dropout: 0.0
    residual_learning: false
```

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
    --config configs/interpolation/chai2020_unet.yaml

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/interpolation/train_interpolation_unet.py \
    --config configs/interpolation/chai2020_unet.yaml
```

## Files Added

| File | Purpose |
|---|---|
| `model/interpolation/chai2020_unet.py` | Model implementation |
| `model/interpolation/__init__.py` | Added import (modified) |
| `configs/interpolation/chai2020_unet.yaml` | Paper-faithful experiment config |
| `tests/test_chai2020_unet_paper.py` | Architecture and functional tests |
| `tests/test_chai2020_unet_extensions.py` | Non-paper option tests |
| `model/interpolation/chai2020_unet_notes.md` | This file |
| `memory/updates.md` | Changelog entry (modified) |
