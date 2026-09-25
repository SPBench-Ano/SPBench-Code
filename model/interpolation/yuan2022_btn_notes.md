# Yuan2022BTN — Implementation Notes

## Paper Reference

Yuan, P., Wang, S., Hu, W., Nadukandi, P., Botero, G. O., Wu, X., Nguyen, H. V.,
and Chen, J. "Self-Supervised Learning for Efficient Antialiasing Seismic Data
Interpolation." *IEEE Transactions on Geoscience and Remote Sensing*, vol. 60,
art. 5913819, 2022. DOI: [10.1109/TGRS.2022.3167546](https://doi.org/10.1109/TGRS.2022.3167546)

## BTN Core Principle

Ordinary U-Nets may simply copy known input traces to the output when performing
seismic interpolation. The Blind-Trace Network (BTN) prevents this by enforcing
a strict receptive-field constraint through network architecture: **the prediction
for trace j must NOT access the input at trace j**. This is achieved by two
half-plane U-Net branches that process left-context and right-context
independently, followed by 1x1 convolution fusion.

This implementation faithfully reproduces the BTN's core network mechanism: two
directional half-plane U-Net branches utilize left and right neighbourhood
information respectively, the network receptive-field constraint strictly excludes
each target trace itself, and two 1x1 convolution layers fuse the predictions.

Automatic Spectrum Suppression and Mix-Training are part of the full paper's loss
and training strategy and are NOT included in this network-architecture-only
implementation.

## Architecture

```
Input [B, C, T, X]
    |
    +-- Rotate 90 --> HalfSidedBlindUNet --> Exclude Current Row --> Rotate back
    |                                                              |
    |                                                     left_feat [B, 32, T, X]
    |
    +-- Rotate 270 -> HalfSidedBlindUNet --> Exclude Current Row --> Rotate back
                                                                   |
                                                          right_feat [B, 32, T, X]
    |
    Concat -> [B, 64, T, X]
    |
    Conv1x1: 64 -> 32, ReLU
    Conv1x1: 32 -> out_channels, LeakyReLU(0.01)
    |
    Output [B, out_channels, T, X]
```

### HalfSidedBlindUNet (per branch)

```
Encoder 1: BlindTraceResidualBlock(1 -> 32)    -> Skip 1
           BlindTraceMaxPool2d
Encoder 2: BlindTraceResidualBlock(32 -> 64)   -> Skip 2
           BlindTraceMaxPool2d
Encoder 3: BlindTraceResidualBlock(64 -> 128)  -> Skip 3
           BlindTraceMaxPool2d
Encoder 4: BlindTraceResidualBlock(128 -> 256) -> Skip 4
           BlindTraceMaxPool2d

Bottleneck: BlindTraceResidualBlock(256 -> 512)

Decoder 4: Upsample -> Concat(Skip 4) -> BlindTraceResidualBlock(768 -> 256)
Decoder 3: Upsample -> Concat(Skip 3) -> BlindTraceResidualBlock(384 -> 128)
Decoder 2: Upsample -> Concat(Skip 2) -> BlindTraceResidualBlock(192 -> 64)
Decoder 1: Upsample -> Concat(Skip 1) -> BlindTraceResidualBlock(96 -> 32)
```

### BlindTraceConv2d

Output row i depends only on input rows 0..i (upper half-plane). Achieved by:
1. Pad k//2 zero rows on TOP
2. Apply standard Conv2d with same-padding
3. Crop k//2 rows from BOTTOM

### BlindTraceMaxPool2d

Preserves half-plane causality: shifts features DOWN by one row (top row zeroed),
then applies standard MaxPool2d(2,2).

### Blind Shift (Exclude Current Row)

After the half-sided U-Net, output row i still has access to input row i. A final
shift (top row zeroed, bottom row discarded) ensures the prediction for row i
depends at most on input rows 0..i-1, fully excluding the current row.

## Input Axis Definition

| Axis | Meaning |
|---|---|
| dim 0 (B) | Batch |
| dim 1 (C) | Channels |
| dim 2 (T) | Time samples (height) |
| dim 3 (X) | Traces (width, spatial direction) |

## Rotation Logic

After `rot90(x, k=1, dims=(-2, -1))` (left branch): the original trace axis X
becomes the vertical axis. The half-sided UNet's "upward-only-visible" constraint
means each output position only accesses traces to its LEFT in the original domain.

After `rot90(x, k=3, dims=(-2, -1))` (right branch): same mechanism, but the
visible half-plane maps to traces to the RIGHT in the original domain.

## Classification of Design Choices

| Setting | Classification | Evidence |
|---|---|---|
| Blind-trace receptive field | paper-explicit | "Blind-Trace Network" in title |
| Two left/right half-plane branches | paper-explicit | Network architecture diagram |
| 1x1 feature aggregation | paper-explicit | Network diagram and text |
| U-Net backbone | paper-explicit | Network description |
| Four encoder/decoder levels | paper-inferred | Author follow-up monograph Fig. 3.11/3.12/4.4 |
| Base channels 32 | paper-inferred | Author follow-up monograph |
| Channel doubling/halving | paper-inferred | Standard U-Net progression |
| Blind residual block | paper-inferred | Author follow-up monograph |
| Decoder BatchNorm | paper-inferred | Author follow-up monograph |
| Encoder no norm | paper-inferred | Author follow-up monograph |
| Kernel size 3 | reproduction-assumption | Default U-Net kernel |
| Shared branch weights | reproduction-assumption | Paper is silent |
| Nearest upsampling | reproduction-assumption | Simplest non-learned option |
| LeakyReLU slope 0.01 | reproduction-assumption | PyTorch default |
| Conv bias=True | reproduction-assumption | No norm in encoder to provide affine |
| Constant zero-pad alignment | repository-adaptation | Paper uses patches divisible by 16 |
| Registry / YAML / tests | repository-adaptation | Benchmark framework integration |

## Distinction from Full BTN-SS / BTN-Mix

This implementation only reproduces the **BTN network architecture**. The full
paper training scheme includes:

- **Automatic Spectrum Suppression**: F-K domain tapered mask suppressing energy
  outside the effective signal cone (loss/regularization, not model).
- **Mix-Training**: Blending different decimation patterns to mitigate aliasing
  artifacts from regular missing traces (data/sampler strategy, not model).

Using only the BTN model with standard MSE loss does NOT equate to reproducing
the full BTN-SS or BTN-Mix training scheme. The user's existing training and data
pipeline handles the loss and sampling strategy.

## Blind-Trace Test Results

The following tests verify the blind-trace constraint (see
`tests/test_yuan2022_btn_profile.py`):

1. **Same-Trace Perturbation Test**: Adding 100.0 to input trace j does not change
   output trace j (atol=1e-5, rtol=1e-5).
2. **Jacobian Leakage Test**: `d(output_j) / d(input_j)` is zero (max < 1e-7).
3. **Half-Sided Conv Test**: Perturbing rows below i does not affect output rows 0..i.
4. **Half-Sided Pool Test**: Same causality verified for BlindTraceMaxPool2d.

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
    --config configs/interpolation/yuan2022_btn.yaml

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/interpolation/train_interpolation_unet.py \
    --config configs/interpolation/yuan2022_btn.yaml
```

## Files Added

| File | Purpose |
|---|---|
| `model/interpolation/yuan2022_btn.py` | Model implementation (7 classes/functions) |
| `model/interpolation/__init__.py` | Added import (modified) |
| `configs/interpolation/yuan2022_btn.yaml` | Conservative reproduction config |
| `tests/test_yuan2022_btn_profile.py` | Architecture, leakage, functional tests |
| `tests/test_yuan2022_btn_extensions.py` | Non-profile option and invalid-arg tests |
| `model/interpolation/yuan2022_btn_notes.md` | This file |
| `memory/updates.md` | Changelog entry (modified) |
