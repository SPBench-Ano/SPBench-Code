# UNet++ Implementation Design

## Goal

Add a UNet++ model (`model/interpolation/unet_plusplus.py`) that follows the "UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation" paper, integrated into the existing registry + factory pattern.

## Architecture

- **Encoder**: Plain CNN encoder identical to the existing `unet.py` — stack of `_DoubleConv` blocks with `MaxPool2d(2)` downsampling.
- **Bottleneck**: `_DoubleConv` at the deepest level.
- **Decoder**: Nested dense skip connections as described in the paper. Each decoder node `X^{i,j}` concatenates:
  1. All preceding same-resolution features from nodes `X^{i,k}` where `k < j`
  2. The upsampled output from `X^{i+1,j-1}`
  Then applies a `_DoubleConv` block.
- **Head**: Single `Conv2d` output layer (no deep supervision).

Implementation is derived from the widely-used `segmentation_models.pytorch` (`UnetPlusPlusDecoder`), stripped of external dependencies (no `smp.base.modules`, no attention modules, no backbone encoders). All conv blocks use the existing codebase style: `Conv2d -> BatchNorm2d -> ReLU` x 2.

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `model/interpolation/unet_plusplus.py` | Add | New UNet++ model |
| `model/interpolation/__init__.py` | Modify | Add import so `@register_model` runs |

## Interface

```yaml
model:
  type: unet_plusplus
  params:
    in_channels: 1
    out_channels: 1
    base_channels: 32
    depth: 4
```

Parameters match the existing `unet` signature for consistency.

## Key Design Decisions

1. **No deep supervision**: Single output head only, matching the current training pipeline which expects one output tensor.
2. **No attention modules**: Keeping parity with the existing plain UNet.
3. **No backbone support**: Using the same plain encoder as `unet.py` to keep the codebase lightweight and registry-agnostic.
4. **Flexible depth**: Support configurable `depth >= 2`, same as `unet.py`.
