# Li2022 CA-Unet — Reproduction Notes

## Paper

- **Title**: CA-Unet: Coordinate Attention U-Net for Seismic Data Reconstruction (exact title behind IEEE paywall)
- **Authors**: Li et al. (2022)
- **Venue**: IEEE (TGRS or GRSL, exact venue behind paywall)
- **DOI**: Unknown (paywalled, no official code repository found)

## What CA-Unet is (and is not)

CA-Unet is a standard U-Net encoder-decoder with Coordinate Attention (CA)
blocks inserted after every double-convolution stage — **not** an SE-Net (no
2D global average pool), not a CBAM-Net (no max-pooling, no 2D spatial
attention), not an Attention U-Net (no gating signals from the decoder), and
not a Transformer (no multi-head attention, no LayerNorm, no positional
encoding).  The CA block from Hou et al. (CVPR 2021) decomposes channel
attention into two 1D directional encodings (height and width strip pooling),
making it lightweight and spatially-aware.

```text
Input [B,1,H,W]  (consecutively missing patch; NO mask channel)
      │
Encoder 0: _DoubleConv(1→32) → CoordAttention2D(32) → MaxPool2d(2)
Encoder 1: _DoubleConv(32→64) → CoordAttention2D(64) → MaxPool2d(2)
Encoder 2: _DoubleConv(64→128) → CoordAttention2D(128) → MaxPool2d(2)
      │
Bottleneck: _DoubleConv(128→256) → CoordAttention2D(256)
      │
Decoder 0: ConvTranspose2d(256→128) → Concat(skip2) → _DoubleConv(256→128) → CoordAttention2D(128)
Decoder 1: ConvTranspose2d(128→64) → Concat(skip1) → _DoubleConv(128→64) → CoordAttention2D(64)
Decoder 2: ConvTranspose2d(64→32) → Concat(skip0) → _DoubleConv(64→32) → CoordAttention2D(32)
      │
Output: Conv2d(32→1, k=1) → [B,1,H,W]
```

7 double-conv stages × 1 CA each = 7 CA blocks total ("7-layer CA-U-Net").

Location: `model/interpolation/li2022_caunet.py`, registered as `li2022_caunet`.

## Coordinate Attention (Hou et al. CVPR 2021 formulation)

For features `x` with shape `(B, C, H, W)`:

```text
1. Directional Pooling:
   z_h = AdaptiveAvgPool2d((H, 1))(x)     ∈ R^(B, C, H, 1)
   z_w = AdaptiveAvgPool2d((1, W))(x)     ∈ R^(B, C, 1, W)

2. Transpose and Concatenate:
   z_h = permute(z_h, [0,1,3,2])          ∈ R^(B, C, 1, H)
   z   = cat([z_h, z_w], dim=-1)          ∈ R^(B, C, 1, H+W)

3. Shared 1×1 Conv → BN → H-Swish (reduction ratio r=16):
   z = Conv1x1(C, C//r)(z) → BN → H-Swish   ∈ R^(B, C//r, 1, H+W)

4. Split and two separate 1×1 Convs → Sigmoid:
   a_h = Sigmoid(Conv1x1(C//r, C)(z_h))   ∈ R^(B, C, 1, H) → permute → (B, C, H, 1)
   a_w = Sigmoid(Conv1x1(C//r, C)(z_w))   ∈ R^(B, C, 1, W)

5. Element-wise multiply:
   output = x × a_h × a_w                 ∈ R^(B, C, H, W)
```

H-Swish = `x * ReLU6(x+3) / 6` (PyTorch `nn.Hardswish`).

Not SE (no 2D global pool to (1,1)), not CBAM (no max-pool, no 2D spatial
attention), not an attention gate (no gating signal).  The entire block is
in-channel — input and output shapes are identical, so it can be inserted
after any convolutional stage.

## Hybrid loss

The CA-Unet paper mentions an SSIM + L1 hybrid loss.  This reproductions uses
the same formula as ANet (`-SSIM + lambda_l1 * L1`, `lambda_l1=1`) via the
shared `ssim_l1` alias in `LOSS_REGISTRY` (which resolves to
`ANetSSIML1Loss`).  See `model/interpolation/yu2022_anet_notes.md` for the
full loss specification.

## Consecutive missing-trace construction

Same infrastructure as ANet: `tools/preprocessing.py::mask_traces(mode="continuous")`
with `mask_ratio_range=(0.1, 0.3)` for per-patch variable missing fraction.
Masks are static per-patch at dataset build time (repository-adaptation).

## Paper profiles

| Setting | Synthetic (SEG C3) | Field (Mobil AVO Viking Graben) |
|---|---|---|
| Config | `configs/interpolation/li2022_caunet_seg_c3_paper.yaml` | `configs/interpolation/li2022_caunet_field_paper.yaml` |
| Patch | 128×128 | 720 time × 120 traces |
| Missing | 10%–30% consecutive traces | same |
| Normalization | min-max [0, 1] | same |
| Optimizer | Adam, lr 1e-3 | same |
| Batch size | 32 | 8 |
| Epochs | 20 | same |
| Scheduler | none | same |

Training command (user-executed):

```bash
python scripts/interpolation/train_interpolation_unet.py \
    --config configs/interpolation/li2022_caunet_seg_c3_paper.yaml
```

## Inference

The paper profile evaluates the **raw network output**
(`inference.overwrite_observed: false`). Observed-trace overwriting
`M·x_obs + (1-M)·pred` improves known-region consistency but changes the
full-patch metrics, so it must be reported separately if enabled.

## Setting classification

| Setting | Classification |
| --- | --- |
| U-Net encoder-decoder backbone | paper-explicit |
| Coordinate Attention blocks | paper-explicit |
| Directional H/W strip pooling | paper-explicit |
| Shared 1×1 conv + BN + H-Swish | paper-explicit |
| Split + two 1×1 convs + Sigmoid | paper-explicit |
| Element-wise attention multiplication | paper-explicit |
| H-Swish activation in CA | paper-explicit |
| SSIM + L1 hybrid loss | paper-inferred |
| lambda_l1 = 1 | paper-inferred |
| 7-layer depth (3 enc + 1 bot + 3 dec) | paper-inferred |
| base_channels = 32 | paper-inferred |
| CA reduction ratio = 16 | paper-inferred |
| CA placement after every double-conv | paper-inferred |
| Kernel size 3×3 | reproduction-assumption |
| Conv → BN → ReLU order | reproduction-assumption |
| ConvTranspose2d upsampling | reproduction-assumption |
| Min-max normalization to [0, 1] | paper-inferred |
| 128×128 synthetic / 720×120 field patches | paper-inferred |
| 10%–30% consecutive missing | paper-inferred |
| Adam, lr 1e-3, batch 32/8, 20 epochs | paper-inferred |
| No mask input channel | paper-explicit |
| No scheduler | paper-inferred |
| Adam betas / weight decay | unresolved (torch defaults) |
| Normalization scope (global stats) | repository-adaptation |
| SSIM global-patch scope; c1=1e-4, c2=9e-4 | reproduction-assumption |
| Per-patch U(0.1, 0.3) ratio sampling | reproduction-assumption |
| Static per-patch masks at dataset build | repository-adaptation |
| ssim_l1 loss alias (shared with ANet) | repository-adaptation |
| SEG-Y volume / split | repository-adaptation |

## Verification status

- Paper is IEEE-paywalled and no official code repository was found.  All
  architecture details (channels, depth, CA placement, kernel sizes,
  hyperparameters) are paper-inferred or classified as
  reproduction-assumptions.
- The Coordinate Attention formula follows Hou et al. CVPR 2021 exactly
  (directional pooling, shared 1×1 conv, BN, H-Swish, split, two 1×1 convs,
  Sigmoid, multiply).
- Local machine has no PyTorch; syntax, YAML, and numpy mask logic verified
  locally. Torch-dependent tests must be run on the training machine:

```bash
python -m pytest tests/test_li2022_caunet_attention.py \
                 tests/test_li2022_caunet_architecture.py \
                 tests/test_li2022_caunet_loss.py \
                 tests/test_li2022_caunet_training.py \
                 tests/test_li2022_caunet_configs.py -q
```

## Unmodified repository modules

Trainer (`utils/train_utils.py` loops), datasets, metrics, preprocessing
(`tools/preprocessing.py`), patching (`tools/patching.py`), and all other
model files are untouched.  The only shared-code changes are the 2-line
`ssim_l1` loss alias in `utils/losses.py` and the 1-line import in
`model/interpolation/__init__.py`.
