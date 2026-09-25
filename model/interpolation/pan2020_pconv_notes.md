# Pan2020 PConv U-Net — Reproduction Notes

## Paper

- **Title**: A Partial Convolution-Based Deep-Learning Network for Seismic Data Regularization
- **Authors**: Pan et al. (2020)
- **Venue**: Computers & Geosciences, vol. 145, 2020
- **DOI**: 10.1016/j.cageo.2020.104609
- **Author code**: https://github.com/shulinpan/seismic-data-regularization

## What Pan2020 PConv U-Net is

A 6-level U-Net where every convolution is replaced by Partial Convolution (PConv).
PConv renormalises the convolution output by the valid-pixel fraction of the
input mask, and propagates an updated binary mask forward.  The mask starts as a
per-trace observation indicator (1=observed, 0=missing) and is gradually filled
(dilated) as each PConv layer's receptive field covers valid pixels.

Key distinction from standard image-inpainting PConv (Liu et al. ECCV 2018):
Pan2020 author code uses **global** mask-mean normalisation (divide by
mean(mask) over all spatial positions) rather than the **local** per-window
valid-pixel count (sum(1)/sum(M)).  See `Pan2020PartialConv2d` for the two modes.

## Architecture

```text
Input: data (B,1,H,W) + mask (B,1,H,W)   [separate mode, author-code default]

Encoder 1: PConv(1→32, k=7, s=2) → ReLU                          → (B,32,H/2,W/2)
Encoder 2: PConv(32→64, k=5, s=2) → BN → ReLU                    → (B,64,H/4,W/4)
Encoder 3: PConv(64→128, k=5, s=2) → BN → ReLU                   → (B,128,H/8,W/8)
Encoder 4: PConv(128→256, k=3, s=2) → BN → ReLU                  → (B,256,H/16,W/16)
Encoder 5: PConv(256→512, k=3, s=2) → BN → ReLU                  → (B,512,H/32,W/32)
Encoder 6: PConv(512→512, k=3, s=2) → BN → ReLU                  → (B,512,H/64,W/64)

Decoder 7:  Upsample(2x) → Concat(enc5 data+mask) → PConv(1024→512, k=3, s=1) → BN → LeakyReLU(0.2)
Decoder 8:  Upsample(2x) → Concat(enc4 data+mask) → PConv(768→256, k=3, s=1)  → BN → LeakyReLU(0.2)
Decoder 9:  Upsample(2x) → Concat(enc3 data+mask) → PConv(384→128, k=3, s=1)  → BN → LeakyReLU(0.2)
Decoder 10: Upsample(2x) → Concat(enc2 data+mask) → PConv(192→64, k=3, s=1)   → BN → LeakyReLU(0.2)
Decoder 11: Upsample(2x) → Concat(enc1 data+mask) → PConv(96→32, k=3, s=1)    → BN → LeakyReLU(0.2)
Decoder 12: Upsample(2x) → Concat(input data+mask) → PConv(33→1, k=3, s=1)    → LeakyReLU(0.2)

Output: Conv2d(1→1, k=1) → Sigmoid → (B,1,H,W)
```

- Encoder 1 has NO BatchNorm (author-code profile); encoders 2–6 and decoders 7–11 have BN.
- Decoder 12 has NO BatchNorm (last decoder stage).
- All decoders use LeakyReLU(0.2); all encoders use ReLU.
- Skip connections concatenate BOTH data and mask from the corresponding encoder.
- Same-padding (Keras `padding='same'`) ensures output spatial size = ceil(input/stride).

## Partial Convolution Formula (author-code global mode)

For data `X` (B,C,H,W) and mask `M` (B,C,H,W):

```text
rho = mean(M, dim=(-2,-1))           # per-channel global mean (B,C,1,1)
Y   = Conv2d((X ⊙ M) / rho, W, b)    # normalised convolution
S   = Conv2d(M, ones_kernel)          # mask count (fixed kernel, no grad)
M'  = (S > 0).float()                 # updated binary mask
```

When M is all-ones (no missing data), rho=1 and PConv degenerates to ordinary Conv2d.

## Composite Loss

```text
L_total = L_valid + 6.0 × L_hole + 0.1 × L_tv
```

- **L_valid**: per-sample spatial SUM of |pred − target| on observed positions; batch MEAN.
- **L_hole**: per-sample spatial SUM of |pred − target| on missing positions; batch MEAN.
- **L_tv**: total variation on hole-neighbourhood composite.  The composite image uses
  target values for observed pixels and pred values for missing pixels.  The hole
  mask is dilated (7×7) so the TV penalty also covers the hole boundary.  Per-sample
  spatial SUM of |diff_x| + |diff_y| within the dilated region; batch MEAN.

Registered as `pan2020_pconv_composite` in `LOSS_REGISTRY`.

## Input / Output convention

- **Input**: masked seismic patch + observed mask (1=observed, 0=missing).
  Full data is normalised to [0,1] BEFORE masking; missing positions are zeroed.
- **Target**: full (unmasked) seismic patch, also [0,1].
- **Output**: reconstructed patch in [0,1] (Sigmoid activation).

## Mask generation (author-code mode)

- Random whole-trace masking WITH replacement.
- Number of observed traces drawn uniformly from [1, 0.5 × n_traces].
- Sampling with replacement means the same trace index can be drawn multiple times;
  unique observed trace count ≤ draws.
- Returns an observed_mask (1=observed, 0=missing) broadcast over the time dimension.
- Applied at patch level (post-patchify), not shot level.

## Paper profiles

| Setting | Synthetic |
|---|---|
| Config | `configs/interpolation/pan2020_pconv_author_code.yaml` |
| Patch | 128×128 |
| Missing | random traces with replacement, 1 to 0.5×n_traces |
| Normalization | min-max [0, 1] (global scope after spherical divergence) |
| Optimizer | Adam, lr 2e-4 |
| Batch size | 4 |
| Epochs | 100 |
| Scheduler | none |

Training command (user-executed):

```bash
python scripts/interpolation/train_interpolation_unet.py \
    --config configs/interpolation/pan2020_pconv_author_code.yaml
```

## Inference

The paper profile evaluates the **raw network output**
(`inference.overwrite_observed: false`).

## Setting classification

| Setting | Classification |
| --- | --- |
| 6-level U-Net encoder-decoder | paper-explicit |
| Partial Convolution (PConv) layers | paper-explicit |
| Global mask-mean normalisation | author-code (differs from paper Eq. 6) |
| Binary mask propagation via fixed ones-kernel | paper-explicit |
| Encoder kernel sizes (7,5,5,3,3,3) | paper-explicit |
| Encoder channels (32,64,128,256,512,512) | paper-explicit |
| Decoder channels (512,256,128,64,32,1) | reproduction-assumption |
| Encoder 1 no BN, rest BN | paper-explicit |
| Decoder LeakyReLU(0.2) | paper-explicit |
| Encoder ReLU | paper-explicit |
| Sigmoid output activation | paper-explicit |
| Skip connections with both data and mask | paper-explicit |
| Composite loss L_valid + 6*L_hole + 0.1*L_tv | paper-explicit |
| TV on hole-neighbourhood composite | paper-explicit |
| Hole mask dilation 7×7 for TV | reproduction-assumption |
| Per-sample spatial sum + batch mean reduction | paper-explicit |
| Random trace masking with replacement | paper-explicit |
| 1 to 0.5×n_traces observed traces | paper-explicit |
| Normalization to [0,1] before masking | paper-explicit |
| Adam, lr 2e-4 | paper-explicit |
| Batch size 4 | paper-explicit |
| 128×128 patches | paper-explicit |
| Same-padding (Keras compatible) | reproduction-assumption |
| Kernel size 3×3 for decoders | reproduction-assumption |
| Upsample mode (nearest) | reproduction-assumption |
| Kaiming uniform init for PConv weights | reproduction-assumption |
| Adam betas / weight decay | unresolved (torch defaults) |
| Number of epochs | reproduction-assumption (100) |
| SEG-Y volume / split | repository-adaptation |

## Verification status

- Author code is publicly available at https://github.com/shulinpan/seismic-data-regularization.
  Architecture (6 encoders, kernel sizes, channels, BN placement, LeakyReLU slope, Sigmoid
  output, composite loss weights) verified against author code.
- The implementation defaults to the **author-code global mask-mean** PConv normalization.
  Paper Eq. 6 describes local per-window normalization; this is available via
  `normalization_mode="standard_local_valid_ratio"` and the ablation config.
- Local machine has no PyTorch; syntax, YAML verified locally.
  Torch-dependent tests must be run on the training machine:

```bash
python -m pytest tests/test_pan2020_partial_conv.py \
                 tests/test_pan2020_pconv_architecture.py \
                 tests/test_pan2020_pconv_loss.py \
                 tests/test_pan2020_pconv_training.py \
                 tests/test_pan2020_pconv_configs.py -q
```

## Unmodified repository modules

Trainer (`utils/train_utils.py` loops), datasets, metrics, patching
(`tools/patching.py`), and all other model files are untouched.  The
trainer's `train_one_epoch` and `evaluate` functions received a
backward-compatible signature-inspection change so that `**extras`
(including the mask) is forwarded to models whose `forward` accepts a
`mask` parameter.
