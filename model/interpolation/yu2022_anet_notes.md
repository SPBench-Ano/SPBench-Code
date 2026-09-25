# Yu2022 ANet — Reproduction Notes

## Paper

- **Title**: Attention and Hybrid Loss Guided Deep Learning for Consecutively Missing Seismic Data Reconstruction
- **Authors**: Jiaxu Yu, Bangyu Wu
- **Venue**: IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–8, 2022 (Early Access 2021)
- **DOI**: [10.1109/TGRS.2021.3068279](https://doi.org/10.1109/TGRS.2021.3068279)

## What ANet is (and is not)

ANet is a plain feed-forward CNN with one exact non-local attention module — **not**
an Attention U-Net, CBAM-U-Net, SE network, or Transformer. There are no
encoder–decoder skip connections, no multi-head attention, no LayerNorm/MLP
blocks, and no positional encoding. The single attention module computes the
full spatial correlation matrix so that wide consecutive gaps can be filled
from globally similar structures.

```text
Input [B,1,H,W]  (consecutively missing patch; NO mask channel)
      │
Down Conv 1  (stride 2, 1 -> 64)   + BN + ReLU
      │
Down Conv 2  (stride 2, 64 -> 128) + BN + ReLU
      │
Residual Block × 6  (two 3×3 convs + identity shortcut each, 128 ch)
      │
Non-local Attention  (1×1 Q/K/V convs, softmax(QK)V, 1×1 out conv, +x)
      │
Upsample ×2 + Conv (128 -> 64) + BN + ReLU
      │
Upsample ×2 + Conv (64 -> 32)  + BN + ReLU
      │
Final Conv (32 -> 1, no BN, no activation)
      │
Output [B,1,H,W]
```

Location: `model/interpolation/yu2022_anet.py`, registered as `yu2022_anet`.

## Non-local attention (paper formulation)

For features `x` with `N = H·W` positions:

```text
Q = W_q x   (B, N, Cq)      K = W_k x   (B, Cq, N)      V = W_v x   (B, N, C)
A = softmax(Q K)            (B, N, N), rows sum to 1, no 1/sqrt(C) scaling
out = x + W_o(A V)
```

Differences from Transformer attention (deliberate): no multi-head split, no
logit scaling by default (`attention_scaling: none`, paper-inferred), no
LayerNorm, no MLP, no positional encoding, no trainable gamma. Q/K keep the
full channel count (no C/8 reduction — that is a SAGAN/non-local-block
convention, not this paper's).

`attention_query_chunk_size` chunks the query dimension only; every chunk
still attends over all keys, so results are mathematically identical
(repository adaptation for the 720×120 field patches where N = 5400).

## Hybrid loss (paper formulation)

```text
L_SSIM   = -SSIM(pred, target)
L_L1     = mean |pred - target|
L_hybrid = L_SSIM + lambda_l1 * L_L1        lambda_l1 = 1 (paper-explicit)
```

Location: `utils/losses.py::ANetSSIML1Loss`, registered as `anet_ssim_l1`.

- SSIM follows the paper's Eq. (1)–(6) merged form with `c3 = c2/2`, computed
  from **per-sample global-patch statistics** (mean/variance/covariance over
  the whole patch). This is a literal equation implementation; whether the
  paper used windowed SSIM is unresolved.
- The loss equals **-1** at perfect reconstruction. This is intentional and
  must not be shifted to `1 - SSIM` (same gradients, different logged value).
- `lambda_l1` weights the **L1 term**, not the SSIM term. λ = 1 works because
  the data is min-max normalized to [0, 1], giving both terms similar ranges.
- The loss is computed on the **full patch** (observed + missing regions),
  with no mask argument.
- Monitoring: `components(pred, target)` returns
  `loss_total / loss_ssim / loss_l1 / ssim` tensors; the trainer contract
  (`forward` returns one Tensor) is unchanged (repository adaptation of the
  dict-return suggestion).

## Consecutive missing-trace construction

`tools/preprocessing.py::mask_traces(mode="continuous", ratio_range=(0.1, 0.3))`
draws each patch's missing fraction uniformly from the range, removes one
contiguous block of **whole traces** (time dimension fully removed), with a
uniformly sampled start position. `True = missing` in the returned mask.
Masks are generated once at dataset build time with a seeded RNG
(`experiment.seed`), so they are fixed across epochs and reproducible across
runs — a repository adaptation of the paper's unspecified regeneration policy;
validation masks are therefore automatically fixed.

## Paper profiles

| Setting | Synthetic (SEG C3) | Field (Mobil AVO Viking Graben) |
|---|---|---|
| Config | `configs/interpolation/yu2022_anet_seg_c3_paper.yaml` | `configs/interpolation/yu2022_anet_field_paper.yaml` |
| Patch | 128×128 | 720 time × 120 traces |
| Missing | 10%–30% consecutive traces | same |
| Normalization | min-max [0, 1] | same |
| Optimizer | Adam, lr 1e-3 | same |
| Batch size | 32 | 8 |
| Epochs | 20 | same |
| Scheduler | none | same |

MSE ablation (`configs/interpolation/yu2022_anet_mse_ablation.yaml`): identical
network/data/optimizer, only `loss: mse` — reproduces the paper's
ANet+MSE vs ANet+SSIM+L1 comparison.

Training command (user-executed):

```bash
python scripts/interpolation/train_interpolation_unet.py \
    --config configs/interpolation/yu2022_anet_seg_c3_paper.yaml
```

## Inference

The paper profile evaluates the **raw network output**
(`inference.overwrite_observed: false`). Observed-trace overwriting
`M·x_obs + (1-M)·pred` improves known-region consistency but changes the
full-patch metrics, so it must be reported separately if enabled.

## Setting classification

| Setting | Classification |
| --- | --- |
| Two downsampling convolutions | paper-explicit |
| Scaling factor 2 (two stride-2 stages) | paper-explicit |
| First conv channels 64 | paper-explicit |
| Channel doubling (64 → 128) | paper-explicit |
| Six residual blocks (2 convs + shortcut) | paper-explicit |
| One non-local attention module | paper-explicit |
| Two upsampling + convolution groups | paper-explicit |
| BN + ReLU except upsampling / final layers | paper-explicit |
| Q/K/V 1×1 convolutions | paper-explicit |
| Full spatial attention matrix | paper-explicit |
| Attention residual output | paper-explicit |
| SSIM + L1 hybrid loss | paper-explicit |
| Negative SSIM (loss = -1 at optimum) | paper-explicit |
| lambda_l1 = 1 | paper-explicit |
| Min-max normalization to [0, 1] | paper-explicit |
| 128×128 synthetic / 720×120 field patches | paper-explicit |
| 10%–30% consecutive missing | paper-explicit |
| Adam, lr 1e-3, batch 32/8, 20 epochs | paper-explicit |
| No mask input channel | paper-explicit |
| Kernel size 3×3 | reproduction-assumption (Fig. 3 / Table V unverified) |
| Decoder channels (64, 32) | reproduction-assumption |
| Nearest + conv upsampling | reproduction-assumption |
| Conv → BN → ReLU order; post-add ReLU | reproduction-assumption |
| Attention channels = full C | paper-inferred |
| No attention logit scaling | paper-inferred |
| No scheduler | paper-inferred |
| SSIM global-patch scope; c1=1e-4, c2=9e-4 | reproduction-assumption |
| Adam betas / weight decay | unresolved (torch defaults) |
| Normalization scope (global stats) | repository-adaptation |
| Per-patch U(0.1, 0.3) ratio sampling | reproduction-assumption |
| Static per-patch masks at dataset build | repository-adaptation |
| Zero padding for arbitrary sizes | repository-adaptation |
| Attention query chunking | repository-adaptation |
| SEG-Y volume / split (vs paper's 25k/10k/10k patches) | repository-adaptation |

## Verification status

- Paper PDF is IEEE-paywalled and no official code repository was found;
  Fig. 1/Fig. 3 per-layer details (kernels, decoder channels, upsampling API)
  could not be machine-verified. Structure above matches the paper text and
  secondary descriptions (two squeeze convs → six residual blocks → attention
  → two upsampling groups; checkerboard artifacts mitigated by post-upsample
  convolution).
- Local machine has no PyTorch; syntax, YAML, and numpy mask logic verified
  locally. Torch-dependent tests
  (`tests/test_yu2022_anet_{architecture,attention,loss,training,configs}.py`)
  must be run on the training machine:

```bash
python -m pytest tests/test_yu2022_anet_architecture.py \
                 tests/test_yu2022_anet_attention.py \
                 tests/test_yu2022_anet_loss.py \
                 tests/test_yu2022_anet_training.py \
                 tests/test_yu2022_anet_configs.py -q
```

## Unmodified repository modules

Trainer (`utils/train_utils.py` loops), datasets, metrics, all other model
files, and all other configs are untouched. The only shared-code changes are
the backward-compatible `ratio_range` keyword in `mask_traces` (legacy
single-ratio path byte-identical) and the `mask_ratio_range` pass-through +
experiment-name suffix in `scripts/interpolation/train_interpolation_unet.py`.
