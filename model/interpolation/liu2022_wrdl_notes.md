# Liu et al. 2022 — WRDL (Wavelet-Based Residual Deep Learning)

## Paper

- **Title:** Seismic Data Reconstruction via Wavelet-Based Residual Deep Learning
- **Authors:** Naihao Liu, Lukun Wu, Jiale Wang, Hao Wu, Jinghuai Gao, Dehua Wang
- **Venue:** IEEE Transactions on Geoscience and Remote Sensing, vol. 60, 2022, article 4508213
- **DOI:** 10.1109/TGRS.2022.3152984
- **Affiliation:** Xi'an Jiaotong University

## Method Overview

WRDL is a supervised seismic data reconstruction network. It uses U-Net as the
backbone with three key innovations:

1. **DWT replaces pooling:** Discrete Wavelet Transform downsamples feature
   maps, decomposing them into four frequency subbands (LL, LH, HL, HH). The
   invertibility of DWT preserves details typically lost in max-pooling.

2. **IWT with expansion convolution replaces upsampling:** Expansion convolution
   produces 4×C_out channels (one set of subbands per output channel), and
   Inverse Wavelet Transform reconstructs the higher-resolution feature map.

3. **Bottleneck residual blocks:** 1×1→3×3→1×1 bottleneck blocks at the
   deepest level improve training stability and accuracy.

The loss is a hybrid of SSIM Loss (local window) and Huber Loss.

## Architecture

```
Input (B, 1, H, W) — incomplete seismic data, missing traces zero-filled
    |
  Stem: Conv3x3 → BN → ReLU — produces (B, 32, H, W)
    |
  Encoder 1: Conv3x3→BN→ReLU→Conv3x3→BN→ReLU → DWT → (B, 256, H/2, W/2)
    |                                                    skip: (B, 64, H, W)
  Encoder 2: Conv3x3→BN→ReLU→Conv3x3→BN→ReLU → DWT → (B, 512, H/4, W/4)
    |                                                    skip: (B, 128, H/2, W/2)
  Encoder 3: Conv3x3→BN→ReLU→Conv3x3→BN→ReLU → DWT → (B, 1024, H/8, W/8)
    |                                                    skip: (B, 256, H/4, W/4)
  Encoder 4: Conv3x3→BN→ReLU→Conv3x3→BN→ReLU → DWT → (B, 2048, H/16, W/16)
    |                                                    skip: (B, 512, H/8, W/8)
  Bottleneck: Conv3x3→BN→ReLU → Residual×2
    |
  Decoder 4: Expansion Conv → IWT → (B, 512, H/8, W/8) + skip → Conv×2
    |
  Decoder 3: Expansion Conv → IWT → (B, 256, H/4, W/4) + skip → Conv×2
    |
  Decoder 2: Expansion Conv → IWT → (B, 128, H/2, W/2) + skip → Conv×2
    |
  Decoder 1: Expansion Conv → IWT → (B, 64, H, W) + skip → Conv×2
    |
  Output: Conv1x1 → (B, 1, H, W)
```

## Wavelet Transform

- **Basis:** Haar (orthonormal, 1/√2 normalisation)
- **Implementation:** Grouped `conv2d` (stride 2) for DWT, grouped
  `conv_transpose2d` (stride 2) for IWT
- **Filters:** Fixed `register_buffer` (not trainable `nn.Parameter`)
- **Subband order:** LL, LH, HL, HH (channel-stacked)
- **Boundary:** Zero padding (no padding for stride-2 conv); odd-size inputs
  are reflect-padded to even before DWT and cropped back at output
- **Perfect reconstruction:** Verified — `IWT(DWT(x)) = x` to within 1e-6

## DWT Mathematical Definition (Haar)

1D kernels (orthonormal):
```
L = 1/√2 * [1, 1]
H = 1/√2 * [-1, 1]
```

2D analysis kernels (K = L^T L, etc.):
```
K_LL = [[ 0.5,  0.5], [ 0.5,  0.5]]
K_LH = [[-0.5,  0.5], [-0.5,  0.5]]
K_HL = [[-0.5, -0.5], [ 0.5,  0.5]]
K_HH = [[ 0.5, -0.5], [-0.5,  0.5]]
```

For orthogonal Haar, synthesis filters = analysis filters (the adjoint equals
the inverse due to orthogonality: A^T A = I).

## Bottleneck Residual Block

```
x → 1×1 Conv(C//4) → BN → ReLU
  → 3×3 Conv(C//4) → BN → ReLU
  → 1×1 Conv(C) → BN
  → + x → ReLU
```

## Expansion Convolution

```
Input (B, C_in, H, W) → 3×3 Conv(4×C_out) → BN → ReLU → IWT → (B, C_out, 2H, 2W)
```

## Loss: SSIM + Huber

```
L_total = ssim_weight × (1 - SSIM) + huber_weight × Huber

Huber(e) = { 0.5 × e²              if |e| ≤ δ
           { δ × (|e| - 0.5×δ)     if |e| > δ

SSIM: local Gaussian window (11×11, σ=1.5), per-channel, mean reduction
```

## Input/Output Convention

- **Input:** Incomplete seismic data with missing traces zero-filled (1 channel)
- **Output:** Reconstructed seismic data (1 channel)
- **No mask channel:** WRDL does not take a separate mask input
- **Global residual:** Not used (output is the direct prediction)

## Parameter Classification

| Setting                          | Classification              |
| -------------------------------- | --------------------------- |
| U-Net backbone                   | paper-explicit              |
| DWT replaces pooling             | paper-explicit              |
| IWT restores feature maps        | paper-explicit              |
| Expansion convolution before IWT | paper-explicit              |
| Residual blocks                  | paper-explicit              |
| Bottleneck residual block        | paper-explicit (keyword)    |
| SSIM + Huber loss                | paper-explicit              |
| Irregular missing experiment     | paper-explicit              |
| Consecutive missing experiment   | paper-explicit              |
| Exact wavelet basis (Haar)       | reproduction-assumption     |
| Encoder channels (32-512)        | reproduction-assumption     |
| Number of levels (5)             | reproduction-assumption     |
| Residual block count (2)         | reproduction-assumption     |
| Bottleneck ratio (4)             | reproduction-assumption     |
| Huber delta (1.0)                | reproduction-assumption     |
| SSIM weight (1.0)                | reproduction-assumption     |
| Huber weight (1.0)               | reproduction-assumption     |
| SSIM window size (11)            | reproduction-assumption     |
| Learning rate (1e-3)             | reproduction-assumption     |
| Batch size (4)                   | reproduction-assumption     |
| Epochs (100)                     | reproduction-assumption     |
| Patch size (128×128)             | reproduction-assumption     |
| Optimizer (Adam)                 | reproduction-assumption     |
| Data normalization (max_abs)     | repository-adaptation       |
| Odd-size padding                 | repository-adaptation       |
| No mask input channel            | reproduction-assumption     |

## Implementation Notes

- The paper PDF could not be independently audited (behind IEEE paywall, no
  author code available). All parameters marked `reproduction-assumption` use
  conservative defaults and must be updated after paper audit.
- The `liu2022_wrdl_conservative.yaml` config must NOT be renamed to
  `liu2022_wrdl_paper.yaml` until all unresolved parameters are verified.
- DWT and IWT use the same Haar filters (orthogonal wavelet: analysis filters
  equal synthesis filters).
- The model does NOT accept a separate mask input — it is a direct supervised
  mapping from incomplete (zero-filled) data to complete data.
- Unlike Pan2020 PConv, WRDL does not need mask passing in the trainer. The
  signature inspection in `train_utils.py` handles this automatically.

## Ablation Configs

| Config | What it tests |
|--------|--------------|
| `liu2022_wrdl_no_residual_ablation` | WRDL with `bottleneck_blocks=0` |
| `liu2022_wrdl_pooling_ablation` | Standard U-Net with MaxPool (same loss) |
| `liu2022_wrdl_ssim_ablation` | SSIM-only loss (`huber_weight=0`) |
| `liu2022_wrdl_huber_ablation` | Huber-only loss (`ssim_weight=0`) |

## References

- Liu et al., "Seismic Data Reconstruction via Wavelet-Based Residual Deep
  Learning," IEEE TGRS, vol. 60, 2022, DOI: 10.1109/TGRS.2022.3152984.
- MWCNN (Liu et al., 2018) — related wavelet CNN but different architecture.
- PyTorch `conv2d` grouped convolution for DWT/IWT implementation pattern.
