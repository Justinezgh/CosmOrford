# Architecture & Training Optimization for CosmOrford

## Context

CosmOrford compresses weak lensing mass maps (1834x88 single-channel fields) through a vision backbone into an 8-dim bottleneck, then predicts cosmological parameters (Omega_m, S8) via a Gaussian prediction head. The bottleneck and head are fixed; this plan optimizes everything upstream.

Key constraint: only ~20k training simulations. Overfitting is the primary risk.

## Current Baseline

- **Backbones**: 6 torchvision models (EfficientNet-B0/V2-S, ConvNeXt Tiny/Small, ResNet18/34), all randomly initialized
- **Input handling**: single channel repeated to 3 channels
- **Augmentation**: noise injection, survey mask, random flips, cyclic shifts
- **LR schedule**: linear warmup + StepLR decay
- **Regularization**: dropout (0.2), weight decay (1e-5)
- **Optimizer**: AdamW
- **Training**: 30 epochs, batch size 128, 16-mixed precision

## Metric

Primary: `val_loss` (Gaussian NLL). Secondary: `val_mse` (MSE in original parameter space).

---

## Phase 1: Foundational Improvements

Apply all together as the new baseline, then validate against random-init.

### 1a. Pretrained Weights
- Pass `weights="DEFAULT"` to all torchvision backbone factories
- Expected to be the single largest improvement for 20k samples

### 1b. Single-Channel Input Adaptation
- At init, replace first conv layer: sum pretrained 3-channel weights into 1-channel weight
- Preserves pretrained feature quality, removes wasteful channel repetition

### 1c. Cosine Annealing LR Schedule
- Replace StepLR with CosineAnnealingLR + linear warmup
- Smoother decay, better generalization in practice

### 1d. Gradient Clipping
- Add `gradient_clip_val: 1.0` to trainer config

### Experiments
| ID | Description | Config |
|----|-------------|--------|
| P1-baseline | Current setup (random init, StepLR) | `configs/experiments/efficientnet_b0.yaml` |
| P1-pretrained | All Phase 1 changes on EfficientNet-B0 | `configs/experiments/phase1_baseline.yaml` |

---

## Phase 2: Backbone Sweep

All pretrained, using Phase 1 training setup. Identical hyperparameters except LR (tuned per backbone).

| ID | Backbone | Params | Why |
|----|----------|--------|-----|
| P2-effnet-b0 | EfficientNet-B0 | 5.3M | Small, strong baseline |
| P2-effnet-v2s | EfficientNet V2-S | 21M | Modern, better training efficiency |
| P2-convnext-t | ConvNeXt Tiny | 28M | Strong modern CNN |
| P2-resnet18 | ResNet-18 | 11M | Smallest, least overfit-prone |
| P2-swin-v2-t | Swin-V2-T | 28M | Hierarchical ViT, varied spatial scales |
| P2-maxvit-t | MaxViT-T | 31M | Hybrid CNN+attention |

Note: Swin-V2-T and MaxViT-T use window-based attention. The 1834x88 input may need padding to be compatible. If they fail or perform poorly due to geometry, drop them.

Top 2-3 by val_loss advance to Phase 3.

---

## Phase 3: Augmentation & Regularization

Test one change at a time against the best Phase 2 backbone.

| ID | Change | Details |
|----|--------|---------|
| P3-mixup | Mixup | alpha=0.2, interpolate both maps and targets |
| P3-cutmix | CutMix | Adapted for regression |
| P3-randeras | Random Erasing | p=0.25, scale=(0.02, 0.15) |
| P3-sam | SAM Optimizer | Sharpness-aware minimization, rho=0.05 |
| P3-stodepth | Stochastic Depth | If backbone supports it, p=0.1 |
| P3-wd | Stronger Weight Decay | Sweep: 1e-4, 1e-3 |
| P3-batchsize | Smaller Batch Size | 64, 32 (implicit regularization) |

---

## Phase 4: Best Combination

Combine winning backbone + augmentations + optimizer from Phases 2-3. Final sweep:
- Learning rate (3 values around Phase 2 best)
- Dropout rate (0.1, 0.2, 0.3)
- Training epochs: increase to 50-100 with early stopping (patience=5)

Produces the final recommended configuration.

---

## Tracking

All experiments logged to W&B project `cosmorford`. Each run tagged with its experiment ID (e.g., `P1-pretrained`). Results summarized in `docs/plans/experiment-log.md` after each phase.
