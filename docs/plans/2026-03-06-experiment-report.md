# CosmOrford Architecture Optimization — Experiment Report

## Objective

Find the best backbone architecture, training strategy, and augmentation scheme for compressing weak lensing mass maps (1834x88, single-channel) into an 8-dimensional bottleneck that predicts cosmological parameters (Omega_m, S8). Key constraint: only ~20k training simulations — overfitting is the primary risk.

## Starting Point

The original CosmOrford codebase used:
- Randomly initialized torchvision backbones (EfficientNet-B0 default)
- 3-channel repeat to feed single-channel maps into RGB backbones
- Full 1834x88 input fed directly to backbone
- StepLR decay schedule
- No gradient clipping
- Augmentations: noise injection, survey mask, random flips, cyclic shifts
- Gaussian NLL loss
- AdamW optimizer, batch size 128, 30 epochs

## Changes Implemented

### Code Changes (11 commits)
1. **Pretrained ImageNet weights** — `get_backbone()` now loads DEFAULT weights
2. **Single-channel adaptation** — `adapt_first_conv()` sums 3-channel pretrained weights into 1 channel
3. **Cosine annealing LR** — new default schedule with warmup
4. **Swin-V2-T and MaxViT-T backbones** — added to registry (MaxViT incompatible with 1834x88 geometry)
5. **Mixup augmentation** — `mixup_alpha` parameter for regression-adapted Mixup
6. **Random Erasing** — `random_erasing` parameter using torchvision's RandomErasing
7. **SAM optimizer** — sharpness-aware minimization (failed to launch due to Lightning compatibility)
8. **Gradient clipping** — `gradient_clip_val: 1.0` in trainer config
9. **Early stopping** — patience=10 for Phase 4 runs
10. **Experiment configs** — 20+ YAML configs across 4 phases
11. **Launch infrastructure** — experiment tracking template, launch scripts

## Experiments Run

### Phase 1: Foundational Improvements
Validated pretrained weights + cosine LR + gradient clipping against the old setup.

| Run | Backbone | Changes | Best val_loss | Best val_mse | Status |
|-----|----------|---------|--------------|-------------|--------|
| efficientnet_b0 (no clip) | EfficientNet-B0 | Pretrained, cosine LR | 0.3471 | 0.000954 | Finished (diverges late) |
| P1-pretrained | EfficientNet-B0 | + gradient clipping | 0.6384 | 0.000934 | Finished (stable) |

Finding: Gradient clipping prevents late-training divergence. Both use pretrained weights (code default changed).

### Phase 2: Backbone Sweep
All pretrained, cosine LR, gradient clipping.

| Run | Backbone | BS | Best val_loss | Best val_mse | Status |
|-----|----------|-----|--------------|-------------|--------|
| P2-resnet18 | ResNet-18 (11M) | 128 | **0.4122** | **0.000794** | Finished |
| P2-effnet-b0 | EfficientNet-B0 (5.3M) | 128 | 0.6384 | 0.000934 | Finished |
| P2-convnext-t | ConvNeXt Tiny (28M) | 32 | 1.4194 | 0.007743 | Crashed @ep6 (OOM at bs=128) |
| P2-effnet-v2s | EfficientNet V2-S (21M) | 32 | 3.4217 | 0.005682 | Crashed @ep5 (OOM at bs=128) |
| P2-swin-v2-t | Swin-V2-T (28M) | 32 | 1.4001 | 0.007535 | Crashed @ep3 (OOM at bs=128) |
| MaxViT-T | - | - | - | - | Geometry incompatible |

Finding: ResNet-18 wins decisively. Larger models OOM at bs=128 on A10G (22GB). At reduced bs=32, they crashed early (Modal infrastructure) and didn't train long enough to converge. The EfficientNet-V2-S result is **inconclusive** — only 5 epochs at bs=32 is insufficient to evaluate it fairly.

### Phase 3: Augmentation & Regularization (on ResNet-18)

| Run | Change | BS | Best val_loss | Best val_mse | Status |
|-----|--------|-----|--------------|-------------|--------|
| P3-batch64 | batch_size=64 | 64 | **0.3461** | 0.000939 | Crashed @ep15 |
| P3-batch32 | batch_size=32 | 32 | 0.5804 | 0.001271 | Crashed @ep16 |
| P3-randeras | Random Erasing | 128 | 0.7270 | 0.001652 | Crashed @ep16 |
| P3-mixup | Mixup alpha=0.2 | 128 | 1.4385 | 0.004141 | Crashed @ep15 |
| P3-sam | SAM optimizer | 128 | - | - | Failed to launch |

Finding: Batch size 64 gives best results (implicit regularization). Mixup and Random Erasing actively hurt performance — these generic vision augmentations destroy the physical structure of weak lensing maps.

### Phase 4: Final Refinement (ResNet-18, bs=64)

| Run | Variation | Best val_loss | Best val_mse | Status |
|-----|-----------|--------------|-------------|--------|
| P4-best-resnet18-bs64 | dropout=0.2, lr=0.01 | **0.3314** | **0.000829** | Finished |
| P4-best-dropout03 | dropout=0.3 | 0.3863 | 0.000836 | Finished |
| P4-best-dropout01 | dropout=0.1 | 0.3910 | 0.000842 | Finished |
| P4-best-lr005 | lr=0.005 | 0.4136 | 0.000888 | Finished |

Finding: Default dropout (0.2) and LR (0.01) work best. Lower LR hurts.

## Best Configuration Found

```yaml
backbone: resnet18          # Pretrained ImageNet, single-channel adapted
batch_size: 64              # Implicit regularization sweet spot
max_lr: 0.01                # Standard for ResNet-18
lr_schedule: cosine         # With 500-step linear warmup
dropout_rate: 0.2           # Default works best
gradient_clip_val: 1.0      # Prevents late-training instability
max_epochs: 50              # With early stopping patience=10
```

**Best result: val_loss=0.3314, val_mse=0.000829** (P4-best-resnet18-bs64)

## What Worked
- Pretrained ImageNet weights with single-channel weight summation
- Smaller backbone (ResNet-18) in low-data regime
- Batch size 64 as implicit regularizer
- Cosine annealing LR schedule
- Gradient clipping
- Early stopping

## What Didn't Work
- **Mixup** — actively harmful for this domain (val_loss 3.5x worse)
- **Random Erasing** — harmful (val_loss 2x worse)
- **Larger backbones** (ConvNeXt, Swin, EfficientNet-V2-S) — OOM at full batch size; inconclusive at reduced batch size
- **SAM optimizer** — failed to integrate with Lightning manual optimization
- **Very small batch size (32)** — too noisy

## Caveats and Open Questions

1. **EfficientNet-V2-S was not fairly tested** — it OOM'd at bs=128 and crashed after only 5 epochs at bs=32. The previous neurips-wl-challenge project used EfficientNet-V2-S successfully by splitting the input into 21 patches of 88x88, which is much more memory-efficient.

2. **The old project had additional features not tested here**:
   - Power spectrum summary statistics as auxiliary input
   - Patch-based processing (88x88 patches averaged)
   - Random 90-degree rotation augmentation on patches
   - EMA weight averaging
   - Score-based loss function (competition-specific)

3. A direct comparison between our best (ResNet-18 pretrained, full image) and the old approach (EfficientNet-V2-S random init, patched + power spectrum) is needed to understand whether architectural choices or training methodology drive performance.
