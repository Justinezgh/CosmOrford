# Architecture & Training Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Systematically optimize CosmOrford's backbone, augmentation, and training strategy to minimize val_loss on 20k weak lensing samples without overfitting.

**Architecture:** Modify `backbones.py` to support pretrained weights + single-channel adaptation + new architectures. Extend `compressor.py` with cosine LR, Mixup/CutMix, Random Erasing, and SAM optimizer options. Create experiment configs for each phase. Launch on Modal, track in W&B.

**Tech Stack:** PyTorch, torchvision, Lightning, Modal, W&B

---

### Task 1: Pretrained Weights Support in Backbone Registry

**Files:**
- Modify: `cosmorford/backbones.py`
- Test: `tests/test_backbones.py` (create)

**Step 1: Write the failing tests**

```python
# tests/test_backbones.py
import torch
import pytest
from cosmorford.backbones import get_backbone, BACKBONES


def test_get_backbone_pretrained():
    """Pretrained backbone returns features with expected output dim."""
    features, dim = get_backbone("resnet18", pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    out = features(x)
    assert out.shape[1] == dim


def test_get_backbone_random_init():
    """Random init still works when pretrained=False."""
    features, dim = get_backbone("resnet18", pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    out = features(x)
    assert out.shape[1] == dim


def test_get_backbone_unknown_raises():
    with pytest.raises(ValueError, match="Unknown backbone"):
        get_backbone("nonexistent")
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_backbones.py -v`
Expected: FAIL — `get_backbone()` does not accept `pretrained` parameter

**Step 3: Implement pretrained weight support**

Replace `cosmorford/backbones.py` with:

```python
"""Backbone registry for vision feature extractors."""
import torch.nn as nn
from torchvision.models.efficientnet import (
    efficientnet_b0, efficientnet_v2_s,
    EfficientNet_B0_Weights, EfficientNet_V2_S_Weights,
)
from torchvision.models.convnext import (
    convnext_tiny, convnext_small,
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights,
)
from torchvision.models.resnet import (
    resnet18, resnet34,
    ResNet18_Weights, ResNet34_Weights,
)

BACKBONES = {
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, 1280),
    "efficientnet_v2_s": (efficientnet_v2_s, EfficientNet_V2_S_Weights.DEFAULT, 1280),
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT, 768),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT, 768),
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT, 512),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT, 512),
}


def get_backbone(name: str, pretrained: bool = True):
    """Return (feature_extractor, output_dim) for a given backbone name."""
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(BACKBONES.keys())}")

    factory, default_weights, out_dim = BACKBONES[name]
    weights = default_weights if pretrained else None
    model = factory(weights=weights)

    if name.startswith("efficientnet"):
        features = model.features
    elif name.startswith("convnext"):
        features = model.features
    elif name.startswith("resnet"):
        features = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        )

    return features, out_dim
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_backbones.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cosmorford/backbones.py tests/test_backbones.py
git commit -m "feat: add pretrained weight support to backbone registry"
```

---

### Task 2: Single-Channel Input Adaptation

**Files:**
- Modify: `cosmorford/backbones.py`
- Modify: `cosmorford/compressor.py`
- Test: `tests/test_backbones.py` (extend)

**Step 1: Write the failing test**

Append to `tests/test_backbones.py`:

```python
def test_adapt_first_conv_single_channel():
    """adapt_first_conv converts 3-channel first conv to 1-channel."""
    from cosmorford.backbones import get_backbone, adapt_first_conv
    features, dim = get_backbone("resnet18", pretrained=True)
    features = adapt_first_conv(features, "resnet18")
    x = torch.randn(1, 1, 224, 224)
    out = features(x)
    assert out.shape[1] == dim


def test_adapt_first_conv_efficientnet():
    from cosmorford.backbones import get_backbone, adapt_first_conv
    features, dim = get_backbone("efficientnet_b0", pretrained=True)
    features = adapt_first_conv(features, "efficientnet_b0")
    x = torch.randn(1, 1, 224, 224)
    out = features(x)
    assert out.shape[1] == dim
```

**Step 2: Run test to verify it fails**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_backbones.py::test_adapt_first_conv_single_channel -v`
Expected: FAIL — `adapt_first_conv` does not exist

**Step 3: Implement adapt_first_conv**

Add to `cosmorford/backbones.py`:

```python
import torch

def adapt_first_conv(features, name: str):
    """Replace first conv layer from 3-channel to 1-channel input.

    Sums the pretrained 3-channel weights into a single channel,
    preserving pretrained feature quality.
    """
    if name.startswith("resnet"):
        old_conv = features[0]  # conv1
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        features[0] = new_conv

    elif name.startswith("efficientnet"):
        old_conv = features[0][0]  # features[0] is the first ConvBNActivation block
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        features[0][0] = new_conv

    elif name.startswith("convnext"):
        old_conv = features[0][0]  # features[0] is the stem
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        features[0][0] = new_conv

    return features
```

Then update `cosmorford/compressor.py` `__init__` to use it — replace the `get_backbone` call:

```python
from cosmorford.backbones import get_backbone, adapt_first_conv

# In __init__:
features, feat_dim = get_backbone(backbone, pretrained=True)
self.backbone = adapt_first_conv(features, backbone)
```

And remove the channel-repeat logic in `_features()` — replace:
```python
def _features(self, x):
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return self.pool(self.backbone(x.float()))
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_backbones.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cosmorford/backbones.py cosmorford/compressor.py tests/test_backbones.py
git commit -m "feat: single-channel input adaptation for pretrained backbones"
```

---

### Task 3: Cosine Annealing LR Schedule

**Files:**
- Modify: `cosmorford/compressor.py`
- Test: `tests/test_compressor.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_compressor.py
import torch
import pytest


def test_compressor_cosine_schedule():
    """CompressorModel with lr_schedule='cosine' creates CosineAnnealingLR."""
    from cosmorford.compressor import CompressorModel
    model = CompressorModel(backbone="resnet18", lr_schedule="cosine")
    assert model.hparams.lr_schedule == "cosine"


def test_compressor_step_schedule():
    """CompressorModel with lr_schedule='step' uses StepLR (backward compat)."""
    from cosmorford.compressor import CompressorModel
    model = CompressorModel(backbone="resnet18", lr_schedule="step")
    assert model.hparams.lr_schedule == "step"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_compressor.py -v`
Expected: FAIL — `lr_schedule` is not a valid parameter

**Step 3: Add lr_schedule parameter to CompressorModel**

In `cosmorford/compressor.py`, update `__init__` signature to add `lr_schedule: str = "cosine"`.

Update `configure_optimizers` to support both schedules:

```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.max_lr, weight_decay=1e-5)
    total_steps = int(self.trainer.estimated_stepping_batches)
    warmup_steps = self.hparams.warmup_steps

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
    )

    if self.hparams.lr_schedule == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps
        )
    else:  # "step"
        steps_per_epoch = total_steps // self.trainer.max_epochs
        step_size = self.hparams.decay_every_epochs * steps_per_epoch
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=self.hparams.decay_rate
        )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, main_scheduler], milestones=[warmup_steps]
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_compressor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cosmorford/compressor.py tests/test_compressor.py
git commit -m "feat: add cosine annealing LR schedule option"
```

---

### Task 4: Phase 1 Experiment Config

**Files:**
- Create: `configs/experiments/phase1_baseline.yaml`

**Step 1: Create config**

```yaml
# configs/experiments/phase1_baseline.yaml
model:
  class_path: cosmorford.compressor.CompressorModel
  init_args:
    backbone: "efficientnet_b0"
    bottleneck_dim: 8
    warmup_steps: 500
    max_lr: 0.008
    decay_rate: 0.85
    dropout_rate: 0.2
    lr_schedule: "cosine"

data:
  class_path: cosmorford.dataset.WLDataModule
  init_args:
    batch_size: 128
    num_workers: 8

trainer:
  max_epochs: 30
  accelerator: gpu
  precision: "16-mixed"
  gradient_clip_val: 1.0
  log_every_n_steps: 1
  check_val_every_n_epoch: 1
  callbacks:
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: "step"
    - class_path: ModelCheckpoint
      init_args:
        monitor: "val_loss"
        mode: "min"
        save_top_k: 3
        save_last: true
  logger:
    class_path: WandbLogger
    init_args:
      name: "P1-pretrained"
      project: "cosmorford"
      tags: ["phase1", "pretrained", "cosine-lr", "single-channel"]
      log_model: true
```

**Step 2: Verify config parses**

Run: `cd /home/francois/repo/CosmOrford && python -c "from cosmorford.trainer import trainer_cli; trainer_cli(['fit', '--config', 'configs/experiments/phase1_baseline.yaml', '--print_config'], run=False)"`
Expected: prints config without error

**Step 3: Commit**

```bash
git add configs/experiments/phase1_baseline.yaml
git commit -m "feat: add Phase 1 experiment config (pretrained + cosine LR)"
```

---

### Task 5: Add Swin-V2-T and MaxViT-T Backbones

**Files:**
- Modify: `cosmorford/backbones.py`
- Test: `tests/test_backbones.py` (extend)

**Step 1: Write the failing tests**

Append to `tests/test_backbones.py`:

```python
def test_swin_v2_t_backbone():
    features, dim = get_backbone("swin_v2_t", pretrained=True)
    x = torch.randn(1, 3, 256, 256)  # Swin needs sizes divisible by window
    out = features(x)
    assert out.shape[1] == dim


def test_maxvit_t_backbone():
    features, dim = get_backbone("maxvit_t", pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    out = features(x)
    assert out.shape[1] == dim


def test_adapt_first_conv_swin():
    from cosmorford.backbones import adapt_first_conv
    features, dim = get_backbone("swin_v2_t", pretrained=True)
    features = adapt_first_conv(features, "swin_v2_t")
    x = torch.randn(1, 1, 256, 256)
    out = features(x)
    assert out.shape[1] == dim


def test_adapt_first_conv_maxvit():
    from cosmorford.backbones import adapt_first_conv
    features, dim = get_backbone("maxvit_t", pretrained=True)
    features = adapt_first_conv(features, "maxvit_t")
    x = torch.randn(1, 1, 224, 224)
    out = features(x)
    assert out.shape[1] == dim
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_backbones.py::test_swin_v2_t_backbone -v`
Expected: FAIL — `swin_v2_t` not in BACKBONES

**Step 3: Add Swin-V2-T and MaxViT-T to registry**

Add imports and entries to `cosmorford/backbones.py`:

```python
from torchvision.models.swin_transformer import swin_v2_t, Swin_V2_T_Weights
from torchvision.models.maxvit import maxvit_t, MaxVit_T_Weights

# Add to BACKBONES dict:
"swin_v2_t": (swin_v2_t, Swin_V2_T_Weights.DEFAULT, 768),
"maxvit_t": (maxvit_t, MaxVit_T_Weights.DEFAULT, 512),
```

Add feature extraction logic in `get_backbone`:

```python
elif name.startswith("swin"):
    features = model.features
    # Swin outputs [B, H, W, C] — need permute + flatten
    # We'll wrap it to output [B, C, H, W] for compatibility with AdaptiveAvgPool2d
    class SwinFeatureWrapper(nn.Module):
        def __init__(self, swin_features, norm):
            super().__init__()
            self.features = swin_features
            self.norm = norm
            self.permute = model.permute
        def forward(self, x):
            x = self.features(x)
            x = self.norm(x)
            x = self.permute(x)
            return x
    features = SwinFeatureWrapper(model.features, model.norm, model.permute)

elif name.startswith("maxvit"):
    features = model.features  # MaxViT features block
```

Add `adapt_first_conv` support for swin and maxvit — inspect the first conv layer at runtime to find it:

```python
elif name.startswith("swin"):
    old_conv = features.features[0][0]  # patch embedding conv
    # ... same pattern: create new 1-channel conv, sum weights
    features.features[0][0] = new_conv

elif name.startswith("maxvit"):
    old_conv = features[0][0]  # stem first conv
    # ... same pattern
    features[0][0] = new_conv
```

Note: The exact layer paths need verification at implementation time by inspecting `print(model)`. The implementer should run `python -c "from torchvision.models import swin_v2_t; print(swin_v2_t())"` to find the correct paths.

**Step 4: Run all backbone tests**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_backbones.py -v`
Expected: PASS

**Step 5: Test with actual input geometry**

Run a quick smoke test with the real 1834x88 input shape to verify no crashes:
```python
python -c "
import torch
from cosmorford.backbones import get_backbone, adapt_first_conv
for name in ['swin_v2_t', 'maxvit_t']:
    f, d = get_backbone(name, pretrained=True)
    f = adapt_first_conv(f, name)
    # Pad 1834x88 to nearest valid size for window attention
    x = torch.randn(1, 1, 1834, 88)
    try:
        out = f(x)
        print(f'{name}: OK, shape={out.shape}')
    except Exception as e:
        print(f'{name}: FAIL — {e}')
"
```

If Swin/MaxViT fails on 1834x88 geometry, add padding logic in `CompressorModel._features()` or drop those backbones.

**Step 6: Commit**

```bash
git add cosmorford/backbones.py tests/test_backbones.py
git commit -m "feat: add Swin-V2-T and MaxViT-T backbone support"
```

---

### Task 6: Phase 2 Experiment Configs

**Files:**
- Create: `configs/experiments/phase2_effnet_b0.yaml`
- Create: `configs/experiments/phase2_effnet_v2s.yaml`
- Create: `configs/experiments/phase2_convnext_t.yaml`
- Create: `configs/experiments/phase2_resnet18.yaml`
- Create: `configs/experiments/phase2_swin_v2_t.yaml`
- Create: `configs/experiments/phase2_maxvit_t.yaml`

**Step 1: Create configs**

All identical to `phase1_baseline.yaml` except backbone name, W&B run name/tags, and per-backbone LR:

| Backbone | max_lr | Rationale |
|----------|--------|-----------|
| efficientnet_b0 | 0.008 | Phase 1 value |
| efficientnet_v2_s | 0.004 | Larger model, lower LR |
| convnext_tiny | 0.004 | Larger model, lower LR |
| resnet18 | 0.01 | Small model, can tolerate higher LR |
| swin_v2_t | 0.002 | Transformers need lower LR |
| maxvit_t | 0.002 | Transformers need lower LR |

Each config has `tags: ["phase2", "<backbone-name>"]` for W&B filtering.

**Step 2: Verify all configs parse**

Run a loop to check each config parses.

**Step 3: Commit**

```bash
git add configs/experiments/phase2_*.yaml
git commit -m "feat: add Phase 2 backbone sweep experiment configs"
```

---

### Task 7: Mixup Augmentation for Regression

**Files:**
- Modify: `cosmorford/compressor.py`
- Test: `tests/test_compressor.py` (extend)

**Step 1: Write the failing test**

Append to `tests/test_compressor.py`:

```python
def test_mixup_augmentation():
    """Mixup produces interpolated inputs and targets."""
    from cosmorford.compressor import mixup_data
    x = torch.randn(8, 1, 64, 64)
    y = torch.randn(8, 2)
    x_mixed, y_mixed = mixup_data(x, y, alpha=0.2)
    assert x_mixed.shape == x.shape
    assert y_mixed.shape == y.shape
    # Mixed data should differ from original (with high probability)
    assert not torch.allclose(x_mixed, x)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_compressor.py::test_mixup_augmentation -v`

**Step 3: Implement mixup_data function**

Add to `cosmorford/compressor.py`:

```python
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for regression: interpolate both inputs and targets."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    x_mixed = lam * x + (1 - lam) * x[index]
    y_mixed = lam * y + (1 - lam) * y[index]
    return x_mixed, y_mixed
```

Add `mixup_alpha: float = 0.0` to `CompressorModel.__init__` and use in `training_step`:

```python
def training_step(self, batch, batch_idx):
    x, y = batch
    x = self._augment(x)
    if self.hparams.mixup_alpha > 0:
        x_in = x.unsqueeze(1) if x.dim() == 3 else x
        x_in, y = mixup_data(x_in, y, self.hparams.mixup_alpha)
        x = x_in.squeeze(1) if x.dim() == 4 and x_in.size(1) == 1 else x_in
    mean, std = self(x)
    loss = -torch.distributions.Normal(loc=mean, scale=std).log_prob(y).mean()
    self.log("train_loss", loss)
    return loss
```

**Step 4: Run tests**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_compressor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cosmorford/compressor.py tests/test_compressor.py
git commit -m "feat: add Mixup augmentation for regression"
```

---

### Task 8: Random Erasing Augmentation

**Files:**
- Modify: `cosmorford/compressor.py`

**Step 1: Add `random_erasing: bool = False` to `CompressorModel.__init__`**

In `_augment`, add at the end:

```python
if self.hparams.random_erasing:
    from torchvision.transforms import RandomErasing
    eraser = RandomErasing(p=0.25, scale=(0.02, 0.15), value=0)
    x = x.unsqueeze(1)  # [B, 1, H, W]
    x = torch.stack([eraser(x[i]) for i in range(x.size(0))])
    x = x.squeeze(1)
```

Note: RandomErasing should be initialized once in `__init__`, not per call. Move it there.

**Step 2: Commit**

```bash
git add cosmorford/compressor.py
git commit -m "feat: add optional Random Erasing augmentation"
```

---

### Task 9: SAM Optimizer

**Files:**
- Create: `cosmorford/sam.py`
- Modify: `cosmorford/compressor.py`
- Test: `tests/test_sam.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_sam.py
import torch
from torch import nn


def test_sam_optimizer_step():
    """SAM optimizer performs two forward-backward passes."""
    from cosmorford.sam import SAM
    model = nn.Linear(10, 2)
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.01, rho=0.05)

    x = torch.randn(4, 10)
    # First forward-backward
    loss = model(x).sum()
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # Second forward-backward
    loss = model(x).sum()
    loss.backward()
    optimizer.second_step(zero_grad=True)
```

**Step 2: Implement SAM optimizer**

```python
# cosmorford/sam.py
"""Sharpness-Aware Minimization (SAM) optimizer."""
import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm
```

Add `use_sam: bool = False` and `sam_rho: float = 0.05` to `CompressorModel.__init__`.

SAM requires manual optimization in Lightning. When `use_sam=True`:
- Set `self.automatic_optimization = False` in `__init__`
- Override `training_step` to do two forward-backward passes manually

```python
# In training_step, when use_sam is True:
opt = self.optimizers()
sch = self.lr_schedulers()

# First forward-backward
x_aug = self._augment(x)
mean, std = self(x_aug)
loss = -torch.distributions.Normal(loc=mean, scale=std).log_prob(y).mean()
self.manual_backward(loss)
opt.first_step(zero_grad=True)

# Second forward-backward
mean, std = self(x_aug)
loss2 = -torch.distributions.Normal(loc=mean, scale=std).log_prob(y).mean()
self.manual_backward(loss2)
opt.second_step(zero_grad=True)
sch.step()

self.log("train_loss", loss)
return loss
```

**Step 3: Run tests**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_sam.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add cosmorford/sam.py cosmorford/compressor.py tests/test_sam.py
git commit -m "feat: add SAM optimizer support"
```

---

### Task 10: Phase 3 Experiment Configs

**Files:**
- Create: `configs/experiments/phase3_mixup.yaml`
- Create: `configs/experiments/phase3_randeras.yaml`
- Create: `configs/experiments/phase3_sam.yaml`
- Create: `configs/experiments/phase3_wd_1e4.yaml`
- Create: `configs/experiments/phase3_wd_1e3.yaml`
- Create: `configs/experiments/phase3_batch64.yaml`
- Create: `configs/experiments/phase3_batch32.yaml`

**Step 1: Create configs**

These are all based on the Phase 2 winner's config (backbone TBD after Phase 2 results). Use placeholder `WINNER_BACKBONE` — the implementer should replace with the actual best backbone from Phase 2.

Each config enables one change at a time:
- `phase3_mixup.yaml`: adds `mixup_alpha: 0.2`
- `phase3_randeras.yaml`: adds `random_erasing: true`
- `phase3_sam.yaml`: adds `use_sam: true`, `sam_rho: 0.05`
- `phase3_wd_1e4.yaml`: override weight_decay to 1e-4 (requires adding `weight_decay` as a model param)
- `phase3_wd_1e3.yaml`: override weight_decay to 1e-3
- `phase3_batch64.yaml`: batch_size: 64
- `phase3_batch32.yaml`: batch_size: 32

Each tagged `["phase3", "<change-name>"]` in W&B.

**Step 2: Commit**

```bash
git add configs/experiments/phase3_*.yaml
git commit -m "feat: add Phase 3 augmentation/regularization experiment configs"
```

---

### Task 11: Experiment Tracking & Launch Script

**Files:**
- Create: `docs/plans/experiment-log.md`
- Create: `scripts/launch_phase.sh`

**Step 1: Create experiment log template**

```markdown
# Experiment Log

## Phase 1: Foundational Improvements
| ID | Run | val_loss | val_mse | Notes |
|----|-----|----------|---------|-------|
| P1-baseline | | | | Random init, StepLR |
| P1-pretrained | | | | Pretrained, cosine LR, 1-ch adapt |

## Phase 2: Backbone Sweep
| ID | Backbone | Run | val_loss | val_mse | Notes |
|----|----------|-----|----------|---------|-------|
| P2-effnet-b0 | EfficientNet-B0 | | | | |
| P2-effnet-v2s | EfficientNet V2-S | | | | |
| P2-convnext-t | ConvNeXt Tiny | | | | |
| P2-resnet18 | ResNet-18 | | | | |
| P2-swin-v2-t | Swin-V2-T | | | | |
| P2-maxvit-t | MaxViT-T | | | | |

## Phase 3: Augmentation & Regularization
(Fill in after Phase 2 winner selected)

## Phase 4: Best Combination
(Fill in after Phase 3)
```

**Step 2: Create launch helper**

```bash
#!/usr/bin/env bash
# scripts/launch_phase.sh — Launch all experiments for a given phase
# Usage: ./scripts/launch_phase.sh 1|2|3
set -euo pipefail
PHASE="${1:?Usage: $0 <phase-number>}"

for config in configs/experiments/phase${PHASE}_*.yaml; do
    name=$(basename "$config" .yaml)
    echo "Launching: $name"
    modal run train_modal.py --config "$config" --name "$name"
done
```

**Step 3: Commit**

```bash
git add docs/plans/experiment-log.md scripts/launch_phase.sh
git commit -m "feat: add experiment log template and launch script"
```

---

## Execution Order

1. Tasks 1-4 (Phase 1 code + config) — then launch Phase 1 experiments on Modal
2. Task 5-6 (Phase 2 backbones + configs) — can be done in parallel with Phase 1 runs
3. Wait for Phase 1 results → validate pretrained >> random init
4. Launch Phase 2 experiments
5. Wait for Phase 2 results → pick top 2-3 backbones
6. Tasks 7-10 (Phase 3 augmentations + configs, fill in winner backbone)
7. Launch Phase 3 experiments
8. Wait for Phase 3 results → combine winners into Phase 4 config
9. Task 11 maintained throughout for tracking
