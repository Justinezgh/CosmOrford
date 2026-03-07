"""Patched compressor model: patch-based backbone + power spectrum features.

Faithfully reproduces the neurips-wl-challenge approach:
- Splits 1834x88 input into 21 patches of 88x88
- Runs each patch through backbone, averages features
- Concatenates with power spectrum MLP features
- Random 90-degree rotation augmentation on patches
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from cosmorford import THETA_MEAN, THETA_STD, NOISE_STD, SURVEY_MASK
from cosmorford.backbones import get_backbone, adapt_first_conv
from cosmorford.summaries import power_spectrum_batch


class PatchedCompressorModel(L.LightningModule):
    def __init__(
        self,
        backbone: str = "efficientnet_v2_s",
        bottleneck_dim: int = 8,
        warmup_steps: int = 500,
        max_lr: float = 0.008,
        decay_rate: float = 0.85,
        decay_every_epochs: int = 1,
        dropout_rate: float = 0.2,
        lr_schedule: str = "cosine",
        pretrained: bool = False,
        use_3channel: bool = True,
        n_val_noise: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

        features, feat_dim = get_backbone(backbone, pretrained=pretrained)
        if not use_3channel:
            features = adapt_first_conv(features, backbone)
        self.backbone = features

        self.mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
        )

        # Power spectrum MLP: 10 bins -> 128 features (matches old project)
        self.ps_head = nn.Sequential(
            nn.Linear(10, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )

        self.bottleneck = nn.Linear(feat_dim + 128, bottleneck_dim)

        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2 * 2),  # mean(2) + log_scale(2)
        )

    def _patch_forward(self, x):
        """Split into 21 patches, run backbone, average, concat power spectrum."""
        # Power spectrum on original input (before patching)
        with torch.no_grad():
            _, ps_features = power_spectrum_batch(x)

        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 3-channel repeat
        if self.hparams.use_3channel and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Pad 1834 -> 1848 (21 * 88) and reshape into patches
        n_channels = x.size(1)
        x = F.pad(x, pad=(0, 0, 0, 14), mode='constant', value=0)
        x = x.reshape(-1, n_channels, 21, 88, 88)
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, n_channels, 88, 88)

        # Random 90-degree rotation during training
        if self.training:
            angles = torch.randint(0, 4, (x.size(0),), device=x.device)
            for k in range(4):
                rot_mask = angles == k
                if rot_mask.any():
                    x[rot_mask] = torch.rot90(x[rot_mask], k=k, dims=(2, 3))

        # Backbone features
        features = self.backbone(x.float())
        # (batch*21, C, H, W) -> (batch, 21, C, H, W) -> average over patches
        features = features.reshape(-1, 21, features.size(1), features.size(2), features.size(3))
        features = features.mean(dim=1)
        features = self.pool(features)

        # Concat with power spectrum features
        ps_features = self.ps_head(ps_features)
        features = torch.cat([features, ps_features], dim=1)

        return features

    def compress(self, x):
        """Return the 8-dim bottleneck representation."""
        return self.bottleneck(self._patch_forward(x))

    def forward(self, x):
        z = self.compress(x)
        out = self.head(z)
        mean = out[..., :2]
        scale = F.softplus(out[..., 2:]) + 0.001
        return mean, scale

    def _augment(self, x):
        """Apply augmentations: noise, survey mask, random flips, cyclic shifts."""
        noise = torch.randn_like(x) * NOISE_STD
        x = x + noise
        x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

        batch_size = x.size(0)
        flip_lr = torch.rand(batch_size, device=x.device) < 0.5
        x[flip_lr] = torch.flip(x[flip_lr], dims=[1])
        flip_ud = torch.rand(batch_size, device=x.device) < 0.5
        x[flip_ud] = torch.flip(x[flip_ud], dims=[2])

        shift_x = torch.randint(0, x.size(1), (batch_size,), device=x.device)
        x = torch.stack([torch.roll(x[i], shifts=(shift_x[i].item(),), dims=(0,)) for i in range(batch_size)])
        shift_y = torch.randint(0, x.size(2), (batch_size,), device=x.device)
        x = torch.stack([torch.roll(x[i], shifts=(shift_y[i].item(),), dims=(1,)) for i in range(batch_size)])

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self._augment(x)
        mean, std = self(x)
        loss = -torch.distributions.Normal(loc=mean, scale=std).log_prob(y).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mask = torch.tensor(self.mask, device=x.device).unsqueeze(0)

        total_loss = 0.0
        total_mean = 0.0
        for _ in range(self.hparams.n_val_noise):
            x_noisy = (x + torch.randn_like(x) * NOISE_STD) * mask
            mean, std = self(x_noisy)
            total_loss += -torch.distributions.Normal(loc=mean, scale=std).log_prob(y).mean()
            total_mean += mean

        loss = total_loss / self.hparams.n_val_noise
        mean = total_mean / self.hparams.n_val_noise

        mean_orig = mean * torch.tensor(THETA_STD[:2], device=mean.device) + torch.tensor(THETA_MEAN[:2], device=mean.device)
        y_orig = y * torch.tensor(THETA_STD[:2], device=y.device) + torch.tensor(THETA_MEAN[:2], device=y.device)
        mse = F.mse_loss(mean_orig, y_orig)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse", mse, prog_bar=True)
        return loss

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
