"""Power spectrum compressor: MLP on power spectrum features only."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from cosmorford import THETA_MEAN, THETA_STD, NOISE_STD, SURVEY_MASK
from cosmorford.summaries import power_spectrum_batch


class PSCompressorModel(L.LightningModule):
    def __init__(
        self,
        bottleneck_dim: int = 8,
        hidden_dim: int = 256,
        n_hidden: int = 3,
        warmup_steps: int = 500,
        max_lr: float = 0.001,
        lr_schedule: str = "cosine",
        dropout_rate: float = 0.1,
        n_val_noise: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

        # MLP: 10 power spectrum bins -> bottleneck
        layers = [nn.Linear(10, hidden_dim), nn.GELU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)]
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2 * 2),  # mean(2) + log_scale(2)
        )

    def compress(self, x):
        """Return the 8-dim bottleneck from power spectrum."""
        with torch.no_grad():
            _, ps = power_spectrum_batch(x)
        return self.encoder(ps)

    def forward(self, x):
        z = self.compress(x)
        out = self.head(z)
        mean = out[..., :2]
        scale = F.softplus(out[..., 2:]) + 0.001
        return mean, scale

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Apply noise and mask (same as other models)
        noise = torch.randn_like(x) * NOISE_STD
        x = x + noise
        x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

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
        else:
            steps_per_epoch = total_steps // self.trainer.max_epochs
            step_size = steps_per_epoch
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=0.85
            )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, main_scheduler], milestones=[warmup_steps]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
