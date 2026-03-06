import os
import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosmoford import THETA_MEAN, THETA_STD, SURVEY_MASK, NOISE_STD
from cosmoford.summaries import power_spectrum_batch
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b2, efficientnet_v2_s, efficientnet_v2_m

class RegressionModel(L.LightningModule):

  def __init__(self, backbone="efficientnet_b0", warmup_steps: int = 1000, max_lr: float = 0.256,
               decay_rate: float = 0.97, decay_every_epochs: int = 2, dropout_rate: float = 0.2,
               loss_type: str = "log_prob", freeze_backbone: bool = False,
               pretrained_checkpoint_path: str = None):
    super().__init__()
    self.save_hyperparameters(ignore=['pretrained_checkpoint_path'])

    # Validate loss_type parameter
    if loss_type not in ["log_prob", "score"]:
      raise ValueError(f"loss_type must be 'log_prob' or 'score', got '{loss_type}'")

    self.mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

    last_dim = 1280  # For efficientnet_b0
    if backbone == "efficientnet_b0":
      vision_model = efficientnet_b0()
    elif backbone == "efficientnet_b2":
      vision_model = efficientnet_b2()
      last_dim = 1408
    elif backbone == "efficientnet_v2_s":
      vision_model = efficientnet_v2_s()
    elif backbone == "efficientnet_v2_m":
      vision_model = efficientnet_v2_m()
    else:
      raise ValueError(f"Backbone {backbone} not supported.")

    self.model = vision_model.features
    self.reshape_head = nn.Sequential(
     nn.AdaptiveAvgPool2d(1),
     nn.Flatten(),
     nn.Dropout(p=self.hparams.dropout_rate, inplace=True),
    )
    self.ps_head = nn.Sequential(
     nn.Linear(10, 256),
     nn.LeakyReLU(),
     nn.Linear(256, 256),
     nn.LeakyReLU(),
     nn.Linear(256, 128),
     nn.LeakyReLU()
    )
    self.head = nn.Sequential(
     nn.Linear(last_dim+128, 128),
     nn.LeakyReLU(),
     nn.Linear(128, 2*2) # Outputing mean and log-variance
    )

    # Load pretrained weights if checkpoint path is provided
    if pretrained_checkpoint_path is not None:
      self.load_pretrained_weights(pretrained_checkpoint_path)

    # Freeze backbone and power spectrum head if in fine-tuning mode
    if freeze_backbone:
      self.freeze_backbone_layers()

  def forward(self, x):

    # Compute power spectrum features
    with torch.no_grad():
      _, ps_features = power_spectrum_batch(x)

    # Add channel dimension if missing
    if x.dim() == 3:
      x = x.unsqueeze(1)
    # Repeat channels to match expected input size (e.g., 3 for RGB)
    if x.size(1) == 1:
      x = x.repeat(1, 3, 1, 1)
    # Reshape data into patches of size 88x88
    x = F.pad(x, pad=(0, 0, 0, 14), mode='constant', value=0)
    x = x.reshape((-1, 3, 21, 88, 88))
    # Combine batch and patch dimensions
    x = x.permute(0, 2, 1, 3, 4).reshape(-1, 3, 88, 88)

    # If this is training, apply random rotation to each patch
    if self.training:
      angles = torch.randint(0, 4, (x.size(0),), device=x.device)  # Random angles: 0, 1, 2, 3
      # Apply different rotations to different samples in the batch
      for k in range(4):
        mask = angles == k
        if mask.any():
          x[mask] = torch.rot90(x[mask], k=k, dims=(2, 3))

    # Compute features
    features = self.model(x.float())
    # Reshape back to (batch_size, num_patches, feature_dim)
    features = features.reshape((-1, 21, features.size(1), features.size(2), features.size(3)))
    features = features.mean(dim=1)  # Average over patches
    features = self.reshape_head(features)

    # Combine with power spectrum features
    ps_features = self.ps_head(ps_features)
    features = torch.cat([features, ps_features], dim=1)

    # Head to get final predictions
    x = self.head(features)
    return x[..., :2], F.softplus(x[..., 2:]) + 0.001  # Return mean and scale

  def load_pretrained_weights(self, checkpoint_path: str):
    """Load weights from a pretrained checkpoint.
    Only loads the model weights, not the hyperparameters or optimizer state."""
    print(f"\nLoading pretrained weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Lightning checkpoints store the model state_dict under 'state_dict' key
    if 'state_dict' in checkpoint:
      state_dict = checkpoint['state_dict']
    else:
      state_dict = checkpoint

    # Load the weights
    self.load_state_dict(state_dict, strict=True)
    print("Pretrained weights loaded successfully!\n")

  def freeze_backbone_layers(self):
    """Freeze the backbone (vision model) and power spectrum head for fine-tuning.
    Only the final regression heads (reshape_head and head) remain trainable."""
    for param in self.model.parameters():
      param.requires_grad = False
    for param in self.ps_head.parameters():
      param.requires_grad = False
    # Keep self.reshape_head and self.head trainable
    print("\nBackbone and ps_head frozen. Only reshape_head and head are trainable.")
    self.print_trainable_parameters()

  def print_trainable_parameters(self):
    """Print statistics about trainable vs frozen parameters"""
    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in self.parameters())
    frozen_params = total_params - trainable_params

    print(f"{'='*60}")
    print(f"Parameter Statistics (Fine-tuning Mode)")
    print(f"{'='*60}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Frozen parameters:    {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)")
    print(f"{'='*60}\n")

  def training_step(self, batch, batch_idx):
    x, y = batch

    # Adding noise to the input convergence maps and applying mask
    noise = torch.randn_like(x) * NOISE_STD
    x = x + noise
    x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

    # Adding augmentations, random left-right and up-down flips (per sample)
    batch_size = x.size(0)
    # Random flips along nx dimension (dim=1)
    flip_lr = torch.rand(batch_size, device=x.device) < 0.5
    x[flip_lr] = torch.flip(x[flip_lr], dims=[1])
    # Random flips along ny dimension (dim=2)
    flip_ud = torch.rand(batch_size, device=x.device) < 0.5
    x[flip_ud] = torch.flip(x[flip_ud], dims=[2])

    # Adding random cyclic shifts (different for each sample) in nx and ny
    shift_x = torch.randint(low=0, high=x.size(1), size=(batch_size,), device=x.device)
    x = torch.stack([torch.roll(x[i], shifts=(shift_x[i].item(),), dims=(0,)) for i in range(batch_size)])
    shift_y = torch.randint(low=0, high=x.size(2), size=(batch_size,), device=x.device)
    x = torch.stack([torch.roll(x[i], shifts=(shift_y[i].item(),), dims=(1,)) for i in range(batch_size)])

    mean, std = self(x)

    if self.hparams.loss_type == "log_prob":
      loss = - torch.distributions.Normal(loc=mean, scale=std).log_prob(y)
      loss = loss.mean()
    elif self.hparams.loss_type == "score":
      # Rescaling back to original parameters
      mean = mean * torch.tensor(THETA_STD[:2], device=mean.device) + torch.tensor(THETA_MEAN[:2], device=mean.device)
      std = std * torch.tensor(THETA_STD[:2], device=std.device)
      y = y[:, :2] * torch.tensor(THETA_STD[:2], device=y.device) + torch.tensor(THETA_MEAN[:2], device=y.device)

      # Use the Phase 1 score as the loss (negative because we minimize)
      sq_error = (y - mean) ** 2
      scale_factor = 1000.0
      score = -torch.sum(sq_error / std**2 + torch.log(std**2) + scale_factor * sq_error, dim=1)
      loss = -torch.mean(score)  # Negative because we want to maximize the score

    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch

    # Adding noise to the input convergence maps and applying mask
    noise = torch.randn_like(x) * NOISE_STD
    x = x + noise
    x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

    mean, std = self(x)

    # Rescaling back to original parameters, for log_prob loss we do it here
    mean = mean * torch.tensor(THETA_STD[:2], device=mean.device) + torch.tensor(THETA_MEAN[:2], device=mean.device)
    std = std * torch.tensor(THETA_STD[:2], device=std.device)
    y = y[:, :2] * torch.tensor(THETA_STD[:2], device=y.device) + torch.tensor(THETA_MEAN[:2], device=y.device)

    # Compute the Phase 1 score (torch version)
    sq_error = (y - mean) ** 2
    scale_factor = 1000.0
    score = -torch.sum(sq_error / std**2 + torch.log(std**2) + scale_factor * sq_error, dim=1)
    score = torch.mean(score)

    mse = F.mse_loss(mean, y)

    self.log('val_score', score, prog_bar=True)
    self.log('val_mse', mse, prog_bar=True)
    return score

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.max_lr, weight_decay=1e-5)

    # Calculate total steps
    total_steps = int(self.trainer.estimated_stepping_batches)
    warmup_steps = self.hparams.warmup_steps

    # Calculate step size for StepLR in terms of steps (not epochs)
    steps_per_epoch = total_steps // self.trainer.max_epochs
    step_size_in_steps = self.hparams.decay_every_epochs * steps_per_epoch

    # Linear warmup from 0 to max_lr
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-10,  # Start from nearly 0
        end_factor=1.0,      # End at max_lr
        total_iters=warmup_steps,
    )

    # Step decay after warmup (in steps, not epochs)
    decay = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size_in_steps,
        gamma=self.hparams.decay_rate,
    )

    # Combine warmup and decay
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[warmup_steps],
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",  # Both warmup and decay operate on steps
            "frequency": 1,
        },
    }
