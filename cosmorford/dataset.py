# cosmorford/dataset.py
import torch
import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader
from cosmorford import THETA_MEAN, THETA_STD


def reshape_field(kappa):
    """Reshape [B, 1424, 176] -> [B, 1834, 88] to remove masked empty space."""
    return torch.concat([kappa[:, :, :88], kappa[:, 620:1030, 88:]], dim=1)


def inverse_reshape_field(kappa_reduced, fill_value=0.0):
    """Inverse of reshape_field: [B, 1834, 88] -> [B, 1424, 176]."""
    B = kappa_reduced.shape[0]
    part1 = kappa_reduced[:, :1424, :]
    part2 = kappa_reduced[:, 1424:, :]
    kappa_full = torch.full((B, 1424, 176), fill_value, dtype=kappa_reduced.dtype, device=kappa_reduced.device)
    kappa_full[:, :, :88] = part1
    kappa_full[:, 620:1030, 88:] = part2
    return kappa_full


class WLDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _collate_fn(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        kappa = reshape_field(batch["kappa"]).float()
        device = batch["theta"].device
        theta = (batch["theta"][:, :2] - torch.tensor(THETA_MEAN[:2], device=device)) / torch.tensor(THETA_STD[:2], device=device)
        theta = theta.float()
        return kappa, theta

    def setup(self, stage=None):
        dset = load_dataset("CosmoStat/neurips-wl-challenge-flat")
        dset = dset.with_format("torch")
        self.train_dataset = dset["train"]
        self.val_dataset = dset["validation"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )
