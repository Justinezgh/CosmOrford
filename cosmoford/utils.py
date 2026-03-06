"""Shared data loading, persistence, and visualization helpers for the challenge."""

from __future__ import annotations

import json
import os
import zipfile
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["Utility", "Data", "Visualization", "Score"]

class Score:
    @staticmethod
    def _score_phase1(true_cosmo, infer_cosmo, errorbar):
        """
        Computes the log-likelihood score for Phase 1 based on predicted cosmological parameters.

        Parameters
        ----------
        true_cosmo : np.ndarray
            Array of true cosmological parameters (shape: [n_samples, n_params]).
        infer_cosmo : np.ndarray
            Array of inferred cosmological parameters from the model (same shape as true_cosmo).
        errorbar : np.ndarray
            Array of standard deviations (uncertainties) for each inferred parameter
            (same shape as true_cosmo).

        Returns
        -------
        np.ndarray
            Array of scores for each sample (shape: [n_samples]).
        """

        sq_error = (true_cosmo - infer_cosmo)**2
        scale_factor = 1000  # This is a constant that scales the error term.
        score = - np.sum(sq_error / errorbar**2 + np.log(errorbar**2) + scale_factor * sq_error, 1)
        score = np.mean(score)
        if score >= -10**6: # Set a minimum of the score (to properly display on Codabench)
            return score
        else:
            return -10**6

class Utility:
    """Collection of static helpers for manipulating convergence maps and persistence."""

    @staticmethod
    def add_noise(
        data: np.ndarray,
        mask: np.ndarray,
        ng: float,
        pixel_size: float = 2.0,
    ) -> np.ndarray:
        """Return a noisy convergence map using the survey noise model."""
        noise = np.random.randn(*data.shape) * 0.4 / (2 * ng * pixel_size**2) ** 0.5
        return data + noise * mask

    @staticmethod
    def load_np(data_dir: str, file_name: str) -> np.ndarray:
        """Load a NumPy array stored in ``data_dir/file_name``."""
        file_path = os.path.join(data_dir, file_name)
        return np.load(file_path)

    @staticmethod
    def save_np(data_dir: str, file_name: str, data: np.ndarray) -> None:
        """Persist a NumPy array to ``data_dir/file_name``."""
        file_path = os.path.join(data_dir, file_name)
        np.save(file_path, data)

    @staticmethod
    def save_json_zip(
        submission_dir: str,
        json_file_name: str,
        zip_file_name: str,
        data: Dict[str, np.ndarray],
    ) -> str:
        """Write ``data`` to JSON and package it inside a ZIP archive."""
        os.makedirs(submission_dir, exist_ok=True)
        json_path = os.path.join(submission_dir, json_file_name)

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(data, file)

        zip_path = os.path.join(submission_dir, zip_file_name)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(json_path, arcname=json_file_name)

        os.remove(json_path)
        return zip_path


class Data:
    """Access training and test convergence maps along with metadata."""

    def __init__(self, data_dir: str, USE_PUBLIC_DATASET: bool):
        """Configure dataset paths and metadata for public or sampled sets."""
        self.USE_PUBLIC_DATASET = USE_PUBLIC_DATASET
        self.use_public_dataset = bool(USE_PUBLIC_DATASET)
        self.data_dir = data_dir
        self.mask_file = "WIDE12H_bin2_2arcmin_mask.npy"
        self.viz_label_file = "label.npy"

        if self.use_public_dataset:
            self.kappa_file = "WIDE12H_bin2_2arcmin_kappa.npy"
            self.label_file = self.viz_label_file
            self.Ncosmo = 101
            self.Nsys = 256
            self.test_kappa_file = "WIDE12H_bin2_2arcmin_kappa_noisy_test.npy"
            self.Ntest = 4000
        else:
            self.kappa_file = "sampled_WIDE12H_bin2_2arcmin_kappa.npy"
            self.label_file = "sampled_label.npy"
            self.Ncosmo = 3
            self.Nsys = 30
            self.test_kappa_file = "sampled_WIDE12H_bin2_2arcmin_kappa_noisy_test.npy"
            self.Ntest = 3

        self.shape = (1424, 176)
        self.pixelsize_arcmin = 2.0
        self.pixelsize_radian = self.pixelsize_arcmin / 60 / 180 * np.pi
        self.ng = 30

        self.mask: np.ndarray | None = None
        self.kappa: np.ndarray | None = None
        self.label: np.ndarray | None = None
        self.viz_label: np.ndarray | None = None
        self.kappa_test: np.ndarray | None = None

    def load_train_data(self) -> None:
        """Load mask, noiseless convergence maps, and labels into memory."""
        self.mask = Utility.load_np(data_dir=self.data_dir, file_name=self.mask_file)
        self.kappa = np.zeros((self.Ncosmo, self.Nsys, *self.shape), dtype=np.float16)
        self.kappa[:, :, self.mask] = Utility.load_np(
            data_dir=self.data_dir,
            file_name=self.kappa_file,
        )
        self.label = Utility.load_np(data_dir=self.data_dir, file_name=self.label_file)
        self.viz_label = Utility.load_np(
            data_dir=self.data_dir,
            file_name=self.viz_label_file,
        )

    def load_test_data(self) -> None:
        """Load noisy test convergence maps into memory."""
        if self.mask is None:
            raise RuntimeError("Call load_train_data before load_test_data to populate mask.")

        self.kappa_test = np.zeros((self.Ntest, *self.shape), dtype=np.float16)
        self.kappa_test[:, self.mask] = Utility.load_np(
            data_dir=self.data_dir,
            file_name=self.test_kappa_file,
        )


class Visualization:
    """Quick-look plotting helpers for convergence maps and labels."""

    @staticmethod
    def plot_mask(mask: np.ndarray) -> None:
        """Display the survey mask."""
        plt.figure(figsize=(30, 100))
        plt.imshow(mask.T)
        plt.show()

    @staticmethod
    def plot_noiseless_training_convergence_map(kappa: np.ndarray) -> None:
        """Plot the first noiseless convergence map."""
        plt.figure(figsize=(30, 100))
        plt.imshow(kappa[0, 0].T, vmin=-0.02, vmax=0.07)
        plt.show()

    @staticmethod
    def plot_noisy_training_convergence_map(
        kappa: np.ndarray,
        mask: np.ndarray,
        pixelsize_arcmin: float,
        ng: float,
    ) -> None:
        """Plot the first convergence map with synthetic noise added."""
        noisy_map = Utility.add_noise(kappa[0, 0], mask, ng, pixelsize_arcmin)
        plt.figure(figsize=(30, 100))
        plt.imshow(noisy_map.T, vmin=-0.02, vmax=0.07)
        plt.show()

    @staticmethod
    def plot_cosmological_parameters_OmegaM_S8(label: np.ndarray) -> None:
        """Scatter plot of ``Omega_m`` versus ``S8`` for the training labels."""
        plt.scatter(label[:, 0, 0], label[:, 0, 1])
        plt.xlabel(r"$\Omega_m$")
        plt.ylabel(r"$S_8$")
        plt.show()

    @staticmethod
    def plot_baryonic_physics_parameters(label: np.ndarray) -> None:
        """Scatter plot of baryonic physics parameters for the first cosmology."""
        plt.scatter(label[0, :, 2], label[0, :, 3])
        plt.xlabel(r"$T_{\mathrm{AGN}}$")
        plt.ylabel(r"$f_0$")
        plt.show()

    @staticmethod
    def plot_photometric_redshift_uncertainty_parameters(label: np.ndarray) -> None:
        """Histogram of photometric redshift uncertainty parameters."""
        plt.hist(label[0, :, 4], bins=20)
        plt.xlabel(r"$\Delta z$")
        plt.show()
