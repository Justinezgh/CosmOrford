"""cosmoford - Shake the Cosmic 8-Ball for cosmological parameter inference."""
import os
import numpy as np
from cosmoford.utils import Data, Utility, Visualization

__all__ = ["Data", "Utility", "Visualization"]

# Normalization factors for the cosmological parameters, computed on the training set
# These values are used internally to normalize the parameters during training and inference
THETA_MEAN = np.array([2.9022e-01, 8.1345e-01, 7.8500e+00, 1.3262e-02, 9.2743e-04])
THETA_STD = np.array([0.1055, 0.0660, 0.3764, 0.0076, 0.0219])
SURVEY_MASK = np.load(os.path.join(os.path.dirname(__file__), "WIDE12H_bin2_2arcmin_mask.npy"))
NOISE_STD = 0.4 / (2 * 30 * 2.0**2) ** 0.5  # Noise std for ng=30 galaxies/arcmin^2 and pixel size=2 arcmin
