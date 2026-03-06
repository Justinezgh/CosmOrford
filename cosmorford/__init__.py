"""CosmOrford - Neural network compression of weak lensing mass maps."""
import numpy as np

# Normalization factors for cosmological parameters (computed on training set)
# 5 parameters: Omega_m, S8, ...  (we only use first 2)
THETA_MEAN = np.array([2.9022e-01, 8.1345e-01, 7.8500e+00, 1.3262e-02, 9.2743e-04])
THETA_STD = np.array([0.1055, 0.0660, 0.3764, 0.0076, 0.0219])
NOISE_STD = 0.4 / (2 * 30 * 2.0**2) ** 0.5
