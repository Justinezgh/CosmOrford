"""
Power spectrum estimation utilities for batched image data.
"""

import torch
import numpy as np

LOG_PS_MEAN = torch.tensor([ -8.676514 ,  -8.953475 ,  -9.1855755,  -9.435738 ,  -9.680172 ,
         -9.947989 , -10.258803 , -10.614475 , -10.918096 , -11.084858], dtype=torch.float32)
LOG_PS_STD = torch.tensor([0.2304908 , 0.22719958, 0.22156876, 0.23032227, 0.2407397 ,
        0.26038605, 0.2909596 , 0.3229235 , 0.35014188, 0.3699048], dtype=torch.float32)


def power_spectrum_batch(x, pixsize=2. / 60 / 180 * np.pi, kedge=np.logspace(2, 4, 11), normalize=True):
    """
    Compute the azimuthally averaged 2D power spectrum of batched real-valued 2D fields.

    Parameters:
    -----------
    x : torch.Tensor
        Input real-space maps with shape (batch, ny, nx).
        Each slice x[i] represents a 2D field (e.g., an image or simulated field).

    pixsize : float
        Physical size of each pixel in the map (e.g., arcmin, Mpc, etc.).
        Units should be consistent with the units used for `kedge`.

    kedge : torch.Tensor or array-like
        Bin edges in wavenumber space (k), used to bin the power spectrum.
        Should be monotonically increasing and cover the k-range of interest.
        Shape: (n_edges,)

    normalize : bool
        If True, normalize the power spectrum for ingestion by ML models.

    Returns:
    --------
    power_k : torch.Tensor
        The average wavenumber in each k bin (excluding the DC bin).
        Shape: (batch, nk) where nk = len(kedge) - 1

    power : torch.Tensor
        The binned, azimuthally averaged power spectrum corresponding to `power_k`.
        Normalized per unit area.
        Shape: (batch, nk)
    """

    # Ensure the input array is 3D: [batch, ny, nx]
    assert x.ndim == 3, f"Expected 3D input [batch, ny, nx], got shape {x.shape}"

    batch_size, ny, nx = x.shape
    device = x.device
    dtype = x.dtype

    # Convert kedge to tensor if needed
    if not isinstance(kedge, torch.Tensor):
        kedge = torch.tensor(kedge, device=device, dtype=dtype)
    else:
        kedge = kedge.to(device=device, dtype=dtype)

    # Compute the 2D FFT and power spectrum
    xk = torch.fft.rfft2(x)  # Shape: [batch, ny, nx//2 + 1]
    xk2 = (xk * xk.conj()).real  # Power spectrum: |FFT|^2

    # Compute the wavenumber grid
    ky = torch.fft.fftfreq(ny, d=pixsize, device=device, dtype=dtype)
    kx = torch.fft.rfftfreq(nx, d=pixsize, device=device, dtype=dtype)

    ky_grid = ky.reshape(-1, 1) ** 2
    kx_grid = kx.reshape(1, -1) ** 2
    k = torch.sqrt(ky_grid + kx_grid) * 2 * np.pi  # Shape: (ny, nx//2 + 1)

    # Bin indices
    index = torch.searchsorted(kedge, k.flatten()).reshape(ny, nx // 2 + 1)

    # Number of bins
    n_bins = len(kedge)
    nk = n_bins - 1

    # Flatten spatial dimensions for binning
    # Shape transformations: [batch, ny, nx//2+1] -> [batch, ny*(nx//2+1)]
    xk2_flat = xk2.reshape(batch_size, -1)
    k_flat = k.flatten().unsqueeze(0).expand(batch_size, -1)  # [batch, ny*(nx//2+1)]
    index_flat = index.flatten().unsqueeze(0).expand(batch_size, -1)  # [batch, ny*(nx//2+1)]

    # Initialize accumulators
    power = torch.zeros(batch_size, n_bins, device=device, dtype=dtype)
    power_k = torch.zeros(batch_size, n_bins, device=device, dtype=dtype)
    Nmode = torch.zeros(batch_size, n_bins, device=device, dtype=dtype)

    # Scatter add to accumulate into bins
    for b in range(batch_size):
        power[b].index_add_(0, index_flat[b], xk2_flat[b])
        power_k[b].index_add_(0, index_flat[b], k_flat[b])
        Nmode[b].index_add_(0, index_flat[b], torch.ones_like(xk2_flat[b]))

    # Add mirror contributions
    if nx % 2 == 0:
        mirror_slice = slice(1, -1)
    else:
        mirror_slice = slice(1, None)

    xk2_mirror = xk2[:, :, mirror_slice].reshape(batch_size, -1)
    k_mirror = k[:, mirror_slice].flatten().unsqueeze(0).expand(batch_size, -1)
    index_mirror = index[:, mirror_slice].flatten().unsqueeze(0).expand(batch_size, -1)

    for b in range(batch_size):
        power[b].index_add_(0, index_mirror[b], xk2_mirror[b])
        power_k[b].index_add_(0, index_mirror[b], k_mirror[b])
        Nmode[b].index_add_(0, index_mirror[b], torch.ones_like(xk2_mirror[b]))

    # Average
    select = Nmode > 0
    power[select] = power[select] / Nmode[select]
    power_k[select] = power_k[select] / Nmode[select]
    power_k = power_k[:, 1:nk+1]  # Exclude DC bin

    # Normalize by map area
    power *= pixsize ** 2 / ny / nx
    power = power[:, 1:nk+1]# Exclude DC bin

    if normalize:
        # Normalize the power spectrum for ML ingestion
        log_power = torch.log10(power + 1e-30)
        log_power = (log_power - LOG_PS_MEAN.unsqueeze(0).to(device=device, dtype=dtype)) / LOG_PS_STD.unsqueeze(0).to(device=device, dtype=dtype)
        power = log_power

    return power_k, power
