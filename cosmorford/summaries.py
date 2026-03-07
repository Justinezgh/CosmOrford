"""Power spectrum estimation utilities for batched image data.
Ported from neurips-wl-challenge/s8ball/summaries.py"""

import torch
import numpy as np

LOG_PS_MEAN = torch.tensor([-8.676514, -8.953475, -9.1855755, -9.435738, -9.680172,
        -9.947989, -10.258803, -10.614475, -10.918096, -11.084858], dtype=torch.float32)
LOG_PS_STD = torch.tensor([0.2304908, 0.22719958, 0.22156876, 0.23032227, 0.2407397,
        0.26038605, 0.2909596, 0.3229235, 0.35014188, 0.3699048], dtype=torch.float32)


def power_spectrum_batch(x, pixsize=2. / 60 / 180 * np.pi, kedge=np.logspace(2, 4, 11), normalize=True):
    """Compute azimuthally averaged 2D power spectrum of batched 2D fields.

    Args:
        x: Input maps [batch, ny, nx].
        pixsize: Physical pixel size.
        kedge: Bin edges in wavenumber space.
        normalize: If True, normalize for ML ingestion.

    Returns:
        power_k: Average wavenumber per bin [batch, nk].
        power: Binned power spectrum [batch, nk].
    """
    assert x.ndim == 3
    batch_size, ny, nx = x.shape
    device = x.device
    dtype = x.dtype

    if not isinstance(kedge, torch.Tensor):
        kedge = torch.tensor(kedge, device=device, dtype=dtype)
    else:
        kedge = kedge.to(device=device, dtype=dtype)

    xk = torch.fft.rfft2(x)
    xk2 = (xk * xk.conj()).real

    ky = torch.fft.fftfreq(ny, d=pixsize, device=device, dtype=dtype)
    kx = torch.fft.rfftfreq(nx, d=pixsize, device=device, dtype=dtype)
    k = torch.sqrt(ky.reshape(-1, 1) ** 2 + kx.reshape(1, -1) ** 2) * 2 * np.pi

    index = torch.searchsorted(kedge, k.flatten()).reshape(ny, nx // 2 + 1)
    n_bins = len(kedge)
    nk = n_bins - 1

    # Bin indices are the same for every sample — compute Nmode and power_k once
    index_1d = index.flatten()
    npix = index_1d.shape[0]
    Nmode = torch.zeros(n_bins, device=device, dtype=dtype)
    Nmode.index_add_(0, index_1d, torch.ones(npix, device=device, dtype=dtype))
    power_k_1d = torch.zeros(n_bins, device=device, dtype=dtype)
    power_k_1d.index_add_(0, index_1d, k.flatten())

    # Accumulate power across batch using scatter_add_ (no Python loop)
    index_batch = index_1d.unsqueeze(0).expand(batch_size, -1)
    power = torch.zeros(batch_size, n_bins, device=device, dtype=dtype)
    power.scatter_add_(1, index_batch, xk2.reshape(batch_size, -1))

    # Mirror contributions (rfft2 only gives half the spectrum)
    if nx % 2 == 0:
        mirror_slice = slice(1, -1)
    else:
        mirror_slice = slice(1, None)

    index_mirror_1d = index[:, mirror_slice].flatten()
    npix_mirror = index_mirror_1d.shape[0]
    Nmode.index_add_(0, index_mirror_1d, torch.ones(npix_mirror, device=device, dtype=dtype))
    power_k_1d.index_add_(0, index_mirror_1d, k[:, mirror_slice].flatten())

    index_mirror_batch = index_mirror_1d.unsqueeze(0).expand(batch_size, -1)
    power.scatter_add_(1, index_mirror_batch, xk2[:, :, mirror_slice].reshape(batch_size, -1))

    # Average
    select = Nmode > 0
    power_k_1d[select] = power_k_1d[select] / Nmode[select]
    power[:, select] = power[:, select] / Nmode[select].unsqueeze(0)

    power_k = power_k_1d[1:nk + 1].unsqueeze(0).expand(batch_size, -1)
    power *= pixsize ** 2 / ny / nx
    power = power[:, 1:nk + 1]

    if normalize:
        log_power = torch.log10(power + 1e-30)
        log_power = (log_power - LOG_PS_MEAN.unsqueeze(0).to(device=device, dtype=dtype)) / LOG_PS_STD.unsqueeze(0).to(device=device, dtype=dtype)
        power = log_power

    return power_k, power
