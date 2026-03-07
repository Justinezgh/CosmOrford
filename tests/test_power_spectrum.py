"""Test that the optimized power_spectrum_batch matches the original loop-based implementation."""
import torch
import numpy as np
import pytest


def power_spectrum_batch_original(x, pixsize=2. / 60 / 180 * np.pi, kedge=np.logspace(2, 4, 11), normalize=True):
    """Original loop-based implementation for reference."""
    from cosmorford.summaries import LOG_PS_MEAN, LOG_PS_STD

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

    Nmode = torch.zeros(batch_size, n_bins, device=device, dtype=dtype)
    power_k = torch.zeros(batch_size, n_bins, device=device, dtype=dtype)
    power = torch.zeros(batch_size, n_bins, device=device, dtype=dtype)

    for b in range(batch_size):
        Nmode[b].index_add_(0, index.flatten(), torch.ones(ny * (nx // 2 + 1), device=device, dtype=dtype))
        power_k[b].index_add_(0, index.flatten(), k.flatten())
        power[b].index_add_(0, index.flatten(), xk2[b].flatten())

    # Mirror
    if nx % 2 == 0:
        mirror_slice = slice(1, -1)
    else:
        mirror_slice = slice(1, None)

    for b in range(batch_size):
        Nmode[b].index_add_(0, index[:, mirror_slice].flatten(),
                            torch.ones(index[:, mirror_slice].numel(), device=device, dtype=dtype))
        power_k[b].index_add_(0, index[:, mirror_slice].flatten(), k[:, mirror_slice].flatten())
        power[b].index_add_(0, index[:, mirror_slice].flatten(), xk2[b, :, mirror_slice].flatten())

    select = Nmode > 0
    power_k[select] = power_k[select] / Nmode[select]
    power[select] = power[select] / Nmode[select]

    power_k = power_k[:, 1:nk + 1]
    power *= pixsize ** 2 / ny / nx
    power = power[:, 1:nk + 1]

    if normalize:
        log_power = torch.log10(power + 1e-30)
        log_power = (log_power - LOG_PS_MEAN.unsqueeze(0).to(device=device, dtype=dtype)) / LOG_PS_STD.unsqueeze(0).to(device=device, dtype=dtype)
        power = log_power

    return power_k, power


def test_power_spectrum_matches_original():
    from cosmorford.summaries import power_spectrum_batch

    torch.manual_seed(42)
    x = torch.randn(8, 88, 88)

    pk_orig, ps_orig = power_spectrum_batch_original(x, normalize=True)
    pk_new, ps_new = power_spectrum_batch(x, normalize=True)

    torch.testing.assert_close(pk_new, pk_orig, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ps_new, ps_orig, atol=1e-5, rtol=1e-5)


def test_power_spectrum_unnormalized():
    from cosmorford.summaries import power_spectrum_batch

    torch.manual_seed(123)
    x = torch.randn(4, 88, 88)

    pk_orig, ps_orig = power_spectrum_batch_original(x, normalize=False)
    pk_new, ps_new = power_spectrum_batch(x, normalize=False)

    torch.testing.assert_close(pk_new, pk_orig, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ps_new, ps_orig, atol=1e-5, rtol=1e-5)


def test_power_spectrum_non_square():
    from cosmorford.summaries import power_spectrum_batch

    torch.manual_seed(7)
    x = torch.randn(2, 64, 128)

    pk_orig, ps_orig = power_spectrum_batch_original(x, normalize=False)
    pk_new, ps_new = power_spectrum_batch(x, normalize=False)

    torch.testing.assert_close(pk_new, pk_orig, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ps_new, ps_orig, atol=1e-5, rtol=1e-5)


def test_power_spectrum_odd_width():
    from cosmorford.summaries import power_spectrum_batch

    torch.manual_seed(99)
    x = torch.randn(3, 88, 89)  # odd nx

    pk_orig, ps_orig = power_spectrum_batch_original(x, normalize=False)
    pk_new, ps_new = power_spectrum_batch(x, normalize=False)

    torch.testing.assert_close(pk_new, pk_orig, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ps_new, ps_orig, atol=1e-5, rtol=1e-5)
