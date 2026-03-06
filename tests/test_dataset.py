# tests/test_dataset.py
import torch
import pytest


def test_reshape_field_output_shape():
    from cosmorford.dataset import reshape_field
    kappa = torch.randn(2, 1424, 176)
    result = reshape_field(kappa)
    assert result.shape == (2, 1834, 88)


def test_reshape_inverse_roundtrip():
    from cosmorford.dataset import reshape_field, inverse_reshape_field
    kappa = torch.randn(2, 1424, 176)
    result = inverse_reshape_field(reshape_field(kappa))
    # Check the regions that survive the roundtrip
    assert torch.allclose(result[:, :, :88], kappa[:, :, :88])
    assert torch.allclose(result[:, 620:1030, 88:], kappa[:, 620:1030, 88:])


def test_data_module_init():
    from cosmorford.dataset import WLDataModule
    dm = WLDataModule(batch_size=32, num_workers=0)
    assert dm.batch_size == 32
