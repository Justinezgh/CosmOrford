# tests/test_compressor.py
import torch
import pytest


def test_compressor_forward_shapes():
    from cosmorford.compressor import CompressorModel
    model = CompressorModel(backbone="resnet18")
    x = torch.randn(4, 1834, 88)
    mean, scale = model(x)
    assert mean.shape == (4, 2)
    assert scale.shape == (4, 2)
    assert (scale > 0).all()


def test_compressor_bottleneck_output():
    from cosmorford.compressor import CompressorModel
    model = CompressorModel(backbone="resnet18")
    x = torch.randn(4, 1834, 88)
    z = model.compress(x)
    assert z.shape == (4, 8)


def test_compressor_training_step():
    from cosmorford.compressor import CompressorModel
    model = CompressorModel(backbone="resnet18")
    x = torch.randn(4, 1834, 88)
    y = torch.randn(4, 2)
    loss = model.training_step((x, y), 0)
    assert loss.shape == ()
    assert loss.requires_grad


def test_compressor_cosine_schedule():
    from cosmorford.compressor import CompressorModel
    model = CompressorModel(backbone="resnet18", lr_schedule="cosine")
    assert model.hparams.lr_schedule == "cosine"


def test_compressor_step_schedule():
    from cosmorford.compressor import CompressorModel
    model = CompressorModel(backbone="resnet18", lr_schedule="step")
    assert model.hparams.lr_schedule == "step"
