import torch

from mrbrains.models import build_model


def test_unet_shape():
    cfg = {
        "name": "unet3d",
        "in_channels": 3,
        "out_channels": 4,
        "base_channels": 4,
        "levels": 3,
        "norm": "instance",
        "dropout": 0.0,
    }
    model = build_model(cfg)
    x = torch.randn(1, 3, 16, 32, 32)
    y = model(x)
    assert y.shape == (1, 4, 16, 32, 32)


def test_resattn_shape():
    cfg = {
        "name": "resattn_unet3d",
        "in_channels": 3,
        "out_channels": 4,
        "base_channels": 4,
        "levels": 3,
        "norm": "instance",
        "dropout": 0.0,
        "deep_supervision": True,
    }
    model = build_model(cfg)
    x = torch.randn(1, 3, 16, 32, 32)
    y = model(x)
    assert isinstance(y, list)
    assert y[0].shape == (1, 4, 16, 32, 32)

