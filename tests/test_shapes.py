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


def test_resattn_deep_supervision_returns_one_per_level_minus_one():
    cfg = {
        "name": "resattn_unet3d",
        "in_channels": 3,
        "out_channels": 4,
        "base_channels": 4,
        "levels": 4,
        "norm": "instance",
        "dropout": 0.0,
        "deep_supervision": True,
    }
    model = build_model(cfg)
    x = torch.randn(1, 3, 16, 32, 32)
    y = model(x)
    assert isinstance(y, list)
    # Final logits + (levels-1) auxiliary heads.
    assert len(y) == 4
    assert y[0].shape == (1, 4, 16, 32, 32)


def test_unet_handles_non_cubic_patch():
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
    x = torch.randn(1, 3, 24, 48, 48)
    y = model(x)
    assert y.shape == (1, 4, 24, 48, 48)


def test_unet_with_gradient_checkpointing_runs():
    cfg = {
        "name": "unet3d",
        "in_channels": 3,
        "out_channels": 4,
        "base_channels": 4,
        "levels": 3,
        "norm": "instance",
        "dropout": 0.0,
        "gradient_checkpointing": True,
    }
    model = build_model(cfg).train()
    x = torch.randn(1, 3, 16, 32, 32, requires_grad=True)
    y = model(x)
    y.sum().backward()
    assert x.grad is not None
