import torch

from mrbrains.infer import _compute_starts, predict_with_tta, sliding_window_predict
from mrbrains.models import build_model


def _tiny_model(num_classes: int = 4) -> torch.nn.Module:
    cfg = {
        "name": "unet3d",
        "in_channels": 3,
        "out_channels": num_classes,
        "base_channels": 4,
        "levels": 2,
    }
    return build_model(cfg)


def test_compute_starts_covers_full_volume():
    starts = _compute_starts(size=10, roi=4, overlap=0.5)
    # Every voxel index in [0, 10) is in at least one [start, start+roi) window.
    coverage = [False] * 10
    for s in starts:
        for i in range(s, min(s + 4, 10)):
            coverage[i] = True
    assert all(coverage)


def test_compute_starts_when_size_smaller_than_roi():
    assert _compute_starts(size=2, roi=4, overlap=0.5) == [0]


def test_sliding_window_predict_shape_matches_input():
    torch.manual_seed(0)
    model = _tiny_model(num_classes=4)
    image = torch.randn(3, 8, 16, 16)
    probs = sliding_window_predict(
        model, image, roi_size=(8, 8, 8), num_classes=4,
        device=torch.device("cpu"), sw_batch_size=1, overlap=0.25, use_amp=False,
    )
    assert probs.shape == (4, 8, 16, 16)
    # softmax probabilities sum to 1 along class dim.
    sums = probs.sum(dim=0)
    assert torch.allclose(sums, torch.ones_like(sums), atol=5e-3)


def test_sliding_window_predict_handles_volume_smaller_than_roi():
    torch.manual_seed(0)
    model = _tiny_model(num_classes=2)
    image = torch.randn(3, 4, 6, 6)
    probs = sliding_window_predict(
        model, image, roi_size=(8, 8, 8), num_classes=2,
        device=torch.device("cpu"), sw_batch_size=1, use_amp=False,
    )
    assert probs.shape == (2, 4, 6, 6)


def test_predict_with_tta_averages_flips():
    torch.manual_seed(0)
    model = _tiny_model(num_classes=2)
    image = torch.randn(3, 8, 16, 16)
    probs = predict_with_tta(
        model, image, roi_size=(8, 8, 8), num_classes=2,
        device=torch.device("cpu"), sw_batch_size=1, tta_flips=[[3]], use_amp=False,
    )
    assert probs.shape == (2, 8, 16, 16)
    sums = probs.sum(dim=0)
    assert torch.allclose(sums, torch.ones_like(sums), atol=5e-3)
