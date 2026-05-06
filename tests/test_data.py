import random

import numpy as np

from mrbrains.data import (
    affine_augment,
    augment_patch,
    class_balanced_center,
    extract_patch,
    foreground_center,
    pad_to_shape,
    random_center,
    robust_normalise,
)


def test_robust_normalise_zero_mean_unit_std_on_foreground():
    rng = np.random.default_rng(0)
    volume = rng.normal(loc=100.0, scale=10.0, size=(8, 8, 8)).astype(np.float32)
    fg = np.ones_like(volume, dtype=bool)
    out = robust_normalise(volume, fg, clip_percentiles=(1.0, 99.0))
    assert abs(out[fg].mean()) < 0.5
    assert abs(out[fg].std() - 1.0) < 0.5


def test_extract_patch_pads_when_outside_bounds():
    array = np.arange(27).reshape(3, 3, 3).astype(np.float32)
    patch = extract_patch(array, center=(0, 0, 0), patch_size=(3, 3, 3))
    assert patch.shape == (3, 3, 3)


def test_extract_patch_with_channels():
    array = np.zeros((2, 4, 4, 4), dtype=np.float32)
    patch = extract_patch(array, center=(2, 2, 2), patch_size=(2, 2, 2))
    assert patch.shape == (2, 2, 2, 2)


def test_pad_to_shape_no_op_when_large_enough():
    array = np.zeros((5, 5, 5), dtype=np.float32)
    padded = pad_to_shape(array, shape=(3, 3, 3))
    assert padded.shape == (5, 5, 5)


def test_random_center_inside_bounds():
    random.seed(0)
    centre = random_center((4, 4, 4))
    assert centre.shape == (3,)
    assert (centre >= 0).all() and (centre < 4).all()


def test_foreground_center_returns_foreground_voxel():
    random.seed(0)
    label = np.zeros((4, 4, 4), dtype=np.int64)
    label[1, 2, 3] = 1
    centre = foreground_center(label, ignore_index=255, fallback_shape=(4, 4, 4))
    assert label[tuple(centre)] == 1


def test_class_balanced_center_picks_target_class():
    random.seed(0)
    label = np.zeros((4, 4, 4), dtype=np.int64)
    label[0, 0, 0] = 1
    label[3, 3, 3] = 2
    centre = class_balanced_center(label, ignore_index=255, classes=[2], fallback_shape=(4, 4, 4))
    assert label[tuple(centre)] == 2


def test_affine_augment_preserves_shape_and_label_dtype():
    image = np.random.randn(2, 4, 8, 8).astype(np.float32)
    label = np.zeros((4, 8, 8), dtype=np.int64)
    label[1, 2:6, 2:6] = 1
    img, lab = affine_augment(image, label, rotate_deg=10.0, scale=1.05, ignore_index=255)
    assert img.shape == image.shape and img.dtype == np.float32
    assert lab.shape == label.shape and lab.dtype == np.int64


def test_augment_patch_disabled_is_identity():
    image = np.ones((3, 4, 4, 4), dtype=np.float32)
    label = np.zeros((4, 4, 4), dtype=np.int64)
    img, lab = augment_patch(image, label, cfg={"enabled": False}, ignore_index=255)
    assert (img == image).all() and (lab == label).all()


def test_augment_patch_intensity_changes_image():
    random.seed(0)
    np.random.seed(0)
    image = np.random.rand(2, 4, 4, 4).astype(np.float32)
    label = np.zeros((4, 4, 4), dtype=np.int64)
    cfg = {"enabled": True, "intensity_probability": 1.0, "noise_std": 0.0, "gamma_range": [0.8, 1.2]}
    img, _ = augment_patch(image.copy(), label, cfg=cfg, ignore_index=255)
    assert not np.allclose(img, image)
