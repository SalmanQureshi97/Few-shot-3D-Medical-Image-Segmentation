import numpy as np

from mrbrains.metrics import (
    assd_per_class,
    avd_per_class,
    confusion_matrix,
    dice_per_class,
    hd95_per_class,
    summarise_metric_dict,
    surface_distances,
    volume_similarity,
)


def test_dice_perfect_overlap_is_one():
    target = np.zeros((4, 4, 4), dtype=np.int64)
    target[1:3, 1:3, 1:3] = 1
    pred = target.copy()
    scores = dice_per_class(pred, target, classes=[1])
    assert abs(scores[1] - 1.0) < 1e-6


def test_dice_zero_overlap_is_zero():
    target = np.zeros((4, 4, 4), dtype=np.int64)
    pred = np.zeros_like(target)
    target[0, 0, 0] = 1
    pred[3, 3, 3] = 1
    scores = dice_per_class(pred, target, classes=[1])
    assert scores[1] == 0.0


def test_dice_class_absent_is_nan():
    target = np.zeros((3, 3, 3), dtype=np.int64)
    pred = np.zeros_like(target)
    scores = dice_per_class(pred, target, classes=[1])
    assert np.isnan(scores[1])


def test_avd_matches_definition():
    target = np.zeros((4, 4, 4), dtype=np.int64)
    target[:2] = 1  # 32 voxels
    pred = np.zeros_like(target)
    pred[:1] = 1  # 16 voxels -> AVD 50%
    scores = avd_per_class(pred, target, classes=[1])
    assert abs(scores[1] - 50.0) < 1e-6


def test_avd_target_zero_returns_nan():
    target = np.zeros((3, 3, 3), dtype=np.int64)
    pred = np.zeros_like(target)
    pred[0, 0, 0] = 1
    assert np.isnan(avd_per_class(pred, target, classes=[1])[1])


def test_volume_similarity_symmetry():
    target = np.zeros((4, 4, 4), dtype=np.int64)
    target[:2] = 1
    pred = np.zeros_like(target)
    pred[:3] = 1
    a = volume_similarity(pred, target, classes=[1])[1]
    b = volume_similarity(target, pred, classes=[1])[1]
    assert abs(a - b) < 1e-6


def test_surface_distances_both_empty_is_empty():
    a = np.zeros((4, 4, 4), dtype=bool)
    b = np.zeros((4, 4, 4), dtype=bool)
    assert surface_distances(a, b).size == 0


def test_surface_distances_one_empty_is_finite_penalty():
    a = np.zeros((4, 4, 4), dtype=bool)
    b = np.zeros((4, 4, 4), dtype=bool)
    b[1:3, 1:3, 1:3] = True
    distances = surface_distances(a, b)
    assert distances.size == 1 and np.isfinite(distances[0])


def test_hd95_returns_zero_on_perfect_match():
    target = np.zeros((6, 6, 6), dtype=np.int64)
    target[2:4, 2:4, 2:4] = 1
    assert hd95_per_class(target, target, classes=[1])[1] == 0.0


def test_assd_finite_for_overlapping_shapes():
    target = np.zeros((6, 6, 6), dtype=np.int64)
    target[2:4, 2:4, 2:4] = 1
    pred = np.zeros_like(target)
    pred[3:5, 3:5, 3:5] = 1
    score = assd_per_class(pred, target, classes=[1])[1]
    assert np.isfinite(score) and score >= 0.0


def test_confusion_matrix_diagonal_for_perfect_pred():
    target = np.array([0, 1, 2, 1, 0])
    pred = target.copy()
    cm = confusion_matrix(pred, target, num_classes=3)
    assert cm.shape == (3, 3) and (cm == np.diag([2, 2, 1])).all()


def test_confusion_matrix_excludes_ignore_index():
    target = np.array([255, 1, 2])
    pred = np.array([0, 1, 2])
    cm = confusion_matrix(pred, target, num_classes=3, ignore_index=255)
    assert cm.sum() == 2


def test_summarise_metric_handles_nan():
    summary = summarise_metric_dict("dice", {1: 0.5, 2: float("nan")})
    assert abs(summary["dice_mean"] - 0.5) < 1e-6
    assert "dice_class_2" in summary
