import numpy as np

from mrbrains.labels import (
    class_voxel_frequencies,
    foreground_classes,
    inverse_frequency_weights,
    map_detailed_to_coarse,
    remap_labels,
)


def test_map_detailed_to_coarse_ignores_hindbrain():
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    mapped = map_detailed_to_coarse(labels, ignore_index=255, ignore_hindbrain=True)
    assert mapped.tolist() == [0, 2, 2, 3, 3, 1, 1, 255, 255]


def test_map_detailed_to_coarse_keeps_hindbrain_when_disabled():
    labels = np.array([0, 7, 8])
    mapped = map_detailed_to_coarse(labels, ignore_index=255, ignore_hindbrain=False)
    # Hindbrain falls through to background under the coarse mapping.
    assert mapped.tolist() == [0, 0, 0]


def test_remap_labels_detailed_passthrough():
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    out = remap_labels(labels, target="detailed")
    assert out.tolist() == labels.tolist()


def test_remap_labels_detailed_can_ignore_hindbrain():
    labels = np.array([0, 1, 7, 8])
    out = remap_labels(labels, target="detailed", ignore_hindbrain_in_detailed=True)
    assert out.tolist() == [0, 1, 255, 255]


def test_foreground_classes_excludes_background():
    assert foreground_classes("coarse") == [1, 2, 3]
    assert foreground_classes("detailed") == [1, 2, 3, 4, 5, 6, 7, 8]
    assert 0 in foreground_classes("coarse", include_background=True)


def test_class_voxel_frequencies_sums_to_one():
    label = np.array([0, 1, 1, 2, 2, 2, 255])
    freqs = class_voxel_frequencies([label], num_classes=4, ignore_index=255)
    assert abs(freqs.sum() - 1.0) < 1e-6
    assert freqs[0] == 1 / 6 and freqs[2] == 3 / 6


def test_inverse_frequency_weights_downweights_background():
    freqs = np.array([0.9, 0.05, 0.04, 0.01])
    weights = inverse_frequency_weights(freqs, background_weight=0.1)
    assert abs(float(weights[0]) - 0.1) < 1e-6
    # The rarest class gets the largest weight.
    assert int(np.argmax(weights[1:])) == 2
