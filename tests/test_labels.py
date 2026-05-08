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


def test_map_detailed_to_coarse_does_not_handle_already_coarse_labels():
    """LabelsForTesting.nii ships in coarse space (0=bg, 1=CSF, 2=GM, 3=WM).

    map_detailed_to_coarse is meant for the 8-class LabelsForTraining.nii. If a
    caller mistakenly applies it to LabelsForTesting, value 1 (CSF) collides with
    detailed class 1 (cortical GM) and gets relabelled to coarse 2 (GM). This
    test pins that behaviour so any future caller knows to skip the remap.
    """
    coarse_input = np.array([0, 1, 2, 3])  # already coarse
    out = map_detailed_to_coarse(coarse_input)
    # If the function ever gains the ability to detect already-coarse input,
    # update this test together with all callers.
    assert out.tolist() == [0, 2, 2, 3]


def test_inverse_frequency_weights_downweights_background():
    freqs = np.array([0.9, 0.05, 0.04, 0.01])
    weights = inverse_frequency_weights(freqs, background_weight=0.1)
    assert abs(float(weights[0]) - 0.1) < 1e-6
    # The rarest class gets the largest weight.
    assert int(np.argmax(weights[1:])) == 2
