import numpy as np

from mrbrains.labels import map_detailed_to_coarse


def test_map_detailed_to_coarse_ignores_hindbrain():
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    mapped = map_detailed_to_coarse(labels, ignore_index=255, ignore_hindbrain=True)
    assert mapped.tolist() == [0, 2, 2, 3, 3, 1, 1, 255, 255]

