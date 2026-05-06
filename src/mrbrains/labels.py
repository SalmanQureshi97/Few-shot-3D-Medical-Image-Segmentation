from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


DETAILED_LABELS: Dict[int, str] = {
    0: "background",
    1: "cortical_gray_matter",
    2: "basal_ganglia",
    3: "white_matter",
    4: "white_matter_lesions",
    5: "extracerebral_csf",
    6: "ventricles",
    7: "cerebellum",
    8: "brainstem",
}

COARSE_LABELS: Dict[int, str] = {
    0: "background",
    1: "csf",
    2: "gray_matter",
    3: "white_matter",
}

COARSE_TO_DETAILED = {
    1: [5, 6],
    2: [1, 2],
    3: [3, 4],
}


def label_names(target: str) -> Dict[int, str]:
    if target == "detailed":
        return DETAILED_LABELS
    if target == "coarse":
        return COARSE_LABELS
    raise ValueError(f"Unknown target: {target}")


def target_num_classes(target: str) -> int:
    return len(label_names(target))


def foreground_classes(target: str, include_background: bool = False) -> List[int]:
    names = label_names(target)
    classes = sorted(names)
    if not include_background:
        classes = [c for c in classes if c != 0]
    return classes


def map_detailed_to_coarse(
    labels: np.ndarray,
    ignore_index: int = 255,
    ignore_hindbrain: bool = True,
) -> np.ndarray:
    """Map MRBrainS detailed labels to background/CSF/GM/WM.

    The official MRBrainS13 test task evaluates CSF, GM and WM. Cerebellum and
    brainstem are excluded by the challenge; using ignore labels during coarse
    training keeps the loss from forcing those voxels into an arbitrary tissue.
    """
    labels = labels.astype(np.int16, copy=False)
    mapped = np.zeros(labels.shape, dtype=np.int64)
    mapped[np.isin(labels, COARSE_TO_DETAILED[1])] = 1
    mapped[np.isin(labels, COARSE_TO_DETAILED[2])] = 2
    mapped[np.isin(labels, COARSE_TO_DETAILED[3])] = 3
    if ignore_hindbrain:
        mapped[np.isin(labels, [7, 8])] = ignore_index
    return mapped


def remap_labels(
    labels: np.ndarray,
    target: str,
    ignore_index: int = 255,
    ignore_hindbrain_in_coarse: bool = True,
) -> np.ndarray:
    if target == "detailed":
        return labels.astype(np.int64)
    if target == "coarse":
        return map_detailed_to_coarse(labels, ignore_index, ignore_hindbrain_in_coarse)
    raise ValueError(f"Unknown target: {target}")


def class_counts(labels: np.ndarray, classes: Iterable[int], ignore_index: int = 255) -> Dict[int, int]:
    valid = labels != ignore_index
    return {int(c): int(np.logical_and(labels == c, valid).sum()) for c in classes}

