from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

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

# MRBrainS13 official 3-class evaluation pools detailed labels into CSF / GM / WM.
COARSE_TO_DETAILED: Dict[int, List[int]] = {
    1: [5, 6],
    2: [1, 2],
    3: [3, 4],
}

# Hindbrain (cerebellum + brainstem) is excluded from the official MRBrainS13 leaderboard.
HINDBRAIN_DETAILED = [7, 8]


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
    """Map MRBrainS13 detailed labels to background/CSF/GM/WM.

    The official MRBrainS13 test task evaluates CSF, GM and WM only. Cerebellum and
    brainstem are excluded by the challenge; using the ignore index during coarse
    training keeps the loss from forcing those voxels into an arbitrary tissue.
    """
    labels = labels.astype(np.int16, copy=False)
    mapped = np.zeros(labels.shape, dtype=np.int64)
    mapped[np.isin(labels, COARSE_TO_DETAILED[1])] = 1
    mapped[np.isin(labels, COARSE_TO_DETAILED[2])] = 2
    mapped[np.isin(labels, COARSE_TO_DETAILED[3])] = 3
    if ignore_hindbrain:
        mapped[np.isin(labels, HINDBRAIN_DETAILED)] = ignore_index
    return mapped


def remap_labels(
    labels: np.ndarray,
    target: str,
    ignore_index: int = 255,
    ignore_hindbrain_in_coarse: bool = True,
    ignore_hindbrain_in_detailed: bool = False,
) -> np.ndarray:
    if target == "detailed":
        out = labels.astype(np.int64)
        if ignore_hindbrain_in_detailed:
            out = np.where(np.isin(out, HINDBRAIN_DETAILED), ignore_index, out)
        return out
    if target == "coarse":
        return map_detailed_to_coarse(labels, ignore_index, ignore_hindbrain_in_coarse)
    raise ValueError(f"Unknown target: {target}")


def class_counts(labels: np.ndarray, classes: Iterable[int], ignore_index: int = 255) -> Dict[int, int]:
    valid = labels != ignore_index
    return {int(c): int(np.logical_and(labels == c, valid).sum()) for c in classes}


def class_voxel_frequencies(
    label_volumes: Sequence[np.ndarray],
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """Voxel frequency per class across a list of label volumes.

    Returns a length-`num_classes` array summing to ~1.0 (the ignore index is
    excluded from the denominator).
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for label in label_volumes:
        valid = label != ignore_index
        flat = label[valid]
        bins = np.bincount(flat.astype(np.int64), minlength=num_classes)
        counts[: len(bins)] += bins[:num_classes]
    total = counts.sum()
    if total <= 0:
        return np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    return counts / total


def inverse_frequency_weights(
    frequencies: np.ndarray,
    smoothing: float = 1e-3,
    background_weight: float = 0.1,
) -> np.ndarray:
    """Inverse-frequency class weights for cross-entropy / weighted Dice.

    A small additive smoothing prevents divide-by-zero on absent classes; the
    background weight is downweighted explicitly because it dominates volume.
    """
    freq = np.maximum(frequencies, smoothing)
    weights = 1.0 / freq
    weights = weights / weights.mean()
    if len(weights) > 0:
        weights[0] = float(background_weight)
    return weights.astype(np.float32)
