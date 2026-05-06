from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
from scipy import ndimage


def dice_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    classes: Iterable[int],
    ignore_index: int = 255,
) -> Dict[int, float]:
    valid = target != ignore_index
    scores: Dict[int, float] = {}
    for cls in classes:
        pred_c = np.logical_and(pred == cls, valid)
        target_c = np.logical_and(target == cls, valid)
        denom = pred_c.sum() + target_c.sum()
        if denom == 0:
            scores[int(cls)] = np.nan
        else:
            scores[int(cls)] = float(2.0 * np.logical_and(pred_c, target_c).sum() / denom)
    return scores


def volume_similarity(
    pred: np.ndarray,
    target: np.ndarray,
    classes: Iterable[int],
    ignore_index: int = 255,
) -> Dict[int, float]:
    valid = target != ignore_index
    scores: Dict[int, float] = {}
    for cls in classes:
        pred_v = np.logical_and(pred == cls, valid).sum()
        target_v = np.logical_and(target == cls, valid).sum()
        denom = pred_v + target_v
        if denom == 0:
            scores[int(cls)] = np.nan
        else:
            scores[int(cls)] = float(1.0 - abs(pred_v - target_v) / denom)
    return scores


def _surface(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(bool)
    eroded = ndimage.binary_erosion(mask, iterations=1, border_value=0)
    return np.logical_xor(mask, eroded)


def surface_distances(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    pred_surface = _surface(pred_mask)
    target_surface = _surface(target_mask)
    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return np.array([], dtype=np.float32)
    dt_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    return np.concatenate([dt_target[pred_surface], dt_pred[target_surface]]).astype(np.float32)


def hd95_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    classes: Iterable[int],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    ignore_index: int = 255,
) -> Dict[int, float]:
    valid = target != ignore_index
    scores: Dict[int, float] = {}
    for cls in classes:
        distances = surface_distances(np.logical_and(pred == cls, valid), np.logical_and(target == cls, valid), spacing)
        scores[int(cls)] = float(np.percentile(distances, 95)) if distances.size else np.nan
    return scores


def assd_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    classes: Iterable[int],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    ignore_index: int = 255,
) -> Dict[int, float]:
    valid = target != ignore_index
    scores: Dict[int, float] = {}
    for cls in classes:
        distances = surface_distances(np.logical_and(pred == cls, valid), np.logical_and(target == cls, valid), spacing)
        scores[int(cls)] = float(distances.mean()) if distances.size else np.nan
    return scores


def summarise_metric_dict(prefix: str, values: Dict[int, float]) -> Dict[str, float]:
    out = {f"{prefix}_class_{cls}": score for cls, score in values.items()}
    finite = [v for v in values.values() if np.isfinite(v)]
    out[f"{prefix}_mean"] = float(np.mean(finite)) if finite else np.nan
    return out

