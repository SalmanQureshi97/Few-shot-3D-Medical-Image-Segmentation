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
            scores[int(cls)] = float("nan")
        else:
            scores[int(cls)] = float(2.0 * np.logical_and(pred_c, target_c).sum() / denom)
    return scores


def volume_similarity(
    pred: np.ndarray,
    target: np.ndarray,
    classes: Iterable[int],
    ignore_index: int = 255,
) -> Dict[int, float]:
    """Symmetric volume similarity: 1 - |Vp - Vt| / (Vp + Vt). Range 0..1."""
    valid = target != ignore_index
    scores: Dict[int, float] = {}
    for cls in classes:
        pred_v = np.logical_and(pred == cls, valid).sum()
        target_v = np.logical_and(target == cls, valid).sum()
        denom = pred_v + target_v
        if denom == 0:
            scores[int(cls)] = float("nan")
        else:
            scores[int(cls)] = float(1.0 - abs(pred_v - target_v) / denom)
    return scores


def avd_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    classes: Iterable[int],
    ignore_index: int = 255,
) -> Dict[int, float]:
    """Absolute volume difference (%) — the metric used in MRBrainS13.

    AVD = 100 * |V_pred - V_target| / V_target. Returns NaN when the target
    volume for a class is zero (cannot divide by zero in a meaningful way).
    """
    valid = target != ignore_index
    scores: Dict[int, float] = {}
    for cls in classes:
        pred_v = float(np.logical_and(pred == cls, valid).sum())
        target_v = float(np.logical_and(target == cls, valid).sum())
        if target_v == 0:
            scores[int(cls)] = float("nan")
        else:
            scores[int(cls)] = float(100.0 * abs(pred_v - target_v) / target_v)
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
    """Symmetric surface distances in mm.

    - Both surfaces empty -> empty array (caller maps to NaN: class absent).
    - Exactly one surface empty -> diagonal of bounding box (caller maps to
      a finite penalty so missed structures register a real failure rather
      than NaN-poisoning the average).
    """
    pred_surface = _surface(pred_mask)
    target_surface = _surface(target_mask)
    p_n = int(pred_surface.sum())
    t_n = int(target_surface.sum())
    if p_n == 0 and t_n == 0:
        return np.array([], dtype=np.float32)
    if p_n == 0 or t_n == 0:
        diag = float(np.sqrt(np.sum((np.array(pred_mask.shape) * np.asarray(spacing)) ** 2)))
        return np.array([diag], dtype=np.float32)
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
        distances = surface_distances(
            np.logical_and(pred == cls, valid),
            np.logical_and(target == cls, valid),
            spacing,
        )
        scores[int(cls)] = float(np.percentile(distances, 95)) if distances.size else float("nan")
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
        distances = surface_distances(
            np.logical_and(pred == cls, valid),
            np.logical_and(target == cls, valid),
            spacing,
        )
        scores[int(cls)] = float(distances.mean()) if distances.size else float("nan")
    return scores


def confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """Voxel-level confusion matrix of shape (C, C). Rows = true, cols = pred."""
    valid = target != ignore_index
    t = target[valid].astype(np.int64).copy()
    p = pred[valid].astype(np.int64).copy()
    np.clip(t, 0, num_classes - 1, out=t)
    np.clip(p, 0, num_classes - 1, out=p)
    flat = t * num_classes + p
    binc = np.bincount(flat, minlength=num_classes * num_classes)
    return binc.reshape(num_classes, num_classes)


def summarise_metric_dict(prefix: str, values: Dict[int, float]) -> Dict[str, float]:
    out = {f"{prefix}_class_{cls}": score for cls, score in values.items()}
    finite = [v for v in values.values() if np.isfinite(v)]
    out[f"{prefix}_mean"] = float(np.mean(finite)) if finite else float("nan")
    return out
