from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


COARSE_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),
    1: (0.0, 0.55, 1.0, 0.55),
    2: (1.0, 0.85, 0.0, 0.55),
    3: (0.95, 0.15, 0.15, 0.55),
    255: (0.6, 0.6, 0.6, 0.25),
}


def label_rgba(label: np.ndarray, colors: Optional[Dict[int, Sequence[float]]] = None) -> np.ndarray:
    colors = colors or COARSE_COLORS
    rgba = np.zeros(label.shape + (4,), dtype=np.float32)
    for cls, color in colors.items():
        rgba[label == cls] = color
    return rgba


def choose_informative_slice(label: Optional[np.ndarray], image: np.ndarray) -> int:
    if label is not None:
        foreground = np.logical_and(label > 0, label != 255)
        counts = foreground.reshape(foreground.shape[0], -1).sum(axis=1)
        if counts.max() > 0:
            return int(np.argmax(counts))
    energy = np.abs(image[0]).reshape(image.shape[1], -1).sum(axis=1)
    return int(np.argmax(energy))


def save_overlay_panel(
    image: np.ndarray,
    target: Optional[np.ndarray],
    pred: Optional[np.ndarray],
    out_path: Union[str, Path],
    subject_id: str,
    slice_index: Optional[int] = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z = slice_index if slice_index is not None else choose_informative_slice(target, image)
    base = image[0, z]
    panels = [("T1/first modality", base, None)]
    if target is not None:
        panels.append(("Ground truth", base, target[z]))
    if pred is not None:
        panels.append(("Prediction", base, pred[z]))
    if target is not None and pred is not None:
        error = np.where(target[z] == 255, 0, (target[z] != pred[z]).astype(np.uint8))
        panels.append(("Error map", error, None))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, img, lab) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        if lab is not None:
            ax.imshow(label_rgba(lab))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{subject_id} | axial slice {z}")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_histograms(
    modality_values: Dict[str, np.ndarray],
    out_path: Union[str, Path],
    bins: int = 80,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for modality, values in modality_values.items():
        values = values[np.isfinite(values)]
        if values.size > 200000:
            values = np.random.default_rng(0).choice(values, 200000, replace=False)
        ax.hist(values, bins=bins, alpha=0.45, density=True, label=modality)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_class_frequency_bar(
    counts: Dict[str, Dict[int, int]],
    out_path: Union[str, Path],
    class_names: Dict[int, str],
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    totals = {cls: 0 for cls in class_names}
    for subject_counts in counts.values():
        for cls, value in subject_counts.items():
            totals[int(cls)] = totals.get(int(cls), 0) + int(value)
    labels = [class_names[c] for c in sorted(totals)]
    values = [totals[c] for c in sorted(totals)]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.bar(labels, values)
    ax.set_yscale("log")
    ax.set_ylabel("Voxel count (log)")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
