from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

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

DETAILED_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),
    1: (1.0, 0.85, 0.0, 0.55),  # cortical GM
    2: (1.0, 0.55, 0.0, 0.55),  # basal ganglia
    3: (0.95, 0.15, 0.15, 0.55),  # WM
    4: (0.7, 0.0, 0.0, 0.6),  # WM lesions
    5: (0.0, 0.55, 1.0, 0.55),  # extracerebral CSF
    6: (0.0, 0.85, 1.0, 0.55),  # ventricles
    7: (0.6, 0.0, 0.7, 0.55),  # cerebellum
    8: (0.4, 0.0, 0.5, 0.6),  # brainstem
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
    label_colors: Optional[Dict[int, Sequence[float]]] = None,
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
            ax.imshow(label_rgba(lab, label_colors))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{subject_id} | axial slice {z}")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_multi_modality_panel(
    image: np.ndarray,
    target: Optional[np.ndarray],
    pred: Optional[np.ndarray],
    modality_names: Sequence[str],
    out_path: Union[str, Path],
    subject_id: str,
    slice_index: Optional[int] = None,
    label_colors: Optional[Dict[int, Sequence[float]]] = None,
) -> None:
    """Side-by-side T1 / IR / FLAIR panels with optional GT and prediction overlays."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z = slice_index if slice_index is not None else choose_informative_slice(target, image)
    n_mods = image.shape[0]
    cols = n_mods + (1 if target is not None else 0) + (1 if pred is not None else 0)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4), constrained_layout=True)
    if cols == 1:
        axes = [axes]
    col = 0
    for m in range(n_mods):
        axes[col].imshow(image[m, z], cmap="gray")
        axes[col].set_title(modality_names[m] if m < len(modality_names) else f"mod{m}")
        axes[col].axis("off")
        col += 1
    if target is not None:
        axes[col].imshow(image[0, z], cmap="gray")
        axes[col].imshow(label_rgba(target[z], label_colors))
        axes[col].set_title("Ground truth")
        axes[col].axis("off")
        col += 1
    if pred is not None:
        axes[col].imshow(image[0, z], cmap="gray")
        axes[col].imshow(label_rgba(pred[z], label_colors))
        axes[col].set_title("Prediction")
        axes[col].axis("off")
    fig.suptitle(f"{subject_id} | axial slice {z}")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_per_class_error_panel(
    target: np.ndarray,
    pred: np.ndarray,
    classes: Sequence[int],
    class_names: Dict[int, str],
    out_path: Union[str, Path],
    subject_id: str,
    slice_index: Optional[int] = None,
) -> None:
    """Per-class FN/FP map: red=missed (FN), blue=hallucinated (FP)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid = target != 255
    if slice_index is None:
        z = int(np.argmax(np.logical_and(target > 0, valid).reshape(target.shape[0], -1).sum(axis=1)))
    else:
        z = int(slice_index)
    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes):
        fn = np.logical_and(target[z] == cls, pred[z] != cls) & valid[z]
        fp = np.logical_and(target[z] != cls, pred[z] == cls) & valid[z]
        canvas = np.zeros(target.shape[1:] + (4,), dtype=np.float32)
        canvas[fn] = (0.95, 0.15, 0.15, 0.85)
        canvas[fp] = (0.15, 0.45, 0.95, 0.85)
        ax.imshow(np.zeros_like(target[z]), cmap="gray")
        ax.imshow(canvas)
        ax.set_title(f"{class_names.get(int(cls), int(cls))}\nFN red / FP blue")
        ax.axis("off")
    fig.suptitle(f"{subject_id} | slice {z}")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_path: Union[str, Path],
    title: str = "Voxel confusion (row-normalised)",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm = cm.astype(np.float64)
    row_sums = norm.sum(axis=1, keepdims=True)
    norm = np.divide(norm, row_sums, out=np.zeros_like(norm), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(0.7 * len(class_names) + 2, 0.7 * len(class_names) + 2), constrained_layout=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            colour = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", color=colour, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_histograms(
    modality_values: Dict[str, np.ndarray],
    out_path: Union[str, Path],
    bins: int = 80,
    title: str = "",
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
    if title:
        ax.set_title(title)
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


def save_per_class_dice_barplot(
    rows: List[Dict[str, float]],
    class_names: Dict[int, str],
    out_path: Union[str, Path],
    title: str = "",
) -> None:
    """Bar plot of per-class Dice. `rows` are method-aggregated dicts.

    Each row must have keys: experiment, dice_class_<i>.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    classes = [c for c in class_names if c != 0]
    methods = [r["experiment"] for r in rows]
    values = np.array(
        [[float(r.get(f"dice_class_{c}", float("nan"))) for c in classes] for r in rows],
        dtype=np.float64,
    )
    n_methods = len(methods)
    n_classes = len(classes)
    width = max(0.1, 0.8 / max(1, n_methods))
    x = np.arange(n_classes)
    fig, ax = plt.subplots(figsize=(2 + 1.5 * n_classes, 4), constrained_layout=True)
    for i, method in enumerate(methods):
        ax.bar(x + i * width - 0.4 + width / 2, values[i], width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([class_names[c] for c in classes], rotation=20, ha="right")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
