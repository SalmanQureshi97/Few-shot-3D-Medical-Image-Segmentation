from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import Subject, VolumeMeta, load_volume, to_dhw
from .labels import remap_labels


@dataclass
class LoadedSubject:
    subject_id: str
    image: np.ndarray
    label: Optional[np.ndarray]
    meta: VolumeMeta
    spacing_dhw: tuple[float, float, float]


def robust_normalise(
    volume: np.ndarray,
    foreground: np.ndarray,
    clip_percentiles: Sequence[float] = (1.0, 99.0),
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-modality robust z-score: clip to foreground percentiles, then standardise.

    Z-scoring on the foreground (non-air) voxels avoids the air peak dominating
    statistics, which is a known issue for MR scans. The clip percentiles
    suppress hyper-intense outliers (e.g. metal artifacts, fat shifts).
    """
    values = volume[foreground]
    if values.size < 32:
        values = volume.reshape(-1)
    lo, hi = np.percentile(values, clip_percentiles)
    volume = np.clip(volume, lo, hi)
    values = volume[foreground] if foreground.any() else volume.reshape(-1)
    mean = float(values.mean())
    std = float(values.std())
    return ((volume - mean) / (std + eps)).astype(np.float32)


def load_subject(
    subject: Subject,
    modalities: Iterable[str],
    target: str,
    ignore_index: int = 255,
    ignore_hindbrain_in_coarse: bool = True,
    clip_percentiles: Sequence[float] = (1.0, 99.0),
    ignore_hindbrain_in_detailed: bool = False,
) -> LoadedSubject:
    images = []
    first_meta: Optional[VolumeMeta] = None
    foreground_mask: Optional[np.ndarray] = None
    raw_by_modality: Dict[str, np.ndarray] = {}

    for modality in modalities:
        array_xyz, meta = load_volume(subject.image_paths[modality])
        if first_meta is None:
            first_meta = meta
        array = to_dhw(array_xyz.astype(np.float32))
        raw_by_modality[modality] = array
        modality_fg = np.isfinite(array) & (np.abs(array) > 1e-6)
        foreground_mask = modality_fg if foreground_mask is None else (foreground_mask | modality_fg)

    assert first_meta is not None
    assert foreground_mask is not None

    for modality in modalities:
        normed = robust_normalise(raw_by_modality[modality], foreground_mask, clip_percentiles)
        images.append(normed)

    label = None
    if subject.label_path is not None:
        label_xyz, _ = load_volume(subject.label_path)
        label = to_dhw(label_xyz).astype(np.int64)
        label = remap_labels(
            label,
            target,
            ignore_index,
            ignore_hindbrain_in_coarse,
            ignore_hindbrain_in_detailed,
        )

    spacing_xyz = first_meta.spacing
    spacing_dhw = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    return LoadedSubject(
        subject_id=subject.subject_id,
        image=np.stack(images, axis=0).astype(np.float32),
        label=label,
        meta=first_meta,
        spacing_dhw=spacing_dhw,
    )


def cached_loaded_subjects(
    subjects: Iterable[Subject],
    modalities: Iterable[str],
    target: str,
    ignore_index: int,
    ignore_hindbrain_in_coarse: bool,
    clip_percentiles: Sequence[float],
    cache_dir: Optional[Path] = None,
    ignore_hindbrain_in_detailed: bool = False,
) -> List[LoadedSubject]:
    """Load subjects, optionally caching the normalised volumes to .npz on disk.

    Useful in LOOCV where the same five subjects are loaded ten times — disk
    caching converts that into a single decode + zscore pass.
    """
    modalities = list(modalities)
    out: List[LoadedSubject] = []
    for subject in subjects:
        cache_path: Optional[Path] = None
        if cache_dir is not None:
            tag = f"{subject.subject_id}__{target}__" + "-".join(modalities)
            cache_path = Path(cache_dir) / f"{tag}.npz"
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as data:
                image = data["image"].astype(np.float32)
                label = data["label"].astype(np.int64) if "label" in data.files else None
                spacing = tuple(float(v) for v in data["spacing_dhw"])
            # We still need a VolumeMeta — load just the affine from the first modality.
            _, meta = load_volume(subject.image_paths[modalities[0]])
            out.append(LoadedSubject(subject.subject_id, image, label, meta, spacing))
            continue
        loaded = load_subject(
            subject,
            modalities,
            target,
            ignore_index,
            ignore_hindbrain_in_coarse,
            clip_percentiles,
            ignore_hindbrain_in_detailed,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload: Dict[str, np.ndarray] = {
                "image": loaded.image.astype(np.float32),
                "spacing_dhw": np.asarray(loaded.spacing_dhw, dtype=np.float32),
            }
            if loaded.label is not None:
                payload["label"] = loaded.label.astype(np.int16)
            np.savez_compressed(cache_path, **payload)
        out.append(loaded)
    return out


def pad_to_shape(array: np.ndarray, shape: Sequence[int], value: Union[float, int] = 0) -> np.ndarray:
    spatial = array.shape[-3:]
    pad_width = [(0, 0)] * (array.ndim - 3)
    for current, target in zip(spatial, shape):
        missing = max(0, int(target) - int(current))
        before = missing // 2
        after = missing - before
        pad_width.append((before, after))
    if all(before == 0 and after == 0 for before, after in pad_width):
        return array
    return np.pad(array, pad_width, mode="constant", constant_values=value)


def extract_patch(
    array: np.ndarray,
    center: Sequence[int],
    patch_size: Sequence[int],
    pad_value: Union[float, int] = 0,
) -> np.ndarray:
    patch_size = np.array(patch_size, dtype=np.int64)
    center = np.array(center, dtype=np.int64)
    spatial = np.array(array.shape[-3:], dtype=np.int64)
    starts = center - patch_size // 2
    ends = starts + patch_size

    pad_before = np.maximum(0, -starts)
    pad_after = np.maximum(0, ends - spatial)
    clipped_starts = np.maximum(0, starts)
    clipped_ends = np.minimum(spatial, ends)

    slices = tuple(slice(int(s), int(e)) for s, e in zip(clipped_starts, clipped_ends))
    patch = array[(...,) + slices]
    pad_width = [(0, 0)] * (array.ndim - 3)
    pad_width.extend((int(b), int(a)) for b, a in zip(pad_before, pad_after))
    if any(b or a for b, a in pad_width):
        patch = np.pad(patch, pad_width, mode="constant", constant_values=pad_value)
    return patch


def random_center(spatial_shape: Sequence[int]) -> np.ndarray:
    return np.array([random.randrange(int(size)) for size in spatial_shape], dtype=np.int64)


def foreground_center(label: np.ndarray, ignore_index: int, fallback_shape: Sequence[int]) -> np.ndarray:
    valid_fg = np.logical_and(label > 0, label != ignore_index)
    coords = np.argwhere(valid_fg)
    if coords.size == 0:
        return random_center(fallback_shape)
    return coords[random.randrange(len(coords))]


def class_balanced_center(
    label: np.ndarray,
    ignore_index: int,
    classes: Sequence[int],
    fallback_shape: Sequence[int],
) -> np.ndarray:
    """Pick a center from a class drawn uniformly at random across `classes`.

    This counters foreground sampling bias toward the most common foreground
    class (white matter), which would otherwise starve small structures
    (lesions, basal ganglia) in the patch stream.
    """
    cls = classes[random.randrange(len(classes))]
    coords = np.argwhere(np.logical_and(label == cls, label != ignore_index))
    if coords.size == 0:
        return foreground_center(label, ignore_index, fallback_shape)
    return coords[random.randrange(len(coords))]


def _affine_grid(
    shape_dhw: Sequence[int],
    rotate_deg: float,
    scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Build a 5D sampling grid for a small rigid+scale augmentation.

    Rotates only in the axial (H,W) plane — slice thickness in MRBrainS13 is
    3 mm and rotating through-slice degrades labels.
    """
    theta = math.radians(rotate_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    s = 1.0 / max(scale, 1e-3)
    affine = torch.tensor(
        [[s, 0.0, 0.0, 0.0],
         [0.0, s * cos, -s * sin, 0.0],
         [0.0, s * sin, s * cos, 0.0]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    grid = torch.nn.functional.affine_grid(
        affine,
        size=(1, 1, int(shape_dhw[0]), int(shape_dhw[1]), int(shape_dhw[2])),
        align_corners=False,
    )
    return grid


def affine_augment(
    image: np.ndarray,
    label: np.ndarray,
    rotate_deg: float,
    scale: float,
    ignore_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if abs(rotate_deg) < 1e-3 and abs(scale - 1.0) < 1e-3:
        return image, label
    device = torch.device("cpu")
    img_t = torch.from_numpy(image).unsqueeze(0).to(device)
    lab_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float().to(device)
    grid = _affine_grid(image.shape[-3:], rotate_deg, scale, device)
    img_warped = torch.nn.functional.grid_sample(img_t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    fill = torch.full_like(lab_t, float(ignore_index))
    lab_warped = torch.nn.functional.grid_sample(
        lab_t,
        grid,
        mode="nearest",
        padding_mode="border",
        align_corners=False,
    )
    return (
        img_warped.squeeze(0).numpy().astype(np.float32),
        lab_warped.squeeze(0).squeeze(0).numpy().astype(np.int64),
    )


def augment_patch(
    image: np.ndarray,
    label: np.ndarray,
    cfg: Dict,
    ignore_index: int = 255,
) -> tuple[np.ndarray, np.ndarray]:
    if not cfg.get("enabled", False):
        return image, label

    flip_probability = float(cfg.get("flip_probability", 0.0))
    for axis in range(3):
        if random.random() < flip_probability:
            image = np.flip(image, axis=axis + 1).copy()
            label = np.flip(label, axis=axis).copy()

    if random.random() < float(cfg.get("intensity_probability", 0.0)):
        scale = random.uniform(*cfg.get("intensity_scale_range", (0.9, 1.1)))
        shift = random.uniform(*cfg.get("intensity_shift_range", (-0.1, 0.1)))
        image = image * scale + shift
        gamma_min, gamma_max = cfg.get("gamma_range", [1.0, 1.0])
        gamma = random.uniform(float(gamma_min), float(gamma_max))
        if abs(gamma - 1.0) > 1e-3:
            min_v = image.min(axis=(1, 2, 3), keepdims=True)
            max_v = image.max(axis=(1, 2, 3), keepdims=True)
            scaled = (image - min_v) / (max_v - min_v + 1e-6)
            image = np.power(np.clip(scaled, 0, 1), gamma) * (max_v - min_v) + min_v
        noise_std = float(cfg.get("noise_std", 0.0))
        if noise_std > 0:
            image = image + np.random.normal(0.0, noise_std, size=image.shape).astype(np.float32)

    if random.random() < float(cfg.get("affine_probability", 0.0)):
        rotate = random.uniform(*cfg.get("rotate_range_deg", (-10.0, 10.0)))
        zoom = random.uniform(*cfg.get("scale_range", (0.9, 1.1)))
        image, label = affine_augment(image, label, rotate, zoom, ignore_index)

    return image.astype(np.float32), label.astype(np.int64)


class PatchDataset(Dataset):
    def __init__(
        self,
        subjects: List[LoadedSubject],
        patch_size: Sequence[int],
        samples_per_epoch: int,
        foreground_patch_ratio: float,
        ignore_index: int,
        augmentation: Optional[Dict] = None,
        class_balanced_classes: Optional[Sequence[int]] = None,
        class_balanced_ratio: float = 0.0,
    ):
        self.subjects = subjects
        self.patch_size = tuple(int(v) for v in patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.foreground_patch_ratio = float(foreground_patch_ratio)
        self.class_balanced_ratio = float(class_balanced_ratio)
        self.class_balanced_classes = list(class_balanced_classes) if class_balanced_classes else []
        self.ignore_index = int(ignore_index)
        self.augmentation = augmentation or {"enabled": False}
        if any(subject.label is None for subject in subjects):
            raise ValueError("PatchDataset requires labels for every subject.")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> Dict[str, object]:
        subject = self.subjects[random.randrange(len(self.subjects))]
        assert subject.label is not None
        spatial = subject.image.shape[-3:]
        roll = random.random()
        if self.class_balanced_classes and roll < self.class_balanced_ratio:
            center = class_balanced_center(
                subject.label, self.ignore_index, self.class_balanced_classes, spatial
            )
        elif roll < self.class_balanced_ratio + self.foreground_patch_ratio:
            center = foreground_center(subject.label, self.ignore_index, spatial)
        else:
            center = random_center(spatial)

        image = extract_patch(subject.image, center, self.patch_size, pad_value=0.0)
        label = extract_patch(subject.label, center, self.patch_size, pad_value=self.ignore_index)
        image, label = augment_patch(image, label, self.augmentation, self.ignore_index)
        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "subject_id": subject.subject_id,
        }


class FullVolumeDataset(Dataset):
    def __init__(self, subjects: List[LoadedSubject]):
        self.subjects = subjects

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int) -> Dict[str, object]:
        subject = self.subjects[index]
        item: Dict[str, object] = {
            "image": torch.from_numpy(subject.image),
            "subject_id": subject.subject_id,
            "meta": subject.meta,
            "spacing_dhw": subject.spacing_dhw,
        }
        if subject.label is not None:
            item["label"] = torch.from_numpy(subject.label)
        return item
