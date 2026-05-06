from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import nibabel as nib
import numpy as np


NIFTI_SUFFIXES = (".nii", ".nii.gz")

MODALITY_ALIASES = {
    "t1": ["reg_T1", "T1_1mm", "T1"],
    "ir": ["T1_IR", "reg_IR", "IR"],
    "flair": ["T2_FLAIR", "FLAIR"],
}

TRAIN_LABEL_ALIASES = ["LabelsForTraining", "segm", "manual", "groundtruth", "gt", "label", "labels"]
TEST_LABEL_ALIASES = ["LabelsForTesting", "label", "labels", "segm"]


@dataclass(frozen=True)
class VolumeMeta:
    affine: np.ndarray
    header: object
    spacing: tuple[float, float, float]
    source_path: Path


@dataclass(frozen=True)
class Subject:
    subject_id: str
    image_paths: Dict[str, Path]
    label_path: Optional[Path] = None


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def load_volume(path: Union[str, Path]) -> tuple[np.ndarray, VolumeMeta]:
    path = Path(path)
    if is_nifti(path):
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
        spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
        return data, VolumeMeta(img.affine, img.header, spacing, path)
    if path.suffix.lower() == ".npy":
        data = np.load(path)
        affine = np.eye(4, dtype=np.float32)
        return data, VolumeMeta(affine, None, (1.0, 1.0, 1.0), path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as npz:
            data = npz["arr_0"]
        affine = np.eye(4, dtype=np.float32)
        return data, VolumeMeta(affine, None, (1.0, 1.0, 1.0), path)
    raise ValueError(f"Unsupported volume format: {path}")


def save_nifti_like(
    data: np.ndarray,
    reference_meta: VolumeMeta,
    out_path: Union[str, Path],
    dtype: np.dtype = np.uint8,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(dtype), reference_meta.affine, reference_meta.header)
    nib.save(img, str(out_path))


def to_dhw(array_xyz: np.ndarray) -> np.ndarray:
    if array_xyz.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {array_xyz.shape}")
    return np.moveaxis(array_xyz, -1, 0)


def from_dhw(array_dhw: np.ndarray) -> np.ndarray:
    if array_dhw.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {array_dhw.shape}")
    return np.moveaxis(array_dhw, 0, -1)


def _normalised_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _candidate_files(root: Path) -> List[Path]:
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and (is_nifti(p) or p.suffix.lower() in {".npy", ".npz"})
    ]


def _score_alias(path: Path, aliases: Iterable[str]) -> int:
    stem = _normalised_stem(path).lower()
    score = 0
    for alias in aliases:
        alias_l = alias.lower()
        if stem == alias_l:
            score = max(score, 100)
        elif stem.endswith(alias_l):
            score = max(score, 80)
        elif alias_l in stem:
            score = max(score, 50)
    if "/pre/" in str(path).lower():
        score += 5
    return score


def _find_best_file(files: List[Path], aliases: Iterable[str]) -> Optional[Path]:
    scored = [(p, _score_alias(p, aliases)) for p in files]
    scored = [(p, score) for p, score in scored if score > 0]
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[1], len(str(item[0]))))
    return scored[0][0]


def discover_subjects(
    data_root: Union[str, Path],
    split: str = "training",
    modalities: Iterable[str] = ("t1", "ir", "flair"),
    require_label: bool = True,
) -> List[Subject]:
    """Discover MRBrainS-like subject folders.

    This intentionally accepts several common layouts because MRBrainS13 mirrors
    and classroom downloads often vary in folder names.
    """
    data_root = Path(data_root)
    split_candidates = [
        data_root / split,
        data_root / split.lower(),
        data_root / split.capitalize(),
        data_root / ("train" if split.startswith("train") else split),
        data_root / ("TrainingData" if split.startswith("train") else "TestData"),
    ]
    split_root = next((p for p in split_candidates if p.exists()), data_root)

    subjects: List[Subject] = []
    folders = [p for p in split_root.iterdir() if p.is_dir()] if split_root.exists() else []
    if not folders:
        folders = [split_root]

    for folder in sorted(folders):
        files = _candidate_files(folder)
        if not files:
            continue
        image_paths: Dict[str, Path] = {}
        for modality in modalities:
            aliases = MODALITY_ALIASES.get(modality.lower(), [modality])
            found = _find_best_file(files, aliases)
            if found is not None:
                image_paths[modality.lower()] = found
        label_aliases = TRAIN_LABEL_ALIASES if split.startswith("train") else TEST_LABEL_ALIASES
        label_path = _find_best_file(files, label_aliases)
        if len(image_paths) != len(list(modalities)):
            continue
        if require_label and label_path is None:
            continue
        subjects.append(Subject(folder.name, image_paths, label_path))

    if not subjects:
        label_msg = " with labels" if require_label else ""
        raise FileNotFoundError(f"No MRBrainS subjects{label_msg} found in {split_root}")
    return subjects


def write_manifest(subjects: List[Subject], out_csv: Union[str, Path]) -> None:
    import pandas as pd

    rows = []
    for subject in subjects:
        row = {"subject_id": subject.subject_id, "label": subject.label_path}
        row.update(subject.image_paths)
        rows.append(row)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
