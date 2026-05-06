from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import nibabel as nib
import numpy as np

from mrbrains.io import from_dhw


def ellipsoid(coords, centre, radii):
    z, y, x = coords
    return (
        ((z - centre[0]) / radii[0]) ** 2
        + ((y - centre[1]) / radii[1]) ** 2
        + ((x - centre[2]) / radii[2]) ** 2
    ) <= 1.0


def make_subject(shape, seed: int):
    rng = np.random.default_rng(seed)
    d, h, w = shape
    z, y, x = np.mgrid[:d, :h, :w]
    centre = np.array([d * 0.52, h * 0.5, w * 0.5]) + rng.normal(0, [0.8, 1.5, 1.5])
    brain = ellipsoid((z, y, x), centre, [d * 0.44, h * 0.40, w * 0.38])
    wm = ellipsoid((z, y, x), centre, [d * 0.25, h * 0.25, w * 0.24])
    gm = brain & ~wm
    ventricles = ellipsoid((z, y, x), centre + np.array([0, 0, -w * 0.07]), [d * 0.12, h * 0.06, w * 0.04])
    ventricles |= ellipsoid((z, y, x), centre + np.array([0, 0, w * 0.07]), [d * 0.12, h * 0.06, w * 0.04])
    basal = ellipsoid((z, y, x), centre + np.array([0, -h * 0.02, -w * 0.12]), [d * 0.10, h * 0.08, w * 0.06])
    basal |= ellipsoid((z, y, x), centre + np.array([0, -h * 0.02, w * 0.12]), [d * 0.10, h * 0.08, w * 0.06])
    csf_outer = brain & ~ellipsoid((z, y, x), centre, [d * 0.39, h * 0.35, w * 0.34])
    lesion = ellipsoid((z, y, x), centre + rng.normal(0, [2, 5, 5]), [d * 0.05, h * 0.04, w * 0.04])
    cerebellum = ellipsoid((z, y, x), centre + np.array([-d * 0.33, h * 0.18, 0]), [d * 0.12, h * 0.14, w * 0.20])
    brainstem = ellipsoid((z, y, x), centre + np.array([-d * 0.28, h * 0.02, 0]), [d * 0.11, h * 0.05, w * 0.06])

    label = np.zeros(shape, dtype=np.uint8)
    label[gm] = 1
    label[basal] = 2
    label[wm] = 3
    label[lesion & wm] = 4
    label[csf_outer] = 5
    label[ventricles] = 6
    label[cerebellum] = 7
    label[brainstem] = 8

    means = {
        "T1": {0: 0, 1: 85, 2: 95, 3: 130, 4: 105, 5: 25, 6: 20, 7: 90, 8: 100},
        "IR": {0: 0, 1: 100, 2: 110, 3: 75, 4: 80, 5: 35, 6: 30, 7: 95, 8: 88},
        "FLAIR": {0: 0, 1: 70, 2: 78, 3: 55, 4: 140, 5: 15, 6: 10, 7: 65, 8: 60},
    }
    images = {}
    bias = 1.0 + 0.12 * (x / max(1, w - 1)) + 0.08 * (y / max(1, h - 1))
    for name, cls_means in means.items():
        img = np.zeros(shape, dtype=np.float32)
        for cls, mean in cls_means.items():
            img[label == cls] = mean
        img = img * bias + rng.normal(0, 6.0, size=shape)
        img[~brain & (label == 0)] = 0
        images[name] = img.astype(np.float32)
    return images, label


def save_nifti(array_dhw: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.diag([0.958, 0.958, 3.0, 1.0]).astype(np.float32)
    nib.save(nib.Nifti1Image(from_dhw(array_dhw), affine), str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic/mrbrains"))
    parser.add_argument("--subjects", type=int, default=5)
    parser.add_argument("--test-subjects", type=int, default=2)
    parser.add_argument("--shape", type=int, nargs=3, default=[24, 48, 48], metavar=("D", "H", "W"))
    args = parser.parse_args()

    for i in range(args.subjects):
        subject_dir = args.out / "training" / f"sub-{i + 1:03d}"
        images, label = make_subject(tuple(args.shape), seed=100 + i)
        save_nifti(images["T1"], subject_dir / "pre" / "T1.nii.gz")
        save_nifti(images["IR"], subject_dir / "pre" / "IR.nii.gz")
        save_nifti(images["FLAIR"], subject_dir / "pre" / "FLAIR.nii.gz")
        save_nifti(label, subject_dir / "segm.nii.gz")

    for i in range(args.test_subjects):
        subject_dir = args.out / "test" / f"test-{i + 1:03d}"
        images, _ = make_subject(tuple(args.shape), seed=500 + i)
        save_nifti(images["T1"], subject_dir / "pre" / "T1.nii.gz")
        save_nifti(images["IR"], subject_dir / "pre" / "IR.nii.gz")
        save_nifti(images["FLAIR"], subject_dir / "pre" / "FLAIR.nii.gz")

    print(f"Synthetic MRBrainS-like dataset written to {args.out}")


if __name__ == "__main__":
    main()

