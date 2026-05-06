from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from mrbrains.data import load_subject
from mrbrains.io import discover_subjects, load_volume, to_dhw, write_manifest
from mrbrains.labels import class_counts, label_names
from mrbrains.viz import save_class_frequency_bar, save_histograms, save_overlay_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/data_inspection"))
    parser.add_argument("--target", choices=["coarse", "detailed"], default="coarse")
    parser.add_argument("--modalities", nargs="+", default=["t1", "ir", "flair"])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    subjects = discover_subjects(args.data_root, "training", args.modalities, require_label=True)
    write_manifest(subjects, args.out / "manifest_training.csv")

    pooled_values = {m: [] for m in args.modalities}
    counts = {}
    for subject in subjects:
        for modality, path in subject.image_paths.items():
            array_xyz, _ = load_volume(path)
            values = array_xyz[np.isfinite(array_xyz)]
            values = values[np.abs(values) > 1e-6]
            if values.size > 50000:
                values = np.random.default_rng(0).choice(values, 50000, replace=False)
            pooled_values[modality].append(values)

        loaded = load_subject(subject, args.modalities, args.target)
        assert loaded.label is not None
        counts[subject.subject_id] = class_counts(loaded.label, label_names(args.target).keys())
        save_overlay_panel(loaded.image, loaded.label, None, args.out / "overlays" / f"{subject.subject_id}.png", subject.subject_id)

    save_histograms({m: np.concatenate(v) for m, v in pooled_values.items() if v}, args.out / "raw_intensity_histograms.png")
    save_class_frequency_bar(counts, args.out / "class_frequency.png", label_names(args.target))
    print(f"Found {len(subjects)} training subjects. Inspection artifacts written to {args.out}")


if __name__ == "__main__":
    main()

