from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from mrbrains.data import load_subject
from mrbrains.io import discover_subjects, load_volume, to_dhw, write_manifest
from mrbrains.labels import (
    DETAILED_LABELS,
    class_counts,
    class_voxel_frequencies,
    inverse_frequency_weights,
    label_names,
)
from mrbrains.viz import (
    DETAILED_COLORS,
    save_class_frequency_bar,
    save_histograms,
    save_multi_modality_panel,
    save_overlay_panel,
)


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

    raw_pooled = {m: [] for m in args.modalities}
    norm_pooled = {m: [] for m in args.modalities}
    counts = {}
    label_volumes = []
    label_palette = DETAILED_COLORS if args.target == "detailed" else None

    for subject in subjects:
        for modality, path in subject.image_paths.items():
            array_xyz, _ = load_volume(path)
            values = array_xyz[np.isfinite(array_xyz)]
            values = values[np.abs(values) > 1e-6]
            if values.size > 50000:
                values = np.random.default_rng(0).choice(values, 50000, replace=False)
            raw_pooled[modality].append(values)

        loaded = load_subject(subject, args.modalities, args.target)
        assert loaded.label is not None
        counts[subject.subject_id] = class_counts(loaded.label, label_names(args.target).keys())
        label_volumes.append(loaded.label)

        for ch, modality in enumerate(args.modalities):
            channel = loaded.image[ch].reshape(-1)
            channel = channel[channel != 0]
            if channel.size > 50000:
                channel = np.random.default_rng(0).choice(channel, 50000, replace=False)
            norm_pooled[modality].append(channel)

        save_overlay_panel(
            loaded.image,
            loaded.label,
            None,
            args.out / "overlays" / f"{subject.subject_id}.png",
            subject.subject_id,
            label_colors=label_palette,
        )
        save_multi_modality_panel(
            loaded.image,
            loaded.label,
            None,
            modality_names=[m.upper() for m in args.modalities],
            out_path=args.out / "modalities" / f"{subject.subject_id}.png",
            subject_id=subject.subject_id,
            label_colors=label_palette,
        )

    save_histograms(
        {m: np.concatenate(v) for m, v in raw_pooled.items() if v},
        args.out / "raw_intensity_histograms.png",
        title="Raw intensities (foreground)",
    )
    save_histograms(
        {m: np.concatenate(v) for m, v in norm_pooled.items() if v},
        args.out / "normalised_intensity_histograms.png",
        title="Robust z-score normalised intensities",
    )
    save_class_frequency_bar(counts, args.out / "class_frequency.png", label_names(args.target))

    # Persist a per-class voxel table + the inverse-frequency loss weight vector.
    num_classes = len(label_names(args.target))
    freq = class_voxel_frequencies(label_volumes, num_classes)
    weights = inverse_frequency_weights(freq)
    with (args.out / "class_voxel_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "name", "voxel_fraction", "loss_weight"])
        for cls_id, name in label_names(args.target).items():
            writer.writerow([cls_id, name, f"{freq[cls_id]:.6f}", f"{weights[cls_id]:.6f}"])
    print(f"Found {len(subjects)} training subjects. Inspection artifacts written to {args.out}")


if __name__ == "__main__":
    main()
