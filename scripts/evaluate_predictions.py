"""Re-compute Dice / HD95 / AVD on already-saved predictions vs. labels.

Useful when you have prediction NIfTIs from `predict_test.py` and want to
re-run metrics without re-loading the model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from mrbrains.io import discover_subjects, load_volume, to_dhw
from mrbrains.labels import foreground_classes, remap_labels
from mrbrains.metrics import (
    assd_per_class,
    avd_per_class,
    dice_per_class,
    hd95_per_class,
    summarise_metric_dict,
    volume_similarity,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=Path("predictions/test/test_metrics.csv"))
    parser.add_argument("--target", choices=["coarse", "detailed"], default="coarse")
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--ignore-hindbrain", action="store_true", default=True)
    args = parser.parse_args()

    subjects = discover_subjects(args.data_root, "test", ("t1", "ir", "flair"), require_label=True)
    classes = foreground_classes(args.target)
    rows = []
    for subject in subjects:
        pred_path = args.predictions / f"{subject.subject_id}_prediction.nii.gz"
        if not pred_path.exists():
            print(f"[skip] no prediction for {subject.subject_id}: {pred_path}")
            continue
        pred_xyz, _ = load_volume(pred_path)
        pred_dhw = to_dhw(pred_xyz).astype(np.int64)
        label_xyz, label_meta = load_volume(subject.label_path)
        label_dhw = to_dhw(label_xyz).astype(np.int64)
        label_remapped = remap_labels(label_dhw, args.target, args.ignore_index, args.ignore_hindbrain)
        spacing_xyz = label_meta.spacing
        spacing_dhw = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        row = {"subject_id": subject.subject_id}
        row.update(summarise_metric_dict("dice", dice_per_class(pred_dhw, label_remapped, classes, args.ignore_index)))
        row.update(summarise_metric_dict("volsim", volume_similarity(pred_dhw, label_remapped, classes, args.ignore_index)))
        row.update(summarise_metric_dict("avd", avd_per_class(pred_dhw, label_remapped, classes, args.ignore_index)))
        row.update(summarise_metric_dict("hd95", hd95_per_class(pred_dhw, label_remapped, classes, spacing_dhw, args.ignore_index)))
        row.update(summarise_metric_dict("assd", assd_per_class(pred_dhw, label_remapped, classes, spacing_dhw, args.ignore_index)))
        rows.append(row)

    if not rows:
        print("No predictions matched any subject.")
        return
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out_csv} ({len(rows)} subjects)")


if __name__ == "__main__":
    main()
