from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from mrbrains.config import apply_overrides, get_device, load_config
from mrbrains.data import load_subject
from mrbrains.infer import predict_with_tta
from mrbrains.io import discover_subjects, from_dhw, save_nifti_like
from mrbrains.labels import foreground_classes, label_names, remap_labels
from mrbrains.metrics import (
    assd_per_class,
    avd_per_class,
    dice_per_class,
    hd95_per_class,
    summarise_metric_dict,
    volume_similarity,
)
from mrbrains.models import build_model


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--out", type=Path, default=Path("predictions/test"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="dotted config overrides, e.g. inference.tta_flips=[[3]]",
    )
    parser.add_argument(
        "--evaluate-against-labels",
        action="store_true",
        help="if LabelsForTesting.nii is present, write a metrics CSV against the coarse labels",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="prefer EMA weights from the checkpoint when present",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)
    cfg["model"]["in_channels"] = len(cfg["data"]["modalities"])
    cfg["model"]["out_channels"] = int(cfg["data"]["num_classes"])
    device = get_device(args.device)
    checkpoint_paths = [Path(p) for p in sorted(glob.glob(args.checkpoint_glob))]
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints matched {args.checkpoint_glob}")

    models = []
    for path in checkpoint_paths:
        model = build_model(cfg["model"]).to(device)
        checkpoint = torch_load(path, device)
        state = checkpoint.get("ema_state") if args.use_ema and "ema_state" in checkpoint else checkpoint["model_state"]
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    subjects = discover_subjects(
        args.data_root,
        "test",
        cfg["data"]["modalities"],
        require_label=False,
    )
    args.out.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    classes = foreground_classes(cfg["data"]["target"])
    num_classes = int(cfg["data"]["num_classes"])
    ignore_index = int(cfg["data"].get("ignore_index", 255))

    for subject in subjects:
        loaded = load_subject(
            subject,
            cfg["data"]["modalities"],
            cfg["data"]["target"],
            ignore_index,
            bool(cfg["data"].get("ignore_hindbrain_in_coarse", True)),
            cfg["data"].get("intensity_clip_percentiles", [1.0, 99.0]),
        )
        probs_sum = None
        for model in models:
            probs = predict_with_tta(
                model,
                torch.from_numpy(loaded.image),
                cfg["inference"]["roi_size"],
                num_classes,
                device,
                int(cfg["inference"].get("sw_batch_size", 2)),
                float(cfg["inference"].get("overlap", 0.5)),
                cfg["inference"].get("tta_flips", []),
                float(cfg["inference"].get("gaussian_sigma_scale", 0.125)),
                bool(cfg["inference"].get("amp", True)),
            )
            probs_sum = probs if probs_sum is None else probs_sum + probs
        assert probs_sum is not None
        pred_dhw = (probs_sum / len(models)).argmax(dim=0).numpy().astype(np.uint8)
        out_path = args.out / f"{subject.subject_id}_prediction.nii.gz"
        save_nifti_like(from_dhw(pred_dhw), loaded.meta, out_path, dtype=np.uint8)
        print(f"Wrote {out_path}")

        if args.evaluate_against_labels and subject.label_path is not None:
            from mrbrains.io import load_volume, to_dhw

            label_xyz, _ = load_volume(subject.label_path)
            label_dhw = to_dhw(label_xyz).astype(np.int64)
            # MRBrainS LabelsForTesting are already in coarse class space (0..3),
            # so the remap is a passthrough; for the detailed task we still remap.
            label_remapped = remap_labels(
                label_dhw,
                cfg["data"]["target"],
                ignore_index,
                bool(cfg["data"].get("ignore_hindbrain_in_coarse", True)),
                bool(cfg["data"].get("ignore_hindbrain_in_detailed", False)),
            )
            row = {"subject_id": subject.subject_id}
            row.update(
                summarise_metric_dict(
                    "dice", dice_per_class(pred_dhw.astype(np.int64), label_remapped, classes, ignore_index)
                )
            )
            row.update(
                summarise_metric_dict(
                    "volsim",
                    volume_similarity(pred_dhw.astype(np.int64), label_remapped, classes, ignore_index),
                )
            )
            row.update(
                summarise_metric_dict(
                    "avd", avd_per_class(pred_dhw.astype(np.int64), label_remapped, classes, ignore_index)
                )
            )
            row.update(
                summarise_metric_dict(
                    "hd95",
                    hd95_per_class(
                        pred_dhw.astype(np.int64),
                        label_remapped,
                        classes,
                        loaded.spacing_dhw,
                        ignore_index,
                    ),
                )
            )
            row.update(
                summarise_metric_dict(
                    "assd",
                    assd_per_class(
                        pred_dhw.astype(np.int64),
                        label_remapped,
                        classes,
                        loaded.spacing_dhw,
                        ignore_index,
                    ),
                )
            )
            metric_rows.append(row)

    if metric_rows:
        out_csv = args.out / "test_metrics.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metric_rows)
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
