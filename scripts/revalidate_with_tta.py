"""Re-validate already-trained checkpoints with a different inference config.

Use case: you trained `improved_resattn_unet3d` and want A4-equivalent numbers
without retraining. This script loads each fold's checkpoint, runs sliding-window
inference with the *target* config's `inference` block (e.g. TTA flips), writes
`metrics_val.csv` and `figures/<subject>_overlay.png` into the target experiment
directory, and copies `config.yaml` so summarise_experiments picks it up as a
proper experiment row.

Example: produce A4 (= A3 + flip TTA at inference) from the A3 checkpoints.

    python3 scripts/revalidate_with_tta.py \\
        --source-config configs/improved_resattn_unet3d.yaml \\
        --target-config configs/ablation_a4_resattn_tta.yaml \\
        --source-runs runs/improved_resattn_unet3d \\
        --target-runs runs/ablation_a4_resattn_tta \\
        --data-root data
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from mrbrains.config import get_device, load_config, save_config
from mrbrains.engine import load_subjects_for_config, split_loocv, validate
from mrbrains.io import discover_subjects
from mrbrains.models import build_model


def _load_checkpoint(path: Path, device: torch.device, use_ema: bool) -> dict:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if use_ema and "ema_state" in ckpt:
        return {"model_state": ckpt["ema_state"]}
    return {"model_state": ckpt["model_state"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True, help="config the source run was trained with")
    parser.add_argument("--target-config", type=Path, required=True, help="config whose inference settings to use")
    parser.add_argument("--source-runs", type=Path, required=True)
    parser.add_argument("--target-runs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-ema", action="store_true", default=True)
    args = parser.parse_args()

    src_cfg = load_config(args.source_config)
    tgt_cfg = load_config(args.target_config)
    src_cfg["model"]["in_channels"] = len(src_cfg["data"]["modalities"])
    src_cfg["model"]["out_channels"] = int(src_cfg["data"]["num_classes"])

    # Build a hybrid config: model + data come from source (must match the trained
    # weights), inference comes from target.
    runtime_cfg = {**src_cfg, "inference": tgt_cfg["inference"]}

    device = get_device(args.device)
    subjects = discover_subjects(args.data_root, "training", src_cfg["data"]["modalities"], require_label=True)
    folds = args.folds if args.folds else list(range(len(subjects)))

    for fold in folds:
        src_fold_dir = args.source_runs / f"fold_{fold}"
        ckpt_path = src_fold_dir / "checkpoints" / "best.pt"
        if not ckpt_path.exists():
            print(f"[skip] fold {fold}: {ckpt_path} missing")
            continue
        tgt_fold_dir = args.target_runs / f"fold_{fold}"
        tgt_fold_dir.mkdir(parents=True, exist_ok=True)

        # Save the resolved config so summarise_experiments treats this as a real run.
        save_config(runtime_cfg, tgt_fold_dir / "config.yaml")

        model = build_model(runtime_cfg["model"]).to(device)
        ckpt = _load_checkpoint(ckpt_path, device, args.use_ema)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        _, val_subjects = split_loocv(subjects, fold)
        val_loaded = load_subjects_for_config(val_subjects, runtime_cfg)
        rows = validate(model, val_loaded, runtime_cfg, device, tgt_fold_dir, epoch=0, save_overlays=True)

        if not rows:
            print(f"[warn] fold {fold} produced no validation rows")
            continue

        metrics_path = tgt_fold_dir / "metrics_val.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        # Copy a single-row history.csv so summarise_experiments can plot.
        history_path = tgt_fold_dir / "history.csv"
        with history_path.open("w", newline="", encoding="utf-8") as f:
            f.write("epoch,train_loss,lr,parameters,epoch_seconds,peak_memory_mb,val_dice_mean,val_avd_mean,val_hd95_mean\n")
            f.write(
                f"0,,,,,,{rows[0].get('dice_mean', '')},{rows[0].get('avd_mean', '')},{rows[0].get('hd95_mean', '')}\n"
            )

        # Symlink the source checkpoint so predict_test can pick it up via the
        # standard glob.
        ckpt_dir = tgt_fold_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        target_ckpt = ckpt_dir / "best.pt"
        if not target_ckpt.exists():
            try:
                target_ckpt.symlink_to(ckpt_path.resolve())
            except OSError:
                shutil.copy2(ckpt_path, target_ckpt)

        print(f"[done] fold {fold}: dice_mean={rows[0].get('dice_mean'):.4f}")


if __name__ == "__main__":
    main()
