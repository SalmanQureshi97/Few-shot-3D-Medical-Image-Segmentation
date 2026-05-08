"""Run a list of ablation configs end-to-end (LOOCV) sequentially.

Usage:
    python scripts/run_ablations.py --data-root data \
        --configs configs/baseline_unet3d.yaml \
                  configs/ablation_a1_fg_aug.yaml \
                  configs/ablation_a2_dicece_focal.yaml \
                  configs/improved_resattn_unet3d.yaml \
                  configs/ablation_a4_resattn_tta.yaml

Each config is run for all 5 LOOCV folds (or the subset you pass via --folds).
After every fold the script prints a checkpoint summary so a tail -f log makes
progress visible. Failed folds do not abort the rest of the run.
"""

from __future__ import annotations

import argparse
import copy
import time
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401

from mrbrains.config import apply_overrides, load_config
from mrbrains.engine import run_training, split_loocv
from mrbrains.io import discover_subjects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overrides", nargs="*", default=None)
    args = parser.parse_args()

    for cfg_path in args.configs:
        print(f"\n==== {cfg_path} ====")
        cfg = load_config(cfg_path)
        cfg = apply_overrides(cfg, args.overrides)
        cfg["model"]["in_channels"] = len(cfg["data"]["modalities"])
        cfg["model"]["out_channels"] = int(cfg["data"]["num_classes"])
        try:
            subjects = discover_subjects(args.data_root, "training", cfg["data"]["modalities"], require_label=True)
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue
        folds = args.folds if args.folds else list(range(len(subjects)))
        for fold in folds:
            fold_cfg = copy.deepcopy(cfg)
            train_subjects, val_subjects = split_loocv(subjects, fold)
            t0 = time.time()
            try:
                run_root = run_training(fold_cfg, train_subjects, val_subjects, fold, args.device)
            except Exception:  # noqa: BLE001 — keep the rest of the suite alive
                print(f"[error] fold {fold} of {cfg_path} crashed:")
                traceback.print_exc()
                continue
            elapsed = (time.time() - t0) / 60.0
            print(f"[done] {cfg_path.stem} fold {fold} in {elapsed:.1f} min -> {run_root}")


if __name__ == "__main__":
    main()
