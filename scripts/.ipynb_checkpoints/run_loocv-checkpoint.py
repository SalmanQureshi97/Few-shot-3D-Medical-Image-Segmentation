from __future__ import annotations

import argparse
import copy
from pathlib import Path

import _bootstrap  # noqa: F401

from mrbrains.config import apply_overrides, load_config
from mrbrains.engine import run_training, split_loocv
from mrbrains.io import discover_subjects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="dotted overrides applied to every fold, e.g. training.epochs=5",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)
    cfg["model"]["in_channels"] = len(cfg["data"]["modalities"])
    cfg["model"]["out_channels"] = int(cfg["data"]["num_classes"])
    subjects = discover_subjects(args.data_root, "training", cfg["data"]["modalities"], require_label=True)
    folds = args.folds if args.folds is not None and len(args.folds) else list(range(len(subjects)))
    for fold in folds:
        fold_cfg = copy.deepcopy(cfg)
        train_subjects, val_subjects = split_loocv(subjects, fold)
        run_root = run_training(fold_cfg, train_subjects, val_subjects, fold, args.device)
        print(f"Fold {fold} complete: {run_root}")


if __name__ == "__main__":
    main()
