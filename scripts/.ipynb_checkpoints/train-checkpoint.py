from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from mrbrains.config import apply_overrides, load_config
from mrbrains.engine import run_training, split_loocv
from mrbrains.io import discover_subjects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="dotted overrides, e.g. training.epochs=5 augmentation.flip_probability=0.0",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="optional checkpoint path to resume from",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)
    cfg["model"]["in_channels"] = len(cfg["data"]["modalities"])
    cfg["model"]["out_channels"] = int(cfg["data"]["num_classes"])
    subjects = discover_subjects(args.data_root, "training", cfg["data"]["modalities"], require_label=True)
    train_subjects, val_subjects = split_loocv(subjects, args.fold)
    run_root = run_training(cfg, train_subjects, val_subjects, args.fold, args.device, resume=args.resume)
    print(f"Finished fold {args.fold}. Run artifacts: {run_root}")


if __name__ == "__main__":
    main()
