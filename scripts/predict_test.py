from __future__ import annotations

import argparse
import glob
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from mrbrains.config import get_device, load_config
from mrbrains.data import load_subject
from mrbrains.infer import predict_with_tta
from mrbrains.io import discover_subjects, from_dhw, save_nifti_like
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
    args = parser.parse_args()

    cfg = load_config(args.config)
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
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)

    subjects = discover_subjects(args.data_root, "test", cfg["data"]["modalities"], require_label=False)
    args.out.mkdir(parents=True, exist_ok=True)

    for subject in subjects:
        loaded = load_subject(
            subject,
            cfg["data"]["modalities"],
            cfg["data"]["target"],
            int(cfg["data"].get("ignore_index", 255)),
            bool(cfg["data"].get("ignore_hindbrain_in_coarse", True)),
            cfg["data"].get("intensity_clip_percentiles", [1.0, 99.0]),
        )
        probs_sum = None
        for model in models:
            probs = predict_with_tta(
                model,
                torch.from_numpy(loaded.image),
                cfg["inference"]["roi_size"],
                int(cfg["data"]["num_classes"]),
                device,
                int(cfg["inference"].get("sw_batch_size", 2)),
                float(cfg["inference"].get("overlap", 0.5)),
                cfg["inference"].get("tta_flips", []),
            )
            probs_sum = probs if probs_sum is None else probs_sum + probs
        assert probs_sum is not None
        pred_dhw = (probs_sum / len(models)).argmax(dim=0).numpy().astype(np.uint8)
        out_path = args.out / f"{subject.subject_id}_prediction.nii.gz"
        save_nifti_like(from_dhw(pred_dhw), loaded.meta, out_path, dtype=np.uint8)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

