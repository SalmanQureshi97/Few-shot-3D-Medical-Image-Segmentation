"""Measure parameters / inference latency / peak GPU memory for each config.

Writes one CSV row per config to artifacts/analysis/efficiency_table.csv. The
volume size mirrors a typical MRBrainS13 subject (~240 x 240 x 48 voxels).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from mrbrains.config import apply_overrides, get_device, load_config
from mrbrains.infer import sliding_window_predict
from mrbrains.models import build_model


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/analysis/efficiency_table.csv"))
    parser.add_argument("--shape", type=int, nargs=3, default=[48, 240, 240], metavar=("D", "H", "W"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--overrides", nargs="*", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    rows = []
    for cfg_path in args.configs:
        cfg = load_config(cfg_path)
        cfg = apply_overrides(cfg, args.overrides)
        cfg["model"]["in_channels"] = len(cfg["data"]["modalities"])
        cfg["model"]["out_channels"] = int(cfg["data"]["num_classes"])
        try:
            model = build_model(cfg["model"]).to(device).eval()
        except ImportError as exc:  # MONAI not installed for swin_unetr
            print(f"[skip] {cfg_path.stem}: {exc}")
            continue
        n_params = count_parameters(model)
        c = int(cfg["data"]["num_classes"])
        in_ch = len(cfg["data"]["modalities"])
        x = torch.randn((in_ch,) + tuple(args.shape), device=device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        # Warmup.
        with torch.no_grad():
            sliding_window_predict(
                model, x, cfg["inference"]["roi_size"], c, device,
                int(cfg["inference"].get("sw_batch_size", 2)),
                float(cfg["inference"].get("overlap", 0.5)),
                float(cfg["inference"].get("gaussian_sigma_scale", 0.125)),
                bool(cfg["inference"].get("amp", True)),
            )
        timings = []
        for _ in range(args.repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.time()
            with torch.no_grad():
                sliding_window_predict(
                    model, x, cfg["inference"]["roi_size"], c, device,
                    int(cfg["inference"].get("sw_batch_size", 2)),
                    float(cfg["inference"].get("overlap", 0.5)),
                    float(cfg["inference"].get("gaussian_sigma_scale", 0.125)),
                    bool(cfg["inference"].get("amp", True)),
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.time() - t0)
        peak_mb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024**2) if device.type == "cuda" else 0.0
        )

        rows.append(
            {
                "config": cfg_path.stem,
                "model": cfg["model"]["name"],
                "params": n_params,
                "params_M": round(n_params / 1e6, 2),
                "median_inference_s": round(sorted(timings)[len(timings) // 2], 3),
                "min_inference_s": round(min(timings), 3),
                "peak_memory_mb": round(peak_mb, 1),
                "shape_dhw": "x".join(str(s) for s in args.shape),
            }
        )
        print(rows[-1])
        del model

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
