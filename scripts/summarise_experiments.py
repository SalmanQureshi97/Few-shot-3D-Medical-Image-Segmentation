from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_run_name(metrics_path: Path) -> tuple[str, str]:
    fold = metrics_path.parent.name
    experiment = metrics_path.parent.parent.name
    return experiment, fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/analysis"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metric_frames = []
    for metrics_path in args.runs.glob("*/fold_*/metrics_val.csv"):
        experiment, fold = parse_run_name(metrics_path)
        df = pd.read_csv(metrics_path)
        df["experiment"] = experiment
        df["fold"] = fold
        metric_frames.append(df)
    if metric_frames:
        metrics = pd.concat(metric_frames, ignore_index=True)
        metrics.to_csv(args.out / "all_validation_metrics.csv", index=False)
        fold_best = (
            metrics.groupby(["experiment", "fold", "epoch"], as_index=False)["dice_mean"]
            .mean()
            .sort_values("dice_mean", ascending=False)
            .groupby(["experiment", "fold"], as_index=False)
            .first()
        )
        fold_best.to_csv(args.out / "best_fold_metrics.csv", index=False)
        summary = fold_best.groupby("experiment")["dice_mean"].agg(["mean", "std", "count"]).reset_index()
        summary.to_csv(args.out / "ablation_summary.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        sns.barplot(data=fold_best, x="experiment", y="dice_mean", errorbar="sd", ax=ax)
        ax.set_ylabel("Best validation mean Dice")
        ax.tick_params(axis="x", rotation=20)
        fig.savefig(args.out / "ablation_barplot.png", dpi=180)
        plt.close(fig)

    history_frames = []
    for history_path in args.runs.glob("*/fold_*/history.csv"):
        experiment, fold = parse_run_name(history_path)
        df = pd.read_csv(history_path)
        df["experiment"] = experiment
        df["fold"] = fold
        history_frames.append(df)
    if history_frames:
        history = pd.concat(history_frames, ignore_index=True)
        history.to_csv(args.out / "all_history.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        sns.lineplot(data=history, x="epoch", y="train_loss", hue="experiment", style="fold", ax=ax)
        ax.set_ylabel("Training loss")
        fig.savefig(args.out / "training_loss_curves.png", dpi=180)
        plt.close(fig)
        if "val_dice_mean" in history.columns:
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            sns.lineplot(data=history.dropna(subset=["val_dice_mean"]), x="epoch", y="val_dice_mean", hue="experiment", style="fold", ax=ax)
            ax.set_ylabel("Validation mean Dice")
            fig.savefig(args.out / "validation_dice_curves.png", dpi=180)
            plt.close(fig)

    print(f"Summary artifacts written to {args.out}")


if __name__ == "__main__":
    main()
