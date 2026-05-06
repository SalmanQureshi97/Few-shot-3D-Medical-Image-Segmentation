from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_run_name(metrics_path: Path) -> tuple[str, str]:
    fold = metrics_path.parent.name
    experiment = metrics_path.parent.parent.name
    return experiment, fold


def _best_per_fold(metrics: pd.DataFrame, by: str = "dice_mean") -> pd.DataFrame:
    if by not in metrics.columns or metrics.empty:
        return pd.DataFrame()
    return (
        metrics.sort_values(by, ascending=False)
        .groupby(["experiment", "fold"], as_index=False)
        .first()
    )


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
        fold_best = _best_per_fold(metrics, by="dice_mean")
        if not fold_best.empty:
            fold_best.to_csv(args.out / "best_fold_metrics.csv", index=False)
            agg_columns = [c for c in fold_best.columns if c.endswith("_mean")]
            summary_rows = []
            for experiment, group in fold_best.groupby("experiment"):
                row = {"experiment": experiment, "folds": len(group)}
                for col in agg_columns:
                    if col in group:
                        row[f"{col}_mean"] = float(np.nanmean(group[col]))
                        row[f"{col}_std"] = float(np.nanstd(group[col]))
                summary_rows.append(row)
            summary = pd.DataFrame(summary_rows)
            summary.to_csv(args.out / "ablation_summary.csv", index=False)

            # Per-class Dice table (mean across folds).
            per_class_cols = [c for c in fold_best.columns if c.startswith("dice_class_")]
            if per_class_cols:
                per_class = fold_best.groupby("experiment")[per_class_cols].mean().reset_index()
                per_class.to_csv(args.out / "per_class_dice.csv", index=False)

            fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
            sns.barplot(data=fold_best, x="experiment", y="dice_mean", errorbar="sd", ax=ax)
            ax.set_ylabel("Best validation mean Dice")
            ax.tick_params(axis="x", rotation=20)
            fig.savefig(args.out / "ablation_barplot.png", dpi=180)
            plt.close(fig)

            # Per-class Dice grouped bar plot.
            if per_class_cols:
                long = per_class.melt(id_vars="experiment", value_vars=per_class_cols, var_name="class", value_name="dice")
                long["class"] = long["class"].str.replace("dice_class_", "", regex=False)
                fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
                sns.barplot(data=long, x="class", y="dice", hue="experiment", ax=ax)
                ax.set_ylim(0, 1)
                ax.set_ylabel("Mean Dice (across folds)")
                ax.set_xlabel("class id")
                fig.savefig(args.out / "per_class_dice_barplot.png", dpi=180)
                plt.close(fig)

            # HD95 / AVD bar plots when present.
            for stat in ("hd95_mean", "avd_mean", "assd_mean"):
                if stat in fold_best.columns:
                    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
                    sns.barplot(data=fold_best, x="experiment", y=stat, errorbar="sd", ax=ax)
                    ax.set_ylabel(stat.replace("_", " "))
                    ax.tick_params(axis="x", rotation=20)
                    fig.savefig(args.out / f"{stat}_barplot.png", dpi=180)
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
        sns.lineplot(data=history, x="epoch", y="train_loss", hue="experiment", style="fold", ax=ax, legend="brief")
        ax.set_ylabel("Training loss")
        fig.savefig(args.out / "training_loss_curves.png", dpi=180)
        plt.close(fig)
        if "val_dice_mean" in history.columns:
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            sns.lineplot(
                data=history.dropna(subset=["val_dice_mean"]),
                x="epoch",
                y="val_dice_mean",
                hue="experiment",
                style="fold",
                ax=ax,
                legend="brief",
            )
            ax.set_ylabel("Validation mean Dice")
            fig.savefig(args.out / "validation_dice_curves.png", dpi=180)
            plt.close(fig)
        if "epoch_seconds" in history.columns:
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            sns.lineplot(data=history, x="epoch", y="epoch_seconds", hue="experiment", style="fold", ax=ax, legend="brief")
            ax.set_ylabel("Epoch wall time (s)")
            fig.savefig(args.out / "epoch_seconds_curves.png", dpi=180)
            plt.close(fig)
        if "peak_memory_mb" in history.columns:
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            sns.lineplot(data=history, x="epoch", y="peak_memory_mb", hue="experiment", style="fold", ax=ax, legend="brief")
            ax.set_ylabel("Peak GPU memory (MiB)")
            fig.savefig(args.out / "peak_memory_curves.png", dpi=180)
            plt.close(fig)

    print(f"Summary artifacts written to {args.out}")


if __name__ == "__main__":
    main()
