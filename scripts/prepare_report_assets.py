from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_ablation_table(csv_path: Path, out_tex: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Mean Dice & Std & Folds \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        method = str(row["experiment"]).replace("_", "\\_")
        mean = float(row["mean"])
        std = float(row["std"]) if pd.notna(row["std"]) else 0.0
        count = int(row["count"])
        lines.append(f"{method} & {mean:.3f} & {std:.3f} & {count} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick_best_worst_overlays(runs_dir: Path, report_figs: Path) -> None:
    metrics_paths = list(runs_dir.glob("*/fold_*/metrics_val.csv"))
    if not metrics_paths:
        return
    rows = []
    for path in metrics_paths:
        df = pd.read_csv(path)
        if df.empty or "dice_mean" not in df:
            continue
        best_epoch = df.sort_values("dice_mean", ascending=False).iloc[0]
        worst_epoch = df.sort_values("dice_mean", ascending=True).iloc[0]
        rows.append(("best", path.parent, best_epoch))
        rows.append(("worst", path.parent, worst_epoch))
    if not rows:
        return
    best = max((r for r in rows if r[0] == "best"), key=lambda item: item[2]["dice_mean"])
    worst = min((r for r in rows if r[0] == "worst"), key=lambda item: item[2]["dice_mean"])
    for tag, fold_dir, row in [best, worst]:
        subject_id = row["subject_id"]
        epoch = int(row["epoch"])
        src = fold_dir / "figures" / f"{subject_id}_epoch_{epoch}_overlay.png"
        copy_if_exists(src, report_figs / f"{tag}_overlay.png")
    copy_if_exists(report_figs / "worst_overlay.png", report_figs / "error_overlay.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=Path("report/ECS795P_Report_Latex_Template"))
    parser.add_argument("--data-inspection", type=Path, default=Path("artifacts/data_inspection"))
    parser.add_argument("--analysis", type=Path, default=Path("artifacts/analysis"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()

    report_figs = args.report_dir / "figs"
    report_tables = args.report_dir / "tables"
    copy_if_exists(args.data_inspection / "raw_intensity_histograms.png", report_figs / "raw_intensity_histograms.png")
    copy_if_exists(args.data_inspection / "class_frequency.png", report_figs / "class_frequency.png")
    copy_if_exists(args.analysis / "ablation_barplot.png", report_figs / "ablation_barplot.png")
    copy_if_exists(args.analysis / "validation_dice_curves.png", report_figs / "validation_dice_curves.png")
    copy_if_exists(args.analysis / "training_loss_curves.png", report_figs / "training_loss_curves.png")
    write_ablation_table(args.analysis / "ablation_summary.csv", report_tables / "ablation_summary.tex")
    pick_best_worst_overlays(args.runs, report_figs)
    print(f"Report assets prepared in {args.report_dir}")


if __name__ == "__main__":
    main()

