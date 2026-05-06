from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
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

    cols = ["experiment", "folds"]
    for stat in ("dice_mean_mean", "dice_mean_std", "hd95_mean_mean", "avd_mean_mean"):
        if stat in df.columns:
            cols.append(stat)
    df = df[cols]

    headers = {
        "experiment": "Method",
        "folds": "Folds",
        "dice_mean_mean": "Dice (mean)",
        "dice_mean_std": "Dice (std)",
        "hd95_mean_mean": "HD95 mm",
        "avd_mean_mean": "AVD %",
    }
    header_row = " & ".join(headers[c] for c in cols) + " \\\\"
    align = "l" + ("c" * (len(cols) - 1))
    lines = [
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        header_row,
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            value = row[c]
            if c == "experiment":
                cells.append(str(value).replace("_", "\\_"))
            elif c == "folds":
                cells.append(str(int(value)))
            else:
                cells.append(f"{float(value):.3f}" if pd.notna(value) else "--")
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_class_table(csv_path: Path, out_tex: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    class_cols = [c for c in df.columns if c.startswith("dice_class_")]
    align = "l" + ("c" * len(class_cols))
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", "Method & " + " & ".join(c.replace("dice_class_", "C") for c in class_cols) + " \\\\", "\\midrule"]
    for _, row in df.iterrows():
        cells = [str(row["experiment"]).replace("_", "\\_")]
        for c in class_cols:
            v = row[c]
            cells.append(f"{float(v):.3f}" if pd.notna(v) else "--")
        lines.append(" & ".join(cells) + " \\\\")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=Path("report"))
    parser.add_argument("--data-inspection", type=Path, default=Path("artifacts/data_inspection"))
    parser.add_argument("--analysis", type=Path, default=Path("artifacts/analysis"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()

    report_figs = args.report_dir / "figs"
    report_tables = args.report_dir / "tables"
    for fig in [
        "raw_intensity_histograms.png",
        "normalised_intensity_histograms.png",
        "class_frequency.png",
    ]:
        copy_if_exists(args.data_inspection / fig, report_figs / fig)
    for fig in [
        "ablation_barplot.png",
        "validation_dice_curves.png",
        "training_loss_curves.png",
        "epoch_seconds_curves.png",
        "peak_memory_curves.png",
        "per_class_dice_barplot.png",
        "hd95_mean_barplot.png",
        "avd_mean_barplot.png",
    ]:
        copy_if_exists(args.analysis / fig, report_figs / fig)

    write_ablation_table(args.analysis / "ablation_summary.csv", report_tables / "ablation_summary.tex")
    write_per_class_table(args.analysis / "per_class_dice.csv", report_tables / "per_class_dice.tex")
    pick_best_worst_overlays(args.runs, report_figs)
    print(f"Report assets prepared in {args.report_dir}")


if __name__ == "__main__":
    main()
