# Few-Shot 3D Brain Tissue Segmentation on MRBrainS13

A research pipeline for brain tissue segmentation on the [MRBrainS13](https://mrbrains13.isi.uu.nl/data/index.html) dataset (5 training subjects, 15 test subjects, 1.5 T multi-modal MRI). The repo provides a reference 3D U-Net baseline plus a residual-attention 3D U-Net with deep supervision, foreground and class-balanced patch sampling, frequency-weighted Dice + (focal) cross-entropy, EMA, mixed precision, and sliding-window inference with optional test-time augmentation.

It is structured for **few-shot evaluation under memory constraints** and for **ablation-driven analysis** rather than a single best number.

## Highlights

- LOOCV training over the 5 MRBrainS13 training subjects with deterministic seeding.
- Models: 3D U-Net, Residual-Attention 3D U-Net with squeeze-excitation skips and deep supervision, optional SwinUNETR adapter (lazy import; MONAI optional).
- Losses: Dice / Tversky / Focal / Dice+CE with auto class-frequency weights, plus an auxiliary boundary loss.
- Sliding-window inference with Gaussian patch blending and optional flip TTA. AMP at both training and validation time.
- Per-class metrics: Dice, MRBrainS13-style AVD (`100·|Vp−Vt|/Vt`), HD95, ASSD, voxel confusion matrix.
- Data inspection: per-modality intensity histograms (raw and post-normalisation), class voxel frequency table, per-subject overlays.
- Ablation runner, efficiency profiler (params, latency, peak memory), test-set predictor with optional metric evaluation against `LabelsForTesting.nii`.

## Repository Layout

```text
configs/                YAML configs (baseline, A0..A5, detailed, smoke)
scripts/                CLI entry points
src/mrbrains/           library (data, models, losses, metrics, engine, infer, viz)
tests/                  pytest unit tests
data/                   place MRBrainS13 here (gitignored)
runs/                   generated; per-experiment training artifacts (gitignored)
predictions/test/       generated; held-out test predictions (gitignored)
artifacts/              generated; data-inspection figures and analysis tables (gitignored)
report/                 generated; figures and LaTeX tables for write-up (gitignored)
```

## Dataset Placement

Place the official MRBrainS13 distribution at `data/`:

```text
data/
  TrainingData/{1..5}/
    T1.nii  T1_1mm.nii  T1_IR.nii  T2_FLAIR.nii
    LabelsForTraining.nii  LabelsForTesting.nii
  TestData/{1..15}/
    T1.nii  T1_1mm.nii  T1_IR.nii  T2_FLAIR.nii
    LabelsForTesting.nii
```

The discovery code accepts the official Dataverse layout (`data/TrainingData/`, `data/TestData/`) and a nested `data/raw/mrbrains13/` layout.

References:
- MRBrainS13 challenge data page: <https://mrbrains13.isi.uu.nl/data/index.html>
- Mendrik et al., *MRBrainS Challenge*, Comp. Intel. Neurosci., 2015.
- Çiçek et al., *3D U-Net*, MICCAI 2016.
- Oktay et al., *Attention U-Net*, MIDL 2018.
- Hatamizadeh et al., *Swin UNETR*, BrainLes 2021.
- Isensee et al., *nnU-Net*, Nat. Methods 2021.

## Install

```bash
python3 -m pip install -r requirements.txt
# optional, only needed for the SwinUNETR config
python3 -m pip install -r requirements-monai.txt
```

## Quick Start

Smoke run on the synthetic dataset (CPU, ~5 minutes):

```bash
python3 scripts/create_synthetic_mrbrains.py --out data/synthetic/mrbrains --subjects 5
python3 scripts/inspect_data.py --data-root data/synthetic/mrbrains --out runs/smoke/inspect
python3 scripts/train.py --config configs/smoke.yaml --data-root data/synthetic/mrbrains --fold 0
```

Real-data sanity run (single fold, 1 epoch):

```bash
python3 scripts/inspect_data.py --data-root data --out artifacts/data_inspection
python3 scripts/run_loocv.py --config configs/baseline_unet3d.yaml --data-root data \
    --folds 0 --overrides training.epochs=1
```

Full LOOCV ablation suite + detailed-class run + test-set predictions + report assets, in one command:

```bash
bash scripts/run_full_pipeline.sh data
# or, with the SwinUNETR comparison row:
INCLUDE_SWIN=1 bash scripts/run_full_pipeline.sh data
```

Individual stages:

```bash
# Sequential ablation
python3 scripts/run_ablations.py --data-root data --configs \
    configs/baseline_unet3d.yaml \
    configs/ablation_a1_fg_aug.yaml \
    configs/ablation_a2_dicece_focal.yaml \
    configs/improved_resattn_unet3d.yaml \
    configs/ablation_a4_resattn_tta.yaml

# Headline LOOCV runs
python3 scripts/run_loocv.py --config configs/baseline_unet3d.yaml --data-root data
python3 scripts/run_loocv.py --config configs/improved_resattn_unet3d.yaml --data-root data

# Detailed (8-class) headline
python3 scripts/run_loocv.py --config configs/detailed_resattn_unet3d.yaml --data-root data

# Test-set predictions (averaged across LOOCV folds, with TTA)
python3 scripts/predict_test.py \
    --config configs/ablation_a4_resattn_tta.yaml \
    --data-root data \
    --checkpoint-glob "runs/ablation_a4_resattn_tta/fold_*/checkpoints/best.pt" \
    --use-ema --evaluate-against-labels --out predictions/test

# Re-evaluate saved predictions vs. LabelsForTesting.nii
python3 scripts/evaluate_predictions.py --data-root data --predictions predictions/test \
    --out-csv predictions/test/test_metrics.csv

# Efficiency profile (params, peak memory, inference latency)
python3 scripts/profile_model.py --configs \
    configs/baseline_unet3d.yaml configs/improved_resattn_unet3d.yaml

# Aggregate run logs into figures and tables
python3 scripts/summarise_experiments.py --runs runs --out artifacts/analysis
python3 scripts/prepare_report_assets.py --report-dir report
```

## Output Layout

```text
runs/<experiment>/fold_<k>/
    config.yaml
    history.csv            # per-epoch loss, lr, peak-memory, epoch wall time
    metrics_val.csv        # per-validation per-subject Dice/AVD/HD95/ASSD + voxel confusion
    checkpoints/best.pt    # best-by-val-Dice checkpoint (gitignored)
    figures/               # per-validation overlays
    tensorboard/           # scalars
artifacts/
    data_inspection/       # histograms, overlays, class table
    analysis/              # ablation_summary.csv, per_class_dice.csv, *.png
predictions/test/          # <subject>_prediction.nii.gz, test_metrics.csv
```

## Reproducibility

- `seed_everything()` seeds Python, NumPy, PyTorch, CUDA; `cudnn.deterministic=True`.
- Each fold uses `seed + fold` so folds differ deterministically.
- Saved configs (`runs/<exp>/fold_<k>/config.yaml`) include resolved class weights so a re-run is exact.
- Resume with `python3 scripts/train.py … --resume runs/<exp>/fold_<k>/checkpoints/best.pt`.

## Tests

```bash
pytest -q
```

Covers label remapping, class frequencies, all losses (Dice/Tversky/Focal/Boundary/DiceCE/DeepSupervision), all metrics (Dice/AVD/HD95/ASSD/confusion), data primitives (sampling, augmentation, affine warp), inference (sliding window, TTA), config overrides, and model shapes.
