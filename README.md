# Mini-Project: Few-Shot 3D Brain Tissue Segmentation

This repository is structured for the mini-project on MRBrainS13 brain tissue segmentation under data and memory constraints.

The implementation is deliberately inspectable: it uses PyTorch for the models, custom losses/metrics/sliding-window inference, and small medical-imaging utilities for NIfTI I/O. The aim is to produce a defensible baseline, an improved method, ablations, visual diagnostics, and report-ready evidence.

## Full-Mark Experiment Plan

1. **Data understanding**
   - Discover MRBrainS13 training/test subjects.
   - Plot per-modality intensity histograms before/after robust normalization.
   - Plot detailed and coarse class frequencies to motivate foreground sampling and imbalance-aware loss.

2. **Baseline**
   - 3D U-Net.
   - Patch-based training.
   - Multiclass Dice loss.
   - Leave-One-Out Cross-Validation (LOOCV) over the 5 training subjects.
   - Sliding-window inference for full-volume validation.

3. **Improved method**
   - Residual 3D U-Net with instance/group normalization, squeeze-excitation skip refinement, attention gates, and deep supervision.
   - Foreground-biased patch sampling.
   - Dice + cross-entropy or focal/Tversky loss for small structures.
   - Medical-safe augmentation: flips, intensity shift/scale, gamma, Gaussian noise.
   - Optional test-time flip augmentation.

4. **Ablations**
   - A0: baseline U-Net + Dice.
   - A1: A0 + foreground sampling + augmentation.
   - A2: A1 + DiceCE or Tversky loss.
   - A3: improved residual-attention architecture.
   - A4 optional: test-time augmentation or LOOCV checkpoint ensemble.

5. **Evaluation**
   - Per-class Dice, mean foreground Dice, volume similarity.
   - HD95/ASSD for validation folds where surfaces are non-empty.
   - Runtime, parameter count, inference time, and peak memory if available.
   - Best/worst fold overlays, error maps, and training curves.

6. **Submission evidence**
   - Submit code, configs, logs, CSV metrics, TensorBoard logs/screenshots, figures, and six-page report PDF.
   - Do not submit trained weights. The helper zip script excludes checkpoint files by default.

## Dataset Placement

Place MRBrainS13 data under:

```text
data/raw/mrbrains13/
  TrainingData/
    <subject folders...>
  TestData/
    <subject folders...>
```

The official Dataverse archive may also be extracted directly into `data/` as `data/TrainingData/` and `data/TestData/`; that layout is supported, so `--data-root data` is valid. The discovery code looks for MRBrainS file names such as `T1.nii`, `T1_IR.nii`, `T2_FLAIR.nii`, `LabelsForTraining.nii`, and `LabelsForTesting.nii`.

Official references:

- MRBrainS13 data page: https://mrbrains13.isi.uu.nl/data/index.html
- MRBrainS13 Dataverse DOI: https://doi.org/10.34894/645ZIN
- MRBrainS evaluation framework paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC4680055/

## Quick Start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Create a tiny synthetic dataset and run the smoke configuration:

```bash
python3 scripts/create_synthetic_mrbrains.py --out data/synthetic/mrbrains --subjects 5
python3 scripts/inspect_data.py --data-root data/synthetic/mrbrains --out runs/smoke_data
python3 scripts/train.py --config configs/smoke.yaml --data-root data/synthetic/mrbrains --fold 0
```

Run the baseline and improved LOOCV on real data:

```bash
python3 scripts/inspect_data.py --data-root data --out artifacts/data_inspection
python3 scripts/run_loocv.py --config configs/baseline_unet3d.yaml --data-root data
python3 scripts/run_loocv.py --config configs/improved_resattn_unet3d.yaml --data-root data
```

Build validation figures/tables:

```bash
python3 scripts/summarise_experiments.py --runs runs --out artifacts/analysis
python3 scripts/prepare_report_assets.py --report-dir report/ECS795P_Report_Latex_Template
```

Predict the official test subjects using the fold checkpoints:

```bash
python3 scripts/predict_test.py --config configs/improved_resattn_unet3d.yaml --data-root data --checkpoint-glob "runs/improved_resattn_unet3d/fold_*/checkpoints/best.pt" --out predictions/test
```

## Output Layout

```text
runs/
  <experiment_name>/fold_<k>/
    config.yaml
    history.csv
    metrics_val.csv
    checkpoints/best.pt
    figures/
    tensorboard/
artifacts/
  analysis/
  submission_manifest.json
```

## Report Skeleton

The report should use the provided double-column LaTeX template and stay within six pages:

1. Introduction and challenge framing.
2. Dataset analysis and preprocessing.
3. Baseline method.
4. Proposed improvements and rationale.
5. Experimental protocol and metrics.
6. Quantitative results and ablations.
7. Qualitative success/failure analysis.
8. Efficiency-performance trade-off.
9. Limitations and future work.
