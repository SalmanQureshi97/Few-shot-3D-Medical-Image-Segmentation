#!/usr/bin/env bash
# One-shot driver: runs data inspection, the A0..A4 LOOCV ablation suite, the
# detailed (8-class) run, profiles the models, predicts on the test set and
# evaluates the predictions, then aggregates everything for the report.
#
# Usage:
#   bash scripts/run_full_pipeline.sh data
#
# Set INCLUDE_SWIN=1 to also include the SwinUNETR config (requires MONAI).
set -euo pipefail

DATA_ROOT=${1:-data}
INCLUDE_SWIN=${INCLUDE_SWIN:-0}
DEVICE=${DEVICE:-auto}

echo "==> data inspection"
python3 scripts/inspect_data.py --data-root "$DATA_ROOT" --out artifacts/data_inspection
python3 scripts/inspect_data.py --data-root "$DATA_ROOT" --out artifacts/data_inspection_detailed --target detailed

CONFIGS=(
  configs/baseline_unet3d.yaml
  configs/ablation_a1_fg_aug.yaml
  configs/ablation_a2_dicece_focal.yaml
  configs/improved_resattn_unet3d.yaml
  configs/ablation_a4_resattn_tta.yaml
)
if [[ "$INCLUDE_SWIN" == "1" ]]; then
  CONFIGS+=(configs/ablation_a5_swin_unetr.yaml)
fi

echo "==> ablation training"
python3 scripts/run_ablations.py --data-root "$DATA_ROOT" --device "$DEVICE" --configs "${CONFIGS[@]}"

echo "==> detailed (8-class) training"
python3 scripts/run_loocv.py --config configs/detailed_resattn_unet3d.yaml --data-root "$DATA_ROOT" --device "$DEVICE"

echo "==> efficiency profile"
python3 scripts/profile_model.py --configs "${CONFIGS[@]}" --device "$DEVICE"

echo "==> predict test set (headline + TTA model)"
python3 scripts/predict_test.py \
  --config configs/ablation_a4_resattn_tta.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint-glob "runs/ablation_a4_resattn_tta/fold_*/checkpoints/best.pt" \
  --use-ema \
  --evaluate-against-labels \
  --out predictions/test

echo "==> summarise + report assets"
python3 scripts/summarise_experiments.py --runs runs --out artifacts/analysis
python3 scripts/prepare_report_assets.py --report-dir report --analysis artifacts/analysis --runs runs --data-inspection artifacts/data_inspection

echo "Done. See runs/, artifacts/, predictions/test/, report/."
