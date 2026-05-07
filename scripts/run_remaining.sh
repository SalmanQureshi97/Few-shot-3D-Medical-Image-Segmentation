#!/usr/bin/env bash
# Continue the pipeline from where it currently is. Idempotent: it skips any
# (config, fold) that already has a checkpoints/best.pt unless you ask it to
# rerun something explicitly via SKIP_DELETE_RESATTN=0 / FORCE_RETRAIN_*.
#
# Usage:
#   bash scripts/run_remaining.sh data
#
# Env knobs:
#   DELETE_RESATTN=1   wipe runs/improved_resattn_unet3d before retraining (default 1)
#   SKIP_A4_TRAIN=1    skip training A4 and instead revalidate from resattn checkpoints (default 1)
#   INCLUDE_DETAILED=1 also train the 8-class detailed config (default 1)
#   INCLUDE_SWIN=1     also train the SwinUNETR config (default 0; needs MONAI)
#   DEVICE=auto        torch device

set -euo pipefail

DATA_ROOT=${1:-data}
DEVICE=${DEVICE:-auto}
DELETE_RESATTN=${DELETE_RESATTN:-1}
SKIP_A4_TRAIN=${SKIP_A4_TRAIN:-1}
INCLUDE_DETAILED=${INCLUDE_DETAILED:-1}
INCLUDE_SWIN=${INCLUDE_SWIN:-0}

needs_train() {
    # needs_train <config_path> [num_folds]
    local cfg=$1
    local n=${2:-5}
    local exp
    exp=$(python3 -c "import yaml; print(yaml.safe_load(open('$cfg'))['experiment']['name'])")
    local missing=0
    for k in $(seq 0 $((n - 1))); do
        if [[ ! -f "runs/$exp/fold_$k/checkpoints/best.pt" ]]; then
            missing=$((missing + 1))
        fi
    done
    [[ $missing -gt 0 ]]
}

train_if_needed() {
    local cfg=$1
    if needs_train "$cfg"; then
        echo "==> training $cfg (some folds missing)"
        python3 scripts/run_loocv.py --config "$cfg" --data-root "$DATA_ROOT" --device "$DEVICE"
    else
        echo "==> $cfg already complete, skipping"
    fi
}

# 1. resattn: optionally wipe so we retrain cleanly with the NaN bug fix.
if [[ "$DELETE_RESATTN" == "1" ]]; then
    echo "==> wiping runs/improved_resattn_unet3d (set DELETE_RESATTN=0 to keep)"
    rm -rf runs/improved_resattn_unet3d
fi
train_if_needed configs/improved_resattn_unet3d.yaml

# 2. A0/A1/A2 should already be done; train_if_needed will skip them.
train_if_needed configs/baseline_unet3d.yaml
train_if_needed configs/ablation_a1_fg_aug.yaml
train_if_needed configs/ablation_a2_dicece_focal.yaml

# 3. A4: identical training to A3 except for inference TTA, so we revalidate from
#    the A3 checkpoints rather than retraining.
if [[ "$SKIP_A4_TRAIN" == "1" ]]; then
    echo "==> revalidating A4 from improved_resattn_unet3d checkpoints (with TTA)"
    rm -rf runs/ablation_a4_resattn_tta
    python3 scripts/revalidate_with_tta.py \
        --source-config configs/improved_resattn_unet3d.yaml \
        --target-config configs/ablation_a4_resattn_tta.yaml \
        --source-runs runs/improved_resattn_unet3d \
        --target-runs runs/ablation_a4_resattn_tta \
        --data-root "$DATA_ROOT" --device "$DEVICE"
else
    train_if_needed configs/ablation_a4_resattn_tta.yaml
fi

# 4. Detailed 8-class run (independent of the coarse ones).
if [[ "$INCLUDE_DETAILED" == "1" ]]; then
    train_if_needed configs/detailed_resattn_unet3d.yaml
fi

# 5. Optional transformer baseline.
if [[ "$INCLUDE_SWIN" == "1" ]]; then
    train_if_needed configs/ablation_a5_swin_unetr.yaml
fi

# 6. Profile, predict, summarise, prepare report assets.
PROFILE_CONFIGS=(
    configs/baseline_unet3d.yaml
    configs/ablation_a1_fg_aug.yaml
    configs/ablation_a2_dicece_focal.yaml
    configs/improved_resattn_unet3d.yaml
    configs/ablation_a4_resattn_tta.yaml
)
[[ "$INCLUDE_DETAILED" == "1" ]] && PROFILE_CONFIGS+=(configs/detailed_resattn_unet3d.yaml)
[[ "$INCLUDE_SWIN" == "1" ]] && PROFILE_CONFIGS+=(configs/ablation_a5_swin_unetr.yaml)

echo "==> efficiency profile"
python3 scripts/profile_model.py --configs "${PROFILE_CONFIGS[@]}" --device "$DEVICE"

echo "==> predict test set with A4 (TTA, EMA)"
python3 scripts/predict_test.py \
    --config configs/ablation_a4_resattn_tta.yaml \
    --data-root "$DATA_ROOT" \
    --checkpoint-glob "runs/improved_resattn_unet3d/fold_*/checkpoints/best.pt" \
    --use-ema --evaluate-against-labels \
    --out predictions/test

echo "==> summarise + report assets"
python3 scripts/summarise_experiments.py --runs runs --out artifacts/analysis
python3 scripts/prepare_report_assets.py --report-dir report

echo "All done. See runs/, artifacts/analysis/, predictions/test/, report/."
