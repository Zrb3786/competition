#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash elder_v15_trained_fusion/scripts/run_v15_trained_fusion.sh
# Optional env:
#   EXPERTS="p:v14v12_p_5x1,audio_controlled:v14v12_audio_controlled_5x1,gait:v14v12_gait_5x1,video:v14v12_video_5x1,audio_big:v14v12_audio_big_5x1"
#   OUT_DIR=outputs/elder_v15_trained_fusion/v15_gated_cls
#   SCORE_MODE=cls

ROOT=${ROOT:-/remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite}
cd "$ROOT"

TRAIN_SPLIT_CSV=${TRAIN_SPLIT_CSV:-/remote-home/yangmz/zhangruibo/MPDD-AVG-2026/MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/split_labels_train.csv}
EXPERT_ROOT=${EXPERT_ROOT:-outputs/elder_v14_v12loader}
OUT_DIR=${OUT_DIR:-outputs/elder_v15_trained_fusion/v15_gated_cls}
SCORE_MODE=${SCORE_MODE:-cls}
EXPERTS=${EXPERTS:-p:v14v12_p_5x1,audio_controlled:v14v12_audio_controlled_5x1,gait:v14v12_gait_5x1,video:v14v12_video_5x1,audio_big:v14v12_audio_big_5x1}

python -u elder_v15_trained_fusion/v15_trained_fusion.py \
  --expert_root "$EXPERT_ROOT" \
  --experts "$EXPERTS" \
  --train_split_csv "$TRAIN_SPLIT_CSV" \
  --output_dir "$OUT_DIR" \
  --score_mode "$SCORE_MODE" \
  --device cuda \
  --folds 5 \
  --seed "${SEED:-42}" \
  --split_seed "${SPLIT_SEED:-2026}" \
  --full_seeds "${FULL_SEEDS:-5}" \
  --epochs "${EPOCHS:-120}" \
  --patience "${PATIENCE:-25}" \
  --batch_size "${BS:-16}" \
  --hidden "${HIDDEN:-48}" \
  --dropout "${DROPOUT:-0.25}" \
  --lr "${LR:-8e-4}" \
  --weight_decay "${WD:-2e-3}" \
  --residual_scale "${RESIDUAL_SCALE:-0.12}" \
  --binary_weight "${BINARY_W:-0.60}" \
  --soft_f1_weight "${SOFTF1_W:-0.20}" \
  --kappa_weight "${KAPPA_W:-0.15}" \
  --phq_weight "${PHQ_W:-0.15}" \
  --ccc_weight "${CCC_W:-0.08}" \
  --gate_entropy_weight "${GATE_ENT_W:-0.005}" \
  --gate_peak_weight "${GATE_PEAK_W:-0.03}" \
  --max_gate "${MAX_GATE:-0.78}" \
  --log_every "${LOG_EVERY:-10}"
