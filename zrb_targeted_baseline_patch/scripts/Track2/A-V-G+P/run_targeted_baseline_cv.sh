#!/usr/bin/env bash
set -euo pipefail

# Run from the MPDD-AVG-2026 repository root.
# This script keeps the official TorchcatBaseline model architecture, but uses
# strict train-only CV, imbalance-aware sampling/loss, and ID-label-PHQ checks.

DATA_ROOT=${DATA_ROOT:-MPDD-AVG2026/MPDD-AVG2026-trainval/Young}
SPLIT_CSV=${SPLIT_CSV:-MPDD-AVG2026/MPDD-AVG2026-trainval/Young/split_labels_train.csv}
PERSONALITY_NPY=${PERSONALITY_NPY:-MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy}

DEVICE=${DEVICE:-cuda}
CV_FOLDS=${CV_FOLDS:-5}
CV_REPEATS=${CV_REPEATS:-3}
MAX_FOLDS=${MAX_FOLDS:-0}
FOLD_IDX=${FOLD_IDX:--1}
EPOCHS=${EPOCHS:-60}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-0.001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
HIDDEN_DIM=${HIDDEN_DIM:-64}
DROPOUT=${DROPOUT:-0.5}
PATIENCE=${PATIENCE:-20}
MIN_SELECT_EPOCH=${MIN_SELECT_EPOCH:-8}
NUM_WORKERS=${NUM_WORKERS:-0}

ENCODER_TYPE=${ENCODER_TYPE:-bilstm_mean}
AUDIO_FEATURE=${AUDIO_FEATURE:-wav2vec}
VIDEO_FEATURE=${VIDEO_FEATURE:-openface}
TARGET_T=${TARGET_T:-128}

# Imbalance/small-data knobs.
CLASS_WEIGHT_MODE=${CLASS_WEIGHT_MODE:-sqrt}
SAMPLER_MODE=${SAMPLER_MODE:-sqrt}
BOUNDARY_POS_WEIGHT_MODE=${BOUNDARY_POS_WEIGHT_MODE:-sqrt}
CE_LOSS_WEIGHT=${CE_LOSS_WEIGHT:-1.0}
ORD_LOSS_WEIGHT=${ORD_LOSS_WEIGHT:-0.5}
PHQ_LOSS_WEIGHT=${PHQ_LOSS_WEIGHT:-0.0}
GE5_LOSS_WEIGHT=${GE5_LOSS_WEIGHT:-1.0}
GE10_LOSS_WEIGHT=${GE10_LOSS_WEIGHT:-1.2}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.05}
FOCAL_GAMMA=${FOCAL_GAMMA:-0.0}

# Main prediction can be argmax or threshold. Start with argmax for the cleanest baseline-like result.
PRED_MODE=${PRED_MODE:-argmax}
PRED_GE5_THRESHOLD=${PRED_GE5_THRESHOLD:-0.50}
PRED_GE10_THRESHOLD=${PRED_GE10_THRESHOLD:-0.45}

# Best-checkpoint selection: protect both middle and severe classes.
KAPPA_WEIGHT=${KAPPA_WEIGHT:-0.1}
RECALL1_WEIGHT=${RECALL1_WEIGHT:-0.05}
RECALL2_WEIGHT=${RECALL2_WEIGHT:-0.05}
ZERO_RECALL_PENALTY=${ZERO_RECALL_PENALTY:-0.1}
CCC_WEIGHT=${CCC_WEIGHT:-0.0}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-track2_avgp_targeted_baseline_cv5x3}
LOGS_DIR=${LOGS_DIR:-logs}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-checkpoints}

python train_targeted_baseline_cv.py \
  --track Track2 \
  --task ternary \
  --subtrack A-V-G+P \
  --encoder_type "${ENCODER_TYPE}" \
  --audio_feature "${AUDIO_FEATURE}" \
  --video_feature "${VIDEO_FEATURE}" \
  --data_root "${DATA_ROOT}" \
  --split_csv "${SPLIT_CSV}" \
  --personality_npy "${PERSONALITY_NPY}" \
  --device "${DEVICE}" \
  --cv_folds "${CV_FOLDS}" \
  --cv_repeats "${CV_REPEATS}" \
  --fold_idx "${FOLD_IDX}" \
  --max_folds "${MAX_FOLDS}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --dropout "${DROPOUT}" \
  --target_t "${TARGET_T}" \
  --patience "${PATIENCE}" \
  --min_select_epoch "${MIN_SELECT_EPOCH}" \
  --num_workers "${NUM_WORKERS}" \
  --class_weight_mode "${CLASS_WEIGHT_MODE}" \
  --sampler_mode "${SAMPLER_MODE}" \
  --boundary_pos_weight_mode "${BOUNDARY_POS_WEIGHT_MODE}" \
  --ce_loss_weight "${CE_LOSS_WEIGHT}" \
  --ord_loss_weight "${ORD_LOSS_WEIGHT}" \
  --phq_loss_weight "${PHQ_LOSS_WEIGHT}" \
  --ge5_loss_weight "${GE5_LOSS_WEIGHT}" \
  --ge10_loss_weight "${GE10_LOSS_WEIGHT}" \
  --label_smoothing "${LABEL_SMOOTHING}" \
  --focal_gamma "${FOCAL_GAMMA}" \
  --pred_mode "${PRED_MODE}" \
  --pred_ge5_threshold "${PRED_GE5_THRESHOLD}" \
  --pred_ge10_threshold "${PRED_GE10_THRESHOLD}" \
  --kappa_weight "${KAPPA_WEIGHT}" \
  --recall1_weight "${RECALL1_WEIGHT}" \
  --recall2_weight "${RECALL2_WEIGHT}" \
  --zero_recall_penalty "${ZERO_RECALL_PENALTY}" \
  --ccc_weight "${CCC_WEIGHT}" \
  --logs_dir "${LOGS_DIR}" \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}"
