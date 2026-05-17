#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# MPDD-AVG-2026 Track1/Elder DepFormerAVP-v2 paired sweep
# ------------------------------------------------------------
# Difference from the previous sweep:
#   Old order:
#     for task in binary ternary; for feature/model/seed ...
#     -> all binary runs finish before ternary starts.
#
#   New order:
#     for feature/model/seed; for task in binary ternary ...
#     -> each configuration runs binary immediately followed by ternary.
#
# This is more convenient for quick validation because one config will produce
# both binary and ternary checkpoints/logs before moving to the next config.
#
# Required placement:
#   cp depformer_avp_v2.py models/depformer_avp.py
#   cp train1_avgp25_v2.py train1_avgp25_v2.py
#   cp run_elder_depformer_v2_sweep_task_inner.sh scripts/run_elder_depformer_v2_sweep.sh
#   bash scripts/run_elder_depformer_v2_sweep.sh
# ============================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "${SCRIPT_DIR}/../train1_avgp25_v2.py" ]; then
  PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
elif [ -f "$(pwd)/train1_avgp25_v2.py" ]; then
  PROJECT_ROOT=$(pwd)
else
  PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
fi
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train1_avgp25_v2.py}"
CONFIG="${CONFIG:-config.json}"
DEVICE="${DEVICE:-cuda}"

TRACK="Track1"
SUBTRACK="${SUBTRACK:-A-V-G+P}"
DATA_ROOT="${DATA_ROOT:-MPDD-AVG2026/MPDD-AVG2026-trainval/Elder}"
SPLIT_CSV="${SPLIT_CSV:-MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/split_labels_train.csv}"
PERSONALITY_NPY="${PERSONALITY_NPY:-MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/descriptions_embeddings_with_ids.npy}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints_elder_v2sweep}"
LOGS_DIR="${LOGS_DIR:-logs_elder_v2sweep}"

# Keep binary before ternary by default. You can override TASKS_STR="ternary binary" if needed.
TASKS_STR="${TASKS_STR:-ternary binary}"
AUDIO_FEATURES_STR="${AUDIO_FEATURES_STR:-mfcc wav2vec}"
VIDEO_FEATURES_STR="${VIDEO_FEATURES_STR:-densenet resnet}"
ENCODERS_STR="${ENCODERS_STR:-report_lstm hybrid_attn}"
HIDDEN_DIMS_STR="${HIDDEN_DIMS_STR:-64 128}"
SEEDS_STR="${SEEDS_STR:-42 3407 2025}"

# Loss / selection controls.
LOSS_TYPE="${LOSS_TYPE:-ce_focal_mse}"
SELECTION_MODE="${SELECTION_MODE:-score}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
FOCAL_LAMBDA="${FOCAL_LAMBDA:-0.5}"
REG_LAMBDA="${REG_LAMBDA:-0.3}"
FORCE_REGRESSION_HEAD="${FORCE_REGRESSION_HEAD:-1}"
USE_P_GATE="${USE_P_GATE:-1}"
AV_ENCODE_PAIRWISE="${AV_ENCODE_PAIRWISE:-1}"
NUM_BCT_LAYERS="${NUM_BCT_LAYERS:-1}"
FFN_MULT="${FFN_MULT:-4}"

VAL_RATIO="${VAL_RATIO:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TARGET_T="${TARGET_T:-128}"
DROPOUT="${DROPOUT:-0.5}"
MIN_DELTA="${MIN_DELTA:-1e-4}"

# Task-specific official-style defaults. Override with BINARY_LR, TERNARY_LR, etc.
BINARY_EPOCHS="${BINARY_EPOCHS:-100}"
BINARY_BATCH_SIZE="${BINARY_BATCH_SIZE:-8}"
BINARY_LR="${BINARY_LR:-7e-4}"
BINARY_WEIGHT_DECAY="${BINARY_WEIGHT_DECAY:-1e-4}"
BINARY_PATIENCE="${BINARY_PATIENCE:-20}"

TERNARY_EPOCHS="${TERNARY_EPOCHS:-140}"
TERNARY_BATCH_SIZE="${TERNARY_BATCH_SIZE:-4}"
TERNARY_LR="${TERNARY_LR:-2e-4}"
TERNARY_WEIGHT_DECAY="${TERNARY_WEIGHT_DECAY:-5e-5}"
TERNARY_PATIENCE="${TERNARY_PATIENCE:-30}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"

# MAX_RUNS counts every task run. MAX_RUNS=2 means one binary+ternary pair if TASKS_STR="binary ternary".
MAX_RUNS="${MAX_RUNS:-0}"

# MAX_GROUPS counts configuration groups. MAX_GROUPS=1 always runs all tasks for the first config.
# A group = one audio/video/encoder/hidden_dim/seed combination.
MAX_GROUPS="${MAX_GROUPS:-0}"

STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# GPU auto-selection. If CUDA_VISIBLE_DEVICES is already set, this will not change it.
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
GPU_THRESHOLD="${GPU_THRESHOLD:-0.5}"
WAIT_INTERVAL="${WAIT_INTERVAL:-60}"

read -r -a TASKS <<< "${TASKS_STR}"
read -r -a AUDIO_FEATURES <<< "${AUDIO_FEATURES_STR}"
read -r -a VIDEO_FEATURES <<< "${VIDEO_FEATURES_STR}"
read -r -a ENCODERS <<< "${ENCODERS_STR}"
read -r -a HIDDEN_DIMS <<< "${HIDDEN_DIMS_STR}"
read -r -a SEEDS <<< "${SEEDS_STR}"

choose_num_heads() {
  local hdim="$1"
  if [ -n "${NUM_HEADS_FORCE:-}" ]; then
    echo "${NUM_HEADS_FORCE}"
  elif (( hdim % 4 == 0 )); then
    echo 4
  else
    echo 2
  fi
}

subtrack_log_dir() {
  case "$1" in
    "A-V+P") echo "A-V-P" ;;
    "A-V-G+P") echo "A-V-G+P" ;;
    "G+P") echo "G-P" ;;
    *) echo "$1" | sed 's/+/-/g' ;;
  esac
}

set_task_defaults() {
  local task="$1"
  case "${task}" in
    binary)
      EPOCHS_USE="${BINARY_EPOCHS}"
      BATCH_SIZE_USE="${BINARY_BATCH_SIZE}"
      LR_USE="${BINARY_LR}"
      WEIGHT_DECAY_USE="${BINARY_WEIGHT_DECAY}"
      PATIENCE_USE="${BINARY_PATIENCE}"
      ;;
    ternary)
      EPOCHS_USE="${TERNARY_EPOCHS}"
      BATCH_SIZE_USE="${TERNARY_BATCH_SIZE}"
      LR_USE="${TERNARY_LR}"
      WEIGHT_DECAY_USE="${TERNARY_WEIGHT_DECAY}"
      PATIENCE_USE="${TERNARY_PATIENCE}"
      ;;
    *)
      echo "ERROR: this sweep script only supports binary/ternary, got ${task}" >&2
      exit 1
      ;;
  esac
}

wait_for_free_gpu() {
  if [ "${WAIT_FOR_GPU}" != "1" ]; then return 0; fi
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "CUDA_VISIBLE_DEVICES already set to ${CUDA_VISIBLE_DEVICES}; skip GPU auto-selection."
    return 0
  fi

  echo "Waiting for free GPU: memory usage ratio < ${GPU_THRESHOLD}"
  while true; do
    local gpu_info best_gpu best_usage
    gpu_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
    if [ -z "${gpu_info}" ]; then
      echo "WARNING: nvidia-smi unavailable. Continue without setting CUDA_VISIBLE_DEVICES."
      return 0
    fi
    best_gpu=-1
    best_usage=1.0
    while read -r line; do
      local idx used total usage
      idx=$(echo "${line}" | awk -F', ' '{print $1}')
      used=$(echo "${line}" | awk -F', ' '{print $2}')
      total=$(echo "${line}" | awk -F', ' '{print $3}')
      [ -z "${idx}" ] && continue
      usage=$(awk "BEGIN {printf \"%.6f\", ${used}/${total}}")
      if (( $(awk "BEGIN {print (${usage} < ${GPU_THRESHOLD})}") )) && \
         (( $(awk "BEGIN {print (${usage} < ${best_usage})}") )); then
        best_gpu="${idx}"
        best_usage="${usage}"
      fi
    done <<< "${gpu_info}"
    if [ "${best_gpu}" -ge 0 ]; then
      export CUDA_VISIBLE_DEVICES="${best_gpu}"
      echo "Selected GPU ${CUDA_VISIBLE_DEVICES}; usage=$(awk "BEGIN {printf \"%.2f%%\", ${best_usage}*100}")"
      return 0
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') no free GPU, retry in ${WAIT_INTERVAL}s"
    sleep "${WAIT_INTERVAL}"
  done
}

check_required_paths() {
  local missing=0
  if [ ! -f "${TRAIN_SCRIPT}" ]; then echo "ERROR: missing ${TRAIN_SCRIPT}" >&2; missing=1; fi
  if [ ! -d "${DATA_ROOT}" ]; then echo "ERROR: missing DATA_ROOT=${DATA_ROOT}" >&2; missing=1; fi
  if [ ! -f "${SPLIT_CSV}" ]; then echo "ERROR: missing SPLIT_CSV=${SPLIT_CSV}" >&2; missing=1; fi
  if [ ! -f "${PERSONALITY_NPY}" ]; then echo "ERROR: missing PERSONALITY_NPY=${PERSONALITY_NPY}" >&2; missing=1; fi
  if [ "${missing}" -ne 0 ]; then exit 1; fi
}

cat <<INFO
============================================================
 Elder DepFormerAVP-v2 paired sweep
------------------------------------------------------------
 PROJECT_ROOT       : ${PROJECT_ROOT}
 TRAIN_SCRIPT       : ${TRAIN_SCRIPT}
 SUBTRACK           : ${SUBTRACK}
 TASK_ORDER         : ${TASKS_STR}
 AUDIO_FEATURES     : ${AUDIO_FEATURES_STR}
 VIDEO_FEATURES     : ${VIDEO_FEATURES_STR}
 ENCODERS           : ${ENCODERS_STR}
 HIDDEN_DIMS        : ${HIDDEN_DIMS_STR}
 SEEDS              : ${SEEDS_STR}
 LOOP_ORDER         : feature/model/seed group -> tasks
 LOSS_TYPE          : ${LOSS_TYPE}
 SELECTION_MODE     : ${SELECTION_MODE}
 LABEL_SMOOTHING    : ${LABEL_SMOOTHING}
 FOCAL_LAMBDA       : ${FOCAL_LAMBDA}
 REG_LAMBDA         : ${REG_LAMBDA}
 CHECKPOINTS_DIR    : ${CHECKPOINTS_DIR}
 LOGS_DIR           : ${LOGS_DIR}
 DRY_RUN            : ${DRY_RUN}
 MAX_RUNS           : ${MAX_RUNS}
 MAX_GROUPS         : ${MAX_GROUPS}
============================================================
INFO

check_required_paths
wait_for_free_gpu
mkdir -p "${CHECKPOINTS_DIR}" "${LOGS_DIR}"

run_count=0
fail_count=0
group_count=0
log_subdir=$(subtrack_log_dir "${SUBTRACK}")

for audio_feature in "${AUDIO_FEATURES[@]}"; do
  for video_feature in "${VIDEO_FEATURES[@]}"; do
    for encoder in "${ENCODERS[@]}"; do
      for hidden_dim in "${HIDDEN_DIMS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          if [ "${MAX_GROUPS}" -gt 0 ] && [ "${group_count}" -ge "${MAX_GROUPS}" ]; then
            echo "Reached MAX_GROUPS=${MAX_GROUPS}. Stop after complete task group(s). attempted=${run_count}, failed=${fail_count}"
            exit 0
          fi
          group_count=$((group_count + 1))
          num_heads=$(choose_num_heads "${hidden_dim}")

          echo "============================================================"
          echo "[GROUP ${group_count}] audio=${audio_feature} video=${video_feature} encoder=${encoder} hidden=${hidden_dim} seed=${seed}"
          echo "Task order: ${TASKS_STR}"
          echo "============================================================"

          for task in "${TASKS[@]}"; do
            set_task_defaults "${task}"

            exp="elder_v2_${task}_${SUBTRACK}_${encoder}_${audio_feature}_${video_feature}_h${hidden_dim}_s${seed}"
            exp=${exp//+/_}; exp=${exp//-/_}; exp=${exp//__/_}
            exp_dir="${LOGS_DIR}/${TRACK}/${log_subdir}/${task}/${exp}"

            if [ "${SKIP_DONE}" = "1" ] && compgen -G "${exp_dir}/train_result_*.json" > /dev/null; then
              echo "[SKIP] ${exp}: found train_result_*.json"
              continue
            fi

            cmd=(
              "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}"
              --config "${CONFIG}"
              --track "${TRACK}"
              --task "${task}"
              --subtrack "${SUBTRACK}"
              --model_type depformer
              --encoder_type "${encoder}"
              --audio_feature "${audio_feature}"
              --video_feature "${video_feature}"
              --experiment_name "${exp}"
              --data_root "${DATA_ROOT}"
              --split_csv "${SPLIT_CSV}"
              --personality_npy "${PERSONALITY_NPY}"
              --seed "${seed}"
              --val_ratio "${VAL_RATIO}"
              --epochs "${EPOCHS_USE}"
              --batch_size "${BATCH_SIZE_USE}"
              --lr "${LR_USE}"
              --weight_decay "${WEIGHT_DECAY_USE}"
              --hidden_dim "${hidden_dim}"
              --dropout "${DROPOUT}"
              --patience "${PATIENCE_USE}"
              --min_delta "${MIN_DELTA}"
              --target_t "${TARGET_T}"
              --device "${DEVICE}"
              --num_workers "${NUM_WORKERS}"
              --checkpoints_dir "${CHECKPOINTS_DIR}"
              --logs_dir "${LOGS_DIR}"
              --loss_type "${LOSS_TYPE}"
              --selection_mode "${SELECTION_MODE}"
              --label_smoothing "${LABEL_SMOOTHING}"
              --focal_gamma "${FOCAL_GAMMA}"
              --focal_lambda "${FOCAL_LAMBDA}"
              --reg_lambda "${REG_LAMBDA}"
              --force_regression_head "${FORCE_REGRESSION_HEAD}"
              --use_p_gate "${USE_P_GATE}"
              --av_encode_pairwise "${AV_ENCODE_PAIRWISE}"
              --num_bct_layers "${NUM_BCT_LAYERS}"
              --num_heads "${num_heads}"
              --ffn_mult "${FFN_MULT}"
            )
            if [ -n "${EXTRA_ARGS}" ]; then
              # shellcheck disable=SC2206
              extra_array=(${EXTRA_ARGS})
              cmd+=("${extra_array[@]}")
            fi

            run_count=$((run_count + 1))
            echo "============================================================"
            echo "[RUN ${run_count} | GROUP ${group_count}] ${exp}"
            printf '%q ' "${cmd[@]}"; echo
            echo "============================================================"

            if [ "${DRY_RUN}" != "1" ]; then
              if ! "${cmd[@]}"; then
                fail_count=$((fail_count + 1))
                echo "[FAILED] ${exp}"
                if [ "${STOP_ON_ERROR}" = "1" ]; then exit 1; fi
              fi
            fi

            if [ "${MAX_RUNS}" -gt 0 ] && [ "${run_count}" -ge "${MAX_RUNS}" ]; then
              echo "Reached MAX_RUNS=${MAX_RUNS}. Stop. attempted=${run_count}, failed=${fail_count}"
              exit 0
            fi
          done
        done
      done
    done
  done
done

echo "============================================================"
echo "Finished. groups=${group_count}, attempted=${run_count}, failed=${fail_count}"
echo "============================================================"
