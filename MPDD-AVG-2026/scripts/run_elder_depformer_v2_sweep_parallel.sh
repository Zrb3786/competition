#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# MPDD-AVG-2026 Track1/young DepFormerAVP-v2 paired sweep
# Multi-GPU parallel scheduler version
# ------------------------------------------------------------
# Core behavior:
#   1) Build one queue item per configuration group:
#      audio/video/encoder/hidden_dim/seed
#   2) Start one worker for each selected GPU.
#   3) Each worker runs binary -> ternary sequentially for its group.
#   4) Different groups are distributed across different GPUs in parallel.
#
# Typical usage:
#   MIN_GPUS=2 MAX_GPUS=4 GPU_THRESHOLD=0.5 \
#   bash scripts/run_young_depformer_v2_sweep_parallel.sh
#
# Use fixed GPUs manually:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_young_depformer_v2_sweep_parallel.sh
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

TRACK="Track2"
SUBTRACK="${SUBTRACK:-A-V-G+P}"
DATA_ROOT="${DATA_ROOT:-MPDD-AVG2026/MPDD-AVG2026-trainval/Young}"
SPLIT_CSV="${SPLIT_CSV:-MPDD-AVG2026/MPDD-AVG2026-trainval/Young/split_labels_train.csv}"
PERSONALITY_NPY="${PERSONALITY_NPY:-MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints_young_v2}"
LOGS_DIR="${LOGS_DIR:-logs_young_v2}"

# Keep binary before ternary inside each worker/group.
TASKS_STR="${TASKS_STR:-binary ternary}"
AUDIO_FEATURES_STR="${AUDIO_FEATURES_STR:-mfcc wav2vec}"
VIDEO_FEATURES_STR="${VIDEO_FEATURES_STR:-densenet resnet}"
ENCODERS_STR="${ENCODERS_STR:- hybrid_attn}"
HIDDEN_DIMS_STR="${HIDDEN_DIMS_STR:- 128 256}"
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
TARGET_T="${TARGET_T:-256}"
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

# In this parallel version, MAX_GROUPS is the clean limiter.
# MAX_RUNS is kept for compatibility, but group-level scheduling is recommended
# because it preserves binary -> ternary pairing inside the same configuration.
MAX_RUNS="${MAX_RUNS:-0}"
MAX_GROUPS="${MAX_GROUPS:-0}"

STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Multi-GPU auto-selection.
# If CUDA_VISIBLE_DEVICES is already set, the listed GPUs are used as workers directly.
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
GPU_THRESHOLD="${GPU_THRESHOLD:-0.5}"        # memory.used / memory.total < threshold
MIN_GPUS="${MIN_GPUS:-2}"
MAX_GPUS="${MAX_GPUS:-4}"
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

check_required_paths() {
  local missing=0
  if [ ! -f "${TRAIN_SCRIPT}" ]; then echo "ERROR: missing ${TRAIN_SCRIPT}" >&2; missing=1; fi
  if [ ! -d "${DATA_ROOT}" ]; then echo "ERROR: missing DATA_ROOT=${DATA_ROOT}" >&2; missing=1; fi
  if [ ! -f "${SPLIT_CSV}" ]; then echo "ERROR: missing SPLIT_CSV=${SPLIT_CSV}" >&2; missing=1; fi
  if [ ! -f "${PERSONALITY_NPY}" ]; then echo "ERROR: missing PERSONALITY_NPY=${PERSONALITY_NPY}" >&2; missing=1; fi
  if ! command -v flock >/dev/null 2>&1; then echo "ERROR: flock is required for the queue lock." >&2; missing=1; fi
  if [ "${missing}" -ne 0 ]; then exit 1; fi
}

select_gpus() {
  SELECTED_GPUS=()

  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a SELECTED_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    echo "CUDA_VISIBLE_DEVICES already set: ${CUDA_VISIBLE_DEVICES}"
    echo "Use these GPUs as parallel workers: ${SELECTED_GPUS[*]}"
    return 0
  fi

  if [ "${WAIT_FOR_GPU}" != "1" ]; then
    echo "WAIT_FOR_GPU=0 and CUDA_VISIBLE_DEVICES is empty. Use one default worker on visible cuda:0."
    SELECTED_GPUS=(0)
    return 0
  fi

  echo "Waiting for ${MIN_GPUS}-${MAX_GPUS} GPUs with memory usage ratio < ${GPU_THRESHOLD}"
  while true; do
    local gpu_info free_lines count
    gpu_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
    if [ -z "${gpu_info}" ]; then
      echo "ERROR: nvidia-smi unavailable and CUDA_VISIBLE_DEVICES is not set." >&2
      exit 1
    fi

    # Output format: usage idx, sorted by usage ascending.
    free_lines=$(
      while read -r line; do
        local idx used total usage
        idx=$(echo "${line}" | awk -F', ' '{print $1}')
        used=$(echo "${line}" | awk -F', ' '{print $2}')
        total=$(echo "${line}" | awk -F', ' '{print $3}')
        [ -z "${idx}" ] && continue
        usage=$(awk "BEGIN {printf \"%.6f\", ${used}/${total}}")
        if (( $(awk "BEGIN {print (${usage} < ${GPU_THRESHOLD})}") )); then
          printf "%s %s\n" "${usage}" "${idx}"
        fi
      done <<< "${gpu_info}" | sort -n | head -n "${MAX_GPUS}"
    )

    count=$(echo "${free_lines}" | sed '/^$/d' | wc -l | awk '{print $1}')
    if [ "${count}" -ge "${MIN_GPUS}" ]; then
      mapfile -t SELECTED_GPUS < <(echo "${free_lines}" | awk '{print $2}')
      echo "Selected GPUs: ${SELECTED_GPUS[*]}"
      return 0
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') only ${count} free GPU(s), need at least ${MIN_GPUS}; retry in ${WAIT_INTERVAL}s"
    sleep "${WAIT_INTERVAL}"
  done
}

build_group_queue() {
  QUEUE_FILE=$(mktemp -p "${PROJECT_ROOT}" young_v2_groups.XXXXXX.tsv)
  local group_count=0
  local generated_runs=0

  for audio_feature in "${AUDIO_FEATURES[@]}"; do
    for video_feature in "${VIDEO_FEATURES[@]}"; do
      for encoder in "${ENCODERS[@]}"; do
        for hidden_dim in "${HIDDEN_DIMS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            if [ "${MAX_GROUPS}" -gt 0 ] && [ "${group_count}" -ge "${MAX_GROUPS}" ]; then
              break 5
            fi
            if [ "${MAX_RUNS}" -gt 0 ] && [ "${generated_runs}" -ge "${MAX_RUNS}" ]; then
              break 5
            fi
            group_count=$((group_count + 1))
            generated_runs=$((generated_runs + ${#TASKS[@]}))
            printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
              "${group_count}" "${audio_feature}" "${video_feature}" "${encoder}" "${hidden_dim}" "${seed}" >> "${QUEUE_FILE}"
          done
        done
      done
    done
  done

  TOTAL_GROUPS=$(wc -l < "${QUEUE_FILE}" | awk '{print $1}')
  STATE_FILE=$(mktemp -p "${PROJECT_ROOT}" young_v2_queue_state.XXXXXX)
  LOCK_FILE="${STATE_FILE}.lock"
  echo 0 > "${STATE_FILE}"

  if [ "${MAX_RUNS}" -gt 0 ]; then
    echo "WARNING: MAX_RUNS=${MAX_RUNS} is approximate in group-parallel mode; queue is generated by complete groups. Prefer MAX_GROUPS."
  fi
}

get_next_group_line() {
  local next line
  {
    flock -x 200
    next=$(cat "${STATE_FILE}")
    next=$((next + 1))
    if [ "${next}" -gt "${TOTAL_GROUPS}" ]; then
      line=""
    else
      echo "${next}" > "${STATE_FILE}"
      line=$(sed -n "${next}p" "${QUEUE_FILE}")
    fi
    printf "%s" "${line}"
  } 200>"${LOCK_FILE}"
}

run_one_task() {
  local worker_id="$1"
  local physical_gpu="$2"
  local group_id="$3"
  local audio_feature="$4"
  local video_feature="$5"
  local encoder="$6"
  local hidden_dim="$7"
  local seed="$8"
  local task="$9"

  set_task_defaults "${task}"

  local num_heads log_subdir exp exp_dir run_log
  num_heads=$(choose_num_heads "${hidden_dim}")
  log_subdir=$(subtrack_log_dir "${SUBTRACK}")

  exp="young_v2_${task}_${SUBTRACK}_${encoder}_${audio_feature}_${video_feature}_h${hidden_dim}_s${seed}"
  exp=${exp//+/_}; exp=${exp//-/_}; exp=${exp//__/_}
  exp_dir="${LOGS_DIR}/${TRACK}/${log_subdir}/${task}/${exp}"
  run_log="${exp_dir}/run_stdout_gpu${physical_gpu}.log"

  if [ "${SKIP_DONE}" = "1" ] && compgen -G "${exp_dir}/train_result_*.json" > /dev/null; then
    echo "[worker ${worker_id} | GPU ${physical_gpu}] [SKIP] ${exp}: found train_result_*.json"
    return 0
  fi

  mkdir -p "${exp_dir}"

  local cmd=(
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
    local extra_array=(${EXTRA_ARGS})
    cmd+=("${extra_array[@]}")
  fi

  {
    echo "============================================================"
    echo "[worker ${worker_id} | physical GPU ${physical_gpu} | group ${group_id}] ${exp}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    printf '%q ' "${cmd[@]}"; echo
    echo "============================================================"
  } | tee "${run_log}"

  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi

  if "${cmd[@]}" >> "${run_log}" 2>&1; then
    echo "[worker ${worker_id} | GPU ${physical_gpu}] [DONE] ${exp}"
    return 0
  else
    echo "[worker ${worker_id} | GPU ${physical_gpu}] [FAILED] ${exp}; see ${run_log}" >&2
    return 1
  fi
}

worker_loop() {
  local worker_id="$1"
  local physical_gpu="$2"
  export CUDA_VISIBLE_DEVICES="${physical_gpu}"

  echo "[worker ${worker_id}] started on physical GPU ${physical_gpu}"

  local line group_id audio_feature video_feature encoder hidden_dim seed
  while true; do
    line=$(get_next_group_line)
    if [ -z "${line}" ]; then
      echo "[worker ${worker_id} | GPU ${physical_gpu}] no more groups."
      return 0
    fi

    IFS=$'\t' read -r group_id audio_feature video_feature encoder hidden_dim seed <<< "${line}"
    echo "============================================================"
    echo "[worker ${worker_id} | GPU ${physical_gpu}] GROUP ${group_id}/${TOTAL_GROUPS}: audio=${audio_feature} video=${video_feature} encoder=${encoder} hidden=${hidden_dim} seed=${seed}"
    echo "Task order inside group: ${TASKS_STR}"
    echo "============================================================"

    local task
    for task in "${TASKS[@]}"; do
      if ! run_one_task "${worker_id}" "${physical_gpu}" "${group_id}" "${audio_feature}" "${video_feature}" "${encoder}" "${hidden_dim}" "${seed}" "${task}"; then
        if [ "${STOP_ON_ERROR}" = "1" ]; then
          return 1
        fi
      fi
    done
  done
}

cleanup_tmp() {
  rm -f "${QUEUE_FILE:-}" "${STATE_FILE:-}" "${LOCK_FILE:-}"
}
trap cleanup_tmp EXIT

cat <<INFO
============================================================
 young DepFormerAVP-v2 paired sweep: multi-GPU parallel
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
 LOOP_ORDER         : group queue -> worker GPU -> binary then ternary
 LOSS_TYPE          : ${LOSS_TYPE}
 SELECTION_MODE     : ${SELECTION_MODE}
 CHECKPOINTS_DIR    : ${CHECKPOINTS_DIR}
 LOGS_DIR           : ${LOGS_DIR}
 DRY_RUN            : ${DRY_RUN}
 MAX_RUNS           : ${MAX_RUNS}
 MAX_GROUPS         : ${MAX_GROUPS}
 MIN_GPUS/MAX_GPUS  : ${MIN_GPUS}/${MAX_GPUS}
 GPU_THRESHOLD      : ${GPU_THRESHOLD}
============================================================
INFO

check_required_paths
select_gpus
mkdir -p "${CHECKPOINTS_DIR}" "${LOGS_DIR}"
build_group_queue

if [ "${TOTAL_GROUPS}" -le 0 ]; then
  echo "No groups to run."
  exit 0
fi

cat <<INFO
============================================================
 Queue ready
------------------------------------------------------------
 TOTAL_GROUPS       : ${TOTAL_GROUPS}
 WORKER_GPUS        : ${SELECTED_GPUS[*]}
 WORKER_COUNT       : ${#SELECTED_GPUS[@]}
 QUEUE_FILE         : ${QUEUE_FILE}
============================================================
INFO

pids=()
worker_id=0
for gpu in "${SELECTED_GPUS[@]}"; do
  worker_id=$((worker_id + 1))
  worker_loop "${worker_id}" "${gpu}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=$((failed + 1))
  fi
done

if [ "${failed}" -gt 0 ]; then
  echo "============================================================"
  echo "Finished with failures. failed_workers=${failed}"
  echo "============================================================"
  exit 1
fi

echo "============================================================"
echo "Finished successfully. groups=${TOTAL_GROUPS}, workers=${#SELECTED_GPUS[@]}"
echo "============================================================"
