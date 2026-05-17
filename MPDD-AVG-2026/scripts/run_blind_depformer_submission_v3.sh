#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# MPDD-AVG-2026 AVG-P / DepFormer blind-test submission wrapper
# ------------------------------------------------------------
# Usage example:
#   TRACK=Track1 \
#   SUBTRACK='A-V-G+P' \
#   BINARY_CKPT='checkpoints_avgp25/Track1/A-V-G+P/binary/.../best_model_*.pth' \
#   TERNARY_CKPT='checkpoints_avgp25/Track1/A-V-G+P/ternary/.../best_model_*.pth' \
#   bash scripts/run_blind_depformer_submission.sh
#
# Optional overrides:
#   TEST_SCRIPT, PYTHON_BIN, DEVICE, BATCH_SIZE, NUM_WORKERS,
#   DATA_ROOT, SPLIT_CSV, PERSONALITY_NPY, SAMPLE_BINARY_CSV,
#   SAMPLE_TERNARY_CSV, ID_COLUMN, OUTPUT_DIR, STRICT_LOAD, PHQ9_TRANSFORM
# ============================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "${SCRIPT_DIR}/../test_blind_depformer_avgp25.py" ]; then
  PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
elif [ -f "$(pwd)/test_blind_depformer_avgp25.py" ]; then
  PROJECT_ROOT=$(pwd)
else
  PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
fi
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TEST_SCRIPT="${TEST_SCRIPT:-test_blind_depformer_avgp25_v3.py}"
CONFIG="${CONFIG:-config.json}"
TRACK="${TRACK:-Track1}"
SUBTRACK="${SUBTRACK:-A-V-G+P}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-blind_submission/${TRACK}/${SUBTRACK//+/_}}"
STRICT_LOAD="${STRICT_LOAD:-0}"
PHQ9_TRANSFORM="${PHQ9_TRANSFORM:-log1p}"
MISSING_REGRESSION_POLICY="${MISSING_REGRESSION_POLICY:-error}"
SAVE_USED_IDS="${SAVE_USED_IDS:-1}"

case "${TRACK}" in
  Track1)
    DEFAULT_DATA_ROOT="MPDD-AVG2026/MPDD-AVG2026-test/Elder"
    DEFAULT_SPLIT_CSV="MPDD-AVG2026/MPDD-AVG2026-test/Elder/split_labels_test.csv"
    DEFAULT_PERSONALITY_NPY="MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/descriptions_embeddings_with_ids.npy"
    ;;
  Track2)
    DEFAULT_DATA_ROOT="MPDD-AVG2026/MPDD-AVG2026-test/Young"
    DEFAULT_SPLIT_CSV="MPDD-AVG2026/MPDD-AVG2026-test/Young/split_labels_test.csv"
    DEFAULT_PERSONALITY_NPY="MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy"
    ;;
  *)
    echo "ERROR: TRACK must be Track1 or Track2, got ${TRACK}" >&2
    exit 1
    ;;
esac

DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
SPLIT_CSV="${SPLIT_CSV:-${DEFAULT_SPLIT_CSV}}"
PERSONALITY_NPY="${PERSONALITY_NPY:-${DEFAULT_PERSONALITY_NPY}}"
SAMPLE_BINARY_CSV="${SAMPLE_BINARY_CSV:-}"
SAMPLE_TERNARY_CSV="${SAMPLE_TERNARY_CSV:-}"
ID_COLUMN="${ID_COLUMN:-auto}"

if [ -z "${BINARY_CKPT:-}" ]; then
  echo "ERROR: please set BINARY_CKPT to your binary best_model_*.pth" >&2
  exit 1
fi
if [ -z "${TERNARY_CKPT:-}" ]; then
  echo "ERROR: please set TERNARY_CKPT to your ternary best_model_*.pth" >&2
  exit 1
fi
if [ ! -f "${TEST_SCRIPT}" ]; then
  echo "ERROR: test script not found: ${PROJECT_ROOT}/${TEST_SCRIPT}" >&2
  exit 1
fi

common_args=(
  --config "${CONFIG}"
  --track "${TRACK}"
  --subtrack "${SUBTRACK}"
  --data_root "${DATA_ROOT}"
  --split_csv "${SPLIT_CSV}"
  --personality_npy "${PERSONALITY_NPY}"
  --output_dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --strict_load "${STRICT_LOAD}"
  --phq9_transform "${PHQ9_TRANSFORM}"
  --missing_regression_policy "${MISSING_REGRESSION_POLICY}"
  --id_column "${ID_COLUMN}"
)

echo "============================================================"
echo "Run binary blind inference"
echo "============================================================"
binary_args=("${common_args[@]}" --task binary --checkpoint "${BINARY_CKPT}" --make_zip 0)
if [ -n "${SAMPLE_BINARY_CSV}" ]; then
  binary_args+=(--sample_csv "${SAMPLE_BINARY_CSV}")
fi
if [ "${SAVE_USED_IDS}" = "1" ]; then
  binary_args+=(--save_used_ids_csv "${OUTPUT_DIR}/binary_used_ids.csv")
fi
"${PYTHON_BIN}" -u "${TEST_SCRIPT}" "${binary_args[@]}"

echo "============================================================"
echo "Run ternary blind inference and package submission.zip"
echo "============================================================"
ternary_args=("${common_args[@]}" --task ternary --checkpoint "${TERNARY_CKPT}" --make_zip 1)
if [ -n "${SAMPLE_TERNARY_CSV}" ]; then
  ternary_args+=(--sample_csv "${SAMPLE_TERNARY_CSV}")
fi
if [ "${SAVE_USED_IDS}" = "1" ]; then
  ternary_args+=(--save_used_ids_csv "${OUTPUT_DIR}/ternary_used_ids.csv")
fi
"${PYTHON_BIN}" -u "${TEST_SCRIPT}" "${ternary_args[@]}"

echo "============================================================"
echo "Finished. Submit: ${OUTPUT_DIR}/submission.zip"
echo "============================================================"
