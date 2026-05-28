#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$PKG_DIR/configs/elder_paths.env"
export CUDA_VISIBLE_DEVICES=4
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/remote-home/yangmz/zhangruibo/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/remote-home/yangmz/zhangruibo/.cache/modelscope}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$FEATURE_DIR"

echo "========== [0] GPU / mirror =========="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
nvidia-smi || true

echo
echo "========== [1] Build merged raw A/V roots + official test IDs =========="
python "$PKG_DIR/tools/build_merged_raw_av_roots.py" \
  --train_split_csv "$TRAIN_SPLIT_CSV" \
  --test_root "$TEST_ROOT" \
  --test_id_csv "$TEST_ID_CSV" \
  --raw_train_video_hf "$RAW_TRAIN_VIDEO_HF" \
  --raw_train_video_old "$RAW_TRAIN_VIDEO_OLD" \
  --raw_test_video_hf "$RAW_TEST_VIDEO_HF" \
  --raw_test_video_old "$RAW_TEST_VIDEO_OLD" \
  --raw_train_audio_hf "$RAW_TRAIN_AUDIO_HF" \
  --raw_train_audio_old "$RAW_TRAIN_AUDIO_OLD" \
  --raw_test_audio_hf "$RAW_TEST_AUDIO_HF" \
  --raw_test_audio_old "$RAW_TEST_AUDIO_OLD" \
  --merged_train_video_root "$MERGED_TRAIN_VIDEO_ROOT" \
  --merged_test_video_root "$MERGED_TEST_VIDEO_ROOT" \
  --merged_train_audio_root "$MERGED_TRAIN_AUDIO_ROOT" \
  --merged_test_audio_root "$MERGED_TEST_AUDIO_ROOT" \
  --report_dir "$FEATURE_DIR"

echo
echo "========== [2] Clean old bad A_big =========="
rm -f "$AUDIO_BIG_TRAIN_NPZ" \
      "$AUDIO_BIG_TEST_NPZ" \
      "$FEATURE_DIR/elder_whisper_transcripts_train.csv" \
      "$FEATURE_DIR/elder_whisper_transcripts_test.csv"

A_TOOL="$PKG_DIR/tools/extract_audio_v10.py"
if [ ! -f "$A_TOOL" ]; then
  echo "[ERROR] cannot find $A_TOOL"
  exit 2
fi

ENABLE_EMOTION2VEC="${ENABLE_EMOTION2VEC:-${ENABLE_E2V:-1}}"

echo
echo "========== [3] Extract train A_big =========="
python "$A_TOOL" \
  --audio_root "$MERGED_TRAIN_AUDIO_ROOT" \
  --id_csv "$TRAIN_SPLIT_CSV" \
  --output_npz "$AUDIO_BIG_TRAIN_NPZ" \
  --transcript_csv "$FEATURE_DIR/elder_whisper_transcripts_train.csv" \
  --device "${AUDIO_DEVICE:-cuda}" \
  --wavlm_model "${WAVLM_MODEL:-microsoft/wavlm-large}" \
  --enable_wavlm "${ENABLE_WAVLM:-1}" \
  --require_wavlm 1 \
  --emotion2vec_model "${E2V_MODEL:-iic/emotion2vec_plus_large}" \
  --enable_emotion2vec "$ENABLE_EMOTION2VEC" \
  --require_emotion2vec "$ENABLE_EMOTION2VEC" \
  --whisper_model "${WHISPER_MODEL:-large-v3}" \
  --whisper_device "${WHISPER_DEVICE:-cuda}" \
  --whisper_compute_type "${WHISPER_COMPUTE_TYPE:-float16}" \
  --whisper_language "${WHISPER_LANGUAGE:-zh}" \
  --enable_whisper "${ENABLE_WHISPER:-1}" \
  --require_whisper "${ENABLE_WHISPER:-1}"

echo
echo "========== [4] Extract test A_big =========="
python "$A_TOOL" \
  --audio_root "$MERGED_TEST_AUDIO_ROOT" \
  --id_csv "$TEST_ID_CSV" \
  --output_npz "$AUDIO_BIG_TEST_NPZ" \
  --transcript_csv "$FEATURE_DIR/elder_whisper_transcripts_test.csv" \
  --device "${AUDIO_DEVICE:-cuda}" \
  --wavlm_model "${WAVLM_MODEL:-microsoft/wavlm-large}" \
  --enable_wavlm "${ENABLE_WAVLM:-1}" \
  --require_wavlm 1 \
  --emotion2vec_model "${E2V_MODEL:-iic/emotion2vec_plus_large}" \
  --enable_emotion2vec "$ENABLE_EMOTION2VEC" \
  --require_emotion2vec "$ENABLE_EMOTION2VEC" \
  --whisper_model "${WHISPER_MODEL:-large-v3}" \
  --whisper_device "${WHISPER_DEVICE:-cuda}" \
  --whisper_compute_type "${WHISPER_COMPUTE_TYPE:-float16}" \
  --whisper_language "${WHISPER_LANGUAGE:-zh}" \
  --enable_whisper "${ENABLE_WHISPER:-1}" \
  --require_whisper "${ENABLE_WHISPER:-1}"

echo
echo "========== [5] Verify A_big =========="
python - <<PY
from pathlib import Path
import numpy as np
import pandas as pd

def id_col(df):
    for c in ["ID", "id", "Id"]:
        if c in df.columns:
            return c
    return df.columns[0]

def ids_csv(p):
    df = pd.read_csv(p)
    return df[id_col(df)].astype(int).tolist()

def show_npz(p):
    p = Path(p)
    print("\\n====", p, "====")
    print("exists:", p.exists(), "size:", p.stat().st_size if p.exists() else 0)
    z = np.load(p, allow_pickle=True)
    print("keys:", z.files)
    for k in z.files:
        a = z[k]
        if hasattr(a, "shape"):
            print(k, a.shape, a.dtype)
    id_key = "ids" if "ids" in z.files else ("ID" if "ID" in z.files else None)
    ids = z[id_key].astype(int).tolist()
    x = z["audio_big_pair"]
    m = z["pair_mask"]
    print("ids n:", len(ids), "first:", ids[:10], "last:", ids[-10:])
    print("feature dim:", x.shape[-1])
    print("valid pairs:", float(m.sum()), "/", m.shape[0] * m.shape[1])
    return ids, x.shape, float(m.sum())

train_ids = ids_csv("$TRAIN_SPLIT_CSV")
test_ids = ids_csv("$TEST_ID_CSV")

a_train_ids, train_shape, train_valid = show_npz("$AUDIO_BIG_TRAIN_NPZ")
a_test_ids, test_shape, test_valid = show_npz("$AUDIO_BIG_TEST_NPZ")

assert sorted(train_ids) == sorted(a_train_ids), "train audio ID set mismatch"
assert test_ids == a_test_ids, "test audio ID order mismatch"
assert train_shape[-1] == test_shape[-1], f"audio dim mismatch: train={train_shape[-1]} test={test_shape[-1]}"

print("\\n[OK] train ID set aligned.")
print("[OK] test ID order aligned.")
print("[OK] train/test dim aligned:", train_shape[-1])

if train_valid < train_shape[0] * train_shape[1]:
    print("[WARN] train has missing/failed audio pairs:", train_valid, "/", train_shape[0] * train_shape[1])
if test_valid < test_shape[0] * test_shape[1]:
    print("[WARN] test has missing/failed audio pairs:", test_valid, "/", test_shape[0] * test_shape[1])
PY

echo
echo "========== DONE =========="
echo "AUDIO_BIG_TRAIN_NPZ=$AUDIO_BIG_TRAIN_NPZ"
echo "AUDIO_BIG_TEST_NPZ=$AUDIO_BIG_TEST_NPZ"
