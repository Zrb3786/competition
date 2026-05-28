#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$PKG_DIR/configs/elder_paths.env"

# ---------- HF / cache mirror ----------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/remote-home/yangmz/zhangruibo/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/remote-home/yangmz/zhangruibo/.cache/modelscope}"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$MODELSCOPE_CACHE" "$FEATURE_DIR"

echo "========== [0] Mirror / cache =========="
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "MODELSCOPE_CACHE=$MODELSCOPE_CACHE"

echo
echo "========== [1] Build merged raw audio/video roots and official test IDs =========="
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
echo "========== [2] Check merged raw audio =========="
python - <<PY
from pathlib import Path
import pandas as pd

for name, root, report in [
    ("train_audio", "$MERGED_TRAIN_AUDIO_ROOT", "$FEATURE_DIR/merged_train_audio_report.csv"),
    ("test_audio", "$MERGED_TEST_AUDIO_ROOT", "$FEATURE_DIR/merged_test_audio_report.csv"),
]:
    root = Path(root)
    print("\\n====", name, root, "====")
    files = sorted(root.glob("*/*.WAV")) + sorted(root.glob("*/*.wav"))
    print("wav files:", len(files))
    if Path(report).exists():
        df = pd.read_csv(report)
        print(df["source"].value_counts().to_string())
        bad = df[df["source"].eq("none")]
        if len(bad):
            print("[MISSING]")
            print(bad[["ID","pair","source","size","dst"]].to_string(index=False))
        else:
            print("[MISSING] none")
PY

echo
echo "========== [3] Extract A_big audio features from raw A_1~A_4 =========="
A_TOOL="$PKG_DIR/tools/extract_audio_v10.py"
if [ ! -f "$A_TOOL" ]; then
  echo "[ERROR] cannot find $A_TOOL"
  exit 2
fi

# 当前 extract_audio_v10.py 是单 split 接口，所以 train/test 分两次提取。
# 兼容变量名：ENABLE_E2V -> enable_emotion2vec。
ENABLE_EMOTION2VEC="${ENABLE_EMOTION2VEC:-${ENABLE_E2V:-1}}"

echo
echo "----- train audio big features -----"
python "$A_TOOL" \
  --audio_root "$MERGED_TRAIN_AUDIO_ROOT" \
  --id_csv "$TRAIN_SPLIT_CSV" \
  --output_npz "$AUDIO_BIG_TRAIN_NPZ" \
  --transcript_csv "$FEATURE_DIR/elder_whisper_transcripts_train.csv" \
  --device "${AUDIO_DEVICE:-cuda}" \
  --wavlm_model "${WAVLM_MODEL:-microsoft/wavlm-large}" \
  --enable_wavlm "${ENABLE_WAVLM:-1}" \
  --emotion2vec_model "${E2V_MODEL:-iic/emotion2vec_plus_large}" \
  --enable_emotion2vec "$ENABLE_EMOTION2VEC" \
  --whisper_model "${WHISPER_MODEL:-large-v3}" \
  --whisper_device "${WHISPER_DEVICE:-cuda}" \
  --whisper_compute_type "${WHISPER_COMPUTE_TYPE:-float16}" \
  --whisper_language "${WHISPER_LANGUAGE:-zh}" \
  --enable_whisper "${ENABLE_WHISPER:-1}"

echo
echo "----- test audio big features -----"
python "$A_TOOL" \
  --audio_root "$MERGED_TEST_AUDIO_ROOT" \
  --id_csv "$TEST_ID_CSV" \
  --output_npz "$AUDIO_BIG_TEST_NPZ" \
  --transcript_csv "$FEATURE_DIR/elder_whisper_transcripts_test.csv" \
  --device "${AUDIO_DEVICE:-cuda}" \
  --wavlm_model "${WAVLM_MODEL:-microsoft/wavlm-large}" \
  --enable_wavlm "${ENABLE_WAVLM:-1}" \
  --emotion2vec_model "${E2V_MODEL:-iic/emotion2vec_plus_large}" \
  --enable_emotion2vec "$ENABLE_EMOTION2VEC" \
  --whisper_model "${WHISPER_MODEL:-large-v3}" \
  --whisper_device "${WHISPER_DEVICE:-cuda}" \
  --whisper_compute_type "${WHISPER_COMPUTE_TYPE:-float16}" \
  --whisper_language "${WHISPER_LANGUAGE:-zh}" \
  --enable_whisper "${ENABLE_WHISPER:-1}"

echo
echo "========== [4] Verify A_big IDs / shapes =========="
python - <<PY
from pathlib import Path
import numpy as np
import pandas as pd

def ids_csv(p):
    df = pd.read_csv(p)
    c = "ID" if "ID" in df.columns else df.columns[0]
    return df[c].astype(int).tolist()

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
    print("ids n:", len(ids), "first:", ids[:10], "last:", ids[-10:])
    return ids

train_ids = ids_csv("$TRAIN_SPLIT_CSV")
test_ids = ids_csv("$TEST_ID_CSV")

a_train_ids = show_npz("$AUDIO_BIG_TRAIN_NPZ")
a_test_ids = show_npz("$AUDIO_BIG_TEST_NPZ")

assert train_ids == a_train_ids, "train audio IDs mismatch"
assert test_ids == a_test_ids, "test audio IDs mismatch"

print("\\n[OK] A_big train/test IDs are aligned.")
print("[OK] train npz:", "$AUDIO_BIG_TRAIN_NPZ")
print("[OK] test npz:", "$AUDIO_BIG_TEST_NPZ")
print("[OK] train transcript:", "$FEATURE_DIR/elder_whisper_transcripts_train.csv")
print("[OK] test transcript:", "$FEATURE_DIR/elder_whisper_transcripts_test.csv")
PY

echo
echo "========== DONE =========="
echo "AUDIO_BIG_TRAIN_NPZ=$AUDIO_BIG_TRAIN_NPZ"
echo "AUDIO_BIG_TEST_NPZ=$AUDIO_BIG_TEST_NPZ"
