#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/configs/elder_paths.env"

mkdir -p "$FEATURE_DIR" "$OUT_DIR/logs"

# The fixed v10 file can be used as project main file.
cp "$ROOT_DIR/src/mpdd_elder_v10_audio_pprior.py" "$CODE_DIR/mpdd_elder_v10_audio_pprior.py"
python -m py_compile "$CODE_DIR/mpdd_elder_v10_audio_pprior.py"

# Build enhanced P features.
python "$ROOT_DIR/tools/build_p_v10_features.py" \
  --desc_csv "$TRAIN_DESC_CSV" \
  --output_csv "$P_EXTRA_TRAIN_CSV"

if [ -f "$TEST_DESC_CSV" ]; then
  python "$ROOT_DIR/tools/build_p_v10_features.py" \
    --desc_csv "$TEST_DESC_CSV" \
    --output_csv "$P_EXTRA_TEST_CSV"
else
  cp "$P_EXTRA_TRAIN_CSV" "$P_EXTRA_TEST_CSV"
fi

# Build official test ID CSV if missing.
if [ ! -f "$TEST_ID_CSV" ]; then
  python - <<PY
from pathlib import Path
import pandas as pd
root=Path("$TEST_ROOT")/"IMU"
ids=[]
for p in root.iterdir():
    if p.is_dir():
        try: ids.append(int(p.name))
        except Exception: pass
ids=sorted(ids)
out=Path("$TEST_ID_CSV"); out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"ID":ids}).to_csv(out,index=False)
print("[OK]", out, len(ids), ids)
PY
fi

# Extract big acoustic features. This can be slow. Use large-v3-turbo if large-v3 is too slow.
# Set env flags to 0 if one component is too heavy:
#   ENABLE_WAVLM=0 ENABLE_E2V=0 ENABLE_WHISPER=1 bash scripts/01_extract_audio_p_features.sh
ENABLE_WAVLM="${ENABLE_WAVLM:-1}"
ENABLE_E2V="${ENABLE_E2V:-1}"
ENABLE_WHISPER="${ENABLE_WHISPER:-1}"
WAVLM_MODEL="${WAVLM_MODEL:-microsoft/wavlm-large}"
EMOTION_MODEL="${EMOTION_MODEL:-iic/emotion2vec_plus_large}"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-float16}"
DEVICE="${DEVICE:-cuda}"

python "$ROOT_DIR/tools/extract_audio_v10.py" \
  --audio_root "$TRAIN_ROOT/audio" \
  --id_csv "$TRAIN_SPLIT_CSV" \
  --output_npz "$AUDIO_BIG_TRAIN_NPZ" \
  --transcript_csv "$AUDIO_BIG_TRAIN_TXT" \
  --device "$DEVICE" \
  --wavlm_model "$WAVLM_MODEL" \
  --enable_wavlm "$ENABLE_WAVLM" \
  --emotion2vec_model "$EMOTION_MODEL" \
  --enable_emotion2vec "$ENABLE_E2V" \
  --whisper_model "$WHISPER_MODEL" \
  --whisper_device "$WHISPER_DEVICE" \
  --whisper_compute_type "$WHISPER_COMPUTE_TYPE" \
  --whisper_language zh \
  --enable_whisper "$ENABLE_WHISPER"

python "$ROOT_DIR/tools/extract_audio_v10.py" \
  --audio_root "$TEST_ROOT/audio" \
  --id_csv "$TEST_ID_CSV" \
  --output_npz "$AUDIO_BIG_TEST_NPZ" \
  --transcript_csv "$AUDIO_BIG_TEST_TXT" \
  --device "$DEVICE" \
  --wavlm_model "$WAVLM_MODEL" \
  --enable_wavlm "$ENABLE_WAVLM" \
  --emotion2vec_model "$EMOTION_MODEL" \
  --enable_emotion2vec "$ENABLE_E2V" \
  --whisper_model "$WHISPER_MODEL" \
  --whisper_device "$WHISPER_DEVICE" \
  --whisper_compute_type "$WHISPER_COMPUTE_TYPE" \
  --whisper_language zh \
  --enable_whisper "$ENABLE_WHISPER"

python - <<PY
from pathlib import Path
import numpy as np
for p in ["$P_EXTRA_TRAIN_CSV", "$P_EXTRA_TEST_CSV", "$AUDIO_BIG_TRAIN_NPZ", "$AUDIO_BIG_TEST_NPZ"]:
    path=Path(p)
    print("[FILE]", path, path.exists(), path.stat().st_size if path.exists() else 0)
    if path.suffix=='.npz' and path.exists():
        z=np.load(path, allow_pickle=True)
        print(" keys", z.files)
        print(" ids", z['ids'].shape, "pair", z['audio_big_pair'].shape, "mask_valid", z['pair_mask'].sum())
PY
