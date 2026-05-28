#!/usr/bin/env bash
set -euo pipefail

# Recommended: use a separate env for feature extraction so mpddavg training env remains stable.
ENV_NAME="${1:-mpddaudio}"
PY_VER="3.10"

if command -v conda >/dev/null 2>&1; then
  conda create -y -n "$ENV_NAME" python="$PY_VER"
  echo "Run: conda activate $ENV_NAME"
else
  echo "[WARN] conda not found; create/activate your own Python $PY_VER env first."
fi

cat <<'MSG'
After activating the env, install packages with one of the following.

GPU / existing CUDA-compatible torch preferred:
  pip install -U pip
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

If torch is already installed, skip the torch line and run:
  pip install -U numpy pandas tqdm scikit-learn soundfile librosa transformers accelerate sentencepiece
  pip install -U faster-whisper ctranslate2
  pip install -U funasr modelscope

Notes:
- WavLM is loaded through transformers, default model microsoft/wavlm-large.
- emotion2vec is loaded through FunASR/ModelScope, default model iic/emotion2vec_plus_large.
- Whisper transcript uses faster-whisper, default model large-v3 with language zh.
MSG
