# MPDD Elder v10 feature extraction + B/C experiments

This package adds two directions on top of the current v3/v9 line:

- **B: `v10_p_prior`** = P-prior residual head + original official audio + gait + raw motion.
- **C: `v10_audio_p_prior`** = B + big acoustic features from WavLM / emotion2vec / Whisper transcript stats.

It keeps the existing raw motion NPZ and official features; it only adds optional acoustic/P features.

## 0. Put package on server

Copy this folder to:

```bash
/remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite/mpdd_v10_feature_scripts
```

Then edit:

```bash
vim configs/elder_paths.env
```

## 1. Create feature extraction environment

```bash
bash scripts/00_setup_audio_env.sh mpddaudio
conda activate mpddaudio
# install packages shown by the script
```

If models are too slow, try:

```bash
export WHISPER_MODEL=large-v3-turbo
```

If emotion2vec install/download is unstable:

```bash
export ENABLE_E2V=0
```

## 2. Extract P and big acoustic features

```bash
conda activate mpddaudio
bash scripts/01_extract_audio_p_features.sh 2>&1 | tee extract_audio_p.log
```

Outputs:

```text
features/elder_p_v10_train.csv
features/elder_p_v10_test.csv
features/elder_audio_big_v10_train.npz
features/elder_audio_big_v10_test.npz
features/elder_whisper_transcripts_train.csv
features/elder_whisper_transcripts_test.csv
```

## 3. Smoke B/C in training env

```bash
conda activate mpddavg
bash scripts/02_smoke_v10_BC.sh
```

## 4. Train B and C

```bash
conda activate mpddavg
bash scripts/03_train_B_pprior_motion.sh 2>&1 | tee v10_B.log
bash scripts/04_train_C_audio_pprior_motion.sh 2>&1 | tee v10_C.log
```

## 5. Analyze predictions before submitting

```bash
bash scripts/05_analyze_BC.sh
```

