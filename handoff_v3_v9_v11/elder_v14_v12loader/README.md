# elder_v14_v12loader

V14 expert probe that **reuses the validated v12/v3/v11 feature loader**. It does not implement a new official-feature directory loader.

Experts:
- `audio_big`: A_big_pca256 only.
- `audio_official`: official audio as loaded by v12 loader.
- `audio`: official audio + A_big.
- `audio_controlled`: A_big pair branch + official audio static reference summary.
- `video`: raw motion + VBeh.
- `gait`: GUnit only by default.
- `p`: P_struct + P_extra by default.
- `av`: A_big × VBeh pair interaction + P static.

## Run

```bash
cd /remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite
unzip elder_v14_v12loader.zip
python -m py_compile elder_v14_v12loader/mpdd_elder_v14_v12loader.py

CUDA_VISIBLE_DEVICES=2 bash elder_v14_v12loader/scripts/05_smoke_all_modalities.sh elder_v14_v12loader/configs/elder_v14_v12loader_paths.env

CUDA_VISIBLE_DEVICES=2 LR=1e-4 WD=5e-4 EPOCHS=80 PATIENCE=15 BS=8 \
  bash elder_v14_v12loader/scripts/06_train_all_modalities_5x1.sh elder_v14_v12loader/configs/elder_v14_v12loader_paths.env
```

Summarize:
```bash
python elder_v14_v12loader/scripts/summarize_v14_v12loader.py --root outputs/elder_v14_v12loader
bash elder_v14_v12loader/scripts/07_check_submission_candidates.sh
```
