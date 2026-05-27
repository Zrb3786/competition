# MPDD Elder AVGP Track Summary

Current best submission:
- v9_no_cross_raw_motion_5x1/predictions_normal
- Score 0.5084
- Binary {0:12,1:11}, Ternary {0:12,1:8,2:3}, severe [18,45,91]

v3:
- Stable baseline, score around 0.5066.
- Uses P_struct/P_embed + official wav2vec/opensmile + raw video motion + gait.

v9_no_cross:
- Current best.
- Same core features as v3.
- Adds P-guided gating, shared/private decomposition, light A-V gated fusion, metric head, F1/Kappa loss.
- No explicit pair cross attention.

v11:
- Adds A_big_pca256, VBeh, GUnit, P_extra.
- Direct fusion did not improve blind score.
- Candidate1 score 0.4509, Candidate2 score 0.3987.
- Main issue: over-predicts positive/severe.

Recommended next direction:
- v12 = v9_no_cross base + residual/stacking calibration from A_big/VBeh/GUnit/P_extra.
- Do not directly replace v9 with v11.
