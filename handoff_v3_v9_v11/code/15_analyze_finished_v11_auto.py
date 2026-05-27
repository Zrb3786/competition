from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite/outputs/elder_v3_lite")

REF_EXP = ROOT / "v9_no_cross_raw_motion_5x1" / "predictions_normal"

def read_pred(pred_dir: Path):
    b = pred_dir / "binary.csv"
    t = pred_dir / "ternary.csv"
    if not b.exists() or not t.exists():
        return None

    bd = pd.read_csv(b)
    td = pd.read_csv(t)

    if "id" not in bd.columns or "binary_pred" not in bd.columns:
        return None
    if "id" not in td.columns or "ternary_pred" not in td.columns:
        return None

    phq_col = "phq9_pred" if "phq9_pred" in bd.columns else None
    if phq_col:
        m = bd[["id", "binary_pred", phq_col]].merge(td[["id", "ternary_pred"]], on="id")
        m = m.rename(columns={phq_col: "phq9_pred"})
    else:
        m = bd[["id", "binary_pred"]].merge(td[["id", "ternary_pred"]], on="id")
        m["phq9_pred"] = np.nan

    return m.sort_values("id").reset_index(drop=True)

def read_cv(exp_dir: Path):
    p = exp_dir / "cv_summary.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {"cv_n": len(df)}

    for c in [
        "best_score", "binary_macro_f1", "binary_kappa",
        "ternary_macro_f1", "ternary_kappa", "phq_ccc",
        "phq_rmse", "phq_mae"
    ]:
        if c in df.columns:
            out[f"cv_{c}_mean"] = float(df[c].mean())
            out[f"cv_{c}_std"] = float(df[c].std()) if len(df) > 1 else 0.0

    if {"binary_macro_f1", "ternary_macro_f1", "binary_kappa", "ternary_kappa"}.issubset(df.columns):
        cls_score = (
            0.30 * df["binary_macro_f1"]
            + 0.30 * df["ternary_macro_f1"]
            + 0.20 * df["binary_kappa"]
            + 0.20 * df["ternary_kappa"]
        )
        out["cv_cls_score_mean"] = float(cls_score.mean())
        out["cv_cls_score_std"] = float(cls_score.std()) if len(df) > 1 else 0.0

    return out

def summarize_pred(m: pd.DataFrame):
    bad = m[
        ((m.binary_pred == 0) & (m.ternary_pred > 0)) |
        ((m.binary_pred == 1) & (m.ternary_pred == 0))
    ]
    return {
        "binary_counts": dict(m.binary_pred.value_counts().sort_index()),
        "ternary_counts": dict(m.ternary_pred.value_counts().sort_index()),
        "inconsistent_n": int(len(bad)),
        "inconsistent_ids": ",".join(map(str, bad.id.astype(int).tolist())),
        "severe_ids": ",".join(map(str, m[m.ternary_pred == 2].id.astype(int).tolist())),
        "phq_min": float(m.phq9_pred.min()) if m.phq9_pred.notna().any() else np.nan,
        "phq_max": float(m.phq9_pred.max()) if m.phq9_pred.notna().any() else np.nan,
        "phq_mean": float(m.phq9_pred.mean()) if m.phq9_pred.notna().any() else np.nan,
    }

ref = read_pred(REF_EXP)

rows = []
details = []

# 先加当前最好参考
if ref is not None:
    row = {
        "name": "v9_no_cross_best",
        "exp": "v9_no_cross_raw_motion_5x1",
        "pred": "predictions_normal",
        "submission": str(REF_EXP / "submission.zip"),
    }
    row.update(read_cv(ROOT / "v9_no_cross_raw_motion_5x1"))
    row.update(summarize_pred(ref))
    row["changed_binary_n_vs_v9best"] = 0
    row["changed_binary_ids_vs_v9best"] = ""
    row["changed_ternary_n_vs_v9best"] = 0
    row["changed_ternary_ids_vs_v9best"] = ""
    rows.append(row)

# 自动扫描所有 v11 输出
for exp_dir in sorted(ROOT.glob("v11_*_5x3")):
    if not exp_dir.is_dir():
        continue

    for pred_dir in sorted(exp_dir.glob("predictions_*")):
        m = read_pred(pred_dir)
        if m is None:
            continue

        row = {
            "name": f"{exp_dir.name}/{pred_dir.name}",
            "exp": exp_dir.name,
            "pred": pred_dir.name,
            "submission": str(pred_dir / "submission.zip"),
        }
        row.update(read_cv(exp_dir))
        row.update(summarize_pred(m))

        if ref is not None:
            mm = ref.rename(columns={
                "binary_pred": "ref_binary",
                "ternary_pred": "ref_ternary",
                "phq9_pred": "ref_phq",
            }).merge(
                m.rename(columns={
                    "binary_pred": "cur_binary",
                    "ternary_pred": "cur_ternary",
                    "phq9_pred": "cur_phq",
                }),
                on="id",
                how="outer",
            ).sort_values("id")

            bc = mm[mm.ref_binary != mm.cur_binary]
            tc = mm[mm.ref_ternary != mm.cur_ternary]
            row["changed_binary_n_vs_v9best"] = int(len(bc))
            row["changed_binary_ids_vs_v9best"] = ",".join(map(str, bc.id.astype(int).tolist()))
            row["changed_ternary_n_vs_v9best"] = int(len(tc))
            row["changed_ternary_ids_vs_v9best"] = ",".join(map(str, tc.id.astype(int).tolist()))

        details.append("\n" + "=" * 100)
        details.append(f"{row['name']}")
        details.append(f"submission: {row['submission']}")
        details.append(f"binary={row['binary_counts']} ternary={row['ternary_counts']}")
        details.append(f"inconsistent={row['inconsistent_n']} ids={row['inconsistent_ids']}")
        details.append(f"severe_ids={row['severe_ids']}")
        details.append(f"changed_binary_vs_v9best={row.get('changed_binary_n_vs_v9best')} ids={row.get('changed_binary_ids_vs_v9best')}")
        details.append(f"changed_ternary_vs_v9best={row.get('changed_ternary_n_vs_v9best')} ids={row.get('changed_ternary_ids_vs_v9best')}")
        details.append("\nPredictions:")
        details.append(m.to_string(index=False))

        rows.append(row)

df = pd.DataFrame(rows)
out_csv = ROOT / "v11_candidate_report_auto.csv"
out_txt = ROOT / "v11_candidate_details_auto.txt"
df.to_csv(out_csv, index=False)
out_txt.write_text("\n".join(details), encoding="utf-8")

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 260)

cols = [
    "name", "cv_n", "cv_best_score_mean", "cv_cls_score_mean",
    "cv_binary_macro_f1_mean", "cv_binary_kappa_mean",
    "cv_ternary_macro_f1_mean", "cv_ternary_kappa_mean", "cv_phq_ccc_mean",
    "binary_counts", "ternary_counts", "inconsistent_n", "inconsistent_ids",
    "severe_ids", "changed_binary_n_vs_v9best", "changed_binary_ids_vs_v9best",
    "changed_ternary_n_vs_v9best", "changed_ternary_ids_vs_v9best",
    "submission",
]
cols = [c for c in cols if c in df.columns]
print(df[cols].to_string(index=False))
print("\n[OK] CSV:", out_csv)
print("[OK] details:", out_txt)
