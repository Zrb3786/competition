from pathlib import Path
import argparse
import re
import zipfile
import tempfile
import numpy as np
import pandas as pd


def read_status(status_path: Path):
    status = {}
    if not status_path.exists():
        return status

    for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[1] in {"DONE", "FAILED"}:
            _, state, exp, outdir = parts[:4]
            status[exp] = {"status": state, "outdir": outdir, "line": line}
    return status


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

    m["id"] = m["id"].astype(int)
    return m.sort_values("id").reset_index(drop=True)


def read_zip_pred(zip_path: Path):
    if not zip_path.exists():
        return None
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(td)
        return read_pred(Path(td))


def summarize_pred(m: pd.DataFrame):
    bad = m[
        ((m["binary_pred"] == 0) & (m["ternary_pred"] > 0)) |
        ((m["binary_pred"] == 1) & (m["ternary_pred"] == 0))
    ]
    return {
        "binary_counts": dict(m["binary_pred"].value_counts().sort_index()),
        "ternary_counts": dict(m["ternary_pred"].value_counts().sort_index()),
        "inconsistent_n": int(len(bad)),
        "inconsistent_ids": ",".join(map(str, bad["id"].astype(int).tolist())),
        "severe_ids": ",".join(map(str, m[m["ternary_pred"] == 2]["id"].astype(int).tolist())),
        "phq_min": float(m["phq9_pred"].min()) if m["phq9_pred"].notna().any() else np.nan,
        "phq_max": float(m["phq9_pred"].max()) if m["phq9_pred"].notna().any() else np.nan,
        "phq_mean": float(m["phq9_pred"].mean()) if m["phq9_pred"].notna().any() else np.nan,
    }


def read_cv(exp_dir: Path):
    p = exp_dir / "cv_summary.csv"
    if not p.exists():
        return {}, None

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
        df = df.copy()
        df["cls_score"] = (
            0.30 * df["binary_macro_f1"]
            + 0.30 * df["ternary_macro_f1"]
            + 0.20 * df["binary_kappa"]
            + 0.20 * df["ternary_kappa"]
        )
        out["cv_cls_score_mean"] = float(df["cls_score"].mean())
        out["cv_cls_score_std"] = float(df["cls_score"].std()) if len(df) > 1 else 0.0

    top = None
    if len(df):
        score_col = "cls_score" if "cls_score" in df.columns else ("best_score" if "best_score" in df.columns else None)
        if score_col:
            top = df.sort_values(score_col, ascending=False).head(1).iloc[0].to_dict()

    return out, top


def find_id_col(df):
    for c in ["ID", "id", "Id"]:
        if c in df.columns:
            return c
    return df.columns[0]


def num_after(pattern, text):
    m = re.search(pattern, str(text), flags=re.I)
    if not m:
        return np.nan
    try:
        return float(m.group(1))
    except Exception:
        return np.nan


def build_p_prior(desc_csv: Path):
    if desc_csv is None or not desc_csv.exists():
        return None

    df = pd.read_csv(desc_csv)
    idc = find_id_col(df)
    desc_col = "Descriptions" if "Descriptions" in df.columns else (
        "Description" if "Description" in df.columns else df.columns[-1]
    )

    rows = []
    for _, r in df.iterrows():
        text = str(r[desc_col])
        item = {"id": int(r[idc])}
        item["age"] = num_after(r"(\d+(?:\.\d+)?)\s*years?\s*old", text)
        item["E"] = num_after(r"Extraversion score is\s*([0-9.]+)", text)
        item["A"] = num_after(r"Agreeableness score is\s*([0-9.]+)", text)
        item["O"] = num_after(r"Openness score is\s*([0-9.]+)", text)
        item["N"] = num_after(r"Neuroticism score is\s*([0-9.]+)", text)
        item["C"] = num_after(r"Conscientiousness score is\s*([0-9.]+)", text)

        m = re.search(r"financial stress is categorized as ([^,\.]+)", text, flags=re.I)
        item["financial"] = m.group(1).strip().lower() if m else "unknown"

        item["family"] = num_after(r"live with\s*([0-9.]+)\s*family", text)

        m = re.search(r"disease classification.*?has ([^\.]+)", text, flags=re.I)
        item["disease"] = m.group(1).strip().lower() if m else "unknown"
        rows.append(item)

    out = pd.DataFrame(rows)

    for c in ["age", "E", "A", "O", "N", "C", "family"]:
        mu = out[c].mean()
        sd = out[c].std()
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        out[c + "_z"] = (out[c] - mu) / sd

    def financial_score(x):
        x = str(x).lower()
        if "no financial" in x:
            return 0.0
        if "mild" in x or "low" in x:
            return 0.5
        if "moderate" in x:
            return 1.0
        if "high" in x or "severe" in x:
            return 1.5
        return 0.0

    def disease_score(x):
        x = str(x).lower()
        if "no" in x and "disease" in x:
            return 0.0
        if "none" in x or "healthy" in x:
            return 0.0
        if "unknown" in x:
            return 0.0
        return 1.0

    out["financial_score"] = out["financial"].map(financial_score)
    out["disease_score"] = out["disease"].map(disease_score)
    out["family_low"] = (out["family"].fillna(out["family"].median()) <= 1).astype(float)

    out["p_risk_score"] = (
        + 1.00 * out["N_z"].fillna(0)
        - 0.45 * out["E_z"].fillna(0)
        - 0.45 * out["C_z"].fillna(0)
        + 0.25 * out["age_z"].fillna(0)
        + 0.45 * out["financial_score"].fillna(0)
        + 0.35 * out["disease_score"].fillna(0)
        + 0.25 * out["family_low"].fillna(0)
    )
    out["p_risk_rank"] = out["p_risk_score"].rank(pct=True)
    out["p_prior_binary"] = (out["p_risk_rank"] >= 0.50).astype(int)
    out["p_prior_ternary"] = pd.cut(
        out["p_risk_rank"],
        bins=[-1, 0.50, 0.85, 1.01],
        labels=[0, 1, 2],
    ).astype(int)

    return out[["id", "p_risk_score", "p_risk_rank", "p_prior_binary", "p_prior_ternary"]]


def add_p_prior_stats(m: pd.DataFrame, pprior: pd.DataFrame):
    if pprior is None:
        return {}

    mm = m.merge(pprior, on="id", how="left")
    if "p_prior_binary" not in mm.columns:
        return {}

    binary_conflict = (mm["binary_pred"] != mm["p_prior_binary"]).sum()
    ternary_gap = (mm["ternary_pred"] - mm["p_prior_ternary"]).abs()

    strong = (
        ((mm["binary_pred"] == 0) & (mm["p_risk_rank"] >= 0.75)) |
        ((mm["ternary_pred"] == 2) & (mm["p_risk_rank"] <= 0.50))
    )

    return {
        "p_prior_binary_counts": dict(mm["p_prior_binary"].value_counts().sort_index()),
        "p_prior_ternary_counts": dict(mm["p_prior_ternary"].value_counts().sort_index()),
        "p_binary_conflicts": int(binary_conflict),
        "p_ternary_gap_sum": int(ternary_gap.sum()),
        "p_strong_suspicious_n": int(strong.sum()),
        "p_strong_suspicious_ids": ",".join(map(str, mm.loc[strong, "id"].astype(int).tolist())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite/outputs/elder_v3_lite")
    ap.add_argument("--status", default=None)
    ap.add_argument("--ref_pred", default=None)
    ap.add_argument("--desc_csv", default=None)
    ap.add_argument("--out_prefix", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    status_path = Path(args.status) if args.status else root / "run_status_v11_5x3.txt"
    ref_pred = Path(args.ref_pred) if args.ref_pred else root / "v9_no_cross_raw_motion_5x1" / "predictions_normal"
    desc_csv = Path(args.desc_csv) if args.desc_csv else None
    out_prefix = Path(args.out_prefix) if args.out_prefix else root / "v11_progress"

    status = read_status(status_path)
    ref = read_pred(ref_pred)
    pprior = build_p_prior(desc_csv) if desc_csv else None

    exp_rows = []
    cand_rows = []
    details = []

    exp_dirs = sorted([p for p in root.glob("v11_*_5x3") if p.is_dir()])
    exp_names_seen = {p.name for p in exp_dirs}

    # 把 status 里已经 START/FAILED/DONE 但目录不存在的也列一下
    for exp in status:
        if exp.startswith("v11_") and exp not in exp_names_seen:
            exp_rows.append({
                "exp": exp,
                "status": status[exp]["status"],
                "exists": False,
                "cv_exists": False,
                "ckpt_n": 0,
                "prediction_dir_n": 0,
                "submission_n": 0,
            })

    for exp_dir in exp_dirs:
        cv, top = read_cv(exp_dir)
        ckpt_n = len(list((exp_dir / "checkpoints").glob("*.pt"))) if (exp_dir / "checkpoints").exists() else 0
        pred_dirs = sorted([p for p in exp_dir.glob("predictions_*") if p.is_dir()])
        sub_n = sum(1 for p in pred_dirs if (p / "submission.zip").exists())

        erow = {
            "exp": exp_dir.name,
            "status": status.get(exp_dir.name, {}).get("status", "RUNNING_OR_UNKNOWN"),
            "exists": True,
            "cv_exists": (exp_dir / "cv_summary.csv").exists(),
            "ckpt_n": ckpt_n,
            "prediction_dir_n": len(pred_dirs),
            "submission_n": sub_n,
        }
        erow.update(cv)

        if top is not None:
            for k in ["fold", "seed", "best_score", "cls_score", "binary_macro_f1", "binary_kappa", "ternary_macro_f1", "ternary_kappa", "phq_ccc", "checkpoint"]:
                if k in top:
                    erow["top_" + k] = top[k]
        exp_rows.append(erow)

        for pred_dir in pred_dirs:
            m = read_pred(pred_dir)
            if m is None:
                continue

            crow = {
                "exp": exp_dir.name,
                "pred": pred_dir.name,
                "status": erow["status"],
                "submission": str(pred_dir / "submission.zip"),
            }
            crow.update(cv)
            crow.update(summarize_pred(m))
            crow.update(add_p_prior_stats(m, pprior))

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
                bc = mm[mm["ref_binary"] != mm["cur_binary"]]
                tc = mm[mm["ref_ternary"] != mm["cur_ternary"]]
                crow["changed_binary_n_vs_v9best"] = int(len(bc))
                crow["changed_binary_ids_vs_v9best"] = ",".join(map(str, bc["id"].astype(int).tolist()))
                crow["changed_ternary_n_vs_v9best"] = int(len(tc))
                crow["changed_ternary_ids_vs_v9best"] = ",".join(map(str, tc["id"].astype(int).tolist()))

            cand_rows.append(crow)

            details.append("\n" + "=" * 110)
            details.append(f"{exp_dir.name}/{pred_dir.name}")
            details.append(f"status={erow['status']}")
            details.append(f"submission={pred_dir / 'submission.zip'}")
            details.append(f"binary={crow.get('binary_counts')} ternary={crow.get('ternary_counts')}")
            details.append(f"inconsistent={crow.get('inconsistent_n')} ids={crow.get('inconsistent_ids')}")
            details.append(f"severe={crow.get('severe_ids')}")
            details.append(f"changed_binary_vs_v9={crow.get('changed_binary_n_vs_v9best')} ids={crow.get('changed_binary_ids_vs_v9best')}")
            details.append(f"changed_ternary_vs_v9={crow.get('changed_ternary_n_vs_v9best')} ids={crow.get('changed_ternary_ids_vs_v9best')}")
            if "p_binary_conflicts" in crow:
                details.append(f"p_conflicts={crow.get('p_binary_conflicts')} strong_suspicious={crow.get('p_strong_suspicious_n')} ids={crow.get('p_strong_suspicious_ids')}")
            details.append("\nPredictions:")
            details.append(m.to_string(index=False))

    exp_df = pd.DataFrame(exp_rows)
    cand_df = pd.DataFrame(cand_rows)

    exp_csv = Path(str(out_prefix) + "_experiments.csv")
    cand_csv = Path(str(out_prefix) + "_candidates.csv")
    txt = Path(str(out_prefix) + "_details.txt")

    exp_df.to_csv(exp_csv, index=False)
    cand_df.to_csv(cand_csv, index=False)
    txt.write_text("\n".join(details), encoding="utf-8")

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 280)

    print("\n========== EXPERIMENT PROGRESS ==========")
    if len(exp_df):
        cols = [
            "exp", "status", "cv_exists", "ckpt_n", "prediction_dir_n", "submission_n",
            "cv_best_score_mean", "cv_cls_score_mean",
            "cv_binary_macro_f1_mean", "cv_binary_kappa_mean",
            "cv_ternary_macro_f1_mean", "cv_ternary_kappa_mean", "cv_phq_ccc_mean",
            "top_fold", "top_seed", "top_best_score", "top_cls_score"
        ]
        cols = [c for c in cols if c in exp_df.columns]
        print(exp_df[cols].sort_values(["status", "exp"]).to_string(index=False))
    else:
        print("No v11_*_5x3 experiment directories found.")

    print("\n========== FINISHED PREDICTION CANDIDATES ==========")
    if len(cand_df):
        cols = [
            "exp", "pred", "status",
            "cv_best_score_mean", "cv_cls_score_mean",
            "binary_counts", "ternary_counts",
            "inconsistent_n", "inconsistent_ids", "severe_ids",
            "changed_binary_n_vs_v9best", "changed_binary_ids_vs_v9best",
            "changed_ternary_n_vs_v9best", "changed_ternary_ids_vs_v9best",
            "p_binary_conflicts", "p_strong_suspicious_n", "p_strong_suspicious_ids",
            "submission"
        ]
        cols = [c for c in cols if c in cand_df.columns]
        print(cand_df[cols].sort_values(
            by=[c for c in ["status", "cv_cls_score_mean", "inconsistent_n"] if c in cand_df.columns],
            ascending=[True, False, True][:len([c for c in ["status", "cv_cls_score_mean", "inconsistent_n"] if c in cand_df.columns])]
        ).to_string(index=False))
    else:
        print("No finished prediction dirs found.")

    print("\n[OK] experiment csv:", exp_csv)
    print("[OK] candidate csv:", cand_csv)
    print("[OK] details txt:", txt)


if __name__ == "__main__":
    main()
