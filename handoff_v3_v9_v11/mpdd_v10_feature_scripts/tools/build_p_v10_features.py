#!/usr/bin/env python3
"""Build enhanced Elder personality / description features.

This keeps original interpretable fields and adds psychology-inspired risk features:
  - neuroticism high, conscientiousness low, extraversion low
  - financial stress, family support, disease flag
  - simple prior-risk combinations
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def read_csv_auto(path: str | Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def get_id_desc(df: pd.DataFrame) -> Tuple[str, str]:
    id_col = "ID" if "ID" in df.columns else ("id" if "id" in df.columns else df.columns[0])
    desc_col = None
    for c in df.columns:
        if c.lower() in {"descriptions", "description", "desc"}:
            desc_col = c
            break
    if desc_col is None:
        desc_col = df.columns[1]
    return id_col, desc_col


def find_float(text: str, patterns: List[str], default: float = 0.0) -> float:
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return default


def financial_stress_score(text: str) -> float:
    t = text.lower()
    if "no financial stress" in t or "no stress" in t:
        return 0.0
    if "mild" in t or "low" in t:
        return 1.0
    if "moderate" in t or "medium" in t:
        return 2.0
    if "severe" in t or "high" in t:
        return 3.0
    if "financial stress" in t:
        return 1.5
    return 0.0


def disease_score(text: str) -> Tuple[float, float, float]:
    t = text.lower()
    has = 1.0 if "disease" in t or "diseases" in t else 0.0
    none = 1.0 if "no disease" in t or "none" in t or "healthy" in t else 0.0
    other = 1.0 if "other disease" in t or "other diseases" in t or "other diseases" in t else 0.0
    return has, none, other


def parse_one(pid: int, desc: str) -> Dict[str, float]:
    text = str(desc)
    age = find_float(text, [r"(\d+(?:\.\d+)?)\s*[- ]year[- ]old", r"patient is\s*(\d+(?:\.\d+)?)\s*years? old"])
    E = find_float(text, [r"Extraversion score (?:is |of )?(\d+(?:\.\d+)?)", r"Extraversion[^0-9]*(\d+(?:\.\d+)?)"])
    A = find_float(text, [r"Agreeableness score (?:is |of )?(\d+(?:\.\d+)?)", r"Agreeableness[^0-9]*(\d+(?:\.\d+)?)"])
    O = find_float(text, [r"Openness score (?:is |of )?(\d+(?:\.\d+)?)", r"Openness[^0-9]*(\d+(?:\.\d+)?)"])
    N = find_float(text, [r"Neuroticism score (?:is |of )?(\d+(?:\.\d+)?)", r"Neuroticism[^0-9]*(\d+(?:\.\d+)?)"])
    C = find_float(text, [r"Conscientiousness score (?:is |of )?(\d+(?:\.\d+)?)", r"Conscientiousness[^0-9]*(\d+(?:\.\d+)?)"])
    fam = find_float(text, [r"live with\s*(\d+(?:\.\d+)?)\s*family", r"with\s*(\d+(?:\.\d+)?)\s*family members"], 0.0)
    fin = financial_stress_score(text)
    dis_has, dis_none, dis_other = disease_score(text)
    # Elder scores often around 0-12. Keep raw and scaled.
    denom = 12.0
    E_s, A_s, O_s, N_s, C_s = [x / denom for x in [E, A, O, N, C]]
    age_s = age / 100.0
    fam_s = min(fam, 6.0) / 6.0
    fin_s = fin / 3.0
    disease_s = dis_has * (1.0 - dis_none)
    risk = 1.2 * N_s - 0.6 * C_s - 0.4 * E_s + 0.5 * fin_s + 0.3 * disease_s - 0.25 * fam_s
    social_withdrawal = (1.0 - E_s) * (1.0 - fam_s)
    stress_sensitivity = N_s * (0.5 + fin_s)
    low_self_reg = (1.0 - C_s) * (0.5 + N_s)
    health_context = disease_s * (0.5 + age_s)
    return {
        "ID": pid,
        "age_raw": age, "age_scaled": age_s,
        "E_raw": E, "A_raw": A, "O_raw": O, "N_raw": N, "C_raw": C,
        "E_scaled": E_s, "A_scaled": A_s, "O_scaled": O_s, "N_scaled": N_s, "C_scaled": C_s,
        "financial_stress_raw": fin, "financial_stress_scaled": fin_s,
        "family_members_raw": fam, "family_support_scaled": fam_s,
        "disease_has": dis_has, "disease_none": dis_none, "disease_other": dis_other, "disease_risk": disease_s,
        "risk_p_prior": risk,
        "social_withdrawal_proxy": social_withdrawal,
        "stress_sensitivity": stress_sensitivity,
        "low_self_regulation": low_self_reg,
        "health_context": health_context,
        "N_minus_C": N_s - C_s,
        "N_minus_E": N_s - E_s,
        "low_E_low_C": (1.0 - E_s) * (1.0 - C_s),
    }


def build(desc_csv: str | Path, out_csv: str | Path) -> None:
    df = read_csv_auto(desc_csv)
    id_col, desc_col = get_id_desc(df)
    rows = []
    for _, r in df.iterrows():
        rows.append(parse_one(int(r[id_col]), str(r[desc_col])))
    out = pd.DataFrame(rows).sort_values("ID")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[OK] saved", out_csv, "shape=", out.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()
    build(args.desc_csv, args.output_csv)


if __name__ == "__main__":
    main()
