#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize MPDD-AVG Elder/Track1 DepFormer-v2 training result JSON files.

Outputs:
  - all_runs.csv
  - best_by_task.csv
  - paired_configs.csv
  - robust_by_config.csv
  - selected_blind_command.sh
  - summary.md

Selection rule:
  task_score = (MacroF1 + Kappa + CCC) / 3
  expected_final_score = (binary_task_score + ternary_task_score) / 2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

METRIC_KEYS = [
    "score", "selection_score", "f1", "acc", "kappa", "ccc",
    "rmse", "mae", "r2", "loss", "cls_loss", "reg_loss", "focal_loss",
]


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


def finite_or_empty(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    return x


def compute_task_score(metrics: dict[str, Any]) -> float:
    """Leaderboard-aware score for one classification task."""
    if "score" in metrics and metrics.get("score") not in (None, ""):
        return safe_float(metrics.get("score"))
    f1 = safe_float(metrics.get("f1"))
    kappa = safe_float(metrics.get("kappa"))
    ccc = safe_float(metrics.get("ccc"))
    if not any(math.isnan(v) for v in (f1, kappa, ccc)):
        return (f1 + kappa + ccc) / 3.0
    if "selection_score" in metrics:
        return safe_float(metrics.get("selection_score"))
    return float("nan")


def resolve_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root / p


def load_one_result(json_path: Path, project_root: Path) -> dict[str, Any] | None:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[WARN] failed to read {json_path}: {exc}")
        return None

    metrics = payload.get("best_val_metrics") or {}
    cfg = payload.get("config") or {}
    checkpoint_path = str(payload.get("checkpoint_path", ""))
    ckpt_abs = resolve_path(checkpoint_path, project_root) if checkpoint_path else Path("")

    row: dict[str, Any] = {
        "json_path": str(json_path),
        "experiment_name": payload.get("experiment_name", ""),
        "timestamp": payload.get("timestamp", ""),
        "track": payload.get("track", cfg.get("track", "")),
        "task": payload.get("task", cfg.get("task", "")),
        "subtrack": payload.get("subtrack", cfg.get("subtrack", "")),
        "model_type": payload.get("model_type", cfg.get("model_type", "")),
        "encoder_type": payload.get("encoder_type", cfg.get("encoder_type", "")),
        "audio_feature": payload.get("audio_feature", cfg.get("audio_feature", "")),
        "video_feature": payload.get("video_feature", cfg.get("video_feature", "")),
        "hidden_dim": safe_int(cfg.get("hidden_dim", "")),
        "num_heads": safe_int(cfg.get("num_heads", "")),
        "seed": safe_int(cfg.get("seed", payload.get("seed", ""))),
        "loss_type": cfg.get("loss_type", payload.get("loss_type", "")),
        "selection_mode": cfg.get("selection_mode", payload.get("selection_metric", "")),
        "label_smoothing": safe_float(cfg.get("label_smoothing", "")),
        "focal_lambda": safe_float(cfg.get("focal_lambda", payload.get("focal_lambda", ""))),
        "reg_lambda": safe_float(cfg.get("reg_lambda", payload.get("reg_lambda", ""))),
        "use_p_gate": str(cfg.get("use_p_gate", "")),
        "av_encode_pairwise": str(cfg.get("av_encode_pairwise", "")),
        "best_epoch": safe_int(payload.get("best_epoch", "")),
        "train_count": safe_int(payload.get("train_count", "")),
        "val_count": safe_int(payload.get("val_count", "")),
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": ckpt_abs.is_file() if checkpoint_path else False,
        "history_path": payload.get("history_path", ""),
    }

    for k in METRIC_KEYS:
        row[k] = safe_float(metrics.get(k, ""))

    row["task_score"] = compute_task_score(metrics)
    row["MacroF1"] = row["f1"]
    row["Kappa"] = row["kappa"]
    row["CCC"] = row["ccc"]
    row["RMSE"] = row["rmse"]
    row["MAE"] = row["mae"]
    return row


def sort_key_for_task(row: dict[str, Any]) -> tuple:
    return (
        safe_float(row.get("task_score"), -999.0),
        safe_float(row.get("f1"), -999.0),
        safe_float(row.get("kappa"), -999.0),
        safe_float(row.get("ccc"), -999.0),
        -safe_float(row.get("rmse"), 999.0),
        -safe_float(row.get("mae"), 999.0),
        -safe_float(row.get("loss"), 999.0),
    )


def config_key(row: dict[str, Any], include_seed: bool = True) -> tuple:
    key = (
        row.get("track", ""), row.get("subtrack", ""), row.get("model_type", ""),
        row.get("encoder_type", ""), row.get("audio_feature", ""), row.get("video_feature", ""),
        row.get("hidden_dim", ""), row.get("num_heads", ""), row.get("loss_type", ""),
        row.get("focal_lambda", ""), row.get("reg_lambda", ""), row.get("use_p_gate", ""),
        row.get("av_encode_pairwise", ""),
    )
    if include_seed:
        key = key + (row.get("seed", ""),)
    return key


def config_key_name(row: dict[str, Any], include_seed: bool = True) -> str:
    parts = [
        str(row.get("encoder_type", "")), str(row.get("audio_feature", "")),
        str(row.get("video_feature", "")), f"h{row.get('hidden_dim', '')}",
    ]
    if include_seed:
        parts.append(f"s{row.get('seed', '')}")
    parts.append(f"fl{row.get('focal_lambda', '')}")
    parts.append(f"rl{row.get('reg_lambda', '')}")
    return "_".join(parts)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: finite_or_empty(r.get(k, "")) for k in fieldnames})


def make_best_by_task(rows: list[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for task in ("binary", "ternary"):
        task_rows = [r for r in rows if r.get("task") == task]
        task_rows = sorted(task_rows, key=sort_key_for_task, reverse=True)
        ranked = []
        for i, r in enumerate(task_rows, start=1):
            rr = dict(r)
            rr["rank_in_task"] = i
            ranked.append(rr)
        out[task] = ranked[:top_k]
    return out


def make_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key_task: dict[tuple, dict[str, dict[str, Any]]] = {}
    for r in rows:
        if r.get("task") not in {"binary", "ternary"}:
            continue
        by_key_task.setdefault(config_key(r, include_seed=True), {})[str(r.get("task"))] = r

    pairs: list[dict[str, Any]] = []
    for d in by_key_task.values():
        b = d.get("binary")
        t = d.get("ternary")
        if b is None or t is None:
            continue
        b_score = safe_float(b.get("task_score"))
        t_score = safe_float(t.get("task_score"))
        pair = {
            "pair_config": config_key_name(b, include_seed=True),
            "expected_final_score": (b_score + t_score) / 2.0,
            "binary_score": b_score,
            "binary_f1": b.get("f1", ""),
            "binary_kappa": b.get("kappa", ""),
            "binary_ccc": b.get("ccc", ""),
            "binary_rmse": b.get("rmse", ""),
            "binary_mae": b.get("mae", ""),
            "binary_checkpoint_path": b.get("checkpoint_path", ""),
            "ternary_score": t_score,
            "ternary_f1": t.get("f1", ""),
            "ternary_kappa": t.get("kappa", ""),
            "ternary_ccc": t.get("ccc", ""),
            "ternary_rmse": t.get("rmse", ""),
            "ternary_mae": t.get("mae", ""),
            "ternary_checkpoint_path": t.get("checkpoint_path", ""),
            "encoder_type": b.get("encoder_type", ""),
            "audio_feature": b.get("audio_feature", ""),
            "video_feature": b.get("video_feature", ""),
            "hidden_dim": b.get("hidden_dim", ""),
            "num_heads": b.get("num_heads", ""),
            "seed": b.get("seed", ""),
            "focal_lambda": b.get("focal_lambda", ""),
            "reg_lambda": b.get("reg_lambda", ""),
        }
        pairs.append(pair)
    pairs.sort(key=lambda x: (
        safe_float(x.get("expected_final_score"), -999.0),
        safe_float(x.get("binary_score"), -999.0),
        safe_float(x.get("ternary_score"), -999.0),
        safe_float(x.get("binary_ccc"), -999.0) + safe_float(x.get("ternary_ccc"), -999.0),
    ), reverse=True)
    for i, p in enumerate(pairs, start=1):
        p["pair_rank"] = i
    return pairs


def make_robust_by_config(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("task") not in {"binary", "ternary"}:
            continue
        k = config_key(r, include_seed=False) + (r.get("task", ""),)
        buckets.setdefault(k, []).append(r)

    out: list[dict[str, Any]] = []
    for rs in buckets.values():
        scores = [safe_float(r.get("task_score")) for r in rs if not math.isnan(safe_float(r.get("task_score")))]
        f1s = [safe_float(r.get("f1")) for r in rs if not math.isnan(safe_float(r.get("f1")))]
        kappas = [safe_float(r.get("kappa")) for r in rs if not math.isnan(safe_float(r.get("kappa")))]
        cccs = [safe_float(r.get("ccc")) for r in rs if not math.isnan(safe_float(r.get("ccc")))]
        if not scores:
            continue
        r0 = rs[0]
        best = sorted(rs, key=sort_key_for_task, reverse=True)[0]
        out.append({
            "task": r0.get("task", ""),
            "config_no_seed": config_key_name(r0, include_seed=False),
            "n_seeds": len(rs),
            "mean_score": mean(scores),
            "std_score": pstdev(scores) if len(scores) > 1 else 0.0,
            "min_score": min(scores),
            "max_score": max(scores),
            "mean_f1": mean(f1s) if f1s else "",
            "mean_kappa": mean(kappas) if kappas else "",
            "mean_ccc": mean(cccs) if cccs else "",
            "best_seed": best.get("seed", ""),
            "best_score": best.get("task_score", ""),
            "best_checkpoint_path": best.get("checkpoint_path", ""),
            "encoder_type": r0.get("encoder_type", ""),
            "audio_feature": r0.get("audio_feature", ""),
            "video_feature": r0.get("video_feature", ""),
            "hidden_dim": r0.get("hidden_dim", ""),
            "focal_lambda": r0.get("focal_lambda", ""),
            "reg_lambda": r0.get("reg_lambda", ""),
        })
    out.sort(key=lambda x: (
        safe_float(x.get("mean_score"), -999.0),
        safe_float(x.get("min_score"), -999.0),
        -safe_float(x.get("std_score"), 999.0),
        safe_float(x.get("best_score"), -999.0),
    ), reverse=True)
    for i, r in enumerate(out, start=1):
        r["robust_rank"] = i
    return out


def markdown_table(rows: list[dict[str, Any]], cols: list[str], max_rows: int = 10) -> str:
    rows = rows[:max_rows]
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_blind_command(path: Path, best_binary: dict[str, Any] | None, best_ternary: dict[str, Any] | None, output_dir: str) -> None:
    if best_binary is None or best_ternary is None:
        content = "# Could not generate command: missing binary or ternary result.\n"
    else:
        content = f'''#!/usr/bin/env bash
set -euo pipefail

# Auto-generated by summarize_elder_v2_results.py
# Independent best checkpoints by validation task_score=(F1+Kappa+CCC)/3.

STRICT_LOAD=1 \\
TRACK=Track1 \\
SUBTRACK="A-V-G+P" \\
OUTPUT_DIR="{output_dir}" \\
BINARY_CKPT="{best_binary.get("checkpoint_path", "")}" \\
TERNARY_CKPT="{best_ternary.get("checkpoint_path", "")}" \\
bash scripts/run_blind_depformer_submission.sh
'''
    path.write_text(content, encoding="utf-8")
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_root", default="logs_elder_v2")
    parser.add_argument("--out_dir", default="elder_v2_summary")
    parser.add_argument("--track", default="Track1")
    parser.add_argument("--subtrack", default="A-V-G+P")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--require_ckpt_exists", type=int, default=0)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--blind_output_dir", default="blind_submission/Track1/A_V_G_P_elder_v2_selected")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    logs_root = Path(args.logs_root)
    if not logs_root.is_absolute():
        logs_root = project_root / logs_root
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(logs_root.rglob("train_result_*.json"))
    print(f"[INFO] Found {len(json_files)} train_result_*.json files under {logs_root}")

    rows = []
    for jp in json_files:
        row = load_one_result(jp, project_root)
        if row is None:
            continue
        if args.track and row.get("track") != args.track:
            continue
        if args.subtrack and row.get("subtrack") != args.subtrack:
            continue
        if row.get("task") not in {"binary", "ternary"}:
            continue
        if args.require_ckpt_exists and not row.get("checkpoint_exists", False):
            continue
        rows.append(row)

    print(f"[INFO] Kept {len(rows)} binary/ternary result rows after filters.")
    if not rows:
        print("[ERROR] No valid rows. Check --logs_root, --track, --subtrack, or --require_ckpt_exists.")
        return

    all_cols = [
        "task", "task_score", "selection_score", "score", "f1", "acc", "kappa", "ccc", "rmse", "mae", "r2",
        "experiment_name", "encoder_type", "audio_feature", "video_feature", "hidden_dim", "num_heads", "seed",
        "focal_lambda", "reg_lambda", "label_smoothing", "loss_type", "best_epoch",
        "checkpoint_exists", "checkpoint_path", "json_path", "history_path",
    ]
    rows_sorted = sorted(rows, key=lambda r: (r.get("task", ""),) + sort_key_for_task(r), reverse=True)
    write_csv(out_dir / "all_runs.csv", rows_sorted, all_cols)

    best_by_task = make_best_by_task(rows, top_k=args.top_k)
    best_task_rows = []
    for task in ("binary", "ternary"):
        best_task_rows.extend(best_by_task.get(task, []))
    write_csv(out_dir / "best_by_task.csv", best_task_rows, ["rank_in_task"] + all_cols)

    pairs = make_pairs(rows)
    write_csv(out_dir / "paired_configs.csv", pairs)
    robust = make_robust_by_config(rows)
    write_csv(out_dir / "robust_by_config.csv", robust)

    best_binary = best_by_task.get("binary", [None])[0] if best_by_task.get("binary") else None
    best_ternary = best_by_task.get("ternary", [None])[0] if best_by_task.get("ternary") else None
    best_pair = pairs[0] if pairs else None
    write_blind_command(out_dir / "selected_blind_command.sh", best_binary, best_ternary, args.blind_output_dir)

    md = []
    md.append("# Elder v2 result summary\n")
    md.append("## Selection rule\n")
    md.append("For one task: `task_score = (MacroF1 + Kappa + CCC) / 3`.\n")
    md.append("For matched binary+ternary configs: `expected_final_score = (binary_task_score + ternary_task_score) / 2`.\n")
    md.append("For CodaBench submission, binary and ternary checkpoints can be selected independently.\n")
    md.append("## Recommended independent checkpoints\n")
    if best_binary:
        md.append(f"**Best binary:** `{best_binary.get('checkpoint_path')}`\n")
        md.append(f"score={safe_float(best_binary.get('task_score')):.6f}, F1={safe_float(best_binary.get('f1')):.6f}, Kappa={safe_float(best_binary.get('kappa')):.6f}, CCC={safe_float(best_binary.get('ccc')):.6f}\n")
    if best_ternary:
        md.append(f"\n**Best ternary:** `{best_ternary.get('checkpoint_path')}`\n")
        md.append(f"score={safe_float(best_ternary.get('task_score')):.6f}, F1={safe_float(best_ternary.get('f1')):.6f}, Kappa={safe_float(best_ternary.get('kappa')):.6f}, CCC={safe_float(best_ternary.get('ccc')):.6f}\n")
    md.append("\n## Best matched binary+ternary config\n")
    if best_pair:
        md.append(f"**Best pair:** `{best_pair.get('pair_config')}`, expected_final_score={safe_float(best_pair.get('expected_final_score')):.6f}\n")
        md.append(f"- binary ckpt: `{best_pair.get('binary_checkpoint_path')}`\n")
        md.append(f"- ternary ckpt: `{best_pair.get('ternary_checkpoint_path')}`\n")
    md.append("\n## Top binary runs\n")
    md.append(markdown_table(best_by_task.get("binary", []), ["rank_in_task", "task_score", "f1", "kappa", "ccc", "rmse", "mae", "encoder_type", "audio_feature", "video_feature", "hidden_dim", "seed", "checkpoint_path"], max_rows=min(args.top_k, 10)))
    md.append("\n## Top ternary runs\n")
    md.append(markdown_table(best_by_task.get("ternary", []), ["rank_in_task", "task_score", "f1", "kappa", "ccc", "rmse", "mae", "encoder_type", "audio_feature", "video_feature", "hidden_dim", "seed", "checkpoint_path"], max_rows=min(args.top_k, 10)))
    md.append("\n## Top matched pairs\n")
    md.append(markdown_table(pairs, ["pair_rank", "expected_final_score", "binary_score", "ternary_score", "encoder_type", "audio_feature", "video_feature", "hidden_dim", "seed"], max_rows=min(args.top_k, 10)))
    md.append("\n## Most robust configs across seeds\n")
    md.append(markdown_table(robust, ["robust_rank", "task", "mean_score", "std_score", "min_score", "max_score", "n_seeds", "encoder_type", "audio_feature", "video_feature", "hidden_dim", "best_seed"], max_rows=min(args.top_k, 15)))
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    for fname in ["all_runs.csv", "best_by_task.csv", "paired_configs.csv", "robust_by_config.csv", "selected_blind_command.sh", "summary.md"]:
        print(f"[OK] Wrote: {out_dir / fname}")

    if best_binary:
        print("\n[SELECT] Best binary checkpoint:")
        print(best_binary.get("checkpoint_path"))
        print(f"score={safe_float(best_binary.get('task_score')):.6f}, f1={safe_float(best_binary.get('f1')):.6f}, kappa={safe_float(best_binary.get('kappa')):.6f}, ccc={safe_float(best_binary.get('ccc')):.6f}")
    if best_ternary:
        print("\n[SELECT] Best ternary checkpoint:")
        print(best_ternary.get("checkpoint_path"))
        print(f"score={safe_float(best_ternary.get('task_score')):.6f}, f1={safe_float(best_ternary.get('f1')):.6f}, kappa={safe_float(best_ternary.get('kappa')):.6f}, ccc={safe_float(best_ternary.get('ccc')):.6f}")
    if best_pair:
        print("\n[SELECT] Best matched pair config:")
        print(best_pair.get("pair_config"))
        print(f"expected_final_score={safe_float(best_pair.get('expected_final_score')):.6f}")


if __name__ == "__main__":
    main()
