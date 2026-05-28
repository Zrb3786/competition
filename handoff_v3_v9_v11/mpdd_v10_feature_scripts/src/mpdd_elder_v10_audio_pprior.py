#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPDD-AVG Elder DepFormerAVP-v3-lite

Subcommands:
  parse-desc       Parse Elder descriptions.csv to structured tabular features.
  extract-video    Extract privacy-friendly raw-video motion features.
  train            Train 5-fold/seed ensemble on Elder train split.
  predict          Predict blind test and package CodaBench submission.zip.
  dummy            Make valid dummy binary/ternary CSVs and submission.zip from test IDs.

The code is intentionally defensive about paths and feature-file conventions.
It follows the official baseline layout when available:
  Elder/Audio/{wav2vec|opensmile}/{ID}/A_1.npy...
  Elder/Video/{openface|resnet|densenet}/{ID}/V_1.npy...
  Elder/IMU-ELDER/{ID}.npy or Elder/IMU/{ID}.npy
  descriptions_embeddings_with_ids.npy
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script needs PyTorch. Please install torch first.") from exc

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

EPS = 1e-8
PAIR_COUNT = 4
TARGET_T = 128
GAIT_KEEP_DIM = 9

AUDIO_ALIASES = {
    "wav2vec": ("wav2vec", "wav2vec2", "wav2vec2-FRA"),
    "wav2vec2": ("wav2vec2", "wav2vec", "wav2vec2-FRA"),
    "opensmile": ("opensmile",),
    "mfcc": ("mfcc", "mfcc64"),
    "mfcc64": ("mfcc64", "mfcc"),
}
VIDEO_ALIASES = {
    "openface": ("openface",),
    "resnet": ("resnet",),
    "densenet": ("densenet",),
}

# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_auto(path: str | Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Failed to read CSV {path}. Last error: {last}")


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def to_int_id(x: Any) -> int:
    return int(float(str(x).strip()))


def normalize_per_sample(arr: np.ndarray, clip: float = 6.0) -> np.ndarray:
    """Official-like per-sample normalization for time-series features."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 1:
        arr = arr[None, :]
    mu = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return np.clip((arr - mu) / std, -clip, clip).astype(np.float32)


def resize_np_time(arr: np.ndarray, target_t: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[0] == target_t:
        return arr.astype(np.float32)
    if arr.shape[0] <= 1:
        return np.repeat(arr, target_t, axis=0).astype(np.float32)
    x_old = np.linspace(0, 1, arr.shape[0], dtype=np.float32)
    x_new = np.linspace(0, 1, target_t, dtype=np.float32)
    out = np.stack([np.interp(x_new, x_old, arr[:, d]) for d in range(arr.shape[1])], axis=1)
    return out.astype(np.float32)


def safe_load_npy(path: Path) -> Optional[np.ndarray]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        arr = np.load(str(path), allow_pickle=True)
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return None


def parse_feature_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s or s.lower() in {"none", "null", "-"}:
        return []
    return [x.strip() for x in s.split(",") if x.strip() and x.strip().lower() not in {"none", "null", "-"}]

# -----------------------------------------------------------------------------
# Official feature path resolvers, compatible with hacilab baseline conventions
# -----------------------------------------------------------------------------

def resolve_modality_base(data_root: Path, modality_names: Sequence[str]) -> Path:
    for name in modality_names:
        p = data_root / name
        if p.exists():
            return p
    return data_root / modality_names[0]


def resolve_audio_feature_root(data_root: Path, feature: str) -> Path:
    base = resolve_modality_base(data_root, ("Audio", "audio"))
    for alias in AUDIO_ALIASES.get(feature, (feature,)):
        p = base / alias
        if p.exists():
            return p
    return base / AUDIO_ALIASES.get(feature, (feature,))[0]


def resolve_video_feature_root(data_root: Path, feature: str) -> Path:
    base = resolve_modality_base(data_root, ("Video", "video"))
    for alias in VIDEO_ALIASES.get(feature, (feature,)):
        p = base / alias
        if p.exists():
            return p
    return base / VIDEO_ALIASES.get(feature, (feature,))[0]


def resolve_gait_root(data_root: Path) -> Path:
    for name in ("IMU-ELDER", "IMU-Elder", "IMU", "Gait", "gait"):
        p = data_root / name
        if p.exists():
            # official may contain train/test split folder, but Elder trainval usually not
            if (p / "train").exists():
                return p / "train"
            return p
    return data_root / "IMU"


def resolve_gait_file(gait_root: Path, pid: int) -> Optional[Path]:
    candidates = [
        gait_root / f"{pid}.npy",
        gait_root / str(pid) / f"{pid}.npy",
        gait_root / str(pid) / "gait.npy",
        gait_root / str(pid) / "imu.npy",
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback recursive, cheap for small N if direct layout differs
    for c in gait_root.rglob("*.npy") if gait_root.exists() else []:
        nums = re.findall(r"\d+", c.stem)
        if nums and int(nums[-1]) == pid:
            return c
    return None


def discover_pair_npy(folder: Path, prefix: str, pair_count: int = PAIR_COUNT) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    if not folder.exists():
        return out
    # Elder official: A_1.npy, V_1.npy, ...
    for i in range(1, pair_count + 1):
        for name in (f"{prefix}_{i}.npy", f"{prefix}{i}.npy", f"event_{i}.npy"):
            p = folder / name
            if p.exists():
                out[i] = p
                break
    if out:
        return out
    # recursive fallback inside ID folder
    pat = re.compile(rf"{re.escape(prefix)}[_-]?(\d+).*\.npy$", re.IGNORECASE)
    for p in sorted(folder.rglob("*.npy")):
        m = pat.search(p.name)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= pair_count:
                out[idx] = p
    return out


def load_official_pair_features(
    data_root: Path,
    pid: int,
    features: Sequence[str],
    modality: str,
    target_t: int,
    pair_count: int = PAIR_COUNT,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return [P,T,C_total], pair_mask, C_total. Missing features become zeros."""
    roots = []
    prefix = "A" if modality == "audio" else "V"
    for feat in features:
        roots.append(resolve_audio_feature_root(data_root, feat) if modality == "audio" else resolve_video_feature_root(data_root, feat))

    per_feature_pairs: List[Dict[int, Path]] = []
    per_feature_dim: List[int] = []
    for root in roots:
        folder = root / str(pid)
        pair_map = discover_pair_npy(folder, prefix=prefix, pair_count=pair_count)
        per_feature_pairs.append(pair_map)
        # infer dim from first existing pair
        dim = 0
        for p in pair_map.values():
            arr = safe_load_npy(p)
            if arr is not None and arr.size > 0:
                if arr.ndim == 1:
                    dim = int(arr.shape[0])
                else:
                    dim = int(arr.shape[-1])
                break
        per_feature_dim.append(dim)

    total_dim = int(sum(per_feature_dim))
    if total_dim <= 0:
        return np.zeros((pair_count, target_t, 0), dtype=np.float32), np.zeros(pair_count, dtype=np.float32), 0

    out_pairs: List[np.ndarray] = []
    mask: List[float] = []
    for i in range(1, pair_count + 1):
        chunks: List[np.ndarray] = []
        valid_any = False
        for pair_map, dim in zip(per_feature_pairs, per_feature_dim):
            if dim <= 0:
                continue
            p = pair_map.get(i)
            arr = safe_load_npy(p) if p else None
            if arr is None or arr.size == 0:
                chunks.append(np.zeros((target_t, dim), dtype=np.float32))
                continue
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape[-1] != dim:
                arr = arr.reshape(arr.shape[0], -1)[:, :dim]
            arr = normalize_per_sample(arr)
            arr = resize_np_time(arr, target_t)
            chunks.append(arr.astype(np.float32))
            valid_any = True
        out_pairs.append(np.concatenate(chunks, axis=-1) if chunks else np.zeros((target_t, 0), dtype=np.float32))
        mask.append(1.0 if valid_any else 0.0)
    return np.stack(out_pairs, axis=0).astype(np.float32), np.asarray(mask, dtype=np.float32), total_dim


def load_gait_seq(data_root: Path, pid: int, target_t: int) -> np.ndarray:
    root = resolve_gait_root(data_root)
    f = resolve_gait_file(root, pid)
    if f is None:
        return np.zeros((target_t, 0), dtype=np.float32)
    arr = safe_load_npy(f)
    if arr is None or arr.size == 0:
        return np.zeros((target_t, 0), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = arr[..., :GAIT_KEEP_DIM]
    arr = normalize_per_sample(arr)
    return resize_np_time(arr, target_t).astype(np.float32)


def load_personality_embedding(path: Optional[str | Path]) -> Dict[int, np.ndarray]:
    if path is None or str(path).strip() == "":
        return {}
    p = Path(path)
    if p.is_dir():
        p = p / "descriptions_embeddings_with_ids.npy"
    if not p.exists():
        print(f"[WARN] personality embedding not found: {p}")
        return {}
    data = np.load(str(p), allow_pickle=True)
    out: Dict[int, np.ndarray] = {}
    for item in data:
        try:
            if isinstance(item, dict):
                pid = int(item.get("id", item.get("ID")))
                emb = np.asarray(item.get("embedding", item.get("emb")), dtype=np.float32)
            else:
                # numpy void / object with fields
                pid = int(item["id"] if "id" in item.dtype.names else item["ID"])
                emb = np.asarray(item["embedding"], dtype=np.float32)
            out[pid] = emb.reshape(-1).astype(np.float32)
        except Exception:
            continue
    return out

# -----------------------------------------------------------------------------
# Elder description parsing
# -----------------------------------------------------------------------------

BIG5 = ["extraversion", "agreeableness", "openness", "neuroticism", "conscientiousness"]


def find_float_near(text: str, key: str) -> float:
    # Handles: "Extraversion score is 12.0", "Extraversion score of 4"
    patterns = [
        rf"{key}\s+score\s+(?:is|of)\s*([0-9]+(?:\.[0-9]+)?)",
        rf"{key}[^0-9]{{0,40}}([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return float(m.group(1))
    return np.nan


def parse_one_description(desc: str) -> Dict[str, Any]:
    text = str(desc or "")
    row: Dict[str, Any] = {}
    m_age = re.search(r"(?:is\s+|This\s+)(\d+)\s*(?:years?\s+old|year-old)", text, flags=re.I)
    if not m_age:
        m_age = re.search(r"(\d+)\s*(?:years?\s+old|year-old)", text, flags=re.I)
    row["age"] = float(m_age.group(1)) if m_age else np.nan
    for k in BIG5:
        row[k] = find_float_near(text, k)

    m_fin = re.search(r"financial stress\s+is\s+categorized\s+as\s+([^,\.]+)", text, flags=re.I)
    row["financial_stress"] = (m_fin.group(1).strip().lower() if m_fin else "unknown")

    m_family = re.search(r"live\s+with\s+([0-9]+)\s+family", text, flags=re.I)
    row["family_members"] = float(m_family.group(1)) if m_family else np.nan

    m_disease = re.search(r"patient\s+has\s+([^\.]+)", text, flags=re.I)
    row["disease_classification"] = (m_disease.group(1).strip().lower() if m_disease else "unknown")
    row["desc_len"] = float(len(text.split()))
    return row


def parse_desc_command(args: argparse.Namespace) -> None:
    inputs: List[Tuple[str, Path]] = []
    if args.train_desc:
        inputs.append(("train", Path(args.train_desc)))
    if args.test_desc:
        inputs.append(("test", Path(args.test_desc)))
    if not inputs:
        raise ValueError("Please provide --train_desc and/or --test_desc")

    raw_frames = []
    for split, p in inputs:
        df = read_csv_auto(p)
        id_col = "ID" if "ID" in df.columns else ("id" if "id" in df.columns else df.columns[0])
        desc_col = "Descriptions" if "Descriptions" in df.columns else ("descriptions" if "descriptions" in df.columns else df.columns[-1])
        records = []
        for _, r in df.iterrows():
            rec = {"id": to_int_id(r[id_col]), "split": split}
            rec.update(parse_one_description(str(r[desc_col])))
            records.append(rec)
        raw_frames.append(pd.DataFrame(records))

    all_df = pd.concat(raw_frames, ignore_index=True)
    num_cols = ["age", *BIG5, "family_members", "desc_len"]
    for c in num_cols:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")
        all_df[c + "_missing"] = all_df[c].isna().astype(float)
        med = all_df.loc[all_df["split"] == "train", c].median() if (all_df["split"] == "train").any() else all_df[c].median()
        if pd.isna(med):
            med = 0.0
        all_df[c] = all_df[c].fillna(float(med))

    cat_cols = ["financial_stress", "disease_classification"]
    all_df[cat_cols] = all_df[cat_cols].fillna("unknown").astype(str).apply(lambda s: s.str.strip().str.lower())
    onehot = pd.get_dummies(all_df[cat_cols], prefix=cat_cols, dtype=float)
    feat_df = pd.concat([all_df[["id", "split"] + num_cols + [c + "_missing" for c in num_cols]], onehot], axis=1)

    out_dir = ensure_dir(args.output_dir)
    for split in sorted(feat_df["split"].unique()):
        split_df = feat_df[feat_df["split"] == split].drop(columns=["split"]).sort_values("id")
        out_path = out_dir / f"elder_descriptions_struct_{split}.csv"
        split_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[OK] saved {out_path} shape={split_df.shape}")
    write_json({"feature_columns": [c for c in feat_df.columns if c not in {"id", "split"}]}, out_dir / "elder_descriptions_struct_meta.json")

# -----------------------------------------------------------------------------
# Raw video motion extraction
# -----------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def discover_raw_video_pairs(video_root: Path, pid: int, pair_count: int = PAIR_COUNT) -> Dict[int, Path]:
    """Elder raw expected: video/<ID>/V_1.mp4 ...; fallback recursive."""
    folder = video_root / str(pid)
    out: Dict[int, Path] = {}
    if folder.exists():
        for i in range(1, pair_count + 1):
            for ext in VIDEO_EXTS:
                p = folder / f"V_{i}{ext}"
                if p.exists():
                    out[i] = p
                    break
            if i not in out:
                # case-insensitive fallback within the folder
                for p in sorted(folder.iterdir()):
                    if p.suffix.lower() in VIDEO_EXTS and re.search(rf"V[_-]?{i}\b", p.stem, re.I):
                        out[i] = p
                        break
    if out:
        return out

    # More general recursive fallback, but keep the exact ID folder check first to avoid matching ID=2 inside ID=20.
    if video_root.exists():
        for p in sorted(video_root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
                continue
            parts = set(p.parts)
            if str(pid) not in parts:
                continue
            m = re.search(r"V[_-]?(\d+)", p.stem, re.I)
            if m:
                idx = int(m.group(1))
                if 1 <= idx <= pair_count:
                    out[idx] = p
    return out


def uniform_frame_indices(frame_count: int, target_t: int) -> np.ndarray:
    if frame_count <= 0:
        return np.arange(target_t)
    return np.linspace(0, max(frame_count - 1, 0), target_t).round().astype(int)


def read_uniform_gray_frames(video_path: Path, target_t: int, resize: int) -> Optional[np.ndarray]:
    if cv2 is None:
        raise RuntimeError("cv2 is not installed. Please `pip install opencv-python`.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    wanted = uniform_frame_indices(frame_count, target_t)
    frames: List[np.ndarray] = []

    if frame_count > 0:
        for idx in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                if frames:
                    frames.append(frames[-1].copy())
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (resize, resize), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            frames.append(gray)
    else:
        # streaming fallback
        all_frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (resize, resize), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            all_frames.append(gray)
        if not all_frames:
            cap.release()
            return None
        idxs = uniform_frame_indices(len(all_frames), target_t)
        frames = [all_frames[min(int(i), len(all_frames) - 1)] for i in idxs]
    cap.release()

    if not frames:
        return None
    while len(frames) < target_t:
        frames.append(frames[-1].copy())
    return np.stack(frames[:target_t], axis=0).astype(np.float32)


def motion_features_from_gray(frames: np.ndarray) -> np.ndarray:
    """Extract per-frame motion descriptors, no identity/facial recognition."""
    if cv2 is None:
        raise RuntimeError("cv2 is not installed. Please `pip install opencv-python`.")
    T, H, W = frames.shape
    feats: List[List[float]] = []
    prev = frames[0]
    yy1, yy2 = int(H * 0.2), int(H * 0.85)
    xx1, xx2 = int(W * 0.2), int(W * 0.8)
    for t in range(T):
        cur = frames[t]
        diff = np.abs(cur - prev)
        center_diff = diff[yy1:yy2, xx1:xx2]
        active = (diff > 0.06).astype(np.float32)
        lap_var = float(cv2.Laplacian((cur * 255).astype(np.uint8), cv2.CV_32F).var() / (255.0 * 255.0))

        if t == 0:
            flow_mag = np.zeros_like(cur, dtype=np.float32)
            flow_x_mean = 0.0
            flow_y_mean = 0.0
        else:
            small_prev = cv2.resize(prev, (64, 64), interpolation=cv2.INTER_AREA)
            small_cur = cv2.resize(cur, (64, 64), interpolation=cv2.INTER_AREA)
            flow = cv2.calcOpticalFlowFarneback(
                (small_prev * 255).astype(np.uint8),
                (small_cur * 255).astype(np.uint8),
                None,
                pyr_scale=0.5,
                levels=2,
                winsize=9,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            flow_mag = np.sqrt(flow_x ** 2 + flow_y ** 2).astype(np.float32)
            flow_x_mean = float(np.mean(flow_x))
            flow_y_mean = float(np.mean(flow_y))

        row = [
            float(cur.mean()),
            float(cur.std()),
            float(lap_var),
            float(diff.mean()),
            float(diff.std()),
            float(np.percentile(diff, 90)),
            float(diff.max()),
            float(center_diff.mean()) if center_diff.size else 0.0,
            float(active.mean()),
            float(flow_mag.mean()),
            float(flow_mag.std()),
            float(np.percentile(flow_mag, 90)),
            float(flow_mag.max()),
            flow_x_mean,
            flow_y_mean,
        ]
        feats.append(row)
        prev = cur
    return np.asarray(feats, dtype=np.float32)


def summarize_motion_seq(seq: np.ndarray) -> np.ndarray:
    if seq.size == 0:
        return np.zeros(1, dtype=np.float32)
    stats = [
        seq.mean(axis=0),
        seq.std(axis=0),
        seq.min(axis=0),
        seq.max(axis=0),
        np.percentile(seq, 25, axis=0),
        np.percentile(seq, 75, axis=0),
    ]
    # a few global activity statistics from diff/flow columns
    global_extra = np.asarray([
        float(seq[:, 3].mean()) if seq.shape[1] > 3 else 0.0,
        float(seq[:, 8].mean()) if seq.shape[1] > 8 else 0.0,
        float((seq[:, 8] > 0.05).mean()) if seq.shape[1] > 8 else 0.0,
        float(seq[:, 9].mean()) if seq.shape[1] > 9 else 0.0,
        float(seq[:, 12].max()) if seq.shape[1] > 12 else 0.0,
    ], dtype=np.float32)
    return np.concatenate([*(s.astype(np.float32) for s in stats), global_extra], axis=0).astype(np.float32)


def extract_video_command(args: argparse.Namespace) -> None:
    if cv2 is None:
        raise RuntimeError("cv2 is not installed. Please run: pip install opencv-python")
    video_root = Path(args.video_root)
    split_csv = Path(args.split_csv)
    df = read_csv_auto(split_csv)
    if "ID" not in df.columns and "id" not in df.columns:
        raise ValueError(f"{split_csv} needs ID or id column")
    id_col = "ID" if "ID" in df.columns else "id"
    ids = [to_int_id(x) for x in df[id_col].tolist()]
    ids = sorted(set(ids))

    feature_names = [
        "gray_mean", "gray_std", "lap_var",
        "diff_mean", "diff_std", "diff_p90", "diff_max", "center_diff_mean", "active_ratio",
        "flow_mag_mean", "flow_mag_std", "flow_mag_p90", "flow_mag_max", "flow_x_mean", "flow_y_mean",
    ]
    all_seq: List[np.ndarray] = []
    all_mask: List[np.ndarray] = []
    all_stat: List[np.ndarray] = []
    missing: List[int] = []

    for pid in tqdm(ids, desc=f"extract video motion ({args.split_name})"):
        pair_map = discover_raw_video_pairs(video_root, pid, pair_count=args.pair_count)
        pair_seqs: List[np.ndarray] = []
        pair_mask: List[float] = []
        pair_stats: List[np.ndarray] = []
        for i in range(1, args.pair_count + 1):
            p = pair_map.get(i)
            if p is None:
                seq = np.zeros((args.target_t, len(feature_names)), dtype=np.float32)
                pair_mask.append(0.0)
            else:
                frames = read_uniform_gray_frames(p, args.target_t, args.resize)
                if frames is None:
                    seq = np.zeros((args.target_t, len(feature_names)), dtype=np.float32)
                    pair_mask.append(0.0)
                else:
                    seq = motion_features_from_gray(frames)
                    pair_mask.append(1.0)
            pair_seqs.append(seq.astype(np.float32))
            pair_stats.append(summarize_motion_seq(seq))
        if sum(pair_mask) == 0:
            missing.append(pid)
        all_seq.append(np.stack(pair_seqs, axis=0).astype(np.float32))
        all_mask.append(np.asarray(pair_mask, dtype=np.float32))
        all_stat.append(np.concatenate(pair_stats + [np.asarray(pair_mask, dtype=np.float32)], axis=0).astype(np.float32))

    out_path = Path(args.output_npz)
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        ids=np.asarray(ids, dtype=np.int64),
        motion_seq=np.stack(all_seq, axis=0).astype(np.float32),
        pair_mask=np.stack(all_mask, axis=0).astype(np.float32),
        motion_stat=np.stack(all_stat, axis=0).astype(np.float32),
        feature_names=np.asarray(feature_names, dtype=object),
        missing_ids=np.asarray(missing, dtype=np.int64),
    )
    print(f"[OK] saved {out_path}")
    print(f"     ids={len(ids)} seq={np.stack(all_seq).shape} stat={np.stack(all_stat).shape} missing_video_ids={len(missing)}")
    if missing:
        print(f"     first missing ids: {missing[:20]}")

# -----------------------------------------------------------------------------
# Data for training/inference
# -----------------------------------------------------------------------------

@dataclass
class FoldScalers:
    p_struct_mean: List[float]
    p_struct_scale: List[float]
    motion_stat_mean: List[float]
    motion_stat_scale: List[float]

    @staticmethod
    def fit(p_struct: np.ndarray, motion_stat: np.ndarray) -> "FoldScalers":
        ps = StandardScaler().fit(p_struct.astype(np.float32)) if p_struct.shape[1] else None
        ms = StandardScaler().fit(motion_stat.astype(np.float32)) if motion_stat.shape[1] else None
        return FoldScalers(
            p_struct_mean=(ps.mean_.astype(float).tolist() if ps is not None else []),
            p_struct_scale=(np.where(ps.scale_ < EPS, 1.0, ps.scale_).astype(float).tolist() if ps is not None else []),
            motion_stat_mean=(ms.mean_.astype(float).tolist() if ms is not None else []),
            motion_stat_scale=(np.where(ms.scale_ < EPS, 1.0, ms.scale_).astype(float).tolist() if ms is not None else []),
        )

    def transform_p(self, x: np.ndarray) -> np.ndarray:
        if not self.p_struct_mean:
            return x.astype(np.float32)
        return ((x - np.asarray(self.p_struct_mean, dtype=np.float32)) / np.asarray(self.p_struct_scale, dtype=np.float32)).astype(np.float32)

    def transform_motion_stat(self, x: np.ndarray) -> np.ndarray:
        if not self.motion_stat_mean:
            return x.astype(np.float32)
        return ((x - np.asarray(self.motion_stat_mean, dtype=np.float32)) / np.asarray(self.motion_stat_scale, dtype=np.float32)).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FoldScalers":
        return FoldScalers(**d)


def read_struct_features(path: Optional[str | Path]) -> Tuple[Dict[int, np.ndarray], List[str]]:
    if path is None or str(path).strip() == "":
        return {}, []
    p = Path(path)
    if not p.exists():
        print(f"[WARN] structured P not found: {p}")
        return {}, []
    df = read_csv_auto(p)
    id_col = "id" if "id" in df.columns else ("ID" if "ID" in df.columns else df.columns[0])
    feat_cols = [c for c in df.columns if c != id_col]
    out = {}
    for _, r in df.iterrows():
        out[to_int_id(r[id_col])] = r[feat_cols].astype(float).values.astype(np.float32)
    return out, feat_cols


def read_motion_npz(path: Optional[str | Path]) -> Dict[int, Dict[str, np.ndarray]]:
    if path is None or str(path).strip() == "":
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] motion npz not found: {p}")
        return {}
    data = np.load(str(p), allow_pickle=True)
    ids = data["ids"].astype(int).tolist()
    seq = data["motion_seq"].astype(np.float32)
    mask = data["pair_mask"].astype(np.float32)
    stat = data["motion_stat"].astype(np.float32)
    out = {}
    for i, pid in enumerate(ids):
        out[int(pid)] = {"seq": seq[i], "mask": mask[i], "stat": stat[i]}
    return out


def read_audio_big_npz(path: Optional[str | Path]) -> Dict[int, Dict[str, np.ndarray]]:
    """Load v10 utterance-level acoustic features.

    Expected NPZ keys:
      ids: [N]
      pair_mask: [N,4]
      audio_big_pair: [N,4,D]
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] audio_big npz not found: {p}")
        return {}
    data = np.load(str(p), allow_pickle=True)
    ids = data["ids"].astype(int).tolist()
    pair = data["audio_big_pair"].astype(np.float32)
    mask = data["pair_mask"].astype(np.float32)
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for i, pid in enumerate(ids):
        out[int(pid)] = {"pair": pair[i], "mask": mask[i]}
    return out


def read_extra_p_csv(path: Optional[str | Path]) -> Tuple[Dict[int, np.ndarray], List[str]]:
    """Load enhanced P features from CSV. First column must be ID/id."""
    if not path:
        return {}, []
    p = Path(path)
    if not p.exists():
        print(f"[WARN] p_extra csv not found: {p}")
        return {}, []
    df = read_csv_auto(p)
    id_col = "ID" if "ID" in df.columns else ("id" if "id" in df.columns else df.columns[0])
    cols = [c for c in df.columns if c != id_col]
    keep: List[str] = []
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().any():
            keep.append(c)
    if keep:
        df[keep] = df[keep].fillna(0.0)
    out: Dict[int, np.ndarray] = {}
    for _, r in df.iterrows():
        out[to_int_id(r[id_col])] = r[keep].astype(float).to_numpy(dtype=np.float32) if keep else np.zeros(0, dtype=np.float32)
    return out, keep


class ElderV3Dataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        data_root: str | Path,
        p_struct_map: Dict[int, np.ndarray],
        p_embed_map: Dict[int, np.ndarray],
        motion_map: Dict[int, Dict[str, np.ndarray]],
        audio_big_map: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
        p_extra_map: Optional[Dict[int, np.ndarray]] = None,
        scalers: Optional[FoldScalers] = None,
        audio_features: Sequence[str] = ("wav2vec", "opensmile"),
        official_video_features: Sequence[str] = (),
        use_gait: bool = True,
        target_t: int = TARGET_T,
        train_mode: bool = True,
    ) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.p_struct_map = p_struct_map
        self.p_embed_map = p_embed_map
        self.motion_map = motion_map
        self.audio_big_map = audio_big_map or {}
        self.p_extra_map = p_extra_map or {}
        self.scalers = scalers
        self.audio_features = list(audio_features)
        self.official_video_features = list(official_video_features)
        self.use_gait = use_gait
        self.target_t = target_t
        self.train_mode = train_mode

        self.ids = [to_int_id(x) for x in self.rows["ID" if "ID" in self.rows.columns else "id"].tolist()]
        self.has_labels = all(c in self.rows.columns for c in ["label2", "label3", "PHQ-9"])
        self._dim_cache = self._infer_dims()

    def _infer_dims(self) -> Dict[str, int]:
        # Find first sample with each feature.
        p_struct_dim = 0
        for v in self.p_struct_map.values():
            p_struct_dim = len(v)
            break
        p_embed_dim = 0
        for v in self.p_embed_map.values():
            p_embed_dim = len(v)
            break
        p_extra_dim = 0
        for v in self.p_extra_map.values():
            p_extra_dim = len(v)
            break
        motion_dim = 0
        motion_stat_dim = 0
        for v in self.motion_map.values():
            motion_dim = int(v["seq"].shape[-1])
            motion_stat_dim = int(v["stat"].shape[-1])
            break
        audio_dim = 0
        audio_big_dim = 0
        for v in self.audio_big_map.values():
            audio_big_dim = int(v["pair"].shape[-1])
            break
        official_video_dim = 0
        gait_dim = 0
        for pid in self.ids[: min(len(self.ids), 20)]:
            if self.audio_features and audio_dim == 0:
                a, _, d = load_official_pair_features(self.data_root, pid, self.audio_features, "audio", self.target_t)
                if d > 0:
                    audio_dim = d
            if self.official_video_features and official_video_dim == 0:
                v, _, d = load_official_pair_features(self.data_root, pid, self.official_video_features, "video", self.target_t)
                if d > 0:
                    official_video_dim = d
            if self.use_gait and gait_dim == 0:
                g = load_gait_seq(self.data_root, pid, self.target_t)
                if g.shape[-1] > 0:
                    gait_dim = g.shape[-1]
        return {
            "p_struct_dim": p_struct_dim,
            "p_embed_dim": p_embed_dim,
            "p_extra_dim": p_extra_dim,
            "audio_dim": audio_dim,
            "audio_big_dim": audio_big_dim,
            "motion_dim": motion_dim,
            "motion_stat_dim": motion_stat_dim,
            "official_video_dim": official_video_dim,
            "gait_dim": gait_dim,
        }

    @property
    def dims(self) -> Dict[str, int]:
        return dict(self._dim_cache)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows.iloc[idx]
        pid = to_int_id(row["ID"] if "ID" in row.index else row["id"])
        dims = self._dim_cache
        item: Dict[str, torch.Tensor] = {"id": torch.tensor(pid, dtype=torch.long)}

        if self.has_labels:
            item["label2"] = torch.tensor(int(row["label2"]), dtype=torch.long)
            item["label3"] = torch.tensor(int(row["label3"]), dtype=torch.long)
            item["phq"] = torch.tensor(float(row["PHQ-9"]), dtype=torch.float32)
        else:
            item["label2"] = torch.tensor(0, dtype=torch.long)
            item["label3"] = torch.tensor(0, dtype=torch.long)
            item["phq"] = torch.tensor(0.0, dtype=torch.float32)

        # Structured P
        p_struct = self.p_struct_map.get(pid, np.zeros(dims["p_struct_dim"], dtype=np.float32))
        if self.scalers is not None:
            p_struct = self.scalers.transform_p(p_struct.reshape(1, -1)).reshape(-1)
        item["p_struct"] = torch.from_numpy(p_struct.astype(np.float32))

        # Official P embedding
        p_embed = self.p_embed_map.get(pid, np.zeros(dims["p_embed_dim"], dtype=np.float32))
        if len(p_embed) != dims["p_embed_dim"]:
            tmp = np.zeros(dims["p_embed_dim"], dtype=np.float32)
            tmp[: min(len(p_embed), len(tmp))] = p_embed[: min(len(p_embed), len(tmp))]
            p_embed = tmp
        item["p_embed"] = torch.from_numpy(p_embed.astype(np.float32))

        # Enhanced P features
        p_extra = self.p_extra_map.get(pid, np.zeros(dims.get("p_extra_dim", 0), dtype=np.float32))
        if len(p_extra) != dims.get("p_extra_dim", 0):
            tmp = np.zeros(dims.get("p_extra_dim", 0), dtype=np.float32)
            tmp[: min(len(p_extra), len(tmp))] = p_extra[: min(len(p_extra), len(tmp))]
            p_extra = tmp
        item["p_extra"] = torch.from_numpy(p_extra.astype(np.float32))

        # Audio official pair seq
        if dims["audio_dim"] > 0 and self.audio_features:
            a, a_mask, _ = load_official_pair_features(self.data_root, pid, self.audio_features, "audio", self.target_t)
            if a.shape[-1] != dims["audio_dim"]:
                tmp = np.zeros((PAIR_COUNT, self.target_t, dims["audio_dim"]), dtype=np.float32)
                tmp[..., : min(a.shape[-1], dims["audio_dim"])] = a[..., : min(a.shape[-1], dims["audio_dim"])]
                a = tmp
        else:
            a = np.zeros((PAIR_COUNT, self.target_t, 0), dtype=np.float32)
            a_mask = np.zeros(PAIR_COUNT, dtype=np.float32)
        item["audio"] = torch.from_numpy(a.astype(np.float32))
        item["audio_pair_mask"] = torch.from_numpy(a_mask.astype(np.float32))

        # Big acoustic utterance-level features: WavLM / emotion2vec / Whisper stats
        if dims.get("audio_big_dim", 0) > 0:
            brec = self.audio_big_map.get(pid)
            if brec is not None:
                audio_big = brec["pair"].astype(np.float32)
                audio_big_mask = brec["mask"].astype(np.float32)
            else:
                audio_big = np.zeros((PAIR_COUNT, dims.get("audio_big_dim", 0)), dtype=np.float32)
                audio_big_mask = np.zeros(PAIR_COUNT, dtype=np.float32)
            if audio_big.shape[-1] != dims.get("audio_big_dim", 0):
                tmp = np.zeros((PAIR_COUNT, dims.get("audio_big_dim", 0)), dtype=np.float32)
                tmp[..., : min(audio_big.shape[-1], dims.get("audio_big_dim", 0))] = audio_big[..., : min(audio_big.shape[-1], dims.get("audio_big_dim", 0))]
                audio_big = tmp
        else:
            audio_big = np.zeros((PAIR_COUNT, 0), dtype=np.float32)
            audio_big_mask = np.zeros(PAIR_COUNT, dtype=np.float32)
        item["audio_big"] = torch.from_numpy(audio_big.astype(np.float32))
        item["audio_big_pair_mask"] = torch.from_numpy(audio_big_mask.astype(np.float32))

        # Raw motion seq + optional official video features concatenated into video branch
        if dims["motion_dim"] > 0:
            mrec = self.motion_map.get(pid)
            if mrec is not None:
                v_motion = mrec["seq"].astype(np.float32)
                v_mask = mrec["mask"].astype(np.float32)
                v_stat = mrec["stat"].astype(np.float32)
            else:
                v_motion = np.zeros((PAIR_COUNT, self.target_t, dims["motion_dim"]), dtype=np.float32)
                v_mask = np.zeros(PAIR_COUNT, dtype=np.float32)
                v_stat = np.zeros(dims["motion_stat_dim"], dtype=np.float32)
        else:
            v_motion = np.zeros((PAIR_COUNT, self.target_t, 0), dtype=np.float32)
            v_mask = np.zeros(PAIR_COUNT, dtype=np.float32)
            v_stat = np.zeros(dims["motion_stat_dim"], dtype=np.float32)

        if self.scalers is not None and v_stat.size:
            v_stat = self.scalers.transform_motion_stat(v_stat.reshape(1, -1)).reshape(-1)

        if dims["official_video_dim"] > 0 and self.official_video_features:
            v_off, v_off_mask, _ = load_official_pair_features(self.data_root, pid, self.official_video_features, "video", self.target_t)
            if v_off.shape[-1] != dims["official_video_dim"]:
                tmp = np.zeros((PAIR_COUNT, self.target_t, dims["official_video_dim"]), dtype=np.float32)
                tmp[..., : min(v_off.shape[-1], dims["official_video_dim"])] = v_off[..., : min(v_off.shape[-1], dims["official_video_dim"])]
                v_off = tmp
            video = np.concatenate([v_motion, v_off], axis=-1)
            v_mask = np.maximum(v_mask, v_off_mask)
        else:
            video = v_motion

        item["video"] = torch.from_numpy(video.astype(np.float32))
        item["video_pair_mask"] = torch.from_numpy(v_mask.astype(np.float32))
        item["motion_stat"] = torch.from_numpy(v_stat.astype(np.float32))

        # Gait
        if dims["gait_dim"] > 0 and self.use_gait:
            g = load_gait_seq(self.data_root, pid, self.target_t)
            if g.shape[-1] != dims["gait_dim"]:
                tmp = np.zeros((self.target_t, dims["gait_dim"]), dtype=np.float32)
                tmp[..., : min(g.shape[-1], dims["gait_dim"])] = g[..., : min(g.shape[-1], dims["gait_dim"])]
                g = tmp
        else:
            g = np.zeros((self.target_t, 0), dtype=np.float32)
        item["gait"] = torch.from_numpy(g.astype(np.float32))
        return item


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or dim
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x [B,L,H], mask [B,L] 1 valid
        logits = self.score(x).squeeze(-1)
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=x.device)
            logits = logits.masked_fill(~mask, -1e4)
            # avoid all-masked NaN
            all_bad = ~mask.any(dim=1)
            if all_bad.any():
                logits[all_bad] = 0.0
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        if mask is not None:
            w = w * mask.float().unsqueeze(-1)
            denom = w.sum(dim=1, keepdim=True).clamp_min(EPS)
            w = w / denom
        return (x * w).sum(dim=1)


class PairTemporalBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.4, use_gru: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        if in_dim <= 0:
            self.enabled = False
            return
        self.enabled = True
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.use_gru = use_gru
        if use_gru:
            self.gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=max(1, min(4, hidden_dim // 32)), dim_feedforward=hidden_dim * 2,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True
            )
            self.tr = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.time_pool = MaskedAttentionPool(hidden_dim)
        self.pair_pool = MaskedAttentionPool(hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, pair_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x [B,P,T,C]
        B = x.shape[0]
        if not self.enabled:
            z = x.new_zeros((B, self.hidden_dim))
            return z, x.new_zeros((B, PAIR_COUNT, self.hidden_dim))
        B, P, T, C = x.shape
        h = self.proj(x.reshape(B * P, T, C))
        if self.use_gru:
            h, _ = self.gru(h)
        else:
            h = self.tr(h)
        pair_feat = self.time_pool(h).reshape(B, P, self.hidden_dim)
        token = self.pair_pool(pair_feat, pair_mask)
        return self.out_norm(token), self.out_norm(pair_feat)


class SeqBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.4):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        if in_dim <= 0:
            self.enabled = False
            return
        self.enabled = True
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.pool = MaskedAttentionPool(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if not self.enabled:
            return x.new_zeros((B, self.hidden_dim))
        h = self.net(x)
        h, _ = self.gru(h)
        return self.norm(self.pool(h))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        if in_dim <= 0:
            self.enabled = False
            self.out_dim = out_dim
            return
        self.enabled = True
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x.new_zeros((x.shape[0], self.out_dim))
        return self.net(x)


class DepFormerAVPV3Lite(nn.Module):
    def __init__(
        self,
        dims: Dict[str, int],
        hidden_dim: int = 96,
        p_embed_bottleneck: int = 48,
        dropout: float = 0.45,
        modality_dropout: float = 0.15,
        num_classes3: int = 3,
    ) -> None:
        super().__init__()
        self.dims = dict(dims)
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout

        p_struct_dim = int(dims.get("p_struct_dim", 0))
        p_embed_dim = int(dims.get("p_embed_dim", 0))
        audio_dim = int(dims.get("audio_dim", 0))
        video_dim = int(dims.get("motion_dim", 0)) + int(dims.get("official_video_dim", 0))
        motion_stat_dim = int(dims.get("motion_stat_dim", 0))
        gait_dim = int(dims.get("gait_dim", 0))

        self.audio_branch = PairTemporalBranch(audio_dim, hidden_dim, dropout=dropout)
        self.video_branch = PairTemporalBranch(video_dim, hidden_dim, dropout=dropout)
        self.gait_branch = SeqBranch(gait_dim, hidden_dim, dropout=dropout)
        self.motion_stat_branch = MLP(motion_stat_dim, hidden_dim, hidden_dim, dropout=dropout)

        self.p_struct_branch = MLP(p_struct_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.p_embed_branch = MLP(p_embed_dim, max(hidden_dim, p_embed_bottleneck * 2), p_embed_bottleneck, dropout=dropout)
        self.p_comb = nn.Sequential(
            nn.Linear(hidden_dim + p_embed_bottleneck, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        # Pair-level audio-video gating, mild fusion only.
        self.av_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.av_res = nn.Sequential(nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout))
        self.pair_pool = MaskedAttentionPool(hidden_dim)
        self.av_norm = nn.LayerNorm(hidden_dim)

        # Modality transformer over P/A/V/AV/G/MotionStat tokens.
        self.token_names = ["P", "A", "V", "AV", "G", "VMstat"]
        self.mod_emb = nn.Parameter(torch.randn(len(self.token_names), hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, min(4, hidden_dim // 32)),
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mod_tr = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.mod_pool = MaskedAttentionPool(hidden_dim)
        self.p_gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        # initialize residual gate near 1: 2*sigmoid(0)=1
        nn.init.zeros_(self.p_gate[-1].weight)
        nn.init.zeros_(self.p_gate[-1].bias)

        self.head_common = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.head3 = nn.Linear(hidden_dim, num_classes3)
        self.head2 = nn.Linear(hidden_dim, 2)
        self.reg = nn.Linear(hidden_dim, 1)  # predicts log1p(PHQ)

    def _token_dropout(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0:
            return tokens, token_mask
        B, L, H = tokens.shape
        keep = torch.ones((B, L), device=tokens.device, dtype=torch.float32)
        # Never drop P token; drop non-P only.
        drop = torch.rand((B, L - 1), device=tokens.device) < self.modality_dropout
        keep[:, 1:] = (~drop).float()
        keep = torch.maximum(keep, 1.0 - token_mask)  # invalid tokens stay zero but mask controls them
        # ensure at least P remains valid
        tokens = tokens * keep.unsqueeze(-1)
        token_mask = token_mask * keep
        token_mask[:, 0] = 1.0
        return tokens, token_mask

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = batch["p_struct"].shape[0]
        device = batch["p_struct"].device
        p_struct_tok = self.p_struct_branch(batch["p_struct"])
        p_emb_tok = self.p_embed_branch(batch["p_embed"])
        p_tok = self.p_comb(torch.cat([p_struct_tok, p_emb_tok], dim=-1))

        a_tok, a_pair = self.audio_branch(batch["audio"], batch["audio_pair_mask"])
        v_tok, v_pair = self.video_branch(batch["video"], batch["video_pair_mask"])
        g_tok = self.gait_branch(batch["gait"])
        ms_tok = self.motion_stat_branch(batch["motion_stat"])

        # mild A/V fusion at pair-token level using shared valid pair mask
        pair_mask = torch.maximum(batch["audio_pair_mask"], batch["video_pair_mask"])
        av_in = torch.cat([a_pair, v_pair, a_pair - v_pair, a_pair * v_pair], dim=-1)
        gate = self.av_gate(av_in)
        av_pair = gate * a_pair + (1.0 - gate) * v_pair + self.av_res(av_in)
        av_pair = self.av_norm(av_pair)
        av_tok = self.pair_pool(av_pair, pair_mask)

        tokens = torch.stack([p_tok, a_tok, v_tok, av_tok, g_tok, ms_tok], dim=1)
        token_mask = torch.ones((B, len(self.token_names)), dtype=torch.float32, device=device)
        token_mask[:, 1] = (batch["audio_pair_mask"].sum(dim=1) > 0).float()
        token_mask[:, 2] = (batch["video_pair_mask"].sum(dim=1) > 0).float()
        token_mask[:, 3] = (pair_mask.sum(dim=1) > 0).float()
        token_mask[:, 4] = 1.0 if self.gait_branch.enabled else 0.0
        token_mask[:, 5] = 1.0 if self.motion_stat_branch.enabled else 0.0
        tokens = tokens + self.mod_emb.unsqueeze(0)
        tokens, token_mask = self._token_dropout(tokens, token_mask)
        fused_tokens = self.mod_tr(tokens)
        fused = self.mod_pool(fused_tokens, token_mask)
        # personality residual gate, keeps P dominant but not exclusive
        fused = fused * (2.0 * torch.sigmoid(self.p_gate(p_tok)))
        h = self.head_common(fused)
        return {
            "logits3": self.head3(h),
            "logits2": self.head2(h),
            "phq_log": self.reg(h).squeeze(-1),
        }



class DepFormerAVPPAnchorV4(nn.Module):
    """P-anchor severity model for tiny Elder split.

    Difference from v3-lite:
    - Personality is the anchor/main path, not just one token in a Transformer.
    - Audio/Gait/VideoMotion/MotionStat are weak residual auxiliaries.
    - No AV pair cross/gate token and no modality Transformer, reducing overfit.
    """
    def __init__(
        self,
        dims: Dict[str, int],
        hidden_dim: int = 64,
        p_embed_bottleneck: int = 16,
        dropout: float = 0.55,
        modality_dropout: float = 0.35,
        num_classes3: int = 3,
        aux_scale: float = 0.30,
    ) -> None:
        super().__init__()
        self.dims = dict(dims)
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout
        self.aux_scale = aux_scale

        p_struct_dim = int(dims.get("p_struct_dim", 0))
        p_embed_dim = int(dims.get("p_embed_dim", 0))
        audio_dim = int(dims.get("audio_dim", 0))
        video_dim = int(dims.get("motion_dim", 0)) + int(dims.get("official_video_dim", 0))
        motion_stat_dim = int(dims.get("motion_stat_dim", 0))
        gait_dim = int(dims.get("gait_dim", 0))

        self.audio_branch = PairTemporalBranch(audio_dim, hidden_dim, dropout=dropout)
        self.video_branch = PairTemporalBranch(video_dim, hidden_dim, dropout=dropout)
        self.gait_branch = SeqBranch(gait_dim, hidden_dim, dropout=dropout)
        self.motion_stat_branch = MLP(motion_stat_dim, hidden_dim, hidden_dim, dropout=dropout)

        self.p_struct_branch = MLP(p_struct_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.p_embed_branch = MLP(
            p_embed_dim,
            max(hidden_dim, p_embed_bottleneck * 2),
            p_embed_bottleneck,
            dropout=dropout,
        )
        self.p_comb = nn.Sequential(
            nn.Linear(hidden_dim + p_embed_bottleneck, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        self.token_names = ["A", "G", "V", "VMstat"]
        self.aux_emb = nn.Parameter(torch.randn(len(self.token_names), hidden_dim) * 0.02)
        self.aux_pool = MaskedAttentionPool(hidden_dim)
        self.aux_norm = nn.LayerNorm(hidden_dim)

        self.aux_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.fuse_norm = nn.LayerNorm(hidden_dim)
        self.head_common = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.head3 = nn.Linear(hidden_dim, num_classes3)
        self.head2 = nn.Linear(hidden_dim, 2)
        self.reg = nn.Linear(hidden_dim, 1)

    def _aux_dropout(self, tokens: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0:
            return tokens, mask
        drop = torch.rand(mask.shape, device=tokens.device) < self.modality_dropout
        keep = (~drop).float() * mask
        tokens = tokens * keep.unsqueeze(-1)
        return tokens, keep

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = batch["p_struct"].shape[0]
        device = batch["p_struct"].device

        p_struct_tok = self.p_struct_branch(batch["p_struct"])
        p_emb_tok = self.p_embed_branch(batch["p_embed"])
        p_tok = self.p_comb(torch.cat([p_struct_tok, p_emb_tok], dim=-1))

        a_tok, _ = self.audio_branch(batch["audio"], batch["audio_pair_mask"])
        v_tok, _ = self.video_branch(batch["video"], batch["video_pair_mask"])
        g_tok = self.gait_branch(batch["gait"])
        ms_tok = self.motion_stat_branch(batch["motion_stat"])

        aux_tokens = torch.stack([a_tok, g_tok, v_tok, ms_tok], dim=1)
        aux_mask = torch.ones((B, len(self.token_names)), dtype=torch.float32, device=device)
        aux_mask[:, 0] = (batch["audio_pair_mask"].sum(dim=1) > 0).float()
        aux_mask[:, 1] = 1.0 if self.gait_branch.enabled else 0.0
        aux_mask[:, 2] = (batch["video_pair_mask"].sum(dim=1) > 0).float()
        aux_mask[:, 3] = 1.0 if self.motion_stat_branch.enabled else 0.0

        aux_tokens = aux_tokens + self.aux_emb.unsqueeze(0)
        aux_tokens, aux_mask = self._aux_dropout(aux_tokens, aux_mask)
        aux = self.aux_norm(self.aux_pool(aux_tokens, aux_mask))

        gate_in = torch.cat([p_tok, aux, p_tok - aux, p_tok * aux], dim=-1)
        gate = self.aux_gate(gate_in)

        fused = self.fuse_norm(p_tok + self.aux_scale * gate * aux)
        h = self.head_common(fused)

        return {
            "logits3": self.head3(h),
            "logits2": self.head2(h),
            "phq_log": self.reg(h).squeeze(-1),
        }


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def build_model(dims: Dict[str, int], config: Any, *, for_predict: bool = False) -> nn.Module:
    arch = str(_cfg_get(config, "model_arch", "lite"))
    hidden_dim = int(_cfg_get(config, "hidden_dim", 96))
    p_embed_bottleneck = int(_cfg_get(config, "p_embed_bottleneck", 48))
    dropout = float(_cfg_get(config, "dropout", 0.45))
    modality_dropout = 0.0 if for_predict else float(_cfg_get(config, "modality_dropout", 0.18))

    if arch in {"v10_p_prior", "v10_b", "p_prior_residual"}:
        return DepFormerAVPV10AudioPprior(
            dims=dims, hidden_dim=hidden_dim, p_embed_bottleneck=p_embed_bottleneck,
            dropout=dropout, modality_dropout=modality_dropout,
            use_audio_big=False, use_p_extra=True, use_p_prior=True,
        )

    if arch in {"v10_audio_p_prior", "v10_c", "audio_p_prior"}:
        return DepFormerAVPV10AudioPprior(
            dims=dims, hidden_dim=hidden_dim, p_embed_bottleneck=p_embed_bottleneck,
            dropout=dropout, modality_dropout=modality_dropout,
            use_audio_big=True, use_p_extra=True, use_p_prior=True,
        )

    if arch in {"v9_depformerv2", "v9", "depformerv2"}:
        return DepFormerAVPV9DepFormerV2(
            dims=dims,
            hidden_dim=hidden_dim,
            p_embed_bottleneck=p_embed_bottleneck,
            dropout=dropout,
            modality_dropout=modality_dropout,
            use_pair_cross=True,
            use_p_guided=True,
            use_shared_private=True,
        )

    if arch in {"v9_no_pguide", "v9_no_p"}:
        return DepFormerAVPV9DepFormerV2(
            dims=dims,
            hidden_dim=hidden_dim,
            p_embed_bottleneck=p_embed_bottleneck,
            dropout=dropout,
            modality_dropout=modality_dropout,
            use_pair_cross=True,
            use_p_guided=False,
            use_shared_private=True,
        )

    if arch in {"v9_no_cross", "v9_nocross"}:
        return DepFormerAVPV9DepFormerV2(
            dims=dims,
            hidden_dim=hidden_dim,
            p_embed_bottleneck=p_embed_bottleneck,
            dropout=dropout,
            modality_dropout=modality_dropout,
            use_pair_cross=False,
            use_p_guided=True,
            use_shared_private=True,
        )

    if arch in {"v9_no_sp", "v9_nosp"}:
        return DepFormerAVPV9DepFormerV2(
            dims=dims,
            hidden_dim=hidden_dim,
            p_embed_bottleneck=p_embed_bottleneck,
            dropout=dropout,
            modality_dropout=modality_dropout,
            use_pair_cross=True,
            use_p_guided=True,
            use_shared_private=False,
        )

    if arch in {"v7_res_hier", "v7_hier", "v7"}:
        return DepFormerAVPV7ResidualHierHead(
            dims=dims,
            hidden_dim=hidden_dim,
            p_embed_bottleneck=p_embed_bottleneck,
            dropout=dropout,
            modality_dropout=modality_dropout,
        )

    if arch in {"v6_cross_motion", "v6_cross", "cross_motion"}:
        return DepFormerAVPV6LiteCrossMotion(
            dims=dims,
            hidden_dim=hidden_dim,
            p_embed_bottleneck=p_embed_bottleneck,
            dropout=dropout,
            modality_dropout=modality_dropout,
            use_p_aux_gate=True,
            use_p_gait_gate=True,
            use_cross_gate=True,
        )

    if arch in {"v6_cross_no_pgate", "v6_no_pgate"}:
        return DepFormerAVPV6LiteCrossMotion(
            dims=dims,
            hidden_dim=hidden_dim,
            p_embed_bottleneck=p_embed_bottleneck,
            dropout=dropout,
            modality_dropout=modality_dropout,
            use_p_aux_gate=False,
            use_p_gait_gate=False,
            use_cross_gate=True,
        )

    if arch in {"v6_cross_no_crossgate", "v6_no_crossgate"}:
        return DepFormerAVPV6LiteCrossMotion(
            dims=dims,
            hidden_dim=hidden_dim,
            p_embed_bottleneck=p_embed_bottleneck,
            dropout=dropout,
            modality_dropout=modality_dropout,
            use_p_aux_gate=True,
            use_p_gait_gate=True,
            use_cross_gate=False,
        )

    if arch in {"v5_hier_ord", "v5", "hier_ord"}:
        return DepFormerAVPV5HierOrd(
            dims=dims,
            hidden_dim=hidden_dim,
            p_embed_bottleneck=p_embed_bottleneck,
            dropout=dropout,
            modality_dropout=modality_dropout,
        )

    if arch in {"p_anchor_v4", "anchor", "p_anchor"}:
        return DepFormerAVPPAnchorV4(
            dims=dims,
            hidden_dim=hidden_dim,
            p_embed_bottleneck=p_embed_bottleneck,
            dropout=dropout,
            modality_dropout=modality_dropout,
        )

    return DepFormerAVPV3Lite(
        dims=dims,
        hidden_dim=hidden_dim,
        p_embed_bottleneck=p_embed_bottleneck,
        dropout=dropout,
        modality_dropout=modality_dropout,
    )



class DepFormerAVPV5HierOrd(nn.Module):
    """v5: v3-like multimodal fusion + explicit ordinal severity head.

    Compared with v4 P-anchor:
    - P is important but not the only anchor.
    - Keeps v3's multi-token fusion.
    - Adds an ordinal head [label>0, label>1] to improve binary/ternary continuity.
    - Keeps pair-level AV interaction light, not frame-level cross attention.
    """
    def __init__(
        self,
        dims: Dict[str, int],
        hidden_dim: int = 96,
        p_embed_bottleneck: int = 48,
        dropout: float = 0.45,
        modality_dropout: float = 0.18,
        num_classes3: int = 3,
    ) -> None:
        super().__init__()
        self.dims = dict(dims)
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout

        p_struct_dim = int(dims.get("p_struct_dim", 0))
        p_embed_dim = int(dims.get("p_embed_dim", 0))
        audio_dim = int(dims.get("audio_dim", 0))
        video_dim = int(dims.get("motion_dim", 0)) + int(dims.get("official_video_dim", 0))
        motion_stat_dim = int(dims.get("motion_stat_dim", 0))
        gait_dim = int(dims.get("gait_dim", 0))

        self.p_struct_branch = MLP(p_struct_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.p_embed_branch = MLP(
            p_embed_dim,
            max(hidden_dim, p_embed_bottleneck * 2),
            p_embed_bottleneck,
            dropout=dropout,
        )
        self.p_comb = nn.Sequential(
            nn.Linear(hidden_dim + p_embed_bottleneck, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        self.audio_branch = PairTemporalBranch(audio_dim, hidden_dim, dropout=dropout)
        self.video_branch = PairTemporalBranch(video_dim, hidden_dim, dropout=dropout)
        self.gait_branch = SeqBranch(gait_dim, hidden_dim, dropout=dropout)
        self.motion_stat_branch = MLP(motion_stat_dim, hidden_dim, hidden_dim, dropout=dropout)

        # light AV pair gate: no frame-level cross attention
        self.av_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.av_resid = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.av_pool = MaskedAttentionPool(hidden_dim)

        # modality fusion tokens: P, A, V, AV, G, VMstat
        self.token_names = ["P", "A", "V", "AV", "G", "VMstat"]
        self.mod_emb = nn.Parameter(torch.randn(len(self.token_names), hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.mod_tr = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.mod_pool = MaskedAttentionPool(hidden_dim)

        # severity refinement: combine global fused with clinically meaningful tokens
        self.sev_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.sev_refine = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.head_common = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.head3 = nn.Linear(hidden_dim, num_classes3)
        self.head2 = nn.Linear(hidden_dim, 2)
        self.ord_head = nn.Linear(hidden_dim, 2)  # [label > 0, label > 1]
        self.reg = nn.Linear(hidden_dim, 1)

    def _modality_dropout(self, tokens: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0:
            return tokens, mask
        # Do not drop P completely; drop auxiliary modalities only.
        aux_mask = mask.clone()
        drop = torch.rand(mask.shape, device=tokens.device) < self.modality_dropout
        drop[:, 0] = False
        keep = (~drop).float() * aux_mask
        tokens = tokens * keep.unsqueeze(-1)
        return tokens, keep

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = batch["p_struct"].shape[0]
        device = batch["p_struct"].device

        p_struct_tok = self.p_struct_branch(batch["p_struct"])
        p_emb_tok = self.p_embed_branch(batch["p_embed"])
        p_tok = self.p_comb(torch.cat([p_struct_tok, p_emb_tok], dim=-1))

        a_tok, a_pair = self.audio_branch(batch["audio"], batch["audio_pair_mask"])
        v_tok, v_pair = self.video_branch(batch["video"], batch["video_pair_mask"])
        g_tok = self.gait_branch(batch["gait"])
        ms_tok = self.motion_stat_branch(batch["motion_stat"])

        pair_mask = batch["audio_pair_mask"] * batch["video_pair_mask"]
        av_in = torch.cat([a_pair, v_pair, a_pair - v_pair, a_pair * v_pair], dim=-1)
        av_gate = self.av_gate(av_in)
        av_pair = av_gate * a_pair + (1.0 - av_gate) * v_pair + 0.25 * self.av_resid(av_in)
        av_tok = self.av_pool(av_pair, pair_mask)

        tokens = torch.stack([p_tok, a_tok, v_tok, av_tok, g_tok, ms_tok], dim=1)
        mask = torch.ones((B, len(self.token_names)), dtype=torch.float32, device=device)
        mask[:, 0] = 1.0
        mask[:, 1] = (batch["audio_pair_mask"].sum(dim=1) > 0).float()
        mask[:, 2] = (batch["video_pair_mask"].sum(dim=1) > 0).float()
        mask[:, 3] = (pair_mask.sum(dim=1) > 0).float()
        mask[:, 4] = 1.0 if self.gait_branch.enabled else 0.0
        mask[:, 5] = 1.0 if self.motion_stat_branch.enabled else 0.0

        tokens = tokens + self.mod_emb.unsqueeze(0)
        tokens, mask = self._modality_dropout(tokens, mask)
        h = self.mod_tr(tokens, src_key_padding_mask=(mask <= 0))
        fused = self.mod_pool(h, mask)

        sev_in = torch.cat([fused, p_tok, av_tok, g_tok, ms_tok], dim=-1)
        sev_gate = self.sev_gate(sev_in)
        sev_delta = self.sev_refine(sev_in)
        severity = fused + 0.35 * sev_gate * sev_delta

        z = self.head_common(severity)
        return {
            "logits3": self.head3(z),
            "logits2": self.head2(z),
            "ord_logits": self.ord_head(z),
            "phq_log": self.reg(z).squeeze(-1),
        }



class GroupedGaitBranch(nn.Module):
    """IMU gait encoder split by physical units:
    acc xyz, gyro xyz, angle xyz. Falls back to single SeqBranch if dim < 9.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout: float = 0.45,
        use_p_gait_gate: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = hidden_dim
        self.use_p_gait_gate = use_p_gait_gate
        if in_dim <= 0:
            self.enabled = False
            return

        self.enabled = True
        self.grouped = in_dim >= 9
        if self.grouped:
            self.acc_branch = SeqBranch(3, hidden_dim, dropout=dropout)
            self.gyro_branch = SeqBranch(3, hidden_dim, dropout=dropout)
            self.angle_branch = SeqBranch(3, hidden_dim, dropout=dropout)
            self.group_emb = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
            self.group_pool = MaskedAttentionPool(hidden_dim)
        else:
            self.single_branch = SeqBranch(in_dim, hidden_dim, dropout=dropout)

        self.p_gait_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.p_gait_delta = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, p_tok: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        if not self.enabled:
            return x.new_zeros((B, self.hidden_dim))

        if not self.grouped:
            g = self.single_branch(x)
        else:
            # x is [B,T,9]: 1-3 acc, 4-6 gyro, 7-9 angle
            acc = x[..., 0:3]
            gyro = x[..., 3:6]
            angle = x[..., 6:9]
            acc_tok = self.acc_branch(acc)
            gyro_tok = self.gyro_branch(gyro)
            angle_tok = self.angle_branch(angle)
            groups = torch.stack([acc_tok, gyro_tok, angle_tok], dim=1) + self.group_emb.unsqueeze(0)
            mask = torch.ones((B, 3), dtype=torch.float32, device=x.device)
            g = self.group_pool(groups, mask)

        if self.use_p_gait_gate and p_tok is not None:
            gate_in = torch.cat([p_tok, g, p_tok - g, p_tok * g], dim=-1)
            gate = self.p_gait_gate(gate_in)
            delta = self.p_gait_delta(gate_in)
            g = g + 0.25 * gate * delta

        return self.out_norm(g)


class DepFormerAVPV6LiteCrossMotion(nn.Module):
    """v6 = v3-lite backbone + stronger pair-level A-motion cross gate
    + grouped IMU gait encoder + P-conditioned light gates.

    This intentionally keeps the effective v3 modality transformer backbone.
    It does not hard-force binary=ternary>0; consistency remains a soft loss.
    """
    def __init__(
        self,
        dims: Dict[str, int],
        hidden_dim: int = 96,
        p_embed_bottleneck: int = 48,
        dropout: float = 0.45,
        modality_dropout: float = 0.18,
        num_classes3: int = 3,
        use_p_aux_gate: bool = True,
        use_p_gait_gate: bool = True,
        use_cross_gate: bool = True,
        cross_scale: float = 0.35,
    ) -> None:
        super().__init__()
        self.dims = dict(dims)
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout
        self.use_p_aux_gate = use_p_aux_gate
        self.use_cross_gate = use_cross_gate
        self.cross_scale = cross_scale

        p_struct_dim = int(dims.get("p_struct_dim", 0))
        p_embed_dim = int(dims.get("p_embed_dim", 0))
        audio_dim = int(dims.get("audio_dim", 0))
        video_dim = int(dims.get("motion_dim", 0)) + int(dims.get("official_video_dim", 0))
        motion_stat_dim = int(dims.get("motion_stat_dim", 0))
        gait_dim = int(dims.get("gait_dim", 0))

        self.p_struct_branch = MLP(p_struct_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.p_embed_branch = MLP(
            p_embed_dim,
            max(hidden_dim, p_embed_bottleneck * 2),
            p_embed_bottleneck,
            dropout=dropout,
        )
        self.p_comb = nn.Sequential(
            nn.Linear(hidden_dim + p_embed_bottleneck, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        self.audio_branch = PairTemporalBranch(audio_dim, hidden_dim, dropout=dropout)
        self.video_branch = PairTemporalBranch(video_dim, hidden_dim, dropout=dropout)
        self.gait_branch = GroupedGaitBranch(
            gait_dim, hidden_dim, dropout=dropout, use_p_gait_gate=use_p_gait_gate
        )
        self.motion_stat_branch = MLP(motion_stat_dim, hidden_dim, hidden_dim, dropout=dropout)

        # C-version: light pair-level A-motion cross gate.
        # It is stricter than v3: the cross token only pools pairs where both A and motion exist.
        self.cross_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.cross_delta = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.pair_pool = MaskedAttentionPool(hidden_dim)

        # P-conditioned auxiliary token gate before modality transformer.
        self.p_aux_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.p_aux_delta = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.token_names = ["P", "A", "V", "AM", "G", "VMstat"]
        self.mod_emb = nn.Parameter(torch.randn(len(self.token_names), hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, min(4, hidden_dim // 32)),
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mod_tr = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.mod_pool = MaskedAttentionPool(hidden_dim)

        # keep v3's personality residual gate, initialized near identity
        self.p_gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        nn.init.zeros_(self.p_gate[-1].weight)
        nn.init.zeros_(self.p_gate[-1].bias)

        self.head_common = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.head3 = nn.Linear(hidden_dim, num_classes3)
        self.head2 = nn.Linear(hidden_dim, 2)
        self.reg = nn.Linear(hidden_dim, 1)

    def _token_dropout(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0:
            return tokens, token_mask
        B, L, H = tokens.shape
        keep = torch.ones((B, L), device=tokens.device, dtype=torch.float32)
        drop = torch.rand((B, L - 1), device=tokens.device) < self.modality_dropout
        keep[:, 1:] = (~drop).float()
        keep = torch.maximum(keep, 1.0 - token_mask)
        tokens = tokens * keep.unsqueeze(-1)
        token_mask = token_mask * keep
        token_mask[:, 0] = 1.0
        return tokens, token_mask

    def _apply_p_aux_gate(self, p_tok: torch.Tensor, aux_tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_p_aux_gate:
            return aux_tokens
        B, L, H = aux_tokens.shape
        p = p_tok.unsqueeze(1).expand(B, L, H)
        gate_in = torch.cat([p, aux_tokens, p - aux_tokens, p * aux_tokens], dim=-1)
        gate = self.p_aux_gate(gate_in)
        delta = self.p_aux_delta(gate_in)
        return aux_tokens + 0.20 * gate * delta

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = batch["p_struct"].shape[0]
        device = batch["p_struct"].device

        p_struct_tok = self.p_struct_branch(batch["p_struct"])
        p_emb_tok = self.p_embed_branch(batch["p_embed"])
        p_tok = self.p_comb(torch.cat([p_struct_tok, p_emb_tok], dim=-1))

        a_tok, a_pair = self.audio_branch(batch["audio"], batch["audio_pair_mask"])
        v_tok, v_pair = self.video_branch(batch["video"], batch["video_pair_mask"])
        g_tok = self.gait_branch(batch["gait"], p_tok=p_tok)
        ms_tok = self.motion_stat_branch(batch["motion_stat"])

        # strict intersection mask for actual A-motion cross interaction
        pair_mask_cross = batch["audio_pair_mask"] * batch["video_pair_mask"]
        cross_in = torch.cat([a_pair, v_pair, a_pair - v_pair, a_pair * v_pair], dim=-1)

        if self.use_cross_gate:
            gate = self.cross_gate(cross_in)
            delta = self.cross_delta(cross_in)
            am_pair = 0.5 * (a_pair + v_pair) + self.cross_scale * gate * delta
        else:
            am_pair = 0.5 * (a_pair + v_pair)

        am_pair = self.cross_norm(am_pair)
        am_tok = self.pair_pool(am_pair, pair_mask_cross)

        tokens = torch.stack([p_tok, a_tok, v_tok, am_tok, g_tok, ms_tok], dim=1)
        token_mask = torch.ones((B, len(self.token_names)), dtype=torch.float32, device=device)
        token_mask[:, 0] = 1.0
        token_mask[:, 1] = (batch["audio_pair_mask"].sum(dim=1) > 0).float()
        token_mask[:, 2] = (batch["video_pair_mask"].sum(dim=1) > 0).float()
        token_mask[:, 3] = (pair_mask_cross.sum(dim=1) > 0).float()
        token_mask[:, 4] = 1.0 if self.gait_branch.enabled else 0.0
        token_mask[:, 5] = 1.0 if self.motion_stat_branch.enabled else 0.0

        # P participates before aggregation through light gating of auxiliary tokens
        aux = self._apply_p_aux_gate(p_tok, tokens[:, 1:, :])
        tokens = torch.cat([tokens[:, :1, :], aux], dim=1)

        tokens = tokens + self.mod_emb.unsqueeze(0)
        tokens, token_mask = self._token_dropout(tokens, token_mask)
        fused_tokens = self.mod_tr(tokens, src_key_padding_mask=(token_mask <= 0))
        fused = self.mod_pool(fused_tokens, token_mask)

        fused = fused * (2.0 * torch.sigmoid(self.p_gate(p_tok)))
        h = self.head_common(fused)
        return {
            "logits3": self.head3(h),
            "logits2": self.head2(h),
            "phq_log": self.reg(h).squeeze(-1),
        }



class DepFormerAVPV7ResidualHierHead(nn.Module):
    """v7 = v3-lite encoder + residual hierarchical binary head.

    Encoder is intentionally kept close to v3-lite:
      [P, A, V, AV, G, VMstat] -> modality Transformer -> fused

    Difference:
      binary logit is derived from ternary depressed-vs-normal logit
      plus a learnable residual. This keeps binary/ternary related but
      avoids hard binary=ternary>0, which was harmful in blind tests.
    """
    def __init__(
        self,
        dims: Dict[str, int],
        hidden_dim: int = 96,
        p_embed_bottleneck: int = 48,
        dropout: float = 0.45,
        modality_dropout: float = 0.18,
        num_classes3: int = 3,
        residual_alpha_init: float = 0.35,
    ) -> None:
        super().__init__()
        self.dims = dict(dims)
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout

        p_struct_dim = int(dims.get("p_struct_dim", 0))
        p_embed_dim = int(dims.get("p_embed_dim", 0))
        audio_dim = int(dims.get("audio_dim", 0))
        video_dim = int(dims.get("motion_dim", 0)) + int(dims.get("official_video_dim", 0))
        motion_stat_dim = int(dims.get("motion_stat_dim", 0))
        gait_dim = int(dims.get("gait_dim", 0))

        self.audio_branch = PairTemporalBranch(audio_dim, hidden_dim, dropout=dropout)
        self.video_branch = PairTemporalBranch(video_dim, hidden_dim, dropout=dropout)
        self.gait_branch = SeqBranch(gait_dim, hidden_dim, dropout=dropout)
        self.motion_stat_branch = MLP(motion_stat_dim, hidden_dim, hidden_dim, dropout=dropout)

        self.p_struct_branch = MLP(p_struct_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.p_embed_branch = MLP(
            p_embed_dim,
            max(hidden_dim, p_embed_bottleneck * 2),
            p_embed_bottleneck,
            dropout=dropout,
        )
        self.p_comb = nn.Sequential(
            nn.Linear(hidden_dim + p_embed_bottleneck, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        # Same light pair-level A/V fusion as v3.
        self.av_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.av_res = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.pair_pool = MaskedAttentionPool(hidden_dim)
        self.av_norm = nn.LayerNorm(hidden_dim)

        self.token_names = ["P", "A", "V", "AV", "G", "VMstat"]
        self.mod_emb = nn.Parameter(torch.randn(len(self.token_names), hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, min(4, hidden_dim // 32)),
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mod_tr = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.mod_pool = MaskedAttentionPool(hidden_dim)

        self.p_gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        nn.init.zeros_(self.p_gate[-1].weight)
        nn.init.zeros_(self.p_gate[-1].bias)

        self.head_common = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.head3 = nn.Linear(hidden_dim, num_classes3)

        # Residual binary head. Binary logit = ternary-derived logit + alpha * residual.
        self.binary_residual = nn.Linear(hidden_dim, 1)
        residual_alpha_init = max(1e-4, min(0.95, float(residual_alpha_init)))
        alpha_logit = math.log(residual_alpha_init / (1.0 - residual_alpha_init))
        self.binary_res_alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))

        self.reg = nn.Linear(hidden_dim, 1)

    def _token_dropout(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0:
            return tokens, token_mask
        B, L, H = tokens.shape
        keep = torch.ones((B, L), device=tokens.device, dtype=torch.float32)
        drop = torch.rand((B, L - 1), device=tokens.device) < self.modality_dropout
        keep[:, 1:] = (~drop).float()
        keep = torch.maximum(keep, 1.0 - token_mask)
        tokens = tokens * keep.unsqueeze(-1)
        token_mask = token_mask * keep
        token_mask[:, 0] = 1.0
        return tokens, token_mask

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = batch["p_struct"].shape[0]
        device = batch["p_struct"].device

        p_struct_tok = self.p_struct_branch(batch["p_struct"])
        p_emb_tok = self.p_embed_branch(batch["p_embed"])
        p_tok = self.p_comb(torch.cat([p_struct_tok, p_emb_tok], dim=-1))

        a_tok, a_pair = self.audio_branch(batch["audio"], batch["audio_pair_mask"])
        v_tok, v_pair = self.video_branch(batch["video"], batch["video_pair_mask"])
        g_tok = self.gait_branch(batch["gait"])
        ms_tok = self.motion_stat_branch(batch["motion_stat"])

        pair_mask = torch.maximum(batch["audio_pair_mask"], batch["video_pair_mask"])
        av_in = torch.cat([a_pair, v_pair, a_pair - v_pair, a_pair * v_pair], dim=-1)
        gate = self.av_gate(av_in)
        av_pair = gate * a_pair + (1.0 - gate) * v_pair + self.av_res(av_in)
        av_pair = self.av_norm(av_pair)
        av_tok = self.pair_pool(av_pair, pair_mask)

        tokens = torch.stack([p_tok, a_tok, v_tok, av_tok, g_tok, ms_tok], dim=1)
        token_mask = torch.ones((B, len(self.token_names)), dtype=torch.float32, device=device)
        token_mask[:, 1] = (batch["audio_pair_mask"].sum(dim=1) > 0).float()
        token_mask[:, 2] = (batch["video_pair_mask"].sum(dim=1) > 0).float()
        token_mask[:, 3] = (pair_mask.sum(dim=1) > 0).float()
        token_mask[:, 4] = 1.0 if self.gait_branch.enabled else 0.0
        token_mask[:, 5] = 1.0 if self.motion_stat_branch.enabled else 0.0

        tokens = tokens + self.mod_emb.unsqueeze(0)
        tokens, token_mask = self._token_dropout(tokens, token_mask)
        fused_tokens = self.mod_tr(tokens)
        fused = self.mod_pool(fused_tokens, token_mask)

        fused = fused * (2.0 * torch.sigmoid(self.p_gate(p_tok)))
        h = self.head_common(fused)

        logits3 = self.head3(h)

        dep_from_ternary = torch.logsumexp(logits3[:, 1:], dim=1) - logits3[:, 0]
        residual = self.binary_residual(h).squeeze(-1)
        alpha = torch.sigmoid(self.binary_res_alpha_logit)
        binary_logit = dep_from_ternary + alpha * residual
        logits2 = torch.stack([torch.zeros_like(binary_logit), binary_logit], dim=1)

        # Ordinal logits derived from unified binary/ternary probabilities.
        sev_logit = logits3[:, 2] - torch.logsumexp(logits3[:, :2], dim=1)
        ord_logits = torch.stack([binary_logit, sev_logit], dim=1)

        return {
            "logits3": logits3,
            "logits2": logits2,
            "ord_logits": ord_logits,
            "phq_log": self.reg(h).squeeze(-1),
        }



class PairMutualCrossFusion(nn.Module):
    """Light pair-level mutual cross fusion for audio-motion tokens.

    It only operates on the 4 pair tokens, not on 128-frame sequences.
    This keeps the mutual-transformer idea small-data friendly.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.45, num_heads: int = 4, use_attn: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_attn = use_attn
        heads = max(1, min(num_heads, hidden_dim // 32))
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.a2v = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.v2a = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 6),
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(hidden_dim * 6),
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, a_pair: torch.Tensor, v_pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        m = pair_mask.unsqueeze(-1)
        a = self.norm_a(a_pair * m)
        v = self.norm_v(v_pair * m)

        if self.use_attn:
            # No key_padding_mask here: all-zero pairs are already masked to zero and
            # are removed again after fusion. This avoids all-masked attention NaNs.
            a2v, _ = self.a2v(a, v, v, need_weights=False)
            v2a, _ = self.v2a(v, a, a, need_weights=False)
        else:
            a2v = v
            v2a = a

        z = torch.cat([a_pair, v_pair, a2v, v2a, a_pair - v_pair, a_pair * v_pair], dim=-1)
        g = self.gate(z)
        d = self.delta(z)
        out = 0.5 * (a_pair + v_pair) + 0.35 * g * d
        return self.out_norm(out) * m


class CosineClassifier(nn.Module):
    """Normalized cosine classifier for small-data class separation."""
    def __init__(self, in_dim: int, num_classes: int, init_scale: float = 12.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        scale = torch.clamp(self.log_scale.exp(), 1.0, 30.0)
        return scale * x.matmul(w.t())


class DepFormerAVPV9DepFormerV2(nn.Module):
    """DepFormerV2-style model for MPDD Elder.

    Keeps v3 feature processing:
      P, Audio, raw motion V, gait, motion_stat.

    Improves fusion:
      1) pair-level audio-motion mutual cross fusion;
      2) P-guided auxiliary gating;
      3) shared/private decomposition before final multimodal transformer;
      4) metric classification heads for MacroF1/Kappa-oriented training.
    """
    def __init__(
        self,
        dims: Dict[str, int],
        hidden_dim: int = 96,
        p_embed_bottleneck: int = 48,
        dropout: float = 0.45,
        modality_dropout: float = 0.18,
        num_classes3: int = 3,
        use_pair_cross: bool = True,
        use_p_guided: bool = True,
        use_shared_private: bool = True,
    ) -> None:
        super().__init__()
        self.dims = dict(dims)
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout
        self.use_p_guided = use_p_guided
        self.use_shared_private = use_shared_private

        p_struct_dim = int(dims.get("p_struct_dim", 0))
        p_embed_dim = int(dims.get("p_embed_dim", 0))
        audio_dim = int(dims.get("audio_dim", 0))
        video_dim = int(dims.get("motion_dim", 0)) + int(dims.get("official_video_dim", 0))
        motion_stat_dim = int(dims.get("motion_stat_dim", 0))
        gait_dim = int(dims.get("gait_dim", 0))

        self.audio_branch = PairTemporalBranch(audio_dim, hidden_dim, dropout=dropout)
        self.video_branch = PairTemporalBranch(video_dim, hidden_dim, dropout=dropout)
        self.gait_branch = SeqBranch(gait_dim, hidden_dim, dropout=dropout)
        self.motion_stat_branch = MLP(motion_stat_dim, hidden_dim, hidden_dim, dropout=dropout)

        self.p_struct_branch = MLP(p_struct_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.p_embed_branch = MLP(
            p_embed_dim,
            max(hidden_dim, p_embed_bottleneck * 2),
            p_embed_bottleneck,
            dropout=dropout,
        )
        self.p_comb = nn.Sequential(
            nn.Linear(hidden_dim + p_embed_bottleneck, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        self.cross_fusion = PairMutualCrossFusion(hidden_dim, dropout=dropout, use_attn=use_pair_cross)
        self.pair_pool = MaskedAttentionPool(hidden_dim)

        self.p_aux_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.p_aux_delta = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.shared_pool = MaskedAttentionPool(hidden_dim)
        self.private_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.token_names = ["P", "Shared", "A_priv", "V_priv", "AV_priv", "G_priv", "MS_priv"]
        self.mod_emb = nn.Parameter(torch.randn(len(self.token_names), hidden_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, min(4, hidden_dim // 32)),
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mod_tr = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.mod_pool = MaskedAttentionPool(hidden_dim)

        self.p_gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        nn.init.zeros_(self.p_gate[-1].weight)
        nn.init.zeros_(self.p_gate[-1].bias)

        self.metric_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        self.head3 = CosineClassifier(hidden_dim, num_classes3, init_scale=12.0)
        self.head2 = CosineClassifier(hidden_dim, 2, init_scale=12.0)
        self.reg = nn.Linear(hidden_dim, 1)

    def _token_dropout(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0:
            return tokens, token_mask
        B, L, H = tokens.shape
        keep = torch.ones((B, L), device=tokens.device, dtype=torch.float32)
        drop = torch.rand((B, L - 1), device=tokens.device) < self.modality_dropout
        keep[:, 1:] = (~drop).float()
        keep = torch.maximum(keep, 1.0 - token_mask)
        tokens = tokens * keep.unsqueeze(-1)
        token_mask = token_mask * keep
        token_mask[:, 0] = 1.0
        return tokens, token_mask

    def _p_guided(self, p_tok: torch.Tensor, aux_tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_p_guided:
            return aux_tokens
        B, L, H = aux_tokens.shape
        p = p_tok.unsqueeze(1).expand(B, L, H)
        z = torch.cat([p, aux_tokens, p - aux_tokens, p * aux_tokens], dim=-1)
        g = self.p_aux_gate(z)
        d = self.p_aux_delta(z)
        return aux_tokens + 0.18 * g * d

    def _shared_private(self, aux_tokens: torch.Tensor, aux_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared_pool(aux_tokens, aux_mask)
        if not self.use_shared_private:
            return shared, aux_tokens
        shared_expand = shared.unsqueeze(1).expand_as(aux_tokens)
        g = self.private_gate(torch.cat([aux_tokens, shared_expand], dim=-1))
        private = aux_tokens - 0.35 * g * shared_expand
        return shared, private

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = batch["p_struct"].shape[0]
        device = batch["p_struct"].device

        p_struct_tok = self.p_struct_branch(batch["p_struct"])
        p_emb_tok = self.p_embed_branch(batch["p_embed"])
        p_tok = self.p_comb(torch.cat([p_struct_tok, p_emb_tok], dim=-1))

        a_tok, a_pair = self.audio_branch(batch["audio"], batch["audio_pair_mask"])
        v_tok, v_pair = self.video_branch(batch["video"], batch["video_pair_mask"])
        g_tok = self.gait_branch(batch["gait"])
        ms_tok = self.motion_stat_branch(batch["motion_stat"])

        pair_mask_cross = batch["audio_pair_mask"] * batch["video_pair_mask"]
        av_pair = self.cross_fusion(a_pair, v_pair, pair_mask_cross)
        av_tok = self.pair_pool(av_pair, pair_mask_cross)

        aux = torch.stack([a_tok, v_tok, av_tok, g_tok, ms_tok], dim=1)
        aux_mask = torch.ones((B, 5), dtype=torch.float32, device=device)
        aux_mask[:, 0] = (batch["audio_pair_mask"].sum(dim=1) > 0).float()
        aux_mask[:, 1] = (batch["video_pair_mask"].sum(dim=1) > 0).float()
        aux_mask[:, 2] = (pair_mask_cross.sum(dim=1) > 0).float()
        aux_mask[:, 3] = 1.0 if self.gait_branch.enabled else 0.0
        aux_mask[:, 4] = 1.0 if self.motion_stat_branch.enabled else 0.0

        aux = self._p_guided(p_tok, aux)
        shared, private = self._shared_private(aux, aux_mask)

        tokens = torch.cat([p_tok.unsqueeze(1), shared.unsqueeze(1), private], dim=1)
        token_mask = torch.cat([
            torch.ones((B, 1), device=device),
            torch.ones((B, 1), device=device),
            aux_mask,
        ], dim=1)

        tokens = tokens + self.mod_emb.unsqueeze(0)
        tokens, token_mask = self._token_dropout(tokens, token_mask)
        fused_tokens = self.mod_tr(tokens, src_key_padding_mask=(token_mask <= 0))
        fused = self.mod_pool(fused_tokens, token_mask)

        fused = fused * (2.0 * torch.sigmoid(self.p_gate(p_tok)))
        h = self.metric_proj(fused)

        logits3 = self.head3(h)
        logits2 = self.head2(h)

        p2_dep_logit = logits2[:, 1] - logits2[:, 0]
        sev_logit = logits3[:, 2] - torch.logsumexp(logits3[:, :2], dim=1)
        ord_logits = torch.stack([p2_dep_logit, sev_logit], dim=1)

        return {
            "logits3": logits3,
            "logits2": logits2,
            "ord_logits": ord_logits,
            "phq_log": self.reg(h).squeeze(-1),
        }



class PairFeatureBranch(nn.Module):
    """Encode per-pair utterance-level features [B,4,D] into pair tokens and a pooled token."""
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.45) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = hidden_dim
        self.enabled = self.in_dim > 0
        self.proj = MLP(self.in_dim, hidden_dim, hidden_dim, dropout=dropout) if self.enabled else None
        self.pool = MaskedAttentionPool(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        if not self.enabled:
            z = x.new_zeros((B, PAIR_COUNT, self.hidden_dim))
            return x.new_zeros((B, self.hidden_dim)), z
        z = self.proj(x)
        tok = self.pool(z, mask)
        return tok, z


class DepFormerAVPV10AudioPprior(nn.Module):
    """v10 = v9_no_cross-style fusion + optional big acoustic features + P-prior residual head."""
    def __init__(self, dims: Dict[str, int], hidden_dim: int = 96, p_embed_bottleneck: int = 48, dropout: float = 0.45,
                 modality_dropout: float = 0.18, num_classes3: int = 3, use_audio_big: bool = True,
                 use_p_extra: bool = True, use_p_prior: bool = True, residual_init: float = 0.55) -> None:
        super().__init__()
        self.dims = dict(dims)
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout
        self.use_audio_big = use_audio_big and int(dims.get("audio_big_dim", 0)) > 0
        self.use_p_extra = use_p_extra and int(dims.get("p_extra_dim", 0)) > 0
        self.use_p_prior = use_p_prior
        p_struct_dim = int(dims.get("p_struct_dim", 0)); p_embed_dim = int(dims.get("p_embed_dim", 0))
        p_extra_dim = int(dims.get("p_extra_dim", 0)) if self.use_p_extra else 0
        audio_dim = int(dims.get("audio_dim", 0)); audio_big_dim = int(dims.get("audio_big_dim", 0)) if self.use_audio_big else 0
        video_dim = int(dims.get("motion_dim", 0)) + int(dims.get("official_video_dim", 0))
        motion_stat_dim = int(dims.get("motion_stat_dim", 0)); gait_dim = int(dims.get("gait_dim", 0))
        self.audio_branch = PairTemporalBranch(audio_dim, hidden_dim, dropout=dropout)
        self.audio_big_branch = PairFeatureBranch(audio_big_dim, hidden_dim, dropout=dropout)
        self.video_branch = PairTemporalBranch(video_dim, hidden_dim, dropout=dropout)
        self.gait_branch = SeqBranch(gait_dim, hidden_dim, dropout=dropout)
        self.motion_stat_branch = MLP(motion_stat_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.p_struct_branch = MLP(p_struct_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.p_embed_branch = MLP(p_embed_dim, max(hidden_dim, p_embed_bottleneck * 2), p_embed_bottleneck, dropout=dropout)
        self.p_extra_branch = MLP(p_extra_dim, hidden_dim, hidden_dim, dropout=dropout) if p_extra_dim > 0 else None
        p_cat_dim = hidden_dim + p_embed_bottleneck + (hidden_dim if p_extra_dim > 0 else 0)
        self.p_comb = nn.Sequential(nn.Linear(p_cat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.cross_fusion = PairMutualCrossFusion(hidden_dim, dropout=dropout, use_attn=False)
        self.pair_pool = MaskedAttentionPool(hidden_dim)
        self.p_aux_gate = nn.Sequential(nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.p_aux_delta = nn.Sequential(nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.shared_pool = MaskedAttentionPool(hidden_dim)
        self.private_gate = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.token_names = ["P", "Shared", "A_priv", "ABig_priv", "V_priv", "AV_priv", "G_priv", "MS_priv"]
        self.mod_emb = nn.Parameter(torch.randn(len(self.token_names), hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=max(1, min(4, hidden_dim // 32)), dim_feedforward=hidden_dim * 2, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.mod_tr = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.mod_pool = MaskedAttentionPool(hidden_dim)
        self.metric_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.p_prior_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.res_head3 = CosineClassifier(hidden_dim, num_classes3, init_scale=12.0)
        self.res_head2 = CosineClassifier(hidden_dim, 2, init_scale=12.0)
        self.p_head3 = CosineClassifier(hidden_dim, num_classes3, init_scale=10.0)
        self.p_head2 = CosineClassifier(hidden_dim, 2, init_scale=10.0)
        self.reg = nn.Linear(hidden_dim, 1); self.p_reg = nn.Linear(hidden_dim, 1)
        residual_init = max(1e-4, min(0.95, float(residual_init)))
        self.residual_alpha_logit = nn.Parameter(torch.tensor(math.log(residual_init / (1.0 - residual_init)), dtype=torch.float32))

    def _token_dropout(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0: return tokens, token_mask
        B, L, H = tokens.shape; keep = torch.ones((B, L), device=tokens.device, dtype=torch.float32)
        drop = torch.rand((B, L - 1), device=tokens.device) < self.modality_dropout
        keep[:, 1:] = (~drop).float(); keep = torch.maximum(keep, 1.0 - token_mask)
        tokens = tokens * keep.unsqueeze(-1); token_mask = token_mask * keep; token_mask[:, 0] = 1.0
        return tokens, token_mask

    def _p_guided(self, p_tok: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        B, L, H = aux.shape; p = p_tok.unsqueeze(1).expand(B, L, H)
        z = torch.cat([p, aux, p - aux, p * aux], dim=-1)
        return aux + 0.15 * self.p_aux_gate(z) * self.p_aux_delta(z)

    def _shared_private(self, aux: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared_pool(aux, mask); shared_expand = shared.unsqueeze(1).expand_as(aux)
        g = self.private_gate(torch.cat([aux, shared_expand], dim=-1))
        return shared, aux - 0.35 * g * shared_expand

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = batch["p_struct"].shape[0]; device = batch["p_struct"].device
        p_parts = [self.p_struct_branch(batch["p_struct"]), self.p_embed_branch(batch["p_embed"])]
        if self.p_extra_branch is not None: p_parts.append(self.p_extra_branch(batch["p_extra"]))
        p_tok = self.p_comb(torch.cat(p_parts, dim=-1))
        a_tok, a_pair = self.audio_branch(batch["audio"], batch["audio_pair_mask"])
        abig_tok, _ = self.audio_big_branch(batch["audio_big"], batch["audio_big_pair_mask"])
        v_tok, v_pair = self.video_branch(batch["video"], batch["video_pair_mask"])
        g_tok = self.gait_branch(batch["gait"]); ms_tok = self.motion_stat_branch(batch["motion_stat"])
        pair_mask_cross = batch["audio_pair_mask"] * batch["video_pair_mask"]
        av_tok = self.pair_pool(self.cross_fusion(a_pair, v_pair, pair_mask_cross), pair_mask_cross)
        aux = torch.stack([a_tok, abig_tok, v_tok, av_tok, g_tok, ms_tok], dim=1)
        aux_mask = torch.ones((B, 6), dtype=torch.float32, device=device)
        aux_mask[:, 0] = (batch["audio_pair_mask"].sum(dim=1) > 0).float()
        aux_mask[:, 1] = (batch["audio_big_pair_mask"].sum(dim=1) > 0).float() if self.audio_big_branch.enabled else 0.0
        aux_mask[:, 2] = (batch["video_pair_mask"].sum(dim=1) > 0).float()
        aux_mask[:, 3] = (pair_mask_cross.sum(dim=1) > 0).float()
        aux_mask[:, 4] = 1.0 if self.gait_branch.enabled else 0.0
        aux_mask[:, 5] = 1.0 if self.motion_stat_branch.enabled else 0.0
        aux = self._p_guided(p_tok, aux); shared, private = self._shared_private(aux, aux_mask)
        tokens = torch.cat([p_tok.unsqueeze(1), shared.unsqueeze(1), private], dim=1)
        token_mask = torch.cat([torch.ones((B, 1), device=device), torch.ones((B, 1), device=device), aux_mask], dim=1)
        tokens = tokens + self.mod_emb.unsqueeze(0); tokens, token_mask = self._token_dropout(tokens, token_mask)
        h = self.metric_proj(self.mod_pool(self.mod_tr(tokens, src_key_padding_mask=(token_mask <= 0)), token_mask))
        p_h = self.p_prior_proj(p_tok)
        res3, res2 = self.res_head3(h), self.res_head2(h); p3, p2 = self.p_head3(p_h), self.p_head2(p_h)
        alpha = torch.sigmoid(self.residual_alpha_logit) if self.use_p_prior else 1.0
        logits3 = p3 + alpha * res3 if self.use_p_prior else res3
        logits2 = p2 + alpha * res2 if self.use_p_prior else res2
        p2_dep_logit = logits2[:, 1] - logits2[:, 0]
        sev_logit = logits3[:, 2] - torch.logsumexp(logits3[:, :2], dim=1)
        ord_logits = torch.stack([p2_dep_logit, sev_logit], dim=1)
        phq_log = 0.5 * self.p_reg(p_h).squeeze(-1) + 0.5 * self.reg(h).squeeze(-1)
        return {"logits3": logits3, "logits2": logits2, "ord_logits": ord_logits, "phq_log": phq_log}

# -----------------------------------------------------------------------------
# Metrics / losses
# -----------------------------------------------------------------------------

def concordance_ccc_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2:
        return 0.0
    mt, mp = y_true.mean(), y_pred.mean()
    vt, vp = y_true.var(), y_pred.var()
    cov = np.mean((y_true - mt) * (y_pred - mp))
    den = vt + vp + (mt - mp) ** 2
    if den <= EPS:
        return 0.0
    return float(2 * cov / den)


def ccc_loss_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true = y_true.float()
    y_pred = y_pred.float()
    mt = y_true.mean()
    mp = y_pred.mean()
    vt = ((y_true - mt) ** 2).mean()
    vp = ((y_pred - mp) ** 2).mean()
    cov = ((y_true - mt) * (y_pred - mp)).mean()
    ccc = 2 * cov / (vt + vp + (mt - mp) ** 2 + EPS)
    return 1.0 - ccc


def compute_metrics(y2: np.ndarray, p2: np.ndarray, y3: np.ndarray, p3: np.ndarray, phq: np.ndarray, phq_pred: np.ndarray) -> Dict[str, float]:
    return {
        "binary_acc": float(accuracy_score(y2, p2)),
        "binary_macro_f1": float(f1_score(y2, p2, average="macro", zero_division=0)),
        "binary_kappa": float(cohen_kappa_score(y2, p2) if len(np.unique(y2)) > 1 and len(np.unique(p2)) > 1 else 0.0),
        "ternary_acc": float(accuracy_score(y3, p3)),
        "ternary_macro_f1": float(f1_score(y3, p3, average="macro", zero_division=0)),
        "ternary_kappa": float(cohen_kappa_score(y3, p3) if len(np.unique(y3)) > 1 and len(np.unique(p3)) > 1 else 0.0),
        "phq_ccc": concordance_ccc_np(phq, phq_pred),
        "phq_rmse": float(math.sqrt(mean_squared_error(phq, phq_pred))),
        "phq_mae": float(mean_absolute_error(phq, phq_pred)),
    }


def validation_score(metrics: Dict[str, float]) -> float:
    # v5 selection score: emphasize classification and kappa.
    # Binary was the main blind-test bottleneck, but ternary still matters.
    return (
        0.30 * metrics["ternary_macro_f1"]
        + 0.25 * metrics["binary_macro_f1"]
        + 0.15 * metrics["ternary_kappa"]
        + 0.15 * metrics["binary_kappa"]
        + 0.15 * max(metrics["phq_ccc"], -1.0)
    )


def class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def expected_phq_by_class(train_df: pd.DataFrame) -> Dict[int, float]:
    return {int(k): float(v) for k, v in train_df.groupby("label3")["PHQ-9"].mean().to_dict().items()}



def binary_ternary_consistency_loss(logits2: torch.Tensor, logits3: torch.Tensor) -> torch.Tensor:
    """Encourage binary depressed prob to agree with ternary mild+severe prob."""
    p2_dep = torch.softmax(logits2, dim=-1)[:, 1]
    p3_dep = torch.softmax(logits3, dim=-1)[:, 1:].sum(dim=-1)
    return F.mse_loss(p2_dep, p3_dep)


def ordinal_cumulative_loss_from_logits3(logits3: torch.Tensor, y3: torch.Tensor) -> torch.Tensor:
    """Ordinal target for ternary: [y>0, y>1].
    Uses cumulative probabilities derived from the ternary softmax.
    This adds severity order without replacing the original ternary CE.
    """
    prob3 = torch.softmax(logits3, dim=-1)
    p_gt0 = torch.clamp(prob3[:, 1] + prob3[:, 2], 1e-5, 1.0 - 1e-5)
    p_gt1 = torch.clamp(prob3[:, 2], 1e-5, 1.0 - 1e-5)
    logits_ord = torch.logit(torch.stack([p_gt0, p_gt1], dim=1))
    target = torch.stack([(y3 > 0).float(), (y3 > 1).float()], dim=1)
    return F.binary_cross_entropy_with_logits(logits_ord, target)



def soft_macro_f1_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-7) -> torch.Tensor:
    """Differentiable macro-F1 loss using soft confusion statistics."""
    probs = torch.softmax(logits, dim=-1)
    y = F.one_hot(target.long(), num_classes=num_classes).float()
    tp = (probs * y).sum(dim=0)
    fp = (probs * (1.0 - y)).sum(dim=0)
    fn = ((1.0 - probs) * y).sum(dim=0)
    f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    return 1.0 - f1.mean()


def soft_cohen_kappa_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-7) -> torch.Tensor:
    """Differentiable unweighted Cohen-kappa loss."""
    probs = torch.softmax(logits, dim=-1)
    y = F.one_hot(target.long(), num_classes=num_classes).float()
    conf = y.t().matmul(probs)  # rows=true, cols=pred
    n = conf.sum().clamp_min(eps)
    po = torch.diag(conf).sum() / n
    true_hist = conf.sum(dim=1)
    pred_hist = conf.sum(dim=0)
    pe = (true_hist * pred_hist).sum() / (n * n + eps)
    kappa = (po - pe) / (1.0 - pe + eps)
    return 1.0 - kappa

# -----------------------------------------------------------------------------
# Train / predict
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    train_data_root: str
    train_split_csv: str
    p_struct_train_csv: str
    p_embed_npy: str
    motion_train_npz: str
    output_dir: str
    audio_big_npz: str = ""
    p_extra_csv: str = ""
    model_arch: str = "lite"
    audio_features: str = "wav2vec,opensmile"
    official_video_features: str = ""
    use_gait: int = 1
    target_t: int = TARGET_T
    folds: int = 5
    seeds: str = "42,43,44"
    epochs: int = 80
    patience: int = 14
    batch_size: int = 8
    lr: float = 7e-4
    weight_decay: float = 2e-3
    hidden_dim: int = 96
    p_embed_bottleneck: int = 48
    dropout: float = 0.45
    modality_dropout: float = 0.18
    reg_weight: float = 0.30
    ccc_weight: float = 0.08
    binary_weight: float = 0.50
    consistency_weight: float = 0.00
    ordinal_weight: float = 0.00
    soft_f1_weight: float = 0.00
    kappa_weight: float = 0.00
    class_weight_power: float = 1.00
    grad_clip: float = 1.0
    device: str = "cuda"
    num_workers: int = 0


def build_rows_for_ids(df: pd.DataFrame, ids: Sequence[int]) -> pd.DataFrame:
    id_col = "ID" if "ID" in df.columns else "id"
    return df[df[id_col].apply(to_int_id).isin(set(map(int, ids)))].copy().reset_index(drop=True)


def build_fold_scalers(rows: pd.DataFrame, p_map: Dict[int, np.ndarray], m_map: Dict[int, Dict[str, np.ndarray]]) -> FoldScalers:
    ids = [to_int_id(x) for x in rows["ID" if "ID" in rows.columns else "id"].tolist()]
    # infer dims
    p_dim = len(next(iter(p_map.values()))) if p_map else 0
    m_dim = len(next(iter(m_map.values()))["stat"]) if m_map else 0
    p_mat = np.stack([p_map.get(pid, np.zeros(p_dim, dtype=np.float32)) for pid in ids], axis=0) if p_dim else np.zeros((len(ids), 0), dtype=np.float32)
    m_mat = np.stack([m_map.get(pid, {"stat": np.zeros(m_dim, dtype=np.float32)})["stat"] for pid in ids], axis=0) if m_dim else np.zeros((len(ids), 0), dtype=np.float32)
    return FoldScalers.fit(p_mat, m_mat)


def train_one_model(
    cfg: TrainConfig,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    p_map: Dict[int, np.ndarray],
    p_embed_map: Dict[int, np.ndarray],
    m_map: Dict[int, Dict[str, np.ndarray]],
    audio_big_map: Optional[Dict[int, Dict[str, np.ndarray]]],
    p_extra_map: Optional[Dict[int, np.ndarray]],
    fold: int,
    seed: int,
    exp_dir: Path,
) -> Dict[str, Any]:
    seed_everything(seed)
    device = torch.device(cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu")
    scalers = build_fold_scalers(train_rows, p_map, m_map)
    audio_features = parse_feature_list(cfg.audio_features)
    official_video_features = parse_feature_list(cfg.official_video_features)

    train_ds = ElderV3Dataset(
        train_rows, cfg.train_data_root, p_map, p_embed_map, m_map,
        audio_big_map=audio_big_map, p_extra_map=p_extra_map, scalers=scalers,
        audio_features=audio_features, official_video_features=official_video_features,
        use_gait=bool(cfg.use_gait), target_t=cfg.target_t, train_mode=True,
    )
    val_ds = ElderV3Dataset(
        val_rows, cfg.train_data_root, p_map, p_embed_map, m_map,
        audio_big_map=audio_big_map, p_extra_map=p_extra_map, scalers=scalers,
        audio_features=audio_features, official_video_features=official_video_features,
        use_gait=bool(cfg.use_gait), target_t=cfg.target_t, train_mode=False,
    )
    dims = train_ds.dims
    print(f"[fold={fold} seed={seed}] dims={dims} train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate)

    model = build_model(dims, cfg).to(device)

    w3 = torch.pow(class_weights(train_rows["label3"].astype(int).tolist(), 3).to(device), cfg.class_weight_power)
    w2 = torch.pow(class_weights(train_rows["label2"].astype(int).tolist(), 2).to(device), cfg.class_weight_power)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(5, cfg.epochs), eta_min=cfg.lr * 0.05)

    best_score = -1e9
    best_payload: Optional[Dict[str, Any]] = None
    best_metrics: Dict[str, float] = {}
    no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_meter: List[float] = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            y3 = batch["label3"]
            y2 = batch["label2"]
            phq = batch["phq"].float()
            phq_log = torch.log1p(torch.clamp(phq, min=0.0))
            loss3 = F.cross_entropy(out["logits3"], y3, weight=w3, label_smoothing=0.03)
            loss2 = F.cross_entropy(out["logits2"], y2, weight=w2, label_smoothing=0.03)
            loss_reg = F.smooth_l1_loss(out["phq_log"], phq_log)
            phq_pred_raw = torch.expm1(out["phq_log"]).clamp(0, 27)
            loss_ccc = ccc_loss_torch(phq, phq_pred_raw)
            loss_cons = binary_ternary_consistency_loss(out["logits2"], out["logits3"])
            if "ord_logits" in out:
                target_ord = torch.stack([(y3 > 0).float(), (y3 > 1).float()], dim=1)
                loss_ord = F.binary_cross_entropy_with_logits(out["ord_logits"], target_ord)
            else:
                loss_ord = ordinal_cumulative_loss_from_logits3(out["logits3"], y3)

            loss_f1 = (
                soft_macro_f1_loss(out["logits2"], y2, 2)
                + soft_macro_f1_loss(out["logits3"], y3, 3)
            )
            loss_kappa = (
                soft_cohen_kappa_loss(out["logits2"], y2, 2)
                + soft_cohen_kappa_loss(out["logits3"], y3, 3)
            )

            loss = (
                loss3
                + cfg.binary_weight * loss2
                + cfg.reg_weight * loss_reg
                + cfg.ccc_weight * loss_ccc
                + cfg.consistency_weight * loss_cons
                + cfg.ordinal_weight * loss_ord
                + cfg.soft_f1_weight * loss_f1
                + cfg.kappa_weight * loss_kappa
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            loss_meter.append(float(loss.item()))
        sched.step()

        metrics, _ = evaluate_model(model, val_loader, device)
        score = validation_score(metrics)
        row = {"epoch": epoch, "loss": float(np.mean(loss_meter)), "score": score, **metrics}
        history.append(row)
        print(f"[fold={fold} seed={seed}] epoch={epoch:03d} loss={row['loss']:.4f} score={score:.4f} tF1={metrics['ternary_macro_f1']:.4f} bF1={metrics['binary_macro_f1']:.4f} CCC={metrics['phq_ccc']:.4f}")
        if score > best_score + 1e-6:
            best_score = score
            best_metrics = metrics
            no_improve = 0
            best_payload = {
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "dims": dims,
                "scalers": scalers.to_dict(),
                "config": asdict(cfg),
                "fold": fold,
                "seed": seed,
                "best_score": best_score,
                "best_metrics": best_metrics,
                "token_names": getattr(model, "token_names", []),
                "expected_phq_by_class": expected_phq_by_class(train_rows),
            }
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"[fold={fold} seed={seed}] early stop at epoch={epoch}, best_score={best_score:.4f}")
                break

    if best_payload is None:
        raise RuntimeError("No checkpoint payload created.")
    ckpt_path = exp_dir / "checkpoints" / f"fold{fold}_seed{seed}.pt"
    ensure_dir(ckpt_path.parent)
    torch.save(best_payload, ckpt_path)
    hist_path = exp_dir / "checkpoints" / f"fold{fold}_seed{seed}_history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"[OK] saved {ckpt_path}")
    return {"checkpoint": str(ckpt_path), "fold": fold, "seed": seed, "best_score": best_score, **best_metrics}


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    model.eval()
    ids: List[int] = []
    y2s: List[int] = []
    y3s: List[int] = []
    phqs: List[float] = []
    prob2s: List[np.ndarray] = []
    prob3s: List[np.ndarray] = []
    phq_preds: List[float] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        prob2 = torch.softmax(out["logits2"], dim=-1).detach().cpu().numpy()
        prob3 = torch.softmax(out["logits3"], dim=-1).detach().cpu().numpy()
        phq_pred = torch.expm1(out["phq_log"]).clamp(0, 27).detach().cpu().numpy()
        ids.extend(batch["id"].detach().cpu().numpy().astype(int).tolist())
        y2s.extend(batch["label2"].detach().cpu().numpy().astype(int).tolist())
        y3s.extend(batch["label3"].detach().cpu().numpy().astype(int).tolist())
        phqs.extend(batch["phq"].detach().cpu().numpy().astype(float).tolist())
        prob2s.append(prob2)
        prob3s.append(prob3)
        phq_preds.extend(phq_pred.astype(float).tolist())
    prob2 = np.concatenate(prob2s, axis=0)
    prob3 = np.concatenate(prob3s, axis=0)
    # Combine independent binary with ternary-derived binary.
    derived2_pos = prob3[:, 1] + prob3[:, 2]
    final2_pos = 0.55 * prob2[:, 1] + 0.45 * derived2_pos
    pred2 = (final2_pos >= 0.5).astype(int)
    pred3 = prob3.argmax(axis=1).astype(int)
    metrics = compute_metrics(np.asarray(y2s), pred2, np.asarray(y3s), pred3, np.asarray(phqs), np.asarray(phq_preds))
    payload = {
        "ids": np.asarray(ids, dtype=int),
        "prob2": prob2,
        "prob3": prob3,
        "pred2": pred2,
        "pred3": pred3,
        "phq_pred": np.asarray(phq_preds, dtype=np.float32),
        "y2": np.asarray(y2s, dtype=int),
        "y3": np.asarray(y3s, dtype=int),
        "phq": np.asarray(phqs, dtype=np.float32),
    }
    return metrics, payload


def train_command(args: argparse.Namespace) -> None:
    # argparse Namespace also contains internal keys like cmd/func.
    # Keep only fields defined in TrainConfig.
    cfg_kwargs = {k: v for k, v in vars(args).items() if k in TrainConfig.__dataclass_fields__}
    cfg = TrainConfig(**cfg_kwargs)
    exp_dir = ensure_dir(cfg.output_dir)
    write_json(asdict(cfg), exp_dir / "train_config.json")
    df = read_csv_auto(cfg.train_split_csv)
    required = {"ID", "label2", "label3", "PHQ-9"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"train_split_csv needs columns {required}, got {df.columns.tolist()}")
    df = df.copy()
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower().eq("train")].copy()
    df = df.sort_values("ID").reset_index(drop=True)

    p_map, p_cols = read_struct_features(cfg.p_struct_train_csv)
    p_embed_map = load_personality_embedding(cfg.p_embed_npy)
    m_map = read_motion_npz(cfg.motion_train_npz)
    audio_big_map = read_audio_big_npz(cfg.audio_big_npz)
    p_extra_map, p_extra_cols = read_extra_p_csv(cfg.p_extra_csv)

    print(f"[INFO] train samples={len(df)} label3={df['label3'].value_counts().to_dict()} label2={df['label2'].value_counts().to_dict()}")
    print(f"[INFO] P_struct={len(p_map)} cols={len(p_cols)} P_embed={len(p_embed_map)} motion={len(m_map)} audio_big={len(audio_big_map)} p_extra={len(p_extra_map)} cols={len(p_extra_cols)}")

    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=2026)
    seeds = [int(x) for x in cfg.seeds.split(",") if x.strip()]
    summary_rows: List[Dict[str, Any]] = []
    ids_all = df["ID"].astype(int).values
    y = df["label3"].astype(int).values
    for fold, (tr_idx, va_idx) in enumerate(skf.split(ids_all, y), start=1):
        train_rows = df.iloc[tr_idx].reset_index(drop=True)
        val_rows = df.iloc[va_idx].reset_index(drop=True)
        for seed in seeds:
            res = train_one_model(cfg, train_rows, val_rows, p_map, p_embed_map, m_map, audio_big_map, p_extra_map, fold, seed, exp_dir)
            summary_rows.append(res)
            pd.DataFrame(summary_rows).to_csv(exp_dir / "cv_summary.csv", index=False)
    print(f"[OK] all done. summary saved to {exp_dir / 'cv_summary.csv'}")


def load_checkpoint_model(ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, FoldScalers, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    model = build_model(ckpt["dims"], config, for_predict=True)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    scalers = FoldScalers.from_dict(ckpt["scalers"])
    return model, scalers, ckpt


@torch.no_grad()
def predict_one_checkpoint(
    ckpt_path: Path,
    test_rows: pd.DataFrame,
    test_data_root: str | Path,
    p_map: Dict[int, np.ndarray],
    p_embed_map: Dict[int, np.ndarray],
    m_map: Dict[int, Dict[str, np.ndarray]],
    audio_big_map: Optional[Dict[int, Dict[str, np.ndarray]]],
    p_extra_map: Optional[Dict[int, np.ndarray]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, np.ndarray]:
    model, scalers, ckpt = load_checkpoint_model(ckpt_path, device)
    cfg = ckpt["config"]
    ds = ElderV3Dataset(
        test_rows, test_data_root, p_map, p_embed_map, m_map,
        audio_big_map=audio_big_map, p_extra_map=p_extra_map, scalers=scalers,
        audio_features=parse_feature_list(cfg.get("audio_features", "wav2vec,opensmile")),
        official_video_features=parse_feature_list(cfg.get("official_video_features", "")),
        use_gait=bool(cfg.get("use_gait", 1)), target_t=int(cfg.get("target_t", TARGET_T)), train_mode=False,
    )

    # IMPORTANT:
    # In blind test, official feature folders may expose a different inferred C
    # from train, e.g. audio 833 vs checkpoint 793.
    # The model was trained with ckpt["dims"], so prediction dataset must
    # pad/truncate every modality to the checkpoint dimensions.
    ds._dim_cache = {k: int(v) for k, v in ckpt["dims"].items()}

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    ids: List[int] = []
    prob2s: List[np.ndarray] = []
    prob3s: List[np.ndarray] = []
    phq_preds: List[np.ndarray] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        ids.extend(batch["id"].detach().cpu().numpy().astype(int).tolist())
        prob2s.append(torch.softmax(out["logits2"], dim=-1).detach().cpu().numpy())
        prob3s.append(torch.softmax(out["logits3"], dim=-1).detach().cpu().numpy())
        phq_preds.append(torch.expm1(out["phq_log"]).clamp(0, 27).detach().cpu().numpy())
    return {
        "ids": np.asarray(ids, dtype=int),
        "prob2": np.concatenate(prob2s, axis=0),
        "prob3": np.concatenate(prob3s, axis=0),
        "phq_pred": np.concatenate(phq_preds, axis=0).astype(np.float32),
    }


def package_submission(ids: np.ndarray, pred2: np.ndarray, pred3: np.ndarray, phq_pred: np.ndarray, output_dir: str | Path) -> None:
    out = ensure_dir(output_dir)
    order = np.argsort(ids.astype(int))
    ids = ids[order].astype(int)
    pred2 = pred2[order].astype(int)
    pred3 = pred3[order].astype(int)
    phq_pred = np.clip(phq_pred[order].astype(float), 0, 27)
    binary_df = pd.DataFrame({"id": ids, "binary_pred": pred2, "phq9_pred": phq_pred})
    ternary_df = pd.DataFrame({"id": ids, "ternary_pred": pred3, "phq9_pred": phq_pred})
    binary_path = out / "binary.csv"
    ternary_path = out / "ternary.csv"
    zip_path = out / "submission.zip"
    binary_df.to_csv(binary_path, index=False, encoding="utf-8")
    ternary_df.to_csv(ternary_path, index=False, encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(binary_path, arcname="binary.csv")
        zf.write(ternary_path, arcname="ternary.csv")
    print(f"[OK] saved {binary_path}")
    print(f"[OK] saved {ternary_path}")
    print(f"[OK] saved {zip_path}")


def predict_command(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    test_df = read_csv_auto(args.test_split_csv)
    if "ID" not in test_df.columns and "id" not in test_df.columns:
        raise ValueError("test_split_csv needs ID/id column")
    id_col = "ID" if "ID" in test_df.columns else "id"
    # Add dummy labels for dataset only; not used for submission.
    test_rows = pd.DataFrame({"ID": test_df[id_col].apply(to_int_id).astype(int).tolist()})
    test_rows["label2"] = 0
    test_rows["label3"] = 0
    test_rows["PHQ-9"] = 0.0

    p_map, _ = read_struct_features(args.p_struct_test_csv)
    p_embed_map = load_personality_embedding(args.p_embed_npy)
    m_map = read_motion_npz(args.motion_test_npz)
    audio_big_map = read_audio_big_npz(getattr(args, "audio_big_npz", ""))
    p_extra_map, _ = read_extra_p_csv(getattr(args, "p_extra_csv", ""))

    ckpt_paths: List[Path] = []
    if args.checkpoints:
        for x in args.checkpoints.split(","):
            if x.strip():
                ckpt_paths.append(Path(x.strip()))
    if args.checkpoint_dir:
        ckpt_paths.extend(sorted(Path(args.checkpoint_dir).glob("fold*_seed*.pt")))
    ckpt_paths = [p for p in ckpt_paths if p.exists()]
    if not ckpt_paths:
        raise RuntimeError("No checkpoint found. Provide --checkpoint_dir or --checkpoints")
    print(f"[INFO] predicting with {len(ckpt_paths)} checkpoints")

    all_prob2: List[np.ndarray] = []
    all_prob3: List[np.ndarray] = []
    all_phq: List[np.ndarray] = []
    base_ids: Optional[np.ndarray] = None
    for p in tqdm(ckpt_paths, desc="predict checkpoints"):
        pred = predict_one_checkpoint(
            p, test_rows, args.test_data_root, p_map, p_embed_map, m_map,
            audio_big_map=audio_big_map, p_extra_map=p_extra_map,
            device=device, batch_size=args.batch_size, num_workers=args.num_workers,
        )
        if base_ids is None:
            base_ids = pred["ids"]
        else:
            if not np.array_equal(base_ids, pred["ids"]):
                raise RuntimeError(f"ID order mismatch for checkpoint {p}")
        all_prob2.append(pred["prob2"])
        all_prob3.append(pred["prob3"])
        all_phq.append(pred["phq_pred"])

    assert base_ids is not None
    prob2 = np.mean(all_prob2, axis=0)
    prob3 = np.mean(all_prob3, axis=0)
    phq_pred = np.mean(all_phq, axis=0)
    derived2_pos = prob3[:, 1] + prob3[:, 2]
    final2_pos = 0.55 * prob2[:, 1] + 0.45 * derived2_pos
    pred2 = (final2_pos >= args.binary_threshold).astype(int)
    pred3 = prob3.argmax(axis=1).astype(int)

    out = ensure_dir(args.output_dir)
    np.savez_compressed(out / "raw_test_predictions.npz", ids=base_ids, prob2=prob2, prob3=prob3, phq_pred=phq_pred, final2_pos=final2_pos)
    package_submission(base_ids, pred2, pred3, phq_pred, out)


def dummy_command(args: argparse.Namespace) -> None:
    df = read_csv_auto(args.test_split_csv)
    id_col = "ID" if "ID" in df.columns else ("id" if "id" in df.columns else df.columns[0])
    ids = np.asarray([to_int_id(x) for x in df[id_col].tolist()], dtype=int)
    # conservative dummy: all normal with low PHQ; only intended to validate format.
    pred2 = np.full(len(ids), int(args.binary_pred), dtype=int)
    pred3 = np.full(len(ids), int(args.ternary_pred), dtype=int)
    phq = np.full(len(ids), float(args.phq9_pred), dtype=np.float32)
    package_submission(ids, pred2, pred3, phq, args.output_dir)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MPDD-AVG Elder DepFormerAVP-v3-lite")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parse-desc", help="parse Elder descriptions.csv")
    p.add_argument("--train_desc", default="")
    p.add_argument("--test_desc", default="")
    p.add_argument("--output_dir", required=True)
    p.set_defaults(func=parse_desc_command)

    p = sub.add_parser("extract-video", help="extract raw video motion features")
    p.add_argument("--video_root", required=True, help="raw Elder video root, e.g. .../privacy-constrained-raw-Elder-train/video")
    p.add_argument("--split_csv", required=True)
    p.add_argument("--split_name", default="train")
    p.add_argument("--output_npz", required=True)
    p.add_argument("--target_t", type=int, default=TARGET_T)
    p.add_argument("--pair_count", type=int, default=PAIR_COUNT)
    p.add_argument("--resize", type=int, default=160)
    p.set_defaults(func=extract_video_command)

    p = sub.add_parser("train", help="train Elder v3-lite CV ensemble")
    p.add_argument("--train_data_root", required=True)
    p.add_argument("--train_split_csv", required=True)
    p.add_argument("--p_struct_train_csv", required=True)
    p.add_argument("--p_embed_npy", required=True)
    p.add_argument("--motion_train_npz", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--audio_big_npz", default="")
    p.add_argument("--p_extra_csv", default="")
    p.add_argument("--model_arch", default="lite", choices=['lite', 'p_anchor_v4', 'anchor', 'p_anchor', 'v5_hier_ord', 'v5', 'hier_ord', 'v6_cross_motion', 'v6_cross', 'cross_motion', 'v6_cross_no_pgate', 'v6_no_pgate', 'v6_cross_no_crossgate', 'v6_no_crossgate', 'v7_res_hier', 'v7_hier', 'v7', 'v9_depformerv2', 'v9', 'depformerv2', 'v9_no_pguide', 'v9_no_p', 'v9_no_cross', 'v9_nocross', 'v9_no_sp', 'v9_nosp', 'v10_p_prior', 'v10_b', 'p_prior_residual', 'v10_audio_p_prior', 'v10_c', 'audio_p_prior'])
    p.add_argument("--audio_features", default="wav2vec,opensmile")
    p.add_argument("--official_video_features", default="")
    p.add_argument("--use_gait", type=int, default=1)
    p.add_argument("--target_t", type=int, default=TARGET_T)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=14)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight_decay", type=float, default=2e-3)
    p.add_argument("--hidden_dim", type=int, default=96)
    p.add_argument("--p_embed_bottleneck", type=int, default=48)
    p.add_argument("--dropout", type=float, default=0.45)
    p.add_argument("--modality_dropout", type=float, default=0.18)
    p.add_argument("--reg_weight", type=float, default=0.30)
    p.add_argument("--ccc_weight", type=float, default=0.08)
    p.add_argument("--binary_weight", type=float, default=0.50)
    p.add_argument("--consistency_weight", type=float, default=0.00)
    p.add_argument("--ordinal_weight", type=float, default=0.00)
    p.add_argument("--soft_f1_weight", type=float, default=0.00)
    p.add_argument("--kappa_weight", type=float, default=0.00)
    p.add_argument("--class_weight_power", type=float, default=1.00)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=0)
    p.set_defaults(func=train_command)

    p = sub.add_parser("predict", help="predict blind test and package submission")
    p.add_argument("--test_data_root", required=True)
    p.add_argument("--test_split_csv", required=True)
    p.add_argument("--p_struct_test_csv", required=True)
    p.add_argument("--p_embed_npy", required=True, help="official descriptions_embeddings_with_ids.npy; use same full train/test file if official provides one")
    p.add_argument("--motion_test_npz", required=True)
    p.add_argument("--audio_big_npz", default="")
    p.add_argument("--p_extra_csv", default="")
    p.add_argument("--checkpoint_dir", default="")
    p.add_argument("--checkpoints", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--binary_threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=0)
    p.set_defaults(func=predict_command)

    p = sub.add_parser("dummy", help="make valid dummy submission from test IDs")
    p.add_argument("--test_split_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--binary_pred", type=int, default=0)
    p.add_argument("--ternary_pred", type=int, default=0)
    p.add_argument("--phq9_pred", type=float, default=3.0)
    p.set_defaults(func=dummy_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
