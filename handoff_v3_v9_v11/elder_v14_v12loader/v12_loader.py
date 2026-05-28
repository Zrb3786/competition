#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean MPDD-AVG Elder v12 trainer/predictor.

Design goal:
  - Use v3/v9 style official feature loading:
      Elder/Audio/{wav2vec2,opensmile,...}/{ID}/A_1.npy ... A_4.npy
      Elder/Video/{openface,resnet,densenet}/{ID}/V_1.npy ... V_4.npy
      Elder/IMU/{ID}/{ID}.npy or Elder/IMU/{ID}.npy
      descriptions_embeddings_with_ids.npy
      raw motion npz
  - Use v11 style new feature loading:
      A_big npz: ids + audio_big_pair + pair_mask
      VBeh npz: ids + motion_extra_pair + motion_extra_stat + pair_mask
      GUnit npz: ids + gait_extra
      P_extra csv: id + numeric columns
  - Only keep v12 model, train, predict, inspect, dummy.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold

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


def parse_feature_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s or s.lower() in {"none", "null", "-"}:
        return []
    return [x.strip() for x in s.split(",") if x.strip() and x.strip().lower() not in {"none", "null", "-"}]


def np_clean(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_per_sample(arr: np.ndarray, clip: float = 6.0) -> np.ndarray:
    arr = np_clean(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    mu = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return np.clip((arr - mu) / std, -clip, clip).astype(np.float32)


def resize_np_time(arr: np.ndarray, target_t: int) -> np.ndarray:
    arr = np_clean(arr)
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


def safe_load_npy(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or (not path.exists()) or path.stat().st_size == 0:
        return None
    try:
        arr = np.load(str(path), allow_pickle=True)
        # Official pair features should be numeric arrays. If object dict, return None here;
        # embedding loader handles dict/object separately.
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return None


def pad_or_truncate_last(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    cur = x.shape[-1] if x.ndim > 0 else 0
    if cur == dim:
        return x.astype(np.float32)
    if cur > dim:
        return x[..., :dim].astype(np.float32)
    pad_shape = list(x.shape)
    pad_shape[-1] = dim - cur
    return np.concatenate([x, np.zeros(pad_shape, dtype=np.float32)], axis=-1).astype(np.float32)

# -----------------------------------------------------------------------------
# v3/v9-style official feature loaders
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
    # fallback recursive, but keep cheap for small Elder data
    if gait_root.exists():
        for c in gait_root.rglob("*.npy"):
            nums = re.findall(r"\d+", c.stem)
            if nums and int(nums[-1]) == pid:
                return c
    return None


def discover_pair_npy(folder: Path, prefix: str, pair_count: int = PAIR_COUNT) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    if not folder.exists():
        return out
    for i in range(1, pair_count + 1):
        for name in (f"{prefix}_{i}.npy", f"{prefix}{i}.npy", f"event_{i}.npy"):
            p = folder / name
            if p.exists():
                out[i] = p
                break
    if out:
        return out
    pat = re.compile(rf"{re.escape(prefix)}[_-]?(\d+).*\.npy$", re.IGNORECASE)
    for p in sorted(folder.rglob("*.npy")):
        m = pat.search(p.name)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= pair_count:
                out[idx] = p
    return out


def infer_pair_feature_dims(data_root: Path, ids: Sequence[int], features: Sequence[str], modality: str) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    prefix = "A" if modality == "audio" else "V"
    for feat in features:
        root = resolve_audio_feature_root(data_root, feat) if modality == "audio" else resolve_video_feature_root(data_root, feat)
        dim = 0
        for pid in ids:
            pair_map = discover_pair_npy(root / str(pid), prefix=prefix)
            for p in pair_map.values():
                arr = safe_load_npy(p)
                if arr is not None and arr.size > 0:
                    if arr.ndim == 1:
                        dim = int(arr.shape[0])
                    else:
                        dim = int(arr.reshape(arr.shape[0], -1).shape[-1])
                    break
            if dim > 0:
                break
        dims[feat] = int(dim)
    return dims


def load_official_pair_features(
    data_root: Path,
    pid: int,
    features: Sequence[str],
    modality: str,
    target_t: int,
    feature_dims: Optional[Dict[str, int]] = None,
    pair_count: int = PAIR_COUNT,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if not features:
        return np.zeros((pair_count, target_t, 0), dtype=np.float32), np.zeros(pair_count, dtype=np.float32), 0
    prefix = "A" if modality == "audio" else "V"
    roots = [resolve_audio_feature_root(data_root, f) if modality == "audio" else resolve_video_feature_root(data_root, f) for f in features]
    dims = []
    pair_maps = []
    for feat, root in zip(features, roots):
        pair_map = discover_pair_npy(root / str(pid), prefix=prefix, pair_count=pair_count)
        pair_maps.append(pair_map)
        dim = int((feature_dims or {}).get(feat, 0))
        if dim <= 0:
            for p in pair_map.values():
                arr = safe_load_npy(p)
                if arr is not None and arr.size > 0:
                    dim = int(arr.shape[0] if arr.ndim == 1 else arr.reshape(arr.shape[0], -1).shape[-1])
                    break
        dims.append(dim)
    total_dim = int(sum(dims))
    if total_dim <= 0:
        return np.zeros((pair_count, target_t, 0), dtype=np.float32), np.zeros(pair_count, dtype=np.float32), 0
    out_pairs: List[np.ndarray] = []
    mask: List[float] = []
    for i in range(1, pair_count + 1):
        chunks = []
        valid_any = False
        for pair_map, dim in zip(pair_maps, dims):
            if dim <= 0:
                continue
            arr = safe_load_npy(pair_map.get(i)) if i in pair_map else None
            if arr is None or arr.size == 0:
                chunks.append(np.zeros((target_t, dim), dtype=np.float32))
                continue
            if arr.ndim == 1:
                arr = arr[None, :]
            arr = arr.reshape(arr.shape[0], -1)
            arr = pad_or_truncate_last(arr, dim)
            arr = normalize_per_sample(arr)
            arr = resize_np_time(arr, target_t)
            chunks.append(arr.astype(np.float32))
            valid_any = True
        out_pairs.append(np.concatenate(chunks, axis=-1) if chunks else np.zeros((target_t, 0), dtype=np.float32))
        mask.append(1.0 if valid_any else 0.0)
    return np.stack(out_pairs, axis=0).astype(np.float32), np.asarray(mask, dtype=np.float32), total_dim


def infer_gait_dim(data_root: Path, ids: Sequence[int]) -> int:
    root = resolve_gait_root(data_root)
    for pid in ids:
        f = resolve_gait_file(root, pid)
        arr = safe_load_npy(f)
        if arr is not None and arr.size > 0:
            if arr.ndim == 1:
                return 1
            return int(min(arr.reshape(arr.shape[0], -1).shape[-1], GAIT_KEEP_DIM))
    return 0


def load_gait_seq(data_root: Path, pid: int, target_t: int, expected_dim: int = 0) -> np.ndarray:
    if expected_dim <= 0:
        return np.zeros((target_t, 0), dtype=np.float32)
    root = resolve_gait_root(data_root)
    f = resolve_gait_file(root, pid)
    arr = safe_load_npy(f)
    if arr is None or arr.size == 0:
        return np.zeros((target_t, expected_dim), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = arr.reshape(arr.shape[0], -1)
    arr = arr[..., :GAIT_KEEP_DIM]
    arr = pad_or_truncate_last(arr, expected_dim)
    arr = normalize_per_sample(arr)
    return resize_np_time(arr, target_t).astype(np.float32)


def _object_npy_to_python(data: Any) -> Any:
    if isinstance(data, np.ndarray) and data.dtype == object:
        if data.shape == ():
            return data.item()
        return [x.item() if isinstance(x, np.ndarray) and x.shape == () else x for x in data.tolist()]
    return data


def load_personality_embedding(path: Optional[str | Path]) -> Dict[int, np.ndarray]:
    if path is None or str(path).strip() == "":
        return {}
    p = Path(path)
    if p.is_dir():
        p = p / "descriptions_embeddings_with_ids.npy"
    if not p.exists():
        print(f"[WARN] personality embedding not found: {p}")
        return {}
    data = _object_npy_to_python(np.load(str(p), allow_pickle=True))
    out: Dict[int, np.ndarray] = {}
    # v3 style: iterable of dict records {id, embedding}
    if isinstance(data, dict):
        keys = set(data.keys())
        id_key = next((k for k in ["ids", "id", "ID", "sample_ids"] if k in keys), None)
        emb_key = next((k for k in ["embeddings", "embedding", "emb", "features", "x"] if k in keys), None)
        if id_key is not None and emb_key is not None:
            ids = np.asarray(data[id_key]).reshape(-1)
            embs = np.asarray(data[emb_key], dtype=np.float32)
            for pid, emb in zip(ids, embs):
                out[to_int_id(pid)] = np.asarray(emb, dtype=np.float32).reshape(-1)
            return out
        # fallback: {id: vector}
        for k, v in data.items():
            try:
                out[to_int_id(k)] = np.asarray(v, dtype=np.float32).reshape(-1)
            except Exception:
                pass
        return out
    if isinstance(data, np.ndarray):
        iterable = data.tolist()
    else:
        iterable = data
    try:
        for item in iterable:
            try:
                if isinstance(item, dict):
                    pid = to_int_id(item.get("id", item.get("ID")))
                    emb = np.asarray(item.get("embedding", item.get("emb", item.get("features"))), dtype=np.float32)
                elif hasattr(item, "dtype") and getattr(item.dtype, "names", None):
                    names = item.dtype.names
                    pid = to_int_id(item["id"] if "id" in names else item["ID"])
                    emb_name = "embedding" if "embedding" in names else ("emb" if "emb" in names else names[-1])
                    emb = np.asarray(item[emb_name], dtype=np.float32)
                else:
                    continue
                out[pid] = emb.reshape(-1).astype(np.float32)
            except Exception:
                continue
    except TypeError:
        pass
    return out

# -----------------------------------------------------------------------------
# raw motion and v11 new feature loaders
# -----------------------------------------------------------------------------

def _pick_npz_key(keys: Sequence[str], candidates: Sequence[str], contains: Optional[Sequence[str]] = None) -> Optional[str]:
    for c in candidates:
        if c in keys:
            return c
    if contains:
        low = [(k, k.lower()) for k in keys]
        for terms in contains:
            terms_l = [t.lower() for t in terms.split("+")]
            for k, kl in low:
                if all(t in kl for t in terms_l):
                    return k
    return None


def read_motion_npz(path: str | Path | None) -> Dict[int, Dict[str, np.ndarray]]:
    if path is None or str(path).strip() == "":
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] raw motion npz not found: {p}")
        return {}
    z = np.load(str(p), allow_pickle=True)
    keys = list(z.files)
    id_key = _pick_npz_key(keys, ["ids", "id", "ID", "sample_ids"])
    pair_key = _pick_npz_key(keys, ["motion_pair", "motion", "video", "video_pair", "raw_motion_pair", "pair"], contains=["motion+pair", "raw+motion", "video+pair"])
    stat_key = _pick_npz_key(keys, ["motion_stat", "stat", "motion_stats", "raw_motion_stat"], contains=["motion+stat"])
    mask_key = _pick_npz_key(keys, ["pair_mask", "motion_pair_mask", "mask", "valid_mask"], contains=["pair+mask"])
    if id_key is None:
        raise ValueError(f"raw motion npz {p} has no ids key. keys={keys}")
    ids = np.asarray(z[id_key]).reshape(-1)
    pair = np.asarray(z[pair_key], dtype=np.float32) if pair_key else None
    stat = np.asarray(z[stat_key], dtype=np.float32) if stat_key else None
    mask = np.asarray(z[mask_key], dtype=np.float32) if mask_key else None
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for idx, pid0 in enumerate(ids):
        pid = to_int_id(pid0)
        rec: Dict[str, np.ndarray] = {}
        if pair is not None:
            arr = np_clean(pair[idx])
            if arr.ndim == 2:  # [T,D] -> [1,T,D]
                arr = arr[None, :, :]
            if arr.ndim == 1:
                arr = arr.reshape(1, 1, -1)
            rec["pair"] = arr.astype(np.float32)
        if stat is not None:
            rec["stat"] = np_clean(stat[idx]).reshape(-1)
        if mask is not None:
            rec["mask"] = np_clean(mask[idx]).reshape(-1)
        out[pid] = rec
    return out


def read_audio_big_npz(path: str | Path | None) -> Dict[int, Dict[str, np.ndarray]]:
    if path is None or str(path).strip() == "":
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] A_big npz not found: {p}")
        return {}
    z = np.load(str(p), allow_pickle=True)
    keys = list(z.files)
    id_key = _pick_npz_key(keys, ["ids", "id", "ID", "sample_ids"])
    pair_key = _pick_npz_key(keys, ["audio_big_pair", "audio_big", "pair", "features"], contains=["audio+big", "big+pair"])
    mask_key = _pick_npz_key(keys, ["pair_mask", "audio_big_pair_mask", "mask", "valid_mask"], contains=["pair+mask"])
    if id_key is None or pair_key is None:
        raise ValueError(f"A_big npz {p} missing ids/audio_big_pair. keys={keys}")
    ids = np.asarray(z[id_key]).reshape(-1)
    pair = np.asarray(z[pair_key], dtype=np.float32)
    mask = np.asarray(z[mask_key], dtype=np.float32) if mask_key else None
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for i, pid0 in enumerate(ids):
        arr = np_clean(pair[i])
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out[to_int_id(pid0)] = {"pair": arr.astype(np.float32), "mask": (np_clean(mask[i]).reshape(-1) if mask is not None else np.ones(arr.shape[0], dtype=np.float32))}
    return out


def read_motion_extra_npz(path: str | Path | None) -> Dict[int, Dict[str, np.ndarray]]:
    if path is None or str(path).strip() == "":
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] motion extra npz not found: {p}")
        return {}
    z = np.load(str(p), allow_pickle=True)
    keys = list(z.files)
    id_key = _pick_npz_key(keys, ["ids", "id", "ID", "sample_ids"])
    pair_key = _pick_npz_key(keys, ["motion_extra_pair", "motion_behavior_pair", "vbeh_pair", "pair", "features_pair"], contains=["motion+extra+pair", "behavior+pair", "vbeh+pair"])
    stat_key = _pick_npz_key(keys, ["motion_extra_stat", "motion_behavior_stat", "vbeh_stat", "stat", "features_stat"], contains=["motion+extra+stat", "behavior+stat", "vbeh+stat"])
    mask_key = _pick_npz_key(keys, ["pair_mask", "motion_extra_pair_mask", "mask", "valid_mask"], contains=["pair+mask"])
    if id_key is None:
        raise ValueError(f"motion extra npz {p} missing ids. keys={keys}")
    ids = np.asarray(z[id_key]).reshape(-1)
    pair = np.asarray(z[pair_key], dtype=np.float32) if pair_key else None
    stat = np.asarray(z[stat_key], dtype=np.float32) if stat_key else None
    mask = np.asarray(z[mask_key], dtype=np.float32) if mask_key else None
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for i, pid0 in enumerate(ids):
        rec: Dict[str, np.ndarray] = {}
        if pair is not None:
            arr = np_clean(pair[i])
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            rec["pair"] = arr.astype(np.float32)
        if stat is not None:
            rec["stat"] = np_clean(stat[i]).reshape(-1)
        if mask is not None:
            rec["mask"] = np_clean(mask[i]).reshape(-1)
        out[to_int_id(pid0)] = rec
    return out


def read_gait_extra_npz(path: str | Path | None) -> Dict[int, np.ndarray]:
    if path is None or str(path).strip() == "":
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] gait extra npz not found: {p}")
        return {}
    z = np.load(str(p), allow_pickle=True)
    keys = list(z.files)
    id_key = _pick_npz_key(keys, ["ids", "id", "ID", "sample_ids"])
    feat_key = _pick_npz_key(keys, ["gait_extra", "gunit", "features", "x"], contains=["gait+extra", "gunit"])
    if id_key is None or feat_key is None:
        raise ValueError(f"gait extra npz {p} missing ids/gait_extra. keys={keys}")
    ids = np.asarray(z[id_key]).reshape(-1)
    x = np.asarray(z[feat_key], dtype=np.float32)
    return {to_int_id(pid): np_clean(x[i]).reshape(-1) for i, pid in enumerate(ids)}


def read_struct_features(path: str | Path | None) -> Tuple[Dict[int, np.ndarray], List[str]]:
    if path is None or str(path).strip() == "":
        return {}, []
    p = Path(path)
    if not p.exists():
        print(f"[WARN] struct csv not found: {p}")
        return {}, []
    df = read_csv_auto(p)
    id_col = "id" if "id" in df.columns else ("ID" if "ID" in df.columns else df.columns[0])
    feat_cols = [c for c in df.columns if c != id_col]
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce")) or True]
    out: Dict[int, np.ndarray] = {}
    for _, r in df.iterrows():
        vals = pd.to_numeric(r[feat_cols], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        out[to_int_id(r[id_col])] = vals.astype(np.float32)
    return out, feat_cols


def read_extra_p_csv(path: str | Path | None) -> Tuple[Dict[int, np.ndarray], List[str]]:
    return read_struct_features(path)

# -----------------------------------------------------------------------------
# Label detection and samples
# -----------------------------------------------------------------------------

def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def prepare_train_rows(split_csv: str | Path) -> pd.DataFrame:
    df = read_csv_auto(split_csv)
    id_col = find_col(df, ["ID", "id", "sample_id", "subject_id"])
    if id_col is None:
        raise ValueError(f"Cannot find ID column in {split_csv}: {df.columns.tolist()}")
    phq_col = find_col(df, ["PHQ-9", "PHQ9", "phq9", "PHQ", "phq", "phq_score", "PHQ_score"])
    y3_col = find_col(df, ["label3", "ternary", "ternary_label", "label_3", "Class3", "class3"])
    y2_col = find_col(df, ["label2", "binary", "binary_label", "label_2", "Class2", "class2"])
    out = pd.DataFrame({"ID": df[id_col].apply(to_int_id).astype(int)})
    if phq_col is not None:
        out["PHQ-9"] = pd.to_numeric(df[phq_col], errors="coerce").fillna(0.0).astype(float)
    else:
        out["PHQ-9"] = 0.0
    if y3_col is not None:
        out["label3"] = pd.to_numeric(df[y3_col], errors="coerce").fillna(0).astype(int).clip(0, 2)
    else:
        # Fallback common PHQ thresholds. Only used if label3 missing.
        phq = out["PHQ-9"].to_numpy()
        out["label3"] = np.where(phq >= 10, 2, np.where(phq >= 5, 1, 0)).astype(int)
    if y2_col is not None:
        out["label2"] = pd.to_numeric(df[y2_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    else:
        out["label2"] = (out["label3"] > 0).astype(int)
    return out.sort_values("ID").reset_index(drop=True)


def prepare_test_rows(test_csv: str | Path) -> pd.DataFrame:
    df = read_csv_auto(test_csv)
    id_col = find_col(df, ["ID", "id", "sample_id", "subject_id"])
    if id_col is None:
        id_col = df.columns[0]
    out = pd.DataFrame({"ID": df[id_col].apply(to_int_id).astype(int)})
    out["label2"] = 0
    out["label3"] = 0
    out["PHQ-9"] = 0.0
    return out.sort_values("ID").reset_index(drop=True)


@dataclass
class FeatureDims:
    audio_dim: int = 0
    video_dim: int = 0
    gait_dim: int = 0
    motion_stat_dim: int = 0
    p_struct_dim: int = 0
    p_embed_dim: int = 0
    audio_big_dim: int = 0
    motion_extra_pair_dim: int = 0
    motion_extra_stat_dim: int = 0
    gait_extra_dim: int = 0
    p_extra_dim: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureDims":
        vals = {k: int(d.get(k, 0)) for k in cls().__dict__.keys()}
        return cls(**vals)


class ElderFeatureStore:
    def __init__(
        self,
        data_root: str | Path,
        ids: Sequence[int],
        *,
        audio_features: Sequence[str],
        official_video_features: Sequence[str],
        use_gait: bool,
        target_t: int,
        p_struct_csv: str | Path | None,
        p_embed_npy: str | Path | None,
        motion_npz: str | Path | None,
        audio_big_npz: str | Path | None,
        motion_extra_npz: str | Path | None,
        gait_extra_npz: str | Path | None,
        p_extra_csv: str | Path | None,
        forced_dims: Optional[Dict[str, int]] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.ids = [int(x) for x in ids]
        self.audio_features = list(audio_features)
        self.official_video_features = list(official_video_features)
        self.use_gait = bool(use_gait)
        self.target_t = int(target_t)
        self.p_struct_map, self.p_struct_cols = read_struct_features(p_struct_csv)
        self.p_embed_map = load_personality_embedding(p_embed_npy)
        self.motion_map = read_motion_npz(motion_npz)
        self.audio_big_map = read_audio_big_npz(audio_big_npz)
        self.motion_extra_map = read_motion_extra_npz(motion_extra_npz)
        self.gait_extra_map = read_gait_extra_npz(gait_extra_npz)
        self.p_extra_map, self.p_extra_cols = read_extra_p_csv(p_extra_csv)
        self.audio_feat_dims = infer_pair_feature_dims(self.data_root, self.ids, self.audio_features, "audio")
        self.official_video_feat_dims = infer_pair_feature_dims(self.data_root, self.ids, self.official_video_features, "video")
        self.gait_dim_infer = infer_gait_dim(self.data_root, self.ids) if self.use_gait else 0
        dims = self._infer_dims()
        if forced_dims:
            # Prediction must match checkpoint dims.
            base = dims.to_dict()
            base.update({k: int(v) for k, v in forced_dims.items() if k in base})
            dims = FeatureDims.from_dict(base)
        self.dims = dims

    def _infer_dims(self) -> FeatureDims:
        d = FeatureDims()
        d.audio_dim = int(sum(self.audio_feat_dims.values()))
        raw_dim = 0
        raw_stat_dim = 0
        for rec in self.motion_map.values():
            if raw_dim <= 0 and "pair" in rec:
                raw_dim = int(rec["pair"].reshape(rec["pair"].shape[0], rec["pair"].shape[1], -1).shape[-1])
            if raw_stat_dim <= 0 and "stat" in rec:
                raw_stat_dim = int(rec["stat"].reshape(-1).shape[0])
            if raw_dim > 0 and raw_stat_dim > 0:
                break
        official_video_dim = int(sum(self.official_video_feat_dims.values()))
        d.video_dim = raw_dim + official_video_dim
        d.motion_stat_dim = raw_stat_dim
        d.gait_dim = int(self.gait_dim_infer)
        if self.p_struct_map:
            d.p_struct_dim = len(next(iter(self.p_struct_map.values())).reshape(-1))
        if self.p_embed_map:
            d.p_embed_dim = len(next(iter(self.p_embed_map.values())).reshape(-1))
        for rec in self.audio_big_map.values():
            if "pair" in rec:
                d.audio_big_dim = int(rec["pair"].reshape(rec["pair"].shape[0], -1).shape[-1])
                break
        for rec in self.motion_extra_map.values():
            if "pair" in rec and d.motion_extra_pair_dim <= 0:
                d.motion_extra_pair_dim = int(rec["pair"].reshape(rec["pair"].shape[0], -1).shape[-1])
            if "stat" in rec and d.motion_extra_stat_dim <= 0:
                d.motion_extra_stat_dim = int(rec["stat"].reshape(-1).shape[0])
            if d.motion_extra_pair_dim > 0 and d.motion_extra_stat_dim > 0:
                break
        if self.gait_extra_map:
            d.gait_extra_dim = len(next(iter(self.gait_extra_map.values())).reshape(-1))
        if self.p_extra_map:
            d.p_extra_dim = len(next(iter(self.p_extra_map.values())).reshape(-1))
        return d

    def report(self) -> Dict[str, Any]:
        def count(m: Dict[int, Any]) -> int:
            return sum(1 for pid in self.ids if pid in m)
        return {
            "n_ids": len(self.ids),
            "ids_head": self.ids[:10],
            "dims": self.dims.to_dict(),
            "audio_features": self.audio_features,
            "audio_feature_dims": self.audio_feat_dims,
            "official_video_features": self.official_video_features,
            "official_video_feature_dims": self.official_video_feat_dims,
            "found": {
                "p_struct": count(self.p_struct_map),
                "p_embed": count(self.p_embed_map),
                "raw_motion": count(self.motion_map),
                "audio_big": count(self.audio_big_map),
                "motion_extra": count(self.motion_extra_map),
                "gait_extra": count(self.gait_extra_map),
                "p_extra": count(self.p_extra_map),
            },
        }

    def make_sample(self, row: pd.Series | Dict[str, Any]) -> Dict[str, Any]:
        pid = to_int_id(row["ID"])
        dims = self.dims
        # official audio
        audio, audio_mask, _ = load_official_pair_features(
            self.data_root, pid, self.audio_features, "audio", self.target_t, self.audio_feat_dims
        )
        audio = pad_or_truncate_last(audio, dims.audio_dim)
        # raw motion + optional official video
        raw_pair = np.zeros((PAIR_COUNT, self.target_t, 0), dtype=np.float32)
        raw_mask = np.zeros(PAIR_COUNT, dtype=np.float32)
        motion_stat = np.zeros(dims.motion_stat_dim, dtype=np.float32)
        rec = self.motion_map.get(pid, {})
        if "pair" in rec and dims.video_dim > 0:
            arr = np_clean(rec["pair"])
            if arr.ndim == 2:
                arr = arr[None, :, :]
            if arr.ndim == 1:
                arr = arr.reshape(1, 1, -1)
            # [P,T,D] -> pad pair and resize T
            pair_list = []
            for i in range(PAIR_COUNT):
                if i < arr.shape[0]:
                    pair_list.append(resize_np_time(arr[i].reshape(arr.shape[1], -1), self.target_t))
                else:
                    pair_list.append(np.zeros((self.target_t, arr.reshape(arr.shape[0], arr.shape[1], -1).shape[-1]), dtype=np.float32))
            raw_pair = np.stack(pair_list, axis=0).astype(np.float32)
            raw_mask = np_clean(rec.get("mask", np.ones(arr.shape[0], dtype=np.float32))).reshape(-1)
            raw_mask = np.pad(raw_mask[:PAIR_COUNT], (0, max(0, PAIR_COUNT - len(raw_mask))), constant_values=0).astype(np.float32)
        raw_dim = raw_pair.shape[-1] if raw_pair.ndim == 3 else 0
        if "stat" in rec and dims.motion_stat_dim > 0:
            motion_stat = pad_or_truncate_last(np_clean(rec["stat"]).reshape(-1), dims.motion_stat_dim)
        offv, offv_mask, _ = load_official_pair_features(
            self.data_root, pid, self.official_video_features, "video", self.target_t, self.official_video_feat_dims
        )
        video = np.concatenate([raw_pair, offv], axis=-1) if offv.shape[-1] > 0 else raw_pair
        video = pad_or_truncate_last(video, dims.video_dim)
        offv_valid = offv_mask if offv_mask.size == PAIR_COUNT else np.zeros(PAIR_COUNT, dtype=np.float32)
        video_mask = np.maximum(raw_mask, offv_valid).astype(np.float32)
        # gait seq
        gait = load_gait_seq(self.data_root, pid, self.target_t, dims.gait_dim) if self.use_gait else np.zeros((self.target_t, dims.gait_dim), dtype=np.float32)
        # p features
        p_struct = pad_or_truncate_last(self.p_struct_map.get(pid, np.zeros(dims.p_struct_dim, dtype=np.float32)), dims.p_struct_dim)
        p_embed = pad_or_truncate_last(self.p_embed_map.get(pid, np.zeros(dims.p_embed_dim, dtype=np.float32)), dims.p_embed_dim)
        p_extra = pad_or_truncate_last(self.p_extra_map.get(pid, np.zeros(dims.p_extra_dim, dtype=np.float32)), dims.p_extra_dim)
        # new A_big
        ab_rec = self.audio_big_map.get(pid, {})
        audio_big = np.zeros((PAIR_COUNT, dims.audio_big_dim), dtype=np.float32)
        audio_big_mask = np.zeros(PAIR_COUNT, dtype=np.float32)
        if "pair" in ab_rec and dims.audio_big_dim > 0:
            arr = np_clean(ab_rec["pair"])
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            arr = arr.reshape(arr.shape[0], -1)
            n = min(PAIR_COUNT, arr.shape[0])
            audio_big[:n] = pad_or_truncate_last(arr[:n], dims.audio_big_dim)
            m = np_clean(ab_rec.get("mask", np.ones(arr.shape[0], dtype=np.float32))).reshape(-1)
            audio_big_mask[:n] = m[:n]
        # new motion extra
        me_rec = self.motion_extra_map.get(pid, {})
        motion_extra_pair = np.zeros((PAIR_COUNT, dims.motion_extra_pair_dim), dtype=np.float32)
        motion_extra_pair_mask = np.zeros(PAIR_COUNT, dtype=np.float32)
        motion_extra_stat = np.zeros(dims.motion_extra_stat_dim, dtype=np.float32)
        if "pair" in me_rec and dims.motion_extra_pair_dim > 0:
            arr = np_clean(me_rec["pair"])
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            arr = arr.reshape(arr.shape[0], -1)
            n = min(PAIR_COUNT, arr.shape[0])
            motion_extra_pair[:n] = pad_or_truncate_last(arr[:n], dims.motion_extra_pair_dim)
            m = np_clean(me_rec.get("mask", np.ones(arr.shape[0], dtype=np.float32))).reshape(-1)
            motion_extra_pair_mask[:n] = m[:n]
        if "stat" in me_rec and dims.motion_extra_stat_dim > 0:
            motion_extra_stat = pad_or_truncate_last(np_clean(me_rec["stat"]).reshape(-1), dims.motion_extra_stat_dim)
        gait_extra = pad_or_truncate_last(self.gait_extra_map.get(pid, np.zeros(dims.gait_extra_dim, dtype=np.float32)), dims.gait_extra_dim)
        phq = float(row.get("PHQ-9", 0.0))
        return {
            "id": int(pid),
            "label2": int(row.get("label2", 0)),
            "label3": int(row.get("label3", 0)),
            "phq": float(phq),
            "phq_log_target": float(math.log1p(max(0.0, phq))),
            "audio": audio.astype(np.float32),
            "audio_pair_mask": audio_mask.astype(np.float32),
            "video": video.astype(np.float32),
            "video_pair_mask": video_mask.astype(np.float32),
            "gait": gait.astype(np.float32),
            "motion_stat": motion_stat.astype(np.float32),
            "p_struct": p_struct.astype(np.float32),
            "p_embed": p_embed.astype(np.float32),
            "audio_big": audio_big.astype(np.float32),
            "audio_big_pair_mask": audio_big_mask.astype(np.float32),
            "motion_extra_pair": motion_extra_pair.astype(np.float32),
            "motion_extra_pair_mask": motion_extra_pair_mask.astype(np.float32),
            "motion_extra_stat": motion_extra_stat.astype(np.float32),
            "gait_extra": gait_extra.astype(np.float32),
            "p_extra": p_extra.astype(np.float32),
        }

# -----------------------------------------------------------------------------
# Scalers and Dataset
# -----------------------------------------------------------------------------

class FeatureScalers:
    def __init__(self) -> None:
        self.stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _flatten_feature(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0 or x.shape[-1] == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return x.reshape(-1, x.shape[-1])

    def fit(self, samples: Sequence[Dict[str, Any]], keys: Sequence[str]) -> "FeatureScalers":
        for k in keys:
            mats = []
            for s in samples:
                x = np.asarray(s[k], dtype=np.float32)
                if x.size == 0 or x.shape[-1] == 0:
                    continue
                mats.append(self._flatten_feature(x))
            if not mats:
                continue
            mat = np.concatenate(mats, axis=0)
            mu = mat.mean(axis=0).astype(np.float32)
            sd = mat.std(axis=0).astype(np.float32)
            sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
            self.stats[k] = (mu, sd)
        return self

    def transform(self, k: str, x: np.ndarray) -> np.ndarray:
        if k not in self.stats:
            return np.asarray(x, dtype=np.float32)
        mu, sd = self.stats[k]
        if x.size == 0 or x.shape[-1] == 0:
            return np.asarray(x, dtype=np.float32)
        return np.clip((x.astype(np.float32) - mu) / sd, -8.0, 8.0).astype(np.float32)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {k: {"mean": v[0].tolist(), "std": v[1].tolist()} for k, v in self.stats.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureScalers":
        obj = cls()
        for k, v in d.items():
            obj.stats[k] = (np.asarray(v["mean"], dtype=np.float32), np.asarray(v["std"], dtype=np.float32))
        return obj


class ElderDataset(Dataset):
    FEATURE_KEYS = [
        "audio", "video", "gait", "motion_stat", "p_struct", "p_embed",
        "audio_big", "motion_extra_pair", "motion_extra_stat", "gait_extra", "p_extra",
    ]

    def __init__(self, samples: Sequence[Dict[str, Any]], scalers: Optional[FeatureScalers] = None) -> None:
        self.samples = list(samples)
        self.scalers = scalers or FeatureScalers()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        out: Dict[str, Any] = {}
        for k, v in s.items():
            if k in self.FEATURE_KEYS:
                out[k] = torch.tensor(self.scalers.transform(k, np.asarray(v, dtype=np.float32)), dtype=torch.float32)
            elif k in {"audio_pair_mask", "video_pair_mask", "audio_big_pair_mask", "motion_extra_pair_mask"}:
                out[k] = torch.tensor(v, dtype=torch.float32)
            elif k in {"id", "label2", "label3"}:
                out[k] = torch.tensor(int(v), dtype=torch.long)
            elif k in {"phq", "phq_log_target"}:
                out[k] = torch.tensor(float(v), dtype=torch.float32)
        return out


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}

# -----------------------------------------------------------------------------
# Model blocks
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.enabled = self.in_dim > 0
        if self.enabled:
            self.net = nn.Sequential(
                nn.LayerNorm(self.in_dim),
                nn.Linear(self.in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(out_dim),
            )
        else:
            self.net = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x.new_zeros((x.shape[0], self.out_dim))
        return self.net(x)


class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x [B,L,H], mask [B,L]
        if x.shape[1] == 0:
            return x.new_zeros((x.shape[0], x.shape[-1]))
        s = self.score(x).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(mask <= 0, -1e4)
        w = torch.softmax(s, dim=1)
        if mask is not None:
            all_bad = (mask.sum(dim=1, keepdim=True) <= 0)
            w = torch.where(all_bad, torch.zeros_like(w), w)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


class PairSeqBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.enabled = self.in_dim > 0
        self.frame = MLP(self.in_dim, hidden_dim, hidden_dim, dropout=dropout) if self.enabled else None
        self.pool = MaskedAttentionPool(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, P = x.shape[:2]
        if not self.enabled:
            z = x.new_zeros((B, P, self.hidden_dim))
            return x.new_zeros((B, self.hidden_dim)), z
        # [B,P,T,D] -> [B*P,T,D]
        y = x.reshape(B * P, x.shape[2], x.shape[3])
        y = self.frame(y.reshape(B * P * x.shape[2], x.shape[3])).reshape(B * P, x.shape[2], self.hidden_dim)
        pair_tok = y.mean(dim=1).reshape(B, P, self.hidden_dim)
        pooled = self.pool(pair_tok, mask)
        return pooled, pair_tok


class PairVectorBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.enabled = self.in_dim > 0
        self.proj = MLP(self.in_dim, hidden_dim, hidden_dim, dropout=dropout) if self.enabled else None
        self.pool = MaskedAttentionPool(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, P = x.shape[:2]
        if not self.enabled:
            z = x.new_zeros((B, P, self.hidden_dim))
            return x.new_zeros((B, self.hidden_dim)), z
        z = self.proj(x.reshape(B * P, x.shape[-1])).reshape(B, P, self.hidden_dim)
        return self.pool(z, mask), z


class SeqBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.enabled = self.in_dim > 0
        self.frame = MLP(self.in_dim, hidden_dim, hidden_dim, dropout=dropout) if self.enabled else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if not self.enabled:
            return x.new_zeros((B, self.hidden_dim))
        y = self.frame(x.reshape(B * x.shape[1], x.shape[2])).reshape(B, x.shape[1], self.hidden_dim)
        return y.mean(dim=1)


class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, init_scale: float = 12.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, in_dim) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return torch.exp(self.log_scale).clamp(1.0, 50.0) * (x @ w.t())


class ElderV12PQueryResidual(nn.Module):
    def __init__(
        self,
        dims: Dict[str, int],
        hidden_dim: int = 96,
        p_embed_bottleneck: int = 48,
        dropout: float = 0.40,
        modality_dropout: float = 0.15,
        s_cls: float = 0.20,
        s_phq: float = 3.0,
        ref_alpha: float = 0.10,
    ) -> None:
        super().__init__()
        self.dims = {k: int(v) for k, v in dims.items()}
        self.hidden_dim = int(hidden_dim)
        self.modality_dropout = float(modality_dropout)
        self.s_cls = float(s_cls)
        self.s_phq = float(s_phq)
        self.ref_alpha = float(ref_alpha)
        H = self.hidden_dim
        # P token combines verified P_struct/P_embed and v11 P_extra.
        self.p_struct = MLP(self.dims.get("p_struct_dim", 0), H, H, dropout)
        self.p_embed = MLP(self.dims.get("p_embed_dim", 0), max(H, p_embed_bottleneck * 2), p_embed_bottleneck, dropout)
        self.p_extra = MLP(self.dims.get("p_extra_dim", 0), H, H, dropout)
        p_cat_dim = H + p_embed_bottleneck + H
        self.p_comb = nn.Sequential(nn.LayerNorm(p_cat_dim), nn.Linear(p_cat_dim, H), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(H))
        # New-feature main branch.
        self.a_big = PairVectorBranch(self.dims.get("audio_big_dim", 0), H, dropout)
        self.vbeh_pair = PairVectorBranch(self.dims.get("motion_extra_pair_dim", 0), H, dropout)
        self.vbeh_stat = MLP(self.dims.get("motion_extra_stat_dim", 0), H, H, dropout)
        self.gunit = MLP(self.dims.get("gait_extra_dim", 0), H, H, dropout)
        self.av_pair = nn.Sequential(
            nn.LayerNorm(H * 4), nn.Linear(H * 4, H), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H, H), nn.GELU(), nn.LayerNorm(H)
        )
        self.q_proj = nn.Linear(H, H)
        self.k_proj = nn.Linear(H, H)
        self.v_proj = nn.Linear(H, H)
        self.new_fuse = nn.Sequential(
            nn.LayerNorm(H * 6), nn.Linear(H * 6, H), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H, H), nn.GELU(), nn.LayerNorm(H)
        )
        self.new_head_common = nn.Sequential(nn.LayerNorm(H), nn.Dropout(dropout), nn.Linear(H, H), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(H))
        self.new_head3 = CosineClassifier(H, 3)
        self.new_head2 = CosineClassifier(H, 2)
        self.new_reg = nn.Linear(H, 1)
        # P-guided correction.
        self.corr_gate = nn.Sequential(nn.LayerNorm(H * 4), nn.Linear(H * 4, H), nn.GELU(), nn.Dropout(dropout), nn.Linear(H, 2), nn.Sigmoid())
        self.delta3 = nn.Sequential(nn.LayerNorm(H * 3), nn.Linear(H * 3, H), nn.GELU(), nn.Dropout(dropout), nn.Linear(H, 3))
        self.delta2 = nn.Sequential(nn.LayerNorm(H * 3), nn.Linear(H * 3, H), nn.GELU(), nn.Dropout(dropout), nn.Linear(H, 2))
        self.delta_phq = nn.Sequential(nn.LayerNorm(H * 3), nn.Linear(H * 3, H), nn.GELU(), nn.Dropout(dropout), nn.Linear(H, 1))
        # Official reference branch: v9-like but light, no strong cross.
        self.ref_audio = PairSeqBranch(self.dims.get("audio_dim", 0), H, dropout)
        self.ref_video = PairSeqBranch(self.dims.get("video_dim", 0), H, dropout)
        self.ref_gait = SeqBranch(self.dims.get("gait_dim", 0), H, dropout)
        self.ref_ms = MLP(self.dims.get("motion_stat_dim", 0), H, H, dropout)
        self.ref_pool = MaskedAttentionPool(H)
        self.ref_common = nn.Sequential(nn.LayerNorm(H), nn.Dropout(dropout), nn.Linear(H, H), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(H))
        self.ref_head3 = CosineClassifier(H, 3)
        self.ref_head2 = CosineClassifier(H, 2)
        self.ref_reg = nn.Linear(H, 1)

    def _p_token(self, b: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.p_comb(torch.cat([self.p_struct(b["p_struct"]), self.p_embed(b["p_embed"]), self.p_extra(b["p_extra"])], dim=-1))

    def _token_dropout(self, tokens: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0:
            return tokens, mask
        keep = torch.ones_like(mask)
        if tokens.shape[1] > 1:
            drop = (torch.rand(mask[:, 1:].shape, device=mask.device) < self.modality_dropout).float()
            keep[:, 1:] = (1.0 - drop) * mask[:, 1:]
        keep = torch.maximum(keep, 1.0 - mask)
        tokens = tokens * keep.unsqueeze(-1)
        return tokens, mask * keep

    def _p_query_attention(self, p_tok: torch.Tensor, av_pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # q=[B,H], k/v=[B,P,H]
        q = self.q_proj(p_tok).unsqueeze(1)
        k = self.k_proj(av_pair)
        v = self.v_proj(av_pair)
        score = (q * k).sum(-1) / math.sqrt(k.shape[-1])
        score = score.masked_fill(mask <= 0, -1e4)
        w = torch.softmax(score, dim=1)
        all_bad = (mask.sum(dim=1, keepdim=True) <= 0)
        w = torch.where(all_bad, torch.zeros_like(w), w)
        return torch.sum(w.unsqueeze(-1) * v, dim=1)

    def forward(self, b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = b["p_struct"].shape[0]
        device = b["p_struct"].device
        p_tok = self._p_token(b)
        # New branch pair-level P-guided AV evidence.
        a_tok, a_pair = self.a_big(b["audio_big"], b["audio_big_pair_mask"])
        v_tok, v_pair = self.vbeh_pair(b["motion_extra_pair"], b["motion_extra_pair_mask"])
        pair_mask = torch.maximum(b["audio_big_pair_mask"], b["motion_extra_pair_mask"])
        av_in = torch.cat([a_pair, v_pair, torch.abs(a_pair - v_pair), a_pair * v_pair], dim=-1)
        av_pair = self.av_pair(av_in.reshape(B * PAIR_COUNT, -1)).reshape(B, PAIR_COUNT, self.hidden_dim)
        pav_tok = self._p_query_attention(p_tok, av_pair, pair_mask)
        vstat_tok = self.vbeh_stat(b["motion_extra_stat"])
        gunit_tok = self.gunit(b["gait_extra"])
        new_h = self.new_fuse(torch.cat([p_tok, pav_tok, a_tok, v_tok, vstat_tok, gunit_tok], dim=-1))
        h = self.new_head_common(new_h)
        new_logits3 = self.new_head3(h)
        new_logits2 = self.new_head2(h)
        new_phq_log = self.new_reg(h).squeeze(-1)
        # Correction controlled by P and P-attended AV evidence.
        gate_in = torch.cat([p_tok, h, pav_tok, p_tok * h], dim=-1)
        gates = self.corr_gate(gate_in)
        gate_cls = gates[:, 0:1]
        gate_phq = gates[:, 1:2]
        d_in = torch.cat([p_tok, h, pav_tok], dim=-1)
        d3 = torch.tanh(self.delta3(d_in))
        d2 = torch.tanh(self.delta2(d_in))
        dphq = torch.tanh(self.delta_phq(d_in)).squeeze(-1)
        corr_logits3 = new_logits3 + self.s_cls * gate_cls * d3
        corr_logits2 = new_logits2 + self.s_cls * gate_cls * d2
        base_phq = torch.expm1(new_phq_log).clamp(0, 27)
        corr_phq = (base_phq + self.s_phq * gate_phq.squeeze(-1) * dphq).clamp(0, 27)
        corr_phq_log = torch.log1p(corr_phq)
        # Official reference branch, auxiliary only + small optional mixing.
        ra_tok, _ = self.ref_audio(b["audio"], b["audio_pair_mask"])
        rv_tok, _ = self.ref_video(b["video"], b["video_pair_mask"])
        rg_tok = self.ref_gait(b["gait"])
        rms_tok = self.ref_ms(b["motion_stat"])
        ref_tokens = torch.stack([p_tok, ra_tok, rv_tok, rg_tok, rms_tok], dim=1)
        ref_mask = torch.ones((B, 5), dtype=torch.float32, device=device)
        ref_mask[:, 1] = (b["audio_pair_mask"].sum(dim=1) > 0).float()
        ref_mask[:, 2] = (b["video_pair_mask"].sum(dim=1) > 0).float()
        ref_mask[:, 3] = 1.0 if self.ref_gait.enabled else 0.0
        ref_mask[:, 4] = 1.0 if self.ref_ms.enabled else 0.0
        ref_tokens, ref_mask = self._token_dropout(ref_tokens, ref_mask)
        ref_h = self.ref_common(self.ref_pool(ref_tokens, ref_mask))
        ref_logits3 = self.ref_head3(ref_h)
        ref_logits2 = self.ref_head2(ref_h)
        ref_phq_log = self.ref_reg(ref_h).squeeze(-1)
        alpha = max(0.0, min(0.35, self.ref_alpha))
        logits3 = (1.0 - alpha) * corr_logits3 + alpha * ref_logits3
        logits2 = (1.0 - alpha) * corr_logits2 + alpha * ref_logits2
        phq = ((1.0 - alpha) * torch.expm1(corr_phq_log).clamp(0, 27) + alpha * torch.expm1(ref_phq_log).clamp(0, 27)).clamp(0, 27)
        phq_log = torch.log1p(phq)
        p2_dep_logit = logits2[:, 1] - logits2[:, 0]
        sev_logit = logits3[:, 2] - torch.logsumexp(logits3[:, :2], dim=1)
        ord_logits = torch.stack([p2_dep_logit, sev_logit], dim=1)
        corr_l2 = (gate_cls * d3).pow(2).mean() + 0.25 * (gate_cls * d2).pow(2).mean() + 0.20 * (gate_phq.squeeze(-1) * dphq).pow(2).mean() + 0.5 * gate_cls.mean() + 0.1 * gate_phq.mean()
        return {
            "logits3": logits3,
            "logits2": logits2,
            "ord_logits": ord_logits,
            "phq_log": phq_log,
            "new_logits3": new_logits3,
            "new_logits2": new_logits2,
            "new_phq_log": new_phq_log,
            "ref_logits3": ref_logits3,
            "ref_logits2": ref_logits2,
            "ref_phq_log": ref_phq_log,
            "corr_l2": corr_l2,
            "gate_cls_mean": gate_cls.mean().detach(),
            "gate_phq_mean": gate_phq.mean().detach(),
        }

# -----------------------------------------------------------------------------
# Loss and metrics
# -----------------------------------------------------------------------------

def class_weights(y: np.ndarray, n: int) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=n).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def soft_macro_f1_loss(logits: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
    p = torch.softmax(logits, dim=-1)
    yoh = F.one_hot(y, num_classes=n_classes).float()
    tp = (p * yoh).sum(dim=0)
    fp = (p * (1 - yoh)).sum(dim=0)
    fn = ((1 - p) * yoh).sum(dim=0)
    f1 = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    return 1.0 - f1.mean()


def soft_kappa_loss(logits: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
    # Differentiable unweighted kappa approximation.
    p = torch.softmax(logits, dim=-1)
    yoh = F.one_hot(y, num_classes=n_classes).float()
    conf = yoh.t() @ p  # [C,C]
    n = conf.sum().clamp_min(EPS)
    po = torch.trace(conf) / n
    row = conf.sum(dim=1)
    col = conf.sum(dim=0)
    pe = (row * col).sum() / (n * n + EPS)
    k = (po - pe) / (1.0 - pe + EPS)
    return 1.0 - k


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    vx = pred.var(unbiased=False)
    vy = target.var(unbiased=False)
    mx = pred.mean()
    my = target.mean()
    cov = ((pred - mx) * (target - my)).mean()
    ccc = (2 * cov) / (vx + vy + (mx - my).pow(2) + EPS)
    return 1.0 - ccc


def compute_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: argparse.Namespace, w2: torch.Tensor, w3: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    y2 = batch["label2"].long()
    y3 = batch["label3"].long()
    phq = batch["phq"].float()
    phq_log_t = batch["phq_log_target"].float()
    w2 = w2.to(y2.device)
    w3 = w3.to(y2.device)
    loss_ce3 = F.cross_entropy(out["logits3"], y3, weight=w3, label_smoothing=cfg.label_smoothing)
    loss_ce2 = F.cross_entropy(out["logits2"], y2, weight=w2, label_smoothing=cfg.label_smoothing)
    loss_f1 = soft_macro_f1_loss(out["logits3"], y3, 3) + 0.5 * soft_macro_f1_loss(out["logits2"], y2, 2)
    loss_kappa = soft_kappa_loss(out["logits3"], y3, 3) + 0.5 * soft_kappa_loss(out["logits2"], y2, 2)
    phq_pred = torch.expm1(out["phq_log"]).clamp(0, 27)
    loss_reg = F.smooth_l1_loss(phq_pred, phq)
    loss_ccc = ccc_loss(phq_pred, phq)
    prob2 = torch.softmax(out["logits2"], dim=-1)[:, 1]
    prob3 = torch.softmax(out["logits3"], dim=-1)
    loss_bt = F.mse_loss(prob2, prob3[:, 1] + prob3[:, 2])
    ord_t = torch.stack([(y3 >= 1).float(), (y3 >= 2).float()], dim=1)
    loss_ord = F.binary_cross_entropy_with_logits(out["ord_logits"], ord_t)
    loss = (
        loss_ce3
        + cfg.binary_weight * loss_ce2
        + cfg.soft_f1_weight * loss_f1
        + cfg.kappa_weight * loss_kappa
        + cfg.reg_weight * loss_reg
        + cfg.ccc_weight * loss_ccc
        + cfg.consistency_weight * loss_bt
        + cfg.ordinal_weight * loss_ord
    )
    # New branch auxiliary: keeps new-feature main predictive, correction cannot be the only predictor.
    if cfg.new_aux_weight > 0 and "new_logits3" in out:
        loss_new = F.cross_entropy(out["new_logits3"], y3, weight=w3, label_smoothing=cfg.label_smoothing) + 0.5 * F.cross_entropy(out["new_logits2"], y2, weight=w2, label_smoothing=cfg.label_smoothing)
        loss_new = loss_new + 0.1 * F.smooth_l1_loss(torch.expm1(out["new_phq_log"]).clamp(0, 27), phq)
        loss = loss + cfg.new_aux_weight * loss_new
    else:
        loss_new = torch.tensor(0.0, device=y2.device)
    # Official reference auxiliary: supervised weak reference, not a teacher.
    if cfg.ref_weight > 0 and "ref_logits3" in out:
        loss_ref = F.cross_entropy(out["ref_logits3"], y3, weight=w3, label_smoothing=cfg.label_smoothing) + 0.5 * F.cross_entropy(out["ref_logits2"], y2, weight=w2, label_smoothing=cfg.label_smoothing)
        loss_ref = loss_ref + 0.1 * F.smooth_l1_loss(torch.expm1(out["ref_phq_log"]).clamp(0, 27), phq)
        loss = loss + cfg.ref_weight * loss_ref
    else:
        loss_ref = torch.tensor(0.0, device=y2.device)
    if cfg.corr_weight > 0 and "corr_l2" in out:
        loss = loss + cfg.corr_weight * out["corr_l2"]
    logs = {
        "loss": float(loss.detach().cpu()),
        "ce3": float(loss_ce3.detach().cpu()),
        "ce2": float(loss_ce2.detach().cpu()),
        "reg": float(loss_reg.detach().cpu()),
        "ccc_loss": float(loss_ccc.detach().cpu()),
        "bt": float(loss_bt.detach().cpu()),
        "ord": float(loss_ord.detach().cpu()),
        "new_aux": float(loss_new.detach().cpu()),
        "ref_aux": float(loss_ref.detach().cpu()),
        "gate_cls": float(out.get("gate_cls_mean", torch.tensor(0.0)).detach().cpu()),
        "gate_phq": float(out.get("gate_phq_mean", torch.tensor(0.0)).detach().cpu()),
    }
    return loss, logs


def eval_predictions(ids: np.ndarray, y2: np.ndarray, y3: np.ndarray, phq: np.ndarray, prob2: np.ndarray, prob3: np.ndarray, phq_pred: np.ndarray) -> Dict[str, float]:
    pred3 = prob3.argmax(axis=1).astype(int)
    # enforce consistency in evaluation report by deriving binary from ternary; still save raw final2_pos.
    pred2 = (pred3 > 0).astype(int)
    out: Dict[str, float] = {}
    out["binary_acc"] = float(accuracy_score(y2, pred2))
    out["binary_macro_f1"] = float(f1_score(y2, pred2, average="macro", zero_division=0))
    out["binary_kappa"] = float(cohen_kappa_score(y2, pred2))
    out["ternary_acc"] = float(accuracy_score(y3, pred3))
    out["ternary_macro_f1"] = float(f1_score(y3, pred3, average="macro", zero_division=0))
    out["ternary_kappa"] = float(cohen_kappa_score(y3, pred3))
    out["phq_mae"] = float(mean_absolute_error(phq, phq_pred))
    out["phq_rmse"] = float(math.sqrt(mean_squared_error(phq, phq_pred)))
    # numpy CCC
    vx = np.var(phq_pred); vy = np.var(phq); mx = np.mean(phq_pred); my = np.mean(phq)
    cov = np.mean((phq_pred - mx) * (phq - my))
    out["phq_ccc"] = float((2 * cov) / (vx + vy + (mx - my) ** 2 + EPS))
    out["inconsistent"] = float(np.sum(pred2 != (pred3 > 0)))
    return out

@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    ids: List[int] = []
    y2s: List[int] = []
    y3s: List[int] = []
    phqs: List[float] = []
    prob2s: List[np.ndarray] = []
    prob3s: List[np.ndarray] = []
    phqps: List[np.ndarray] = []
    gate_cls: List[float] = []
    gate_phq: List[float] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        ids.extend(batch["id"].detach().cpu().numpy().astype(int).tolist())
        y2s.extend(batch["label2"].detach().cpu().numpy().astype(int).tolist())
        y3s.extend(batch["label3"].detach().cpu().numpy().astype(int).tolist())
        phqs.extend(batch["phq"].detach().cpu().numpy().astype(float).tolist())
        prob2s.append(torch.softmax(out["logits2"], dim=-1).detach().cpu().numpy())
        prob3s.append(torch.softmax(out["logits3"], dim=-1).detach().cpu().numpy())
        phqps.append(torch.expm1(out["phq_log"]).clamp(0, 27).detach().cpu().numpy())
        if "gate_cls_mean" in out:
            gate_cls.append(float(out["gate_cls_mean"].detach().cpu()))
        if "gate_phq_mean" in out:
            gate_phq.append(float(out["gate_phq_mean"].detach().cpu()))
    return {
        "ids": np.asarray(ids, dtype=int),
        "y2": np.asarray(y2s, dtype=int),
        "y3": np.asarray(y3s, dtype=int),
        "phq": np.asarray(phqs, dtype=np.float32),
        "prob2": np.concatenate(prob2s, axis=0),
        "prob3": np.concatenate(prob3s, axis=0),
        "phq_pred": np.concatenate(phqps, axis=0).astype(np.float32),
        "gate_cls_mean": np.asarray(gate_cls, dtype=np.float32),
        "gate_phq_mean": np.asarray(gate_phq, dtype=np.float32),
    }

# -----------------------------------------------------------------------------
# Train / predict commands
# -----------------------------------------------------------------------------

def make_store_from_args(args: argparse.Namespace, rows: pd.DataFrame, split: str, forced_dims: Optional[Dict[str, int]] = None) -> ElderFeatureStore:
    if split == "train":
        data_root = args.train_data_root
        p_struct = args.p_struct_train_csv
        motion = args.motion_train_npz
        audio_big = args.audio_big_train_npz
        motion_extra = args.motion_extra_train_npz
        gait_extra = args.gait_extra_train_npz
        p_extra = args.p_extra_train_csv
    else:
        data_root = args.test_data_root
        p_struct = args.p_struct_test_csv
        motion = args.motion_test_npz
        audio_big = args.audio_big_test_npz
        motion_extra = args.motion_extra_test_npz
        gait_extra = args.gait_extra_test_npz
        p_extra = args.p_extra_test_csv
    return ElderFeatureStore(
        data_root=data_root,
        ids=rows["ID"].tolist(),
        audio_features=parse_feature_list(args.audio_features),
        official_video_features=parse_feature_list(args.official_video_features),
        use_gait=bool(args.use_gait),
        target_t=int(args.target_t),
        p_struct_csv=p_struct,
        p_embed_npy=args.p_embed_npy if split == "train" else (args.p_embed_test_npy or args.p_embed_npy),
        motion_npz=motion,
        audio_big_npz=audio_big,
        motion_extra_npz=motion_extra,
        gait_extra_npz=gait_extra,
        p_extra_csv=p_extra,
        forced_dims=forced_dims,
    )


def build_model_from_args(dims: Dict[str, int], args: argparse.Namespace, for_predict: bool = False) -> nn.Module:
    return ElderV12PQueryResidual(
        dims=dims,
        hidden_dim=args.hidden_dim,
        p_embed_bottleneck=args.p_embed_bottleneck,
        dropout=args.dropout,
        modality_dropout=0.0 if for_predict else args.modality_dropout,
        s_cls=args.s_cls,
        s_phq=args.s_phq,
        ref_alpha=args.ref_alpha,
    )


def train_command(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    out_dir = ensure_dir(args.output_dir)
    rows = prepare_train_rows(args.train_split_csv)
    if args.smoke:
        rows = rows.head(min(len(rows), args.smoke_n)).copy()
        args.epochs = min(args.epochs, 2)
        args.folds = min(args.folds, 2)
        print(f"[SMOKE] rows={len(rows)} folds={args.folds} epochs={args.epochs}")
    store = make_store_from_args(args, rows, "train")
    write_json(store.report(), out_dir / "feature_report_train.json")
    print("[INFO] dims", store.dims.to_dict())
    samples = [store.make_sample(r) for _, r in tqdm(rows.iterrows(), total=len(rows), desc="build train samples")]
    y2 = np.asarray([s["label2"] for s in samples], dtype=int)
    y3 = np.asarray([s["label3"] for s in samples], dtype=int)
    w2 = class_weights(y2, 2)
    w3 = class_weights(y3, 3)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_rows = []
    oof_records: List[pd.DataFrame] = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(samples)), y3)):
        print(f"\n[Fold {fold}] train={len(tr_idx)} val={len(va_idx)}")
        seed_everything(args.seed + fold)
        train_samples = [samples[i] for i in tr_idx]
        val_samples = [samples[i] for i in va_idx]
        scalers = FeatureScalers().fit(train_samples, ElderDataset.FEATURE_KEYS)
        train_ds = ElderDataset(train_samples, scalers)
        val_ds = ElderDataset(val_samples, scalers)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
        model = build_model_from_args(store.dims.to_dict(), args).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)
        best_score = -1e9
        best_metrics: Dict[str, float] = {}
        best_pred: Optional[Dict[str, np.ndarray]] = None
        ckpt_path = out_dir / f"fold{fold}_seed{args.seed}.pt"
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_vals = []
            gatec_vals = []
            gatep_vals = []
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                out = model(batch)
                loss, logs = compute_loss(out, batch, args, w2, w3)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                loss_vals.append(logs["loss"])
                gatec_vals.append(logs.get("gate_cls", 0.0))
                gatep_vals.append(logs.get("gate_phq", 0.0))
            sched.step()
            pred = predict_loader(model, val_loader, device)
            metrics = eval_predictions(pred["ids"], pred["y2"], pred["y3"], pred["phq"], pred["prob2"], pred["prob3"], pred["phq_pred"])
            score = metrics["ternary_macro_f1"] + metrics["ternary_kappa"] + 0.6 * metrics["binary_macro_f1"] + 0.3 * metrics["phq_ccc"]
            if score > best_score:
                best_score = score
                best_metrics = metrics
                best_pred = pred
                # Save only plain serializable config values. argparse subparser injects
                # args.func=train_command, which is a Python function object and breaks
                # torch.load(weights_only=True) on PyTorch >= 2.6.
                safe_config = {
                    k: v for k, v in vars(args).items()
                    if k != "func" and not callable(v)
                }
                torch.save({
                    "model_state": model.state_dict(),
                    "dims": store.dims.to_dict(),
                    "config": safe_config,
                    "scalers": scalers.to_dict(),
                    "fold": fold,
                    "seed": args.seed,
                    "best_metrics": best_metrics,
                }, ckpt_path)
            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                print(f"[Fold {fold}][Epoch {epoch:03d}] loss={np.mean(loss_vals):.4f} val_f1={metrics['ternary_macro_f1']:.4f} val_kappa={metrics['ternary_kappa']:.4f} val_ccc={metrics['phq_ccc']:.4f} gates=({np.mean(gatec_vals):.3f},{np.mean(gatep_vals):.3f})")
        row = {"fold": fold, "seed": args.seed, "best_score": best_score, **best_metrics}
        fold_rows.append(row)
        if best_pred is not None:
            order = np.argsort(best_pred["ids"])
            dfp = pd.DataFrame({
                "id": best_pred["ids"][order],
                "y2": best_pred["y2"][order],
                "y3": best_pred["y3"][order],
                "phq": best_pred["phq"][order],
                "pred2": (best_pred["prob3"][order].argmax(axis=1) > 0).astype(int),
                "pred3": best_pred["prob3"][order].argmax(axis=1).astype(int),
                "phq_pred": best_pred["phq_pred"][order],
                "prob3_0": best_pred["prob3"][order, 0],
                "prob3_1": best_pred["prob3"][order, 1],
                "prob3_2": best_pred["prob3"][order, 2],
                "fold": fold,
                "seed": args.seed,
            })
            oof_records.append(dfp)
        print(f"[Fold {fold}] best {row}")
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    if oof_records:
        oof = pd.concat(oof_records, ignore_index=True).sort_values("id")
        oof.to_csv(out_dir / "oof_predictions.csv", index=False)
        oof_metrics = eval_predictions(oof["id"].to_numpy(), oof["y2"].to_numpy(), oof["y3"].to_numpy(), oof["phq"].to_numpy(),
                                       np.stack([1-oof["pred2"].to_numpy(), oof["pred2"].to_numpy()], axis=1).astype(float),
                                       oof[["prob3_0","prob3_1","prob3_2"]].to_numpy(), oof["phq_pred"].to_numpy())
        write_json(oof_metrics, out_dir / "oof_metrics.json")
    print(f"[OK] saved outputs to {out_dir}")


def torch_load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load our own training checkpoint across PyTorch versions.

    PyTorch 2.6 changed torch.load default weights_only from False to True.
    Old smoke checkpoints may contain argparse args.func, a function object, so they
    require weights_only=False. This is safe here because the checkpoint is produced
    by this local training script. New checkpoints no longer store function objects.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return torch.load(path, map_location="cpu")


def clean_config_dict(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg_dict or {})
    cfg.pop("func", None)
    for k in list(cfg.keys()):
        if callable(cfg[k]):
            cfg.pop(k, None)
    return cfg


def load_ckpt(path: Path, device: torch.device) -> Tuple[nn.Module, FeatureScalers, Dict[str, Any]]:
    ckpt = torch_load_checkpoint(path)
    cfg_dict = clean_config_dict(ckpt.get("config", {}))
    ckpt["config"] = cfg_dict
    ns = argparse.Namespace(**cfg_dict)
    model = build_model_from_args(ckpt["dims"], ns, for_predict=True)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    scalers = FeatureScalers.from_dict(ckpt.get("scalers", {}))
    return model, scalers, ckpt


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
    test_rows = prepare_test_rows(args.test_split_csv)
    ckpts: List[Path] = []
    if args.checkpoints:
        ckpts.extend([Path(x.strip()) for x in args.checkpoints.split(",") if x.strip()])
    if args.checkpoint_dir:
        ckpts.extend(sorted(Path(args.checkpoint_dir).glob("fold*_seed*.pt")))
    ckpts = [p for p in ckpts if p.exists()]
    if not ckpts:
        raise RuntimeError("No checkpoints found. Use --checkpoint_dir or --checkpoints")
    print(f"[INFO] predicting with {len(ckpts)} checkpoints")
    all_prob2 = []
    all_prob3 = []
    all_phq = []
    base_ids = None
    feature_reports = []
    for ckpt_path in tqdm(ckpts, desc="ckpt"):
        model, scalers, ckpt = load_ckpt(ckpt_path, device)
        cfg = argparse.Namespace(**ckpt["config"])
        # Runtime args may override test paths/device/batch size, but model feature switches are from ckpt.
        for name in [
            "test_data_root", "test_split_csv", "p_struct_test_csv", "motion_test_npz", "audio_big_test_npz",
            "motion_extra_test_npz", "gait_extra_test_npz", "p_extra_test_csv", "p_embed_test_npy", "p_embed_npy",
        ]:
            setattr(cfg, name, getattr(args, name))
        store = make_store_from_args(cfg, test_rows, "test", forced_dims=ckpt["dims"])
        feature_reports.append({"ckpt": str(ckpt_path), "report": store.report()})
        samples = [store.make_sample(r) for _, r in test_rows.iterrows()]
        ds = ElderDataset(samples, scalers)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
        pred = predict_loader(model, loader, device)
        if base_ids is None:
            base_ids = pred["ids"]
        elif not np.array_equal(base_ids, pred["ids"]):
            raise RuntimeError(f"ID order mismatch at {ckpt_path}")
        all_prob2.append(pred["prob2"])
        all_prob3.append(pred["prob3"])
        all_phq.append(pred["phq_pred"])
    assert base_ids is not None
    prob2 = np.mean(all_prob2, axis=0)
    prob3 = np.mean(all_prob3, axis=0)
    phq_pred = np.mean(all_phq, axis=0)
    pred3 = prob3.argmax(axis=1).astype(int)
    # Consistency-first: binary derived from ternary, with optional high-confidence binary override disabled by default.
    pred2 = (pred3 > 0).astype(int)
    out = ensure_dir(args.output_dir)
    np.savez_compressed(out / "raw_test_predictions.npz", ids=base_ids, prob2=prob2, prob3=prob3, phq_pred=phq_pred)
    pd.DataFrame({
        "id": base_ids,
        "pred2": pred2,
        "pred3": pred3,
        "phq_pred": phq_pred,
        "prob2_1": prob2[:,1],
        "prob3_0": prob3[:,0],
        "prob3_1": prob3[:,1],
        "prob3_2": prob3[:,2],
    }).sort_values("id").to_csv(out / "test_predictions.csv", index=False)
    dist = {
        "n": int(len(base_ids)),
        "binary_dist": {str(k): int(v) for k, v in zip(*np.unique(pred2, return_counts=True))},
        "ternary_dist": {str(k): int(v) for k, v in zip(*np.unique(pred3, return_counts=True))},
        "inconsistent": int(np.sum(pred2 != (pred3 > 0))),
        "severe_ids": [int(x) for x in base_ids[pred3 == 2].tolist()],
    }
    write_json(dist, out / "distribution_report.json")
    write_json(feature_reports[:1], out / "feature_report_predict_first_ckpt.json")
    package_submission(base_ids, pred2, pred3, phq_pred, out / "predictions_normal")


def inspect_command(args: argparse.Namespace) -> None:
    out = ensure_dir(args.output_dir)
    train_rows = prepare_train_rows(args.train_split_csv)
    test_rows = prepare_test_rows(args.test_split_csv) if args.test_split_csv else train_rows.head(0)
    train_store = make_store_from_args(args, train_rows, "train")
    write_json(train_store.report(), out / "feature_report_train.json")
    print(json.dumps(train_store.report(), indent=2, ensure_ascii=False))
    if len(test_rows) > 0:
        test_store = make_store_from_args(args, test_rows, "test", forced_dims=train_store.dims.to_dict())
        write_json(test_store.report(), out / "feature_report_test.json")
        print(json.dumps(test_store.report(), indent=2, ensure_ascii=False))
    # Try building a few samples to catch shape errors.
    for _, r in train_rows.head(3).iterrows():
        s = train_store.make_sample(r)
        print("[SAMPLE]", s["id"], {k: list(v.shape) for k, v in s.items() if isinstance(v, np.ndarray)})
    print(f"[OK] inspect saved to {out}")


def dummy_command(args: argparse.Namespace) -> None:
    rows = prepare_test_rows(args.test_split_csv)
    ids = rows["ID"].to_numpy(dtype=int)
    pred3 = np.zeros(len(ids), dtype=int) + int(args.ternary_pred)
    pred2 = (pred3 > 0).astype(int) if args.binary_pred < 0 else np.zeros(len(ids), dtype=int) + int(args.binary_pred)
    phq = np.zeros(len(ids), dtype=np.float32) + float(args.phq9_pred)
    package_submission(ids, pred2, pred3, phq, args.output_dir)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train_data_root", default="")
    p.add_argument("--test_data_root", default="")
    p.add_argument("--train_split_csv", default="")
    p.add_argument("--test_split_csv", default="")
    p.add_argument("--p_struct_train_csv", default="")
    p.add_argument("--p_struct_test_csv", default="")
    p.add_argument("--p_embed_npy", default="")
    p.add_argument("--p_embed_test_npy", default="")
    p.add_argument("--motion_train_npz", default="")
    p.add_argument("--motion_test_npz", default="")
    p.add_argument("--audio_big_train_npz", default="")
    p.add_argument("--audio_big_test_npz", default="")
    p.add_argument("--motion_extra_train_npz", default="")
    p.add_argument("--motion_extra_test_npz", default="")
    p.add_argument("--gait_extra_train_npz", default="")
    p.add_argument("--gait_extra_test_npz", default="")
    p.add_argument("--p_extra_train_csv", default="")
    p.add_argument("--p_extra_test_csv", default="")
    p.add_argument("--audio_features", default="wav2vec,opensmile")
    p.add_argument("--official_video_features", default="")
    p.add_argument("--use_gait", type=int, default=1)
    p.add_argument("--target_t", type=int, default=TARGET_T)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean MPDD-AVG Elder v12 p-guided trainer/predictor")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect")
    add_common_args(p)
    p.add_argument("--output_dir", required=True)
    p.set_defaults(func=inspect_command)
    p = sub.add_parser("train")
    add_common_args(p)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--hidden_dim", type=int, default=96)
    p.add_argument("--p_embed_bottleneck", type=int, default=48)
    p.add_argument("--dropout", type=float, default=0.40)
    p.add_argument("--modality_dropout", type=float, default=0.15)
    p.add_argument("--s_cls", type=float, default=0.20)
    p.add_argument("--s_phq", type=float, default=3.0)
    p.add_argument("--ref_alpha", type=float, default=0.10)
    p.add_argument("--label_smoothing", type=float, default=0.03)
    p.add_argument("--binary_weight", type=float, default=0.50)
    p.add_argument("--soft_f1_weight", type=float, default=0.15)
    p.add_argument("--kappa_weight", type=float, default=0.10)
    p.add_argument("--reg_weight", type=float, default=0.20)
    p.add_argument("--ccc_weight", type=float, default=0.10)
    p.add_argument("--consistency_weight", type=float, default=0.30)
    p.add_argument("--ordinal_weight", type=float, default=0.20)
    p.add_argument("--new_aux_weight", type=float, default=0.30)
    p.add_argument("--ref_weight", type=float, default=0.20)
    p.add_argument("--corr_weight", type=float, default=0.05)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_n", type=int, default=24)
    p.set_defaults(func=train_command)
    p = sub.add_parser("predict")
    add_common_args(p)
    p.add_argument("--checkpoint_dir", default="")
    p.add_argument("--checkpoints", default="")
    p.add_argument("--output_dir", required=True)
    p.set_defaults(func=predict_command)
    p = sub.add_parser("dummy")
    p.add_argument("--test_split_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--binary_pred", type=int, default=-1)
    p.add_argument("--ternary_pred", type=int, default=0)
    p.add_argument("--phq9_pred", type=float, default=2.0)
    p.set_defaults(func=dummy_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
