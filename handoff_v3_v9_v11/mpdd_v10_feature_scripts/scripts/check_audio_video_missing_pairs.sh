#!/usr/bin/env bash
set -euo pipefail

set -a
source configs/elder_paths.env
set +a

: "${AUDIO_BIG_TRAIN_NPZ:?missing AUDIO_BIG_TRAIN_NPZ}"
: "${AUDIO_BIG_TEST_NPZ:?missing AUDIO_BIG_TEST_NPZ}"
: "${TRAIN_MOTION_NPZ:?missing TRAIN_MOTION_NPZ}"
: "${TEST_MOTION_NPZ:?missing TEST_MOTION_NPZ}"
: "${FEATURE_DIR:?missing FEATURE_DIR}"

echo "AUDIO_BIG_TRAIN_NPZ=$AUDIO_BIG_TRAIN_NPZ"
echo "AUDIO_BIG_TEST_NPZ=$AUDIO_BIG_TEST_NPZ"
echo "TRAIN_MOTION_NPZ=$TRAIN_MOTION_NPZ"
echo "TEST_MOTION_NPZ=$TEST_MOTION_NPZ"
echo "FEATURE_DIR=$FEATURE_DIR"

python - <<PY
from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_DIR = Path("$FEATURE_DIR")

paths = {
    "train_audio": Path("$AUDIO_BIG_TRAIN_NPZ"),
    "test_audio": Path("$AUDIO_BIG_TEST_NPZ"),
    "train_motion": Path("$TRAIN_MOTION_NPZ"),
    "test_motion": Path("$TEST_MOTION_NPZ"),
}

reports = {
    "train_audio_report": FEATURE_DIR / "merged_train_audio_report.csv",
    "test_audio_report": FEATURE_DIR / "merged_test_audio_report.csv",
    "train_video_report": FEATURE_DIR / "merged_train_video_report.csv",
    "test_video_report": FEATURE_DIR / "merged_test_video_report.csv",
}

def load_mask(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    id_key = "ids" if "ids" in z.files else ("ID" if "ID" in z.files else None)
    if id_key is None:
        raise ValueError(f"cannot find ids in {npz_path}, keys={z.files}")
    ids = z[id_key].astype(int).tolist()

    mask = None
    for k in ["pair_mask", "mask", "video_pair_mask", "audio_pair_mask"]:
        if k in z.files:
            mask = z[k].astype(float)
            break

    if mask is None:
        for k in z.files:
            arr = z[k]
            if hasattr(arr, "ndim") and arr.ndim == 2 and arr.shape[1] == 4:
                mask = arr.astype(float)
                break

    if mask is None:
        raise ValueError(f"Cannot find pair mask in {npz_path}, keys={z.files}")

    return ids, mask

def missing_pairs(ids, mask):
    rows = []
    for i, pid in enumerate(ids):
        for pair in range(1, mask.shape[1] + 1):
            if float(mask[i, pair - 1]) <= 0:
                rows.append((int(pid), int(pair)))
    return set(rows)

def report_missing(name, path):
    print(f"\\n==== raw report: {name} ====")
    if not path.exists():
        print("[NO REPORT]", path)
        return

    df = pd.read_csv(path)
    if "source" in df.columns:
        print("source counts:")
        print(df["source"].value_counts().to_string())

        bad = df[df["source"].eq("none")]
        if len(bad):
            cols = [c for c in ["ID", "pair", "source", "size", "path", "dst"] if c in bad.columns]
            print("\\nsource=none rows:")
            print(bad[cols].sort_values(["ID", "pair"]).to_string(index=False))
        else:
            print("source=none: none")
    else:
        print("[WARN] no source column:", path)

def show_split(split):
    print("\\n" + "=" * 90)
    print(f"COMPARE {split.upper()}")
    print("=" * 90)

    a_ids, a_mask = load_mask(paths[f"{split}_audio"])
    v_ids, v_mask = load_mask(paths[f"{split}_motion"])

    print(f"audio ids n={len(a_ids)} first={a_ids[:10]} last={a_ids[-10:]}")
    print(f"video ids n={len(v_ids)} first={v_ids[:10]} last={v_ids[-10:]}")
    print("ID set same:", set(a_ids) == set(v_ids))
    print("ID order same:", a_ids == v_ids)

    a_miss = missing_pairs(a_ids, a_mask)
    v_miss = missing_pairs(v_ids, v_mask)

    both = sorted(a_miss & v_miss)
    audio_only = sorted(a_miss - v_miss)
    video_only = sorted(v_miss - a_miss)

    print(f"\\naudio valid pairs: {float(a_mask.sum())} / {a_mask.size}")
    print(f"video valid pairs: {float(v_mask.sum())} / {v_mask.size}")

    print(f"\\naudio missing n={len(a_miss)}")
    print(sorted(a_miss))

    print(f"\\nvideo missing n={len(v_miss)}")
    print(sorted(v_miss))

    print(f"\\nboth missing n={len(both)}")
    print(both)

    print(f"\\naudio_only missing n={len(audio_only)}")
    print(audio_only)

    print(f"\\nvideo_only missing n={len(video_only)}")
    print(video_only)

    rows = []
    for pid in sorted(set([x[0] for x in a_miss | v_miss])):
        rows.append({
            "ID": pid,
            "audio_missing_pairs": ",".join(str(p) for i, p in sorted(a_miss) if i == pid),
            "video_missing_pairs": ",".join(str(p) for i, p in sorted(v_miss) if i == pid),
        })

    if rows:
        print("\\nper-ID missing summary:")
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("\\nper-ID missing summary: no missing pairs")

show_split("train")
show_split("test")

report_missing("train audio", reports["train_audio_report"])
report_missing("test audio", reports["test_audio_report"])
report_missing("train video", reports["train_video_report"])
report_missing("test video", reports["test_video_report"])
PY
