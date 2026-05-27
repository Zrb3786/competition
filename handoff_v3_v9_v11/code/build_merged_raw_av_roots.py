from pathlib import Path
import argparse
import os
import shutil
import pandas as pd


def id_col(df):
    for c in ["ID", "id", "Id"]:
        if c in df.columns:
            return c
    return df.columns[0]


def parse_csv_ids(path):
    df = pd.read_csv(path)
    return sorted(df[id_col(df)].astype(int).tolist())


def parse_test_ids_from_imu(test_root, out_csv):
    imu = Path(test_root) / "IMU"
    ids = []
    for p in imu.iterdir():
        if p.is_dir():
            try:
                ids.append(int(p.name))
            except ValueError:
                pass
    ids = sorted(ids)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": ids}).to_csv(out, index=False)
    print("[OK] official test ids:", len(ids), ids)
    print("[OK] saved:", out)
    return ids


def reset_dir(p):
    p = Path(p)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def link_file(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def pick_file(candidates, pid, fname):
    for tag, root in candidates:
        root = Path(root)
        if not str(root) or not root.exists():
            continue
        p = root / str(pid) / fname
        if p.exists() and p.stat().st_size > 0:
            return tag, p
    return "none", None


def build_one(ids, modality, out_root, train_candidates, test_candidates=None, train_fallback_root=None):
    out_root = Path(out_root)
    reset_dir(out_root)

    rows = []

    if modality == "audio":
        names = [f"A_{i}.WAV" for i in range(1, 5)]
    elif modality == "video":
        names = [f"V_{i}.mp4" for i in range(1, 5)]
    else:
        raise ValueError(modality)

    for pid in ids:
        for pair, fname in enumerate(names, 1):
            tag, src = "none", None

            if test_candidates is not None:
                tag, src = pick_file(test_candidates, pid, fname)

            if src is None:
                tag, src = pick_file(train_candidates, pid, fname)

            if src is None and train_fallback_root is not None:
                fallback = Path(train_fallback_root) / str(pid) / fname
                if fallback.exists() and fallback.stat().st_size > 0:
                    tag, src = "train_fallback_merged", fallback

            dst = out_root / str(pid) / fname
            size = 0
            if src is not None:
                link_file(src, dst)
                size = src.stat().st_size

            rows.append({
                "ID": pid,
                "pair": pair,
                "modality": modality,
                "source": tag,
                "size": size,
                "path": str(src) if src is not None else "",
                "dst": str(dst),
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_split_csv", required=True)
    ap.add_argument("--test_root", required=True)
    ap.add_argument("--test_id_csv", required=True)

    ap.add_argument("--raw_train_video_hf", default="")
    ap.add_argument("--raw_train_video_old", default="")
    ap.add_argument("--raw_test_video_hf", default="")
    ap.add_argument("--raw_test_video_old", default="")

    ap.add_argument("--raw_train_audio_hf", default="")
    ap.add_argument("--raw_train_audio_old", default="")
    ap.add_argument("--raw_test_audio_hf", default="")
    ap.add_argument("--raw_test_audio_old", default="")

    ap.add_argument("--merged_train_video_root", required=True)
    ap.add_argument("--merged_test_video_root", required=True)
    ap.add_argument("--merged_train_audio_root", required=True)
    ap.add_argument("--merged_test_audio_root", required=True)

    ap.add_argument("--report_dir", required=True)
    args = ap.parse_args()

    train_ids = parse_csv_ids(args.train_split_csv)
    test_ids = parse_test_ids_from_imu(args.test_root, args.test_id_csv)

    train_video_candidates = [
        ("hf_train_video", args.raw_train_video_hf),
        ("old_train_video", args.raw_train_video_old),
    ]
    test_video_candidates = [
        ("hf_test_video", args.raw_test_video_hf),
        ("old_test_video", args.raw_test_video_old),
    ]
    train_audio_candidates = [
        ("hf_train_audio", args.raw_train_audio_hf),
        ("old_train_audio", args.raw_train_audio_old),
    ]
    test_audio_candidates = [
        ("hf_test_audio", args.raw_test_audio_hf),
        ("old_test_audio", args.raw_test_audio_old),
    ]

    tr_v = build_one(
        train_ids, "video", args.merged_train_video_root,
        train_candidates=train_video_candidates,
        test_candidates=None,
    )
    tr_a = build_one(
        train_ids, "audio", args.merged_train_audio_root,
        train_candidates=train_audio_candidates,
        test_candidates=None,
    )
    te_v = build_one(
        test_ids, "video", args.merged_test_video_root,
        train_candidates=train_video_candidates,
        test_candidates=test_video_candidates,
        train_fallback_root=args.merged_train_video_root,
    )
    te_a = build_one(
        test_ids, "audio", args.merged_test_audio_root,
        train_candidates=train_audio_candidates,
        test_candidates=test_audio_candidates,
        train_fallback_root=args.merged_train_audio_root,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    for name, df in [
        ("merged_train_video_report.csv", tr_v),
        ("merged_train_audio_report.csv", tr_a),
        ("merged_test_video_report.csv", te_v),
        ("merged_test_audio_report.csv", te_a),
    ]:
        p = report_dir / name
        df.to_csv(p, index=False)
        print("\n[REPORT]", p)
        print(df["source"].value_counts().to_string())
        bad = df[df["source"].eq("none")]
        if len(bad):
            print("[MISSING]")
            print(bad[["ID", "pair", "source", "size", "dst"]].to_string(index=False))
        else:
            print("[MISSING] none")


if __name__ == "__main__":
    main()
