from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

PAIR_COUNT = 4


def id_col(df):
    for c in ["ID", "id", "Id"]:
        if c in df.columns:
            return c
    return df.columns[0]


def read_ids_from_csv(path):
    df = pd.read_csv(path)
    return sorted(df[id_col(df)].astype(int).tolist())


def load_motion_npz(path):
    z = np.load(path, allow_pickle=True)
    ids = z["ids"].astype(int)
    seq = z["motion_seq"].astype(np.float32)
    stat = z["motion_stat"].astype(np.float32) if "motion_stat" in z.files else np.zeros((len(ids), 0), dtype=np.float32)
    mask = z["pair_mask"].astype(np.float32) if "pair_mask" in z.files else (np.abs(seq).sum(axis=(2, 3)) > 0).astype(np.float32)
    return ids, seq, stat, mask


def zscore_fit(x):
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[~np.isfinite(sd)] = 1.0
    sd[sd < 1e-6] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    return mu.astype(np.float32), sd.astype(np.float32)


def zscore_apply(x, mu, sd):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return ((x - mu) / (sd + 1e-6)).astype(np.float32)


def slope_1d(x):
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    t = np.arange(len(x), dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-6)
    y = (x - x.mean()) / (x.std() + 1e-6)
    return float(np.polyfit(t, y, 1)[0])


def longest_run(mask):
    cur = 0
    best = 0
    for v in mask:
        if bool(v):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def motion_energy(seq):
    # seq [P,T,C], use motion-like channels if enough dims; otherwise all dims.
    C = seq.shape[-1]
    if C >= 10:
        x = seq[..., 2:min(C, 10)]
    else:
        x = seq
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.mean(np.abs(x), axis=-1).astype(np.float32)  # [P,T]


def fit_motion_thresholds(seq, mask):
    vals = []
    for i in range(seq.shape[0]):
        e = motion_energy(seq[i])
        m = mask[i].astype(bool)
        if m.any():
            vals.append(e[m].reshape(-1))
    if vals:
        x = np.concatenate(vals)
        x = x[np.isfinite(x)]
    else:
        x = np.array([0.0, 1.0], dtype=np.float32)
    return float(np.percentile(x, 20)), float(np.percentile(x, 80))


def compute_motion_extra(ids, seq, mask, still_thr, burst_thr):
    rows_pair = []
    rows_stat = []
    C = seq.shape[-1]
    for i, pid in enumerate(ids):
        s = seq[i]
        pm = mask[i].astype(bool)
        e = motion_energy(s)  # [4,T]
        pair_feats = []
        global_vals = []
        burst_counts = []
        long_stills = []
        slopes = []
        for p in range(PAIR_COUNT):
            if p >= e.shape[0] or not pm[p]:
                pf = np.zeros(20 + C * 3, dtype=np.float32)
                pair_feats.append(pf)
                continue
            ep = e[p]
            sp = np.nan_to_num(s[p], nan=0.0, posinf=0.0, neginf=0.0)
            still = (ep <= still_thr)
            burst = (ep >= burst_thr)
            b_edges = np.diff(np.concatenate([[0], burst.astype(int)]))
            b_count = float((b_edges == 1).sum())
            lr = float(longest_run(still) / max(1, len(still)))
            sl = slope_1d(ep)
            # pair-level behavior-retardation stats
            base = [
                np.mean(ep), np.std(ep), np.percentile(ep, 10), np.percentile(ep, 50), np.percentile(ep, 90), np.max(ep),
                np.mean(still), np.mean(burst), b_count, lr, sl,
                np.mean(np.diff(ep)) if len(ep) > 1 else 0.0,
                np.std(np.diff(ep)) if len(ep) > 1 else 0.0,
                float(np.mean(ep[:len(ep)//2]) if len(ep) > 2 else np.mean(ep)),
                float(np.mean(ep[len(ep)//2:]) if len(ep) > 2 else np.mean(ep)),
                float(np.mean(ep[len(ep)//2:]) - np.mean(ep[:len(ep)//2]) if len(ep) > 2 else 0.0),
                float(np.mean(np.abs(np.diff(ep))) if len(ep) > 1 else 0.0),
                float(np.percentile(ep, 75) - np.percentile(ep, 25)),
                float(np.mean(ep > np.mean(ep) + np.std(ep))),
                float(np.mean(ep < np.mean(ep) - 0.5 * np.std(ep))),
            ]
            ch = []
            for c in range(C):
                col = sp[:, c]
                ch.extend([float(np.mean(col)), float(np.std(col)), float(np.percentile(col, 90))])
            pf = np.asarray(base + ch, dtype=np.float32)
            pair_feats.append(pf)
            global_vals.append(ep)
            burst_counts.append(b_count)
            long_stills.append(lr)
            slopes.append(sl)
        pair_feats = np.stack(pair_feats, axis=0).astype(np.float32)
        if global_vals:
            gv = np.concatenate(global_vals)
            stat = [
                float(pm.sum()), float(PAIR_COUNT - pm.sum()),
                float(np.mean(gv)), float(np.std(gv)), float(np.percentile(gv, 10)), float(np.percentile(gv, 50)),
                float(np.percentile(gv, 90)), float(np.max(gv)), float(np.mean(gv <= still_thr)), float(np.mean(gv >= burst_thr)),
                float(np.mean(burst_counts)), float(np.sum(burst_counts)), float(np.mean(long_stills)), float(np.max(long_stills)),
                float(np.mean(slopes)), float(np.std(slopes)),
            ]
            pair_energy = pair_feats[:, 0]
            pair_still = pair_feats[:, 6]
            pair_burst = pair_feats[:, 7]
            stat += [float(np.std(pair_energy)), float(np.std(pair_still)), float(np.std(pair_burst))]
        else:
            stat = [0.0] * 19
        rows_pair.append(pair_feats)
        rows_stat.append(np.asarray(stat, dtype=np.float32))
    return np.stack(rows_pair, axis=0), np.stack(rows_stat, axis=0)


def read_numeric_table(path):
    path = Path(path)
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(path, allow_pickle=True)
            return np.asarray(arr, dtype=float)
        if path.suffix.lower() == ".npz":
            z = np.load(path, allow_pickle=True)
            for k in z.files:
                a = z[k]
                if hasattr(a, "ndim") and np.issubdtype(a.dtype, np.number):
                    return np.asarray(a, dtype=float)
            return None
        if path.suffix.lower() in [".csv"]:
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, sep=None, engine="python")
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            return None
        return num.to_numpy(dtype=float)
    except Exception:
        return None


def load_gait_array(root, pid):
    d = Path(root) / "IMU" / str(pid)
    if not d.exists():
        return None
    mats = []
    for f in sorted(d.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in [".csv", ".txt", ".tsv", ".npy", ".npz"]:
            continue
        arr = read_numeric_table(f)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.size:
            mats.append(arr)
    if not mats:
        return None
    x = np.concatenate(mats, axis=0)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[1] < 9:
        x = np.concatenate([x, np.zeros((x.shape[0], 9 - x.shape[1]), dtype=np.float32)], axis=1)
    return x[:, :9]


def group_stats(x):
    x = np.asarray(x, dtype=np.float32)
    mag = np.linalg.norm(x, axis=1) if x.size else np.zeros(1, dtype=np.float32)
    diff = np.diff(x, axis=0)
    jerk = np.linalg.norm(diff, axis=1) if len(diff) else np.zeros(1, dtype=np.float32)
    return [
        float(np.mean(mag)), float(np.std(mag)), float(np.percentile(mag, 10)), float(np.percentile(mag, 50)),
        float(np.percentile(mag, 90)), float(np.max(mag)), float(np.mean(np.std(x, axis=0))),
        float(np.mean(jerk)), float(np.std(jerk)), float(np.percentile(jerk, 90)),
        float(np.mean(mag <= np.percentile(mag, 20))), float(np.mean(mag >= np.percentile(mag, 80))),
        slope_1d(mag), float(np.mean(np.abs(np.diff(mag))) if len(mag) > 1 else 0.0),
    ]


def compute_gait_extra(root, ids):
    rows = []
    for pid in ids:
        x = load_gait_array(root, pid)
        if x is None:
            x = np.zeros((1, 9), dtype=np.float32)
            exists = 0.0
        else:
            exists = 1.0
        acc, gyro, angle = x[:, 0:3], x[:, 3:6], x[:, 6:9]
        acc_s = group_stats(acc)
        gyro_s = group_stats(gyro)
        angle_s = group_stats(angle)
        ratios = [
            gyro_s[0] / (acc_s[0] + 1e-6), angle_s[0] / (acc_s[0] + 1e-6), gyro_s[0] / (angle_s[0] + 1e-6),
            gyro_s[7] / (acc_s[7] + 1e-6), angle_s[7] / (acc_s[7] + 1e-6),
        ]
        rows.append(np.asarray([exists] + acc_s + gyro_s + angle_s + ratios, dtype=np.float32))
    return np.stack(rows, axis=0)


def save_motion(ids, pair_mask, pair, stat, out_path, scaler=None, fit=False):
    N, P, Dp = pair.shape
    Ds = stat.shape[-1]
    pair2 = pair.reshape(N * P, Dp)
    if fit:
        pmu, psd = zscore_fit(pair2)
        smu, ssd = zscore_fit(stat)
        scaler = {"pair_mean": pmu.tolist(), "pair_std": psd.tolist(), "stat_mean": smu.tolist(), "stat_std": ssd.tolist()}
    else:
        pmu, psd = np.asarray(scaler["pair_mean"], dtype=np.float32), np.asarray(scaler["pair_std"], dtype=np.float32)
        smu, ssd = np.asarray(scaler["stat_mean"], dtype=np.float32), np.asarray(scaler["stat_std"], dtype=np.float32)
    pair_z = zscore_apply(pair2, pmu, psd).reshape(N, P, Dp)
    stat_z = zscore_apply(stat, smu, ssd)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ids=ids.astype(np.int64), pair_mask=pair_mask.astype(np.float32), motion_extra_pair=pair_z, motion_extra_stat=stat_z)
    return scaler


def save_gait(ids, feat, out_path, scaler=None, fit=False):
    if fit:
        mu, sd = zscore_fit(feat)
        scaler = {"mean": mu.tolist(), "std": sd.tolist()}
    else:
        mu, sd = np.asarray(scaler["mean"], dtype=np.float32), np.asarray(scaler["std"], dtype=np.float32)
    feat_z = zscore_apply(feat, mu, sd)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ids=ids.astype(np.int64), gait_extra=feat_z)
    return scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--test_root", required=True)
    ap.add_argument("--motion_train_npz", required=True)
    ap.add_argument("--motion_test_npz", required=True)
    ap.add_argument("--out_motion_train_npz", required=True)
    ap.add_argument("--out_motion_test_npz", required=True)
    ap.add_argument("--out_gait_train_npz", required=True)
    ap.add_argument("--out_gait_test_npz", required=True)
    ap.add_argument("--scaler_json", required=True)
    args = ap.parse_args()

    tr_ids, tr_seq, _, tr_mask = load_motion_npz(args.motion_train_npz)
    te_ids, te_seq, _, te_mask = load_motion_npz(args.motion_test_npz)
    still_thr, burst_thr = fit_motion_thresholds(tr_seq, tr_mask)
    print("[motion thresholds]", {"still_thr": still_thr, "burst_thr": burst_thr})

    tr_pair, tr_stat = compute_motion_extra(tr_ids, tr_seq, tr_mask, still_thr, burst_thr)
    te_pair, te_stat = compute_motion_extra(te_ids, te_seq, te_mask, still_thr, burst_thr)
    motion_scaler = save_motion(tr_ids, tr_mask, tr_pair, tr_stat, args.out_motion_train_npz, fit=True)
    save_motion(te_ids, te_mask, te_pair, te_stat, args.out_motion_test_npz, scaler=motion_scaler, fit=False)

    tr_gait = compute_gait_extra(args.train_root, tr_ids)
    te_gait = compute_gait_extra(args.test_root, te_ids)
    gait_scaler = save_gait(tr_ids, tr_gait, args.out_gait_train_npz, fit=True)
    save_gait(te_ids, te_gait, args.out_gait_test_npz, scaler=gait_scaler, fit=False)

    meta = {
        "motion_thresholds": {"still_thr": still_thr, "burst_thr": burst_thr},
        "motion_pair_dim": int(tr_pair.shape[-1]),
        "motion_stat_dim": int(tr_stat.shape[-1]),
        "gait_extra_dim": int(tr_gait.shape[-1]),
        "motion_scaler": motion_scaler,
        "gait_scaler": gait_scaler,
    }
    Path(args.scaler_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] motion train", args.out_motion_train_npz, tr_pair.shape, tr_stat.shape)
    print("[OK] motion test ", args.out_motion_test_npz, te_pair.shape, te_stat.shape)
    print("[OK] gait train  ", args.out_gait_train_npz, tr_gait.shape)
    print("[OK] gait test   ", args.out_gait_test_npz, te_gait.shape)
    print("[OK] scaler", args.scaler_json)


if __name__ == "__main__":
    main()
