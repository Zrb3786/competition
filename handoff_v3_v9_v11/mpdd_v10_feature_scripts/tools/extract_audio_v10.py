#!/usr/bin/env python3
"""Extract v10 acoustic features for MPDD Elder.

Outputs an NPZ:
  ids: [N]
  pair_mask: [N,4]
  audio_big_pair: [N,4,D]

Features can include:
  - WavLM hidden-state statistics
  - emotion2vec utterance embedding
  - Whisper / faster-whisper transcript statistics

This script is intentionally robust: if a model fails and --require_* is 0,
it fills that component with zeros/empty features and continues.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

PAIR_COUNT = 4
SR = 16000


def read_ids(csv_path: str | Path) -> List[int]:
    df = pd.read_csv(csv_path)
    id_col = "ID" if "ID" in df.columns else ("id" if "id" in df.columns else df.columns[0])
    return sorted([int(x) for x in df[id_col].tolist()])


def find_audio_file(audio_root: Path, pid: int, pair: int) -> Optional[Path]:
    d = audio_root / str(pid)
    candidates = [
        d / f"A_{pair}.WAV", d / f"A_{pair}.wav", d / f"a_{pair}.wav",
        d / f"audio_{pair}.wav", d / f"{pair}.wav",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    # fallback glob
    if d.exists():
        pats = [f"*{pair}*.wav", f"*{pair}*.WAV"]
        for pat in pats:
            xs = sorted(d.glob(pat))
            if xs:
                return xs[0]
    return None


def load_audio(path: Path, target_sr: int = SR) -> Tuple[np.ndarray, float]:
    import soundfile as sf
    import librosa

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    wav = np.asarray(wav, dtype=np.float32)
    dur = float(len(wav) / target_sr) if len(wav) else 0.0
    if len(wav) == 0:
        wav = np.zeros(target_sr, dtype=np.float32)
    return wav, dur


def basic_audio_stats(wav: np.ndarray, dur: float) -> np.ndarray:
    if wav.size == 0:
        wav = np.zeros(SR, dtype=np.float32)
    absw = np.abs(wav)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(wav).astype(np.float32))))) if len(wav) > 1 else 0.0
    energy = wav.astype(np.float64) ** 2
    return np.asarray([
        dur,
        float(np.mean(wav)), float(np.std(wav)),
        float(np.mean(absw)), float(np.std(absw)), float(np.percentile(absw, 90)), float(np.max(absw)),
        float(np.mean(energy)), float(np.std(energy)), zcr,
    ], dtype=np.float32)


def init_wavlm(model_name: str, device: str):
    import torch
    from transformers import AutoFeatureExtractor, WavLMModel
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return fe, model


def wavlm_features(wav: np.ndarray, fe: Any, model: Any, layers: Sequence[int], device: str) -> np.ndarray:
    import torch
    with torch.no_grad():
        inp = fe(wav, sampling_rate=SR, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        out = model(**inp, output_hidden_states=True)
        hs = out.hidden_states
        feats: List[np.ndarray] = []
        for l in layers:
            li = int(l)
            if li < 0:
                li = len(hs) + li
            li = max(0, min(li, len(hs) - 1))
            h = hs[li].detach().float().cpu().numpy()[0]
            feats.extend([
                h.mean(axis=0),
                h.std(axis=0),
                h.max(axis=0),
            ])
        return np.concatenate(feats).astype(np.float32)


def init_emotion2vec(model_name: str):
    from funasr import AutoModel
    return AutoModel(model=model_name)


def _find_array(obj: Any) -> Optional[np.ndarray]:
    """Robustly find the first feature-like numeric array in FunASR outputs."""
    import torch
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        arr = obj
    elif "torch" in str(type(obj)) and isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
    elif isinstance(obj, (list, tuple)):
        # prefer nested arrays with larger size
        cands = []
        for x in obj:
            a = _find_array(x)
            if a is not None:
                cands.append(a)
        if not cands:
            return None
        return max(cands, key=lambda a: a.size)
    elif isinstance(obj, dict):
        preferred = ["feats", "feat", "embedding", "emb", "xvector", "hidden_states"]
        for k in preferred:
            if k in obj:
                a = _find_array(obj[k])
                if a is not None:
                    return a
        cands = []
        for v in obj.values():
            a = _find_array(v)
            if a is not None:
                cands.append(a)
        if not cands:
            return None
        return max(cands, key=lambda a: a.size)
    else:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size < 8:
        return None
    return arr


def emotion2vec_features(path: Path, model: Any) -> np.ndarray:
    res = model.generate(input=str(path), granularity="utterance", extract_embedding=True)
    arr = _find_array(res)
    if arr is None:
        return np.zeros(0, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    # frame-level or [1,T,D], summarize
    arr = arr.reshape(-1, arr.shape[-1])
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr.max(axis=0)]).astype(np.float32)


def init_whisper(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def count_zh(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def whisper_features(path: Path, model: Any, language: str = "zh") -> Tuple[np.ndarray, str]:
    segments, info = model.transcribe(
        str(path),
        language=language if language else None,
        task="transcribe",
        vad_filter=True,
        beam_size=5,
    )
    texts: List[str] = []
    seg_durs: List[float] = []
    avg_logprobs: List[float] = []
    no_speech_probs: List[float] = []
    comp_ratios: List[float] = []
    for seg in segments:
        txt = (seg.text or "").strip()
        texts.append(txt)
        seg_durs.append(max(0.0, float(seg.end - seg.start)))
        avg_logprobs.append(float(getattr(seg, "avg_logprob", 0.0) or 0.0))
        no_speech_probs.append(float(getattr(seg, "no_speech_prob", 0.0) or 0.0))
        comp_ratios.append(float(getattr(seg, "compression_ratio", 0.0) or 0.0))
    text = " ".join(texts).strip()
    speech_dur = float(sum(seg_durs))
    nseg = len(seg_durs)
    chars = len(text)
    zh_chars = count_zh(text)
    ascii_words = len(re.findall(r"[A-Za-z]+", text))
    feat = np.asarray([
        nseg,
        speech_dur,
        np.mean(seg_durs) if seg_durs else 0.0,
        np.std(seg_durs) if seg_durs else 0.0,
        np.mean(avg_logprobs) if avg_logprobs else 0.0,
        np.std(avg_logprobs) if avg_logprobs else 0.0,
        np.mean(no_speech_probs) if no_speech_probs else 0.0,
        np.mean(comp_ratios) if comp_ratios else 0.0,
        chars,
        zh_chars,
        ascii_words,
        zh_chars / max(speech_dur, 1e-6),
        ascii_words / max(speech_dur, 1e-6),
    ], dtype=np.float32)
    return feat, text


def pad_vecs(vecs: List[List[Optional[np.ndarray]]]) -> Tuple[np.ndarray, np.ndarray]:
    # vecs: N x 4 list
    maxd = 0
    for row in vecs:
        for v in row:
            if v is not None:
                maxd = max(maxd, int(v.shape[0]))
    arr = np.zeros((len(vecs), PAIR_COUNT, maxd), dtype=np.float32)
    mask = np.zeros((len(vecs), PAIR_COUNT), dtype=np.float32)
    for i, row in enumerate(vecs):
        for j, v in enumerate(row):
            if v is None:
                continue
            arr[i, j, : v.shape[0]] = v.astype(np.float32)
            mask[i, j] = 1.0
    return arr, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_root", required=True, help=".../Elder/audio")
    ap.add_argument("--id_csv", required=True)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--transcript_csv", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wavlm_model", default="microsoft/wavlm-large")
    ap.add_argument("--wavlm_layers", default="6,12,18,24")
    ap.add_argument("--enable_wavlm", type=int, default=1)
    ap.add_argument("--require_wavlm", type=int, default=0)
    ap.add_argument("--emotion2vec_model", default="iic/emotion2vec_plus_large")
    ap.add_argument("--enable_emotion2vec", type=int, default=1)
    ap.add_argument("--require_emotion2vec", type=int, default=0)
    ap.add_argument("--whisper_model", default="large-v3")
    ap.add_argument("--whisper_device", default="cuda")
    ap.add_argument("--whisper_compute_type", default="float16")
    ap.add_argument("--whisper_language", default="zh")
    ap.add_argument("--enable_whisper", type=int, default=1)
    ap.add_argument("--require_whisper", type=int, default=0)
    args = ap.parse_args()

    audio_root = Path(args.audio_root)
    ids = read_ids(args.id_csv)
    layers = [int(x) for x in args.wavlm_layers.split(",") if x.strip()]

    wavlm = None
    if args.enable_wavlm:
        try:
            wavlm = init_wavlm(args.wavlm_model, args.device)
            print("[OK] loaded WavLM", args.wavlm_model)
        except Exception as e:
            print("[WARN] failed loading WavLM:", repr(e))
            if args.require_wavlm:
                raise

    e2v = None
    if args.enable_emotion2vec:
        try:
            e2v = init_emotion2vec(args.emotion2vec_model)
            print("[OK] loaded emotion2vec", args.emotion2vec_model)
        except Exception as e:
            print("[WARN] failed loading emotion2vec:", repr(e))
            if args.require_emotion2vec:
                raise

    whisper = None
    if args.enable_whisper:
        try:
            whisper = init_whisper(args.whisper_model, args.whisper_device, args.whisper_compute_type)
            print("[OK] loaded faster-whisper", args.whisper_model)
        except Exception as e:
            print("[WARN] failed loading Whisper:", repr(e))
            if args.require_whisper:
                raise

    all_vecs: List[List[Optional[np.ndarray]]] = []
    rows: List[Dict[str, Any]] = []

    for pid in tqdm(ids, desc="audio ids"):
        row_vecs: List[Optional[np.ndarray]] = []
        for pair in range(1, PAIR_COUNT + 1):
            path = find_audio_file(audio_root, pid, pair)
            if path is None:
                row_vecs.append(None)
                rows.append({"ID": pid, "pair": pair, "path": "", "text": "", "status": "missing"})
                continue
            try:
                wav, dur = load_audio(path, SR)
                feats: List[np.ndarray] = [basic_audio_stats(wav, dur)]
                text = ""
                if wavlm is not None:
                    feats.append(wavlm_features(wav, wavlm[0], wavlm[1], layers, args.device))
                if e2v is not None:
                    try:
                        feats.append(emotion2vec_features(path, e2v))
                    except Exception as e:
                        print(f"[WARN] emotion2vec failed ID={pid} pair={pair}: {e}")
                        if args.require_emotion2vec:
                            raise
                if whisper is not None:
                    try:
                        wf, text = whisper_features(path, whisper, args.whisper_language)
                        feats.append(wf)
                    except Exception as e:
                        print(f"[WARN] whisper failed ID={pid} pair={pair}: {e}")
                        if args.require_whisper:
                            raise
                vec = np.concatenate([f.reshape(-1).astype(np.float32) for f in feats if f.size > 0]).astype(np.float32)
                row_vecs.append(vec)
                rows.append({"ID": pid, "pair": pair, "path": str(path), "text": text, "status": "ok", "dim": int(vec.shape[0])})
            except Exception as e:
                print(f"[WARN] failed ID={pid} pair={pair}: {e}")
                row_vecs.append(None)
                rows.append({"ID": pid, "pair": pair, "path": str(path), "text": "", "status": f"error:{type(e).__name__}"})
        all_vecs.append(row_vecs)

    arr, mask = pad_vecs(all_vecs)
    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=np.asarray(ids, dtype=np.int64), pair_mask=mask, audio_big_pair=arr)
    pd.DataFrame(rows).to_csv(args.transcript_csv, index=False, encoding="utf-8-sig")
    meta = {"ids": len(ids), "dim": int(arr.shape[-1]), "shape": list(arr.shape), "wavlm_model": args.wavlm_model,
            "wavlm_layers": layers, "emotion2vec_model": args.emotion2vec_model, "whisper_model": args.whisper_model}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] saved", out, "shape=", arr.shape, "mask valid=", int(mask.sum()))
    print("[OK] transcripts", args.transcript_csv)


if __name__ == "__main__":
    main()
