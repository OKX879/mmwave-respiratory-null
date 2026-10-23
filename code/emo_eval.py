#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""emo_dataset 跨硬件测试（IWR6843 60 GHz，15 受试者 × 19 trial，PPG 金标）。
emo_dataset cross-hardware test: IWR6843 at 60 GHz, 15 subjects x 19 trials, PPG gold standard."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))

from real_eval import (BASES, LEVELS, THRESHOLDS, BOUNDARY_MAIN, BOUNDARY_EXTRA,
                       FS, build_pool, eval_case, max_level)          # noqa: E402

BASE = Path(r"C:\Users\yxh\Desktop\emo_dataset\processed\02_processed_data")
FS_PPG = 200.0
WIN_S = 20.0


# ---------------------------------------------------------------------------
# 金标（与 产品/diag_external_validation.py 的 ppg_gold_beats / gold_hr_from_beats 同参数）
# ---------------------------------------------------------------------------
def ppg_gold_beats(ppg: np.ndarray, fs: float = FS_PPG) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt, find_peaks
    nyq = fs / 2.0
    sos = butter(4, [0.6 / nyq, 5.0 / nyq], btype="band", output="sos")
    sig = sosfiltfilt(sos, ppg - np.mean(ppg))
    pk, _ = find_peaks(sig, distance=int(0.4 * fs), prominence=0.3 * np.std(sig))
    return pk / fs


def gold_hr_from_beats(beat_s: np.ndarray) -> float:
    if len(beat_s) < 5:
        return float("nan")
    ibi = np.diff(beat_s)
    ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]
    if len(ibi) < 4:
        return float("nan")
    return float(60.0 / np.median(ibi))


def subject_of(stem: str) -> str:
    return stem.split("_")[1]


def list_trials(max_trials: int) -> list[Path]:
    mw = sorted((BASE / "mmwave").glob("mmwave_*.csv"))
    if max_trials <= 0:
        return mw
    # 受试者分层抽样：先按受试者分组，再轮流取，保证覆盖全部 15 人
    by_sub: dict[str, list[Path]] = {}
    for p in mw:
        by_sub.setdefault(subject_of(p.stem), []).append(p)
    subs = sorted(by_sub)
    out, k = [], 0
    while len(out) < max_trials:
        added = False
        for s in subs:
            if k < len(by_sub[s]) and len(out) < max_trials:
                out.append(by_sub[s][k])
                added = True
        if not added:
            break
        k += 1
    return sorted(out)


# ---------------------------------------------------------------------------
def load_trial(mw_path: Path, mode: str, window: str) -> dict | None:
    d = pd.read_csv(mw_path)
    if "vital" not in d.columns:
        return None
    x_all = np.asarray(d["vital"], dtype=float)
    n20 = int(WIN_S * FS)
    if window == "full":
        x = x_all
        n_ppg = 2 * len(x_all)          # 实测 PPG:mmwave 行数比恒为 2.0000
    else:
        if len(x_all) < n20:
            return None
        x = x_all[:n20]
        n_ppg = 2 * n20
    if mode == "cumsum":
        x = np.cumsum(x)
    x = x - np.mean(x)

    pp_path = BASE / "ppg" / f"ppg_{subject_of(mw_path.stem)}_{mw_path.stem.split('_')[2]}.csv"
    if not pp_path.exists():
        return None
    ppg = np.asarray(pd.read_csv(pp_path, header=None).iloc[:, 0], dtype=float)
    ppg = ppg[:min(n_ppg, len(ppg))]
    if len(ppg) < int(2.0 * FS_PPG):
        return None
    beats = ppg_gold_beats(ppg)
    gold = gold_hr_from_beats(beats)
    if not np.isfinite(gold):
        return None
    return {"wide": x, "gold_hr": gold, "ecg_lbl": beats * FS,
            "src": mw_path.stem, "subject": subject_of(mw_path.stem)}


# ---------------------------------------------------------------------------
# 并行工作进程（按【候选】切分；信号只 pickle 一次/进程）
# ---------------------------------------------------------------------------
_WCTX: dict = {}


def _init_worker(recs: list[dict]) -> None:
    _WCTX.clear()
    _WCTX["recs"] = recs


def _worker_cand(cand: dict) -> list[dict]:
    recs = _WCTX["recs"]
    mxl_cache: dict[tuple[str, int], int] = {}
    rows = []
    for rec in recs:
        row = eval_case(cand, rec)
        if row is None:
            continue
        key = (cand["wavelet"], len(rec["wide"]))
        if key not in mxl_cache:      # full 窗口下各 trial 长度不同 → 必须逐例算
            mxl_cache[key] = max_level(*key)
        mxl = mxl_cache[key]
        row.update({"wavelet": cand["wavelet"], "level": cand["level"],
                    "threshold": cand["threshold"], "boundary": cand["boundary"],
                    "separation": cand["separation"], "max_level": mxl,
                    "level_over_max": int(cand["level"] > mxl) if mxl > 0 else 0,
                    "subject": rec["subject"]})
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="T5-4 emo_dataset 跨硬件测试")
    ap.add_argument("--mode", choices=["vital", "cumsum"], default="vital")
    ap.add_argument("--window", choices=["20", "full"], default="20")
    ap.add_argument("--max-trials", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=1,
                    help="并行进程数（按候选切分）。单进程约 18 ms/候选/例；"
                         "285 trial × 525 候选单进程约 45 min，4 进程约 12 min")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else HERE / "emo_matrix"
    out.mkdir(parents=True, exist_ok=True)

    trials = list_trials(args.max_trials)
    recs = []
    for i, p in enumerate(trials):
        r = load_trial(p, args.mode, args.window)
        if r is None:
            continue
        r["sid"] = i
        recs.append(r)
    subs = sorted({r["subject"] for r in recs})
    print(f"[emo] mode={args.mode} window={args.window}  载入 {len(recs)} trial / {len(subs)} 受试者")
    golds = np.array([r["gold_hr"] for r in recs])
    print(f"[emo] PPG 金标 HR: 中位 {np.median(golds):.1f} BPM  范围 [{golds.min():.1f}, {golds.max():.1f}]")
    print(f"[emo] ⚠ 两变量变化：硬件 IWR6843 + 参考论文自己的空间前端（非我们生产链）")

    pool_cands = build_pool()
    print(f"[emo] 候选池 {len(pool_cands)}（与 capsule / 合成同口径）；"
          f"评估量 {len(pool_cands)*len(recs)}")

    raw_rows, t0 = [], time.time()
    if args.jobs and args.jobs > 1:
        import multiprocessing as mp
        print(f"[emo] 并行 {args.jobs} 进程（按候选切分）")
        with mp.Pool(processes=args.jobs, initializer=_init_worker,
                     initargs=(recs,)) as mp_pool:
            for ci, rows in enumerate(mp_pool.imap_unordered(_worker_cand, pool_cands,
                                                             chunksize=2)):
                raw_rows.extend(rows)
                if (ci + 1) % 50 == 0 or ci == 0:
                    el = time.time() - t0
                    print(f"  [{ci+1}/{len(pool_cands)}] rows={len(raw_rows)} "
                          f"({el:.0f}s elapsed, "
                          f"~{el/(ci+1)*(len(pool_cands)-ci-1):.0f}s left)")
    else:
        _init_worker(recs)               # 串行路径：直接把 recs 注入上下文
        for ci, cand in enumerate(pool_cands):
            raw_rows.extend(_worker_cand(cand))
            if (ci + 1) % 50 == 0 or ci == 0:
                el = time.time() - t0
                print(f"  [{ci+1}/{len(pool_cands)}] rows={len(raw_rows)} ({el:.0f}s elapsed, "
                      f"~{el/(ci+1)*(len(pool_cands)-ci-1):.0f}s left)")

    raw = pd.DataFrame(raw_rows)
    if raw.empty:
        print("[emo] 无有效结果")
        return 1
    tag = f"{args.mode}_{args.window}"
    raw.to_csv(out / f"emo_raw_{tag}.csv", index=False, encoding="utf-8")

    metric_cols = [c for c in raw.columns
                   if c not in ("candidate_id", "sample_id", "wavelet", "level",
                                "threshold", "boundary", "separation", "max_level",
                                "level_over_max", "subject")]
    agg = raw.groupby("candidate_id")[metric_cols].median(numeric_only=True)
    meta = raw.groupby("candidate_id")[["wavelet", "level", "threshold", "boundary",
                                        "separation", "max_level", "level_over_max"]].first()
    cnt = raw.groupby("candidate_id")["hr_mae_bpm"].count().rename("n_ok")
    res = pd.concat([meta, cnt, agg], axis=1).reset_index()
    res.to_csv(out / f"emo_results_{tag}.csv", index=False, encoding="utf-8")

    (out / f"emo_config_{tag}.json").write_text(json.dumps({
        "dataset": "emo_dataset (IWR6843, 60 GHz, 100 Hz), PPG 200 Hz gold",
        "warning": "two-variable change: hardware + reference-paper spatial front-end (NOT our chain)",
        "mode": args.mode, "window": args.window, "n_trials": len(recs),
        "n_subjects": len(subs), "subjects": subs, "fs": FS,
        "gold": "PPG 0.6-5Hz BPF + peak detection -> median IBI -> BPM (same as diag_external_validation)",
        "pool_n": len(pool_cands), "bases": BASES, "levels": LEVELS,
        "thresholds": THRESHOLDS, "boundary_main": BOUNDARY_MAIN,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[emo] 候选级最优前 8（{tag}）")
    print(f"  {'候选':46s}{'HR_MAE':>9s}{'beat_F1':>9s}{'n_ok':>6s}")
    for _, r in res.sort_values("hr_mae_bpm").head(8).iterrows():
        print(f"  {r['candidate_id']:46s}{r['hr_mae_bpm']:9.2f}"
              f"{r.get('beat_f1', float('nan')):9.3f}{int(r['n_ok']):6d}")
    print(f"  池中位 {res['hr_mae_bpm'].median():.2f}  min {res['hr_mae_bpm'].min():.2f}"
          f"  max {res['hr_mae_bpm'].max():.2f}")
    print(f"\n[emo] 输出：{out}/emo_raw_{tag}.csv / emo_results_{tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
