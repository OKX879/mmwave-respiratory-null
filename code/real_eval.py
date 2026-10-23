#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T5-2 真实数据矩阵评估：在 capsule 的真实【宽带位移】上跑缩减候选池。

与合成矩阵的关系
----------------
复用完全相同的实现（`matrix_eval.decompose_dwt` + `criteria_lib.evaluate_all` +
`criteria_lib.estimate_hr_psd`），**只换输入信号**。这样"基 × 层数 × 阈值 × 边界"
的比较口径与合成侧逐字一致，差异只能归因于数据本身。

与合成侧的**实质差别（必须如实声明）**
--------------------------------------
1. **没有干净源** → 合成侧可算的 SDR / corr_card 在真实数据上不可得；
   真实侧金标只有 **HR MAE（对 ECG R 峰）**，以及逐拍 beat F1（简化检测器）。
2. **单通道** → rc1（跨通道相干）/ rc2（跨距离格相干）在真实侧仍为 nan
   （rc2 需要原始立方体的邻 bin，本步骤不做，见 T5-3）。
3. **信号长度 20 s**（生产链 history），比合成侧的 30 s 短。

候选池为何缩减
--------------
T5 是**定向迁移检验**，不是重跑 2025 候选矩阵：20 例样本上跑满池统计功效不足。
因此池 = 15 基 × L∈{2..6} × 5 阈值 × 主边界(symmetric) 375 个，
外加 L=4 × 其余两种边界 150 个（供边界零效应的**配对**检验）。

用法
----
    python real_eval.py                      # 全量
    python real_eval.py --only 1 2           # 冒烟
    python real_eval.py --jobs 4             # 并行（可选）
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pywt

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))

from matrix_eval import decompose_dwt                        # noqa: E402
from criteria_lib import (                                    # noqa: E402
    DISPLAY_BAND, evaluate_all, estimate_hr_psd,
)

BASES = ["haar", "db4", "db6", "db8", "db10",
         "sym4", "sym6", "sym8", "coif2", "coif4",
         "bior3.5", "bior4.4", "bior6.8", "dmey", "rbio6.8"]
LEVELS = [2, 3, 4, 5, 6]
THRESHOLDS = ["none", "universal-soft", "universal-hard", "sure", "minimax"]
BOUNDARY_MAIN = "symmetric"          # 主边界（合成侧已证边界零效应）
BOUNDARY_EXTRA = ["zero", "periodic"]  # 仅在 L=4 上做配对检验
FS = 100.0


# ---------------------------------------------------------------------------
def build_pool() -> list[dict]:
    cands: list[dict] = []
    def add(b, lv, th, bd):
        cands.append({"kind": "dwt", "wavelet": b, "level": lv, "threshold": th,
                      "boundary": bd, "separation": "bandpass",
                      "id": f"dwt|{b}|L{lv}|{th}|{bd}|bandpass"})
    for b in BASES:
        for lv in LEVELS:
            for th in THRESHOLDS:
                add(b, lv, th, BOUNDARY_MAIN)
    for b in BASES:
        for th in THRESHOLDS:
            for bd in BOUNDARY_EXTRA:
                add(b, 4, th, bd)
    # 去重
    seen, out = set(), []
    for c in cands:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        out.append(c)
    return out


def max_level(w: str, n: int) -> int:
    try:
        return int(pywt.dwt_max_level(n, pywt.Wavelet(w).dec_len))
    except Exception:
        return -1


# ---------------------------------------------------------------------------
def beat_f1(pred_ms: np.ndarray, gold_ms: np.ndarray, fs: float,
            tol_ms: float = 150.0, lag_max_ms: float = 400.0,
            lag_step_ms: float = 20.0) -> tuple[float, float, float, float]:
    """简化逐拍 F1：全局滞后对齐（±400 ms，20 ms 步长）+ ±150 ms 容差配对。

    与 benchmark_v1 的 beat 口径同构，但检测器为简化实现（见 detect_beats），
    **不能与生产链的 beat F1 直接比较**。
    """
    if len(pred_ms) < 3 or len(gold_ms) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    best = (float("nan"), float("nan"), float("nan"), float("nan"))
    best_f1 = -1.0
    lags = np.arange(-lag_max_ms, lag_max_ms + 1e-9, lag_step_ms)
    for lag in lags:
        p = pred_ms - lag
        used = np.zeros(len(gold_ms), dtype=bool)
        tp = 0
        for x in p:
            d = np.abs(gold_ms - x)
            j = int(np.argmin(d))
            if d[j] <= tol_ms and not used[j]:
                used[j] = True
                tp += 1
        fp = len(p) - tp
        fn = len(gold_ms) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best = (f1, prec, rec, float(lag))
    return best


def detect_beats(x: np.ndarray, fs: float) -> np.ndarray:
    """简化心搏检测：对心搏分量做自适应阈值峰检测，返回峰时刻（ms）。

    生产链的 EventDetector 使用范围为 [333, 1500] ms 的 IBI 门控；此处做同样的
    生理过滤，但阈值取 max(0.35σ, 0.6×局部窗最大值)（项目既有的弱峰改进思路）。

    性能：局部最大值窗用 `maximum_filter1d`（C 实现，O(n)），而不是 Python 逐点
    循环 —— 后者在原实现里是 O(n·w)，实测单次 5.4 ms，是 285 trial 全量评估的
    主要瓶颈之一（单候选单例 24 ms 里占 22%）。
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import maximum_filter1d
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    sd = float(np.std(x))
    if sd <= 0:
        return np.array([])
    w = max(int(2.0 * fs), 8)
    loc_max = maximum_filter1d(np.abs(x), size=2 * w + 1, mode="nearest")
    thr = np.maximum(0.35 * sd, 0.6 * np.where(loc_max > 0, loc_max, np.inf))
    pk, _ = find_peaks(np.abs(x), height=thr, distance=int(0.3 * fs))
    if len(pk) == 0:
        return np.array([])
    # IBI 生理过滤：迭代剔除越界间隔
    keep = np.ones(len(pk), dtype=bool)
    for _ in range(3):
        ms = pk[keep] / fs * 1000.0
        if len(ms) < 3:
            break
        d = np.diff(ms)
        bad = (d < 333) | (d > 1500)
        if not bad.any():
            break
        idx = np.where(keep)[0]
        for k in np.where(bad)[0]:
            keep[idx[k + 1]] = False
    return pk[keep] / fs * 1000.0


# ---------------------------------------------------------------------------
def eval_case(cand: dict, rec: dict) -> dict | None:
    x = rec["wide"]
    n = len(x)
    d = decompose_dwt(x, cand["wavelet"], cand["level"], cand["threshold"],
                      cand["boundary"], FS, separation="bandpass", need_full=True)
    if d is None:
        return None
    res = {
        "card": d["card"], "resp": d["resp"], "mix": x, "x_fresp": x,
        "recon": d["recon"], "card_ch": [d["card"]],
        "card_per_res": d.get("card_per_res") or [], "coeffs": d.get("coeffs"),
    }
    crit = evaluate_all(res, FS)
    hr = estimate_hr_psd(d["card"], FS, band=DISPLAY_BAND)
    gold = rec["gold_hr"]
    row = {"candidate_id": cand["id"], "sample_id": rec["sid"],
           **crit,
           "hr_mae_bpm": abs(hr - gold) if np.isfinite(hr) else float("nan"),
           "hr_est": hr, "gold_hr": gold}
    # 逐拍（简化检测器）
    try:
        gms = np.asarray(rec["ecg_lbl"], dtype=float) / FS * 1000.0
        pms = detect_beats(d["card"], FS)
        f1, prec, rec_, lag = beat_f1(pms, gms, FS)
        row.update({"beat_f1": f1, "beat_prec": prec, "beat_rec": rec_, "beat_lag_ms": lag})
    except Exception:
        row.update({"beat_f1": float("nan"), "beat_prec": float("nan"),
                    "beat_rec": float("nan"), "beat_lag_ms": float("nan")})
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="T5-2 真实数据矩阵评估")
    ap.add_argument("--data", type=str, default=str(HERE / "real_data"))
    ap.add_argument("--out", type=str, default=str(HERE / "real_matrix"))
    ap.add_argument("--only", type=int, nargs="*", default=None)
    args = ap.parse_args(argv)

    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("patient_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if args.only:
        keep = {f"patient_{i}" for i in args.only}
        files = [f for f in files if f.stem in keep]
    recs = []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        rec = {k: d[k] for k in d.files}
        rec["sid"] = int(fp.stem.split("_")[1])
        rec["wide"] = np.asarray(rec["wide"], dtype=float).ravel()
        rec["gold_hr"] = float(rec["gold_hr"])
        if not np.isfinite(rec["gold_hr"]) or len(rec["wide"]) < 400:
            print(f"[real_eval] 跳过 {fp.stem}（gold 或长度无效）")
            continue
        recs.append(rec)
    print(f"[real_eval] {len(recs)} 例真实数据；信号长度 "
          f"{sorted({len(r['wide']) for r in recs})} 采样 @ {FS}Hz")

    pool = build_pool()
    print(f"[real_eval] 候选池 {len(pool)} 个（{len(BASES)} 基 × L{LEVELS} × "
          f"{len(THRESHOLDS)} 阈值 × 主边界 + L4 边界配对）")
    print(f"[real_eval] 总评估量 {len(pool)*len(recs)}（单进程，预计数分钟）")

    raw_rows: list[dict] = []
    t0 = time.time()
    for ci, cand in enumerate(pool):
        n_ok = 0
        mxl = max_level(cand["wavelet"], len(recs[0]["wide"]))
        for rec in recs:
            row = eval_case(cand, rec)
            if row is None:
                continue
            row["wavelet"] = cand["wavelet"]
            row["level"] = cand["level"]
            row["threshold"] = cand["threshold"]
            row["boundary"] = cand["boundary"]
            row["separation"] = cand["separation"]
            row["max_level"] = mxl
            row["level_over_max"] = int(cand["level"] > mxl) if mxl > 0 else 0
            raw_rows.append(row)
            n_ok += 1
        if (ci + 1) % 50 == 0 or ci == 0:
            el = time.time() - t0
            eta = el / (ci + 1) * (len(pool) - ci - 1)
            print(f"  [{ci+1}/{len(pool)}] {cand['id'][:40]:42s} ok={n_ok} "
                  f"({el:.0f}s elapsed, ~{eta:.0f}s left)")

    import pandas as pd
    raw = pd.DataFrame(raw_rows)
    if raw.empty:
        print("[real_eval] 无有效结果")
        return 1
    raw.to_csv(out / "real_matrix_raw.csv", index=False, encoding="utf-8")

    metric_cols = [c for c in raw.columns
                   if c not in ("candidate_id", "sample_id", "wavelet", "level",
                                "threshold", "boundary", "separation", "max_level",
                                "level_over_max")]
    agg = raw.groupby("candidate_id")[metric_cols].median(numeric_only=True)
    meta = raw.groupby("candidate_id")[["wavelet", "level", "threshold", "boundary",
                                        "separation", "max_level",
                                        "level_over_max"]].first()
    counts = raw.groupby("candidate_id")["hr_mae_bpm"].count().rename("n_ok")
    res = pd.concat([meta, counts, agg], axis=1).reset_index()
    res.to_csv(out / "real_matrix_results.csv", index=False, encoding="utf-8")

    (out / "real_matrix_config.json").write_text(json.dumps({
        "data_dir": str(data_dir), "n_cases": len(recs), "fs": FS,
        "pool": {"n": len(pool), "bases": BASES, "levels": LEVELS,
                 "thresholds": THRESHOLDS, "boundary_main": BOUNDARY_MAIN,
                 "boundary_extra_L4": BOUNDARY_EXTRA},
        "gold": "HR MAE vs ECG R peaks (no clean source -> SDR/corr unavailable)",
        "beat": "simplified detector, lag +-400ms / tol 150ms (NOT comparable to production chain)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[real_eval] 候选级汇总（前 10，按中位 HR MAE）")
    print(f"  {'候选':46s}{'HR_MAE':>9s}{'beat_F1':>9s}{'n_ok':>6s}")
    for _, r in res.sort_values("hr_mae_bpm").head(10).iterrows():
        print(f"  {r['candidate_id']:46s}{r['hr_mae_bpm']:9.2f}"
              f"{r.get('beat_f1', float('nan')):9.3f}{int(r['n_ok']):6d}")
    print(f"  全体候选 HR_MAE 中位 = {res['hr_mae_bpm'].median():.2f} BPM"
          f"（min {res['hr_mae_bpm'].min():.2f} / max {res['hr_mae_bpm'].max():.2f}）")
    print(f"\n[real_eval] 输出：{out}/real_matrix_raw.csv / real_matrix_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
