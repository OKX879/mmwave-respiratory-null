#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T5-1 真实数据回放：从 capsule（RS6240，8 通道，ECG 金标）捕获生产链的位移信号。

为什么需要这一步
----------------
`matrix_eval.py` 的输入是"位移序列"；而真实数据的原始素材是复数数据立方体
（`patient_*.mat` 的 `data_all`）。二者之间隔着生产链的空间前端
（静态杂波消除 → BinScorer/Viterbi 选 bin → 8 通道复数相干融合）。

为了遵守"只变一件事"原则，**前端必须由生产链本身提供**，不能自己重写：
因此本脚本直接 import `产品/benchmark_v1.py` 的 `run_pipeline_capture()`
（它已被项目用于统一基准，含 t0 全局时间对齐），只做"跑一遍 + 落盘"。

产出（每例一个 npz）
--------------------
<out>/patient_N.npz
    wide       完整链路融合的【宽带位移】(呼吸+心搏) —— 本研究的"mix"
    cardiac    窄带心搏通路（1-2.5 Hz），供对照
    gold_hr    ECG 金标心率（60000 / median(IBI)）
    ecg_lbl    ECG R 峰样本索引（逐拍金标，供 beat F1 / IBI RMSE）
    center_bin / angle / t0_wide / t0_card / final_hr / hr_conf

用法
----
    python real_capture.py                # 全量 20 例
    python real_capture.py --only 1 2     # 只跑指定例（用于冒烟）
    python real_capture.py --force        # 覆盖已有缓存
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PRODUCT = Path(r"c:\Users\yxh\Desktop\2\runtime\产品")
sys.path.insert(0, str(PRODUCT))

from benchmark_v1 import DATA, FS, load_patient, run_pipeline_capture  # noqa: E402

# 生产链在短信号上会打印大量 "信号过短" 警告（对回放无关）——静音
logging.disable(logging.WARNING)

HERE = Path(__file__).resolve().parent
OUT = HERE / "real_data"


def capture_one(fp: Path) -> dict:
    z, lbl = load_patient(fp)
    n_angles = z.shape[1]
    ang = []
    for ai in range(n_angles):
        r = run_pipeline_capture(z[:, ai, :, :])
        r["angle"] = ai
        ang.append(r)
    # 角度选择：与 benchmark_v1 完全一致（force_relock 优先，hr_conf 次之；全无 HR 时按 fallback 最近）
    gold_hr = (60000.0 / np.median(np.diff(lbl) / FS * 1000.0)) if len(lbl) > 1 else float("nan")
    fl = [x for x in ang if x["n_force_relock"] > 0]
    best = (max(fl, key=lambda x: float(x["hr_conf"] or 0.0)) if fl
            else max(ang, key=lambda x: float(x["hr_conf"] or 0.0)))
    if all(x["final_hr"] is None for x in ang):
        best = max(ang, key=lambda x: float(
            -abs((x["fallback"] if x["fallback"] is not None else 1e9) - gold_hr)))
    return {
        "wide": np.asarray(best["wide"], dtype=float),
        "cardiac": np.asarray(best["cardiac"], dtype=float),
        "gold_hr": float(gold_hr),
        "ecg_lbl": np.asarray(lbl, dtype=float).ravel(),
        "center_bin": int(best["center_bin"]),
        "angle": int(best["angle"]),
        "t0_wide": int(best["t0_wide"] if best["t0_wide"] is not None else -1),
        "t0_card": int(best["t0_card"] if best["t0_card"] is not None else -1),
        "final_hr": float(best["final_hr"] if best["final_hr"] is not None else np.nan),
        "hr_conf": float(best["hr_conf"] if best["hr_conf"] is not None else np.nan),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="T5-1 capsule 位移信号回放捕获")
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--only", type=int, nargs="*", default=None, help="只跑这些例号")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA.glob("patient_*.mat"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if args.only:
        keep = {f"patient_{i}" for i in args.only}
        files = [f for f in files if f.stem in keep]
    print(f"[capture] {len(files)} 例，fs={FS}Hz → {out}")

    for fp in files:
        dst = out / f"{fp.stem}.npz"
        if dst.exists() and not args.force:
            print(f"[{fp.stem}] 已有缓存，跳过")
            continue
        t0 = time.time()
        try:
            rec = capture_one(fp)
        except Exception as e:
            print(f"[{fp.stem}] FAIL: {type(e).__name__}: {e}")
            continue
        np.savez(dst, **rec)
        print(f"[{fp.stem}] ok  gold={rec['gold_hr']:.2f} BPM  bin={rec['center_bin']}  "
              f"angle={rec['angle']}  len_wide={len(rec['wide'])/FS:.1f}s  "
              f"final_hr={rec['final_hr']:.1f}  ({time.time()-t0:.1f}s)")
    print("[capture] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
