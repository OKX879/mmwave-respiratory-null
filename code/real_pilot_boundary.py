#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速 pilot：真实信号上检验【层数结构性安全边界 L ≤ 4】是否成立。
Quick pilot: does the structural level bound L <= 4 hold on real signals?"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from matrix_eval import decompose_dwt                       # noqa: E402
from criteria_lib import estimate_hr_psd, DISPLAY_BAND       # noqa: E402

CACHE = Path(r"c:\Users\yxh\Desktop\2\runtime\产品\band_sweep_cache")
FS = 100.0

BASES = ["bior3.5", "db4", "sym4", "haar", "db8"]
LEVELS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
THRESHOLDS = ["none", "universal-soft", "sure", "minimax"]


def load_cases() -> list[dict]:
    cases = []
    for fp in sorted(CACHE.glob("patient_*.npz"),
                     key=lambda p: int(p.stem.split("_")[1])):
        d = np.load(fp, allow_pickle=True)
        x = np.asarray(d["cardiac"], dtype=float).ravel()
        g = d["gold"]
        gold = float(g) if np.ndim(g) == 0 else float(np.asarray(g).ravel()[0])
        if not np.isfinite(gold) or len(x) < 400:
            continue
        cases.append({"id": fp.stem, "x": x, "gold": gold})
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="T5-0 真实信号上的 L<=4 边界 pilot")
    ap.add_argument("--out", type=str, default=str(HERE / "real_pilot"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    print(f"[pilot] 载入 {len(cases)} 例（窄带 cardiac，100 Hz）")
    golds = np.array([c["gold"] for c in cases])
    print(f"[pilot] ECG 金标 HR: 中位 {np.median(golds):.1f} BPM，范围 "
          f"[{golds.min():.1f}, {golds.max():.1f}]")

    rows = []
    for base in BASES:
        for lv in LEVELS:
            for thr in THRESHOLDS:
                errs = []
                for c in cases:
                    d = decompose_dwt(c["x"], base, lv, thr, "symmetric", FS,
                                      separation="bandpass", need_full=False)
                    if d is None:
                        continue
                    hr = estimate_hr_psd(d["card"], FS, band=DISPLAY_BAND)
                    if np.isfinite(hr):
                        errs.append(abs(hr - c["gold"]))
                rows.append({"base": base, "level": lv, "threshold": thr,
                             "n": len(errs),
                             "hr_mae_med": float(np.median(errs)) if errs else float("nan")})
    import pandas as pd
    r = pd.DataFrame(rows)
    r.to_csv(out / "real_pilot_boundary.csv", index=False, encoding="utf-8")

    # ---- 呈现 1：仅按层数聚合（全部基/阈值）----
    print("\n[pilot] 各层数上的中位 HR MAE（跨 5 基 × 4 阈值 × 20 例的中位）")
    print(f"  {'L':>3s}{'aL上界/Hz':>12s}{'覆盖心搏带':>11s}{'中位HR_MAE':>12s}")
    for lv in LEVELS:
        aL = (FS / 2) / 2 ** lv
        cov = "是" if aL >= max(DISPLAY_BAND) else "否"
        sub = r[(r.level == lv) & (r.threshold != "none")]
        print(f"  {lv:>3d}{aL:>12.4f}{cov:>11s}{sub['hr_mae_med'].median():>12.2f}")

    # ---- 呈现 2：配对检验（同一 base×level×case，阈值化 vs none）----
    print("\n[pilot] 配对 ΔHR（阈值化 − none，中位；负值 = 阈值化更好）")
    piv = r.pivot_table(index=["base", "level"], columns="threshold",
                        values="hr_mae_med", aggfunc="median")
    if "none" in piv.columns:
        for lv in LEVELS:
            sub = piv.xs(lv, level="level")
            deltas = {t: float(np.nanmedian(sub[t] - sub["none"]))
                      for t in ["universal-soft", "sure", "minimax"] if t in sub.columns}
            print(f"  L{lv:>2d}  " + "  ".join(f"Δ({t})={v:+7.2f}"
                                               for t, v in deltas.items()))

    # ---- 呈现 3：最优配置 ----
    print("\n[pilot] 真实数据上的最优候选（候选级中位 HR MAE，前 8）")
    for _, x in r.sort_values("hr_mae_med").head(8).iterrows():
        print(f"  {x['base']:>8s} L{int(x['level'])} {x['threshold']:>15s}  "
              f"HR_MAE={x['hr_mae_med']:6.2f}  (n={int(x['n'])})")
    print(f"\n[pilot] 输出：{out / 'real_pilot_boundary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
