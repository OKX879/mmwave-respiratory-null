#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q_D 在 20 条 capsule 记录上到底有没有触发？——逐 bin 复核 D 的取值。
Does Q_D ever trigger on the 20 capsule records? A per-bin review of D."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

DATA = Path(r"C:\Users\yxh\Desktop\capsule-4604022-data\Minimum_dataset")
NPZ = Path(__file__).resolve().parent / "real_data"
DARK = 1e-3
D_ONSET, D_SAT = 0.01, 0.05      # 与 bin_scorer.py: 1% 起罚、5% 饱和，w_D=0.25
WIN = 2000


def dip(z: np.ndarray) -> float:
    """z: (frames, channels) → 通道凹陷占比的中位数（同 bin_scorer.dip_ratio）。
z of shape (frames, channels) -> median over channels of the dip ratio, same as bin_scorer.dip_ratio."""
    a = np.abs(z)
    vals = []
    for ci in range(a.shape[1]):
        m = float(np.median(a[:, ci]))
        if m > DARK:
            vals.append(float(np.mean(a[:, ci] < 0.3 * m)))
    return float(np.median(vals)) if vals else 0.0


def main() -> int:
    rows = []
    for i in range(1, 21):
        with h5py.File(DATA / f"patient_{i}.mat", "r") as f:
            r = f[f["data_all"][0, 0]][:]
        z = r["real"] + 1j * r["imag"]                # (ch, ang, bin, frame)
        d = np.load(NPZ / f"patient_{i}.npz", allow_pickle=True)
        cb, ang, t0 = int(d["center_bin"]), int(d["angle"]), int(d["t0_wide"])
        zf = np.transpose(z, (3, 1, 2, 0))            # (frame, ang, bin, ch)

        full = np.array([[dip(zf[:, a, b, :]) for b in range(zf.shape[2])]
                         for a in range(zf.shape[1])])
        win = np.array([[dip(zf[t0:t0 + WIN, a, b, :]) for b in range(zf.shape[2])]
                        for a in range(zf.shape[1])])
        rows.append(dict(
            patient=i, center_bin=cb, angle=ang,
            D_sel_full=float(full[ang, cb]), D_sel_win=float(win[ang, cb]),
            D_max_full=float(full.max()), D_max_win=float(win.max()),
            n_bin_ge_onset=int((win >= D_ONSET).sum()),
            n_bin_sat=int((win >= D_SAT).sum()),
            penalty_sel=float(np.clip(win[ang, cb] / D_SAT, 0, 1)) * 0.25,
        ))

    print(f"{'pat':>4}{'bin':>5}{'ang':>4}{'D@sel_full':>12}{'D@sel_win':>11}"
          f"{'Dmax_full':>11}{'Dmax_win':>10}{'n>=1%':>7}{'n>=5%':>7}{'pen@sel':>9}")
    for r in rows:
        print(f"{r['patient']:>4}{r['center_bin']:>5}{r['angle']:>4}"
              f"{r['D_sel_full']*100:>11.3f}%{r['D_sel_win']*100:>10.3f}%"
              f"{r['D_max_full']*100:>10.3f}%{r['D_max_win']*100:>9.3f}%"
              f"{r['n_bin_ge_onset']:>7}{r['n_bin_sat']:>7}{r['penalty_sel']:>9.3f}")

    ds = np.array([r["D_sel_full"] for r in rows])
    dw = np.array([r["D_sel_win"] for r in rows])
    dm = np.array([r["D_max_win"] for r in rows])
    print("\n--- 汇总 ---")
    print(f"D@sel(full) >0 的记录: {int((ds > 0).sum())}/20；>1%: {int((ds > D_ONSET).sum())}/20")
    print(f"D@sel(win)  >0 的记录: {int((dw > 0).sum())}/20；>1%: {int((dw > D_ONSET).sum())}/20")
    print(f"任一 bin 的 D(win) >=1% 的记录: {int((dm >= D_ONSET).sum())}/20；"
          f">=5%: {int((dm >= D_SAT).sum())}/20")
    print(f"惩罚项在所选 bin 上非零的记录: "
          f"{int(sum(1 for r in rows if r['penalty_sel'] > 0))}/20")
    print(f"D@sel(win) 中位 {np.median(dw)*100:.3f}%，最大 {dw.max()*100:.3f}%")
    print("\n注：D 在 30 s 整条上常为 0、在 20 s 窗上更常为 0，是口径性质；"
          "\n    惩罚项用哪个窗决定了它是否可能触发——这一点必须在正文写明。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
