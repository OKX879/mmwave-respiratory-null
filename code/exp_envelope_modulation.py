#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""包络调制的呼吸带占比：解释"为什么常圆心压不平"。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from lei_pilot import load, ROOT, R_BIN
from exp_center_removal import circle_geometric, EXPECT

OUT = Path(__file__).resolve().parent / 'exp_out'
R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']


def main() -> int:
    rows = []
    for base in BASES:
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            pos = int(d.name.replace('position_ (', '').replace(')', ''))
            for bw in (3, 2.5, 2):
                tag = {3: '3', 2.5: '2_5', 2: '2'}[bw]
                rb = R_BIN[bw]
                for k in range(1, 7):
                    fp = d / f'adc_{tag}GHZ_position{pos}_ ({k}).bin'
                    if not fp.exists():
                        fp = d / f'adc_{tag}GHZ_positon{pos}_ ({k}).bin'
                    if not fp.exists() or fp.stat().st_size != EXPECT:
                        continue
                    zf = np.fft.fft(load(fp), axis=3).mean(axis=(1, 2))
                    idx = []
                    for r in R_TRUE[pos]:
                        idx += list(range(max(1, int((r - .25) / rb)),
                                          int((r + .05) / rb) + 1))
                    idx = sorted(set(idx))
                    h = len(zf) // 2
                    b = idx[int(np.argmax(np.std(zf[:h][:, idx], axis=0)))]
                    z = zf[:, b]
                    c, r, _ = circle_geometric(z)
                    env = np.abs(z - c)
                    f, P = welch(env - env.mean(), fs=20, nperseg=256)
                    tot = P[(f > 0.05) & (f < 5)].sum()
                    rows.append({
                        'bw_GHz': bw, 'pos': pos, 'repeat': k, 'bin': b,
                        'resp_frac': float(P[(f >= 0.1) & (f <= 0.6)].sum() / tot),
                        'mod': float((np.percentile(env, 95) - np.percentile(env, 5))
                                     / np.median(env)),
                        'q05': float(np.percentile(env, 5) / np.median(env)),
                    })
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / 'envelope_modulation.csv', index=False, encoding='utf-8')
    print(f'n = {len(df)} 条记录（每条取选中 bin）')
    print(f'包络波动能量落在 0.1–0.6 Hz 呼吸带的占比：'
          f'中位 {df.resp_frac.median():.2f}，'
          f'IQR {df.resp_frac.quantile(.25):.2f}–{df.resp_frac.quantile(.75):.2f}')
    print(f'减圆心后半径调制 (p95−p5)/中位：中位 {df["mod"].median():.2f}')
    print('分带宽 (呼吸带占比中位):',
          {k: round(v, 2) for k, v in df.groupby('bw_GHz').resp_frac.median().items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
