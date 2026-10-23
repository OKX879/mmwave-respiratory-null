#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G 组：角度维是否真的携带可分信息？"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import mannwhitneyu

from lei_pilot import load, ROOT, R_BIN, FS
from exp_center_removal import EXPECT

OUT = Path(__file__).resolve().parent / 'exp_out'
GEO = {1: (40, 20, 1.0, 1.0), 2: (20, 40, 1.4, 1.4), 3: (40, 40, 1.4, 1.0),
       4: (20, 20, 1.0, 0.6), 5: (20, 40, 1.4, 1.0), 6: (40, 20, 0.6, 1.0),
       7: (40, 40, 0.6, 0.6), 8: (40, 40, 1.0, 1.0), 9: (20, 20, 1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']


def build():
    rows = []
    for base in BASES:
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            pos = int(d.name.replace('position_ (', '').replace(')', ''))
            th1, th2, r1, r2 = GEO[pos]
            for bw in (3, 2.5, 2):
                tag = {3: '3', 2.5: '2_5', 2: '2'}[bw]
                rb = R_BIN[bw]
                for k in range(1, 7):
                    fp = d / f'adc_{tag}GHZ_position{pos}_ ({k}).bin'
                    if not fp.exists():
                        fp = d / f'adc_{tag}GHZ_positon{pos}_ ({k}).bin'
                    if not fp.exists() or fp.stat().st_size != EXPECT:
                        continue
                    X = load(fp)                                  # (frame, tx, rx, sample)
                    Z = np.fft.fft(X, axis=3)                     # 保留通道
                    zf12 = Z.mean(axis=(1, 2))
                    idx = []
                    for r in (r1, r2):
                        idx += list(range(max(1, int((r - .25) / rb)),
                                          int((r + .05) / rb) + 1))
                    idx = sorted(set(idx))
                    b = idx[int(np.argmax(np.std(zf12[:, idx], axis=0)))]

                    # 同一 TX 内 4 个 RX 的相位差漂移
                    ipw = []
                    for tx in range(Z.shape[1]):
                        zc = Z[:, tx, :, b]                       # (frame, rx)
                        ph = np.unwrap(np.angle(zc), axis=0)
                        for i in range(ph.shape[1]):
                            for j in range(i + 1, ph.shape[1]):
                                ipw.append(float(np.std(ph[:, i] - ph[:, j])))
                    rows.append({
                        'pos': pos, 'bw_GHz': bw, 'repeat': k, 'bin': b,
                        'th1': th1, 'th2': th2, 'r1': r1, 'r2': r2,
                        'same_bin': int(round(r1 / rb) == int(round(r2 / rb))),
                        'same_ang': int(th1 == th2),
                        'IPW': float(np.median(ipw)),
                        'IPW_p75': float(np.percentile(ipw, 75)),
                    })
            print('.', end='', flush=True)
    print()
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    cache = OUT / 'G_angular.csv'
    df = pd.read_csv(cache) if cache.exists() else build()
    if not cache.exists():
        df.to_csv(cache, index=False, encoding='utf-8')
    print(f'[data] {len(df)} 条\n')

    df['group'] = np.where(df.same_bin == 1,
                           np.where(df.same_ang == 1, 'same-range, same-angle',
                                    'same-range, diff-angle'),
                           np.where(df.same_ang == 1, 'diff-range, same-angle',
                                    'diff-range, diff-angle'))
    print('=' * 74)
    print('G1. 通道间相位漂移 IPW（同一 TX 内 4 个 RX，无需校准）')
    print('=' * 74)
    for g in ('same-range, same-angle', 'same-range, diff-angle',
              'diff-range, same-angle', 'diff-range, diff-angle'):
        x = df[df.group == g]
        print(f'  {g:28} n={len(x):>3}  IPW 中位 {x.IPW.median():.3f} rad'
              f'  P75 {x.IPW_p75.median():.3f}')

    a = df[df.group == 'same-range, same-angle'].IPW
    b = df[df.group == 'same-range, diff-angle'].IPW
    u = mannwhitneyu(a, b, alternative='less')
    print(f'\n  同距内 同角度 vs 异角度：{a.median():.3f} vs {b.median():.3f}'
          f'  (one-sided p = {u.pvalue:.4f})')
    print('  预期：异角度 > 同角度（两源混合使通道相位差随时间摆动）')

    df.to_csv(cache, index=False, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
