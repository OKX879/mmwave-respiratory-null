#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反推 Table II 的选 bin 协议。
Reverse-engineer the Table II bin-selection protocol."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from lei_pilot import load, ROOT, R_BIN, FS, bp
from exp_center_removal import EXPECT

OUT = Path(__file__).resolve().parent / 'exp_out'
R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']
TARGET_NULL = {3.0: 7, 2.5: 26, 2.0: 33}
TARGET_ASR = {3.0: 19.7, 2.5: 4.65, 2.0: 2.57}


def n_deep(seg, depth=0.35, prom=0.15, gap=0.25):
    env = np.abs(seg)
    med = float(np.median(env))
    if med <= 0:
        return 0
    pk, _ = find_peaks(-env, distance=int(gap * FS), prominence=prom * np.std(env))
    return int(np.sum(env[pk] < depth * med))


def as_over_r(seg):
    a = bp(seg)
    return float(abs(complex(np.mean(seg))) / (np.mean(np.abs(a)) + 1e-12))


def window(pos, rb, lo_off, hi_off):
    idx = []
    for r in R_TRUE[pos]:
        idx += list(range(max(1, int((r + lo_off) / rb)),
                          int((r + hi_off) / rb) + 1))
    return sorted(set(idx))


def pick(zf, idx, mode, h):
    if mode == 'std1':
        return idx[int(np.argmax(np.std(zf[:h][:, idx], axis=0)))]
    if mode == 'stdall':
        return idx[int(np.argmax(np.std(zf[:, idx], axis=0)))]
    if mode == 'std20':
        return idx[int(np.argmax(np.std(zf[:int(20 * FS)][:, idx], axis=0)))]
    if mode == 'asr':
        vals = [abs(complex(np.mean(zf[:, b]))) for b in idx]
        return idx[int(np.argmax(vals))]
    raise ValueError(mode)


VARIANTS = {
    'V1 std1 [-.25,+.05]': ('std1', -0.25, 0.05),
    'V2 stdall [-.25,+.05]': ('stdall', -0.25, 0.05),
    'V3 std20 [-.25,+.05]': ('std20', -0.25, 0.05),
    'V4 std1 [-.25,+.15]': ('std1', -0.25, 0.15),
    'V5 std1 [-.30,+.10]': ('std1', -0.30, 0.10),
    'V6 asr  [-.25,+.05]': ('asr', -0.25, 0.05),
    'V7 std1 [-.25,+.00]': ('std1', -0.25, 0.00),
    'V8 std1 [-.20,+.05]': ('std1', -0.20, 0.05),
}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rows = []
    for base in BASES:
        if not base.exists():
            continue
        layout = base.parent.name
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
                    h = len(zf) // 2
                    for name, (mode, lo, hi) in VARIANTS.items():
                        idx = window(pos, rb, lo, hi)
                        if not idx:
                            continue
                        b = pick(zf, idx, mode, h)
                        z = zf[:, b]
                        rows.append({'layout': layout, 'pos': pos, 'bw_GHz': bw,
                                     'repeat': k, 'variant': name, 'bin': b,
                                     'range_m': round(b * rb, 3),
                                     'n_deep': n_deep(z), 'AsR': as_over_r(z),
                                     'D': float(np.mean(np.abs(z) < 0.3 * np.median(np.abs(z))))})
                    print('.', end='', flush=True)
    print()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'pin_protocol.csv', index=False, encoding='utf-8')

    print('目标指纹：零点记录 7 / 26 / 33；|As|/R 中位 19.7 / 4.65 / 2.57\n')
    print(f'{"变体":>24}{"零点数 3/2.5/2":>20}{"|As|/R 3/2.5/2":>24}{"偏差":>8}')
    best = []
    for name, g in df.groupby('variant'):
        cnt = [int((g[g.bw_GHz == bw].n_deep >= 5).sum()) for bw in (3, 2.5, 2)]
        asr = [round(float(g[g.bw_GHz == bw].AsR.median()), 2) for bw in (3, 2.5, 2)]
        err = sum(abs(c - t) for c, t in zip(cnt, TARGET_NULL.values()))
        err += sum(abs(np.log(max(a, 1e-6) / t)) for a, t in
                   zip(asr, TARGET_ASR.values()))
        best.append((err, name, cnt, asr))
    best.sort()
    for err, name, cnt, asr in best:
        print(f'{name:>24}{str(cnt):>20}{str(asr):>24}{err:>8.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
