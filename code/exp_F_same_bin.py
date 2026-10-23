#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F 组：两个人落在同一个距离门里时，深零点会被抑制吗？"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu

from lei_pilot import load, ROOT, R_BIN, FS
from exp_center_removal import EXPECT, circle_geometric

OUT = Path(__file__).resolve().parent / 'exp_out'
R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']
DEPTH, PROM, R_TH = 0.35, 0.15, 0.30


def build():
    rows = []
    for base in BASES:
        if not base.exists():
            continue
        layout = base.parent.name
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            pos = int(d.name.replace('position_ (', '').replace(')', ''))
            r1, r2 = R_TRUE[pos]
            for bw in (3, 2.5, 2):
                tag = {3: '3', 2.5: '2_5', 2: '2'}[bw]
                rb = R_BIN[bw]
                same_bin = int(round(r1 / rb) == int(round(r2 / rb)))
                for k in range(1, 7):
                    fp = d / f'adc_{tag}GHZ_position{pos}_ ({k}).bin'
                    if not fp.exists():
                        fp = d / f'adc_{tag}GHZ_positon{pos}_ ({k}).bin'
                    if not fp.exists() or fp.stat().st_size != EXPECT:
                        continue
                    zf = np.fft.fft(load(fp), axis=3).mean(axis=(1, 2))
                    idx = []
                    for r in (r1, r2):
                        idx += list(range(max(1, int((r - .25) / rb)),
                                          int((r + .05) / rb) + 1))
                    idx = sorted(set(idx))
                    b = idx[int(np.argmax(np.std(zf[:, idx], axis=0)))]
                    z = zf[:, b]
                    env = np.abs(z)
                    med = float(np.median(env))
                    pk, _ = find_peaks(-env, distance=int(0.25 * FS),
                                       prominence=PROM * np.std(env))
                    pk = pk[env[pk] < DEPTH * med]
                    cf, rf, _ = circle_geometric(z)
                    th = np.unwrap(np.angle(z - cf))
                    rows.append({
                        'layout': layout, 'pos': pos, 'bw_GHz': bw, 'repeat': k,
                        'r1': r1, 'r2': r2, 'dr_m': abs(r1 - r2),
                        'same_bin': same_bin,
                        'bin': b, 'range_m': round(b * rb, 3),
                        'n_deep': len(pk),
                        'D': float(np.mean(env < R_TH * med)),
                        'AsR': float(abs(complex(np.mean(z))) / (med + 1e-12)),
                        'th_max': float((th.max() - th.min()) / 2),
                    })
            print('.', end='', flush=True)
    print()
    return pd.DataFrame(rows)


def report(df: pd.DataFrame):
    print('=' * 84)
    print('F1. 分组概况')
    print('=' * 84)
    for flag, name in [(1, '同 bin（两人径向同距）'), (0, '异 bin（两人径向相差 0.4 m）')]:
        g = df[df.same_bin == flag]
        pos = sorted(g.pos.unique())
        print(f'  {name}: {len(g)} 条记录, positions {pos}')
        print(f'      |A_s|/R 中位 {g.AsR.median():6.2f} | θ_max 中位 {g.th_max.median():5.2f} rad'
              f' | D 中位 {g.D.median():.3f} | n_deep 中位 {g.n_deep.median():.0f}')

    print()
    print('=' * 84)
    print('F2. 未控制 ρ 的原始比较（预期会混杂，仅作参考）')
    print('=' * 84)
    a = df[df.same_bin == 1]; b = df[df.same_bin == 0]
    for col in ('D', 'n_deep'):
        u = mannwhitneyu(a[col], b[col], alternative='two-sided')
        print(f'  {col:8}: 同 bin {a[col].median():8.3f} vs 异 bin {b[col].median():8.3f}'
              f'  (Mann–Whitney p = {u.pvalue:.3f})')

    print()
    print('=' * 84)
    print('F3. 按 |A_s|/R 分层（层内同 bin vs 异 bin）')
    print('=' * 84)
    d = df.copy()
    d['logAsR'] = np.log(d.AsR.clip(lower=0.15))
    edges = d.logAsR.quantile([0, .25, .5, .75, 1.0]).values
    d['stratum'] = pd.cut(d.logAsR, bins=edges, include_lowest=True, duplicates='drop')
    print(f'  {"|As|/R 区间":>20}{"n(同)":>7}{"n(异)":>7}{"D 同":>8}{"D 异":>8}'
          f'{"n_deep 同":>11}{"n_deep 异":>11}')
    for s, g in d.groupby('stratum', observed=True):
        ga, gb = g[g.same_bin == 1], g[g.same_bin == 0]
        if len(ga) == 0 or len(gb) == 0:
            continue
        lo = np.exp(s.left); hi = np.exp(s.right)
        print(f'  [{lo:6.2f}, {hi:6.2f}]{len(ga):>9}{len(gb):>7}'
              f'{ga.D.median():>8.3f}{gb.D.median():>8.3f}'
              f'{ga.n_deep.median():>11.0f}{gb.n_deep.median():>11.0f}')

    print()
    print('=' * 84)
    print('F4. 最小二乘：控制 log|As|/R 与 θ_max 后，"同 bin" 的偏效应')
    print('=' * 84)
    X = np.column_stack([np.ones(len(d)), d.logAsR, d.th_max, d.same_bin])
    for ycol, lab in [('D', 'D'), ('n_deep', 'n_deep')]:
        y = d[ycol].values.astype(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        s2 = resid @ resid / (len(y) - X.shape[1])
        cov = s2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        t = beta / se
        print(f'  {lab:7}: 常数 {beta[0]:+.3f}  log|As|/R {beta[1]:+.3f}  '
              f'θ_max {beta[2]:+.3f}  同bin {beta[3]:+.3f} (t = {t[3]:+.2f})')

    print()
    print('  读法：若"同 bin"系数为负且 |t| 较大，说明落在同一格的两人确实抑制了深零点，')
    print('        与三相量仿真的预测（0.126 → 0.07）方向一致。')


def main() -> int:
    OUT.mkdir(exist_ok=True)
    cache = OUT / 'F_same_bin.csv'
    df = pd.read_csv(cache) if cache.exists() else build()
    if not cache.exists():
        df.to_csv(cache, index=False, encoding='utf-8')
    print(f'[data] {len(df)} 条记录')
    report(df)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
