#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一口径后的减圆心分析（与 Table II 完全同口径）。
Centre-removal analysis under the unified protocol, identical to Table II."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from scipy.special import j0 as bessel_j0

from lei_pilot import load, ROOT, R_BIN, FS, bp
from exp_center_removal import (circle_geometric, circle_windowed, dip_stats,
                                residual_series, q05, radius_modulation,
                                EXPECT, ESTIMATORS)

OUT = Path(__file__).resolve().parent / 'exp_out'
R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']
NULL_RECORD = 5
TABLE2_NULL = {3.0: '13.0% (7)', 2.5: '48.1% (26)', 2.0: '62.3% (33)'}


def as_over_r(seg):
    a = bp(seg)
    return float(abs(complex(np.mean(seg))) / (np.mean(np.abs(a)) + 1e-12))


def env_resp_fraction(env):
    f, P = welch(env - env.mean(), fs=FS, nperseg=int(min(len(env), 256)))
    tot = float(P[(f > 0.05) & (f < 5)].sum())
    return float(P[(f >= 0.1) & (f <= 0.6)].sum() / tot) if tot > 0 else np.nan


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
                    idx = []
                    for r in R_TRUE[pos]:
                        idx += list(range(max(1, int((r - .25) / rb)),
                                          int((r + .05) / rb) + 1))
                    idx = sorted(set(idx))
                    b = idx[int(np.argmax(np.std(zf[:, idx], axis=0)))]   # V2
                    z = zf[:, b]
                    h = len(z) // 2
                    row = {'layout': layout, 'pos': pos, 'bw_GHz': bw, 'repeat': k,
                           'bin': b, 'range_m': round(b * rb, 3),
                           'AsR': as_over_r(z)}
                    for est in ESTIMATORS:
                        res = residual_series(z, est)
                        D, n = dip_stats(res)
                        row[f'D_{est}'], row[f'n_{est}'] = D, n
                        row[f'q05_{est}'] = q05(res)
                        row[f'mod_{est}'] = radius_modulation(res)
                    # 圆心稳定性：前后半段各自独立拟合
                    c1, _, _ = circle_geometric(z[:h])
                    c2, _, _ = circle_geometric(z[h:])
                    cf, rf, rms = circle_geometric(z)
                    row['leak'] = float(abs(c1 - c2) / (rf + 1e-12))
                    row['d_mean'] = float(abs(complex(np.mean(z)) - cf) / (rf + 1e-12))
                    th = np.unwrap(np.angle(z - cf))
                    row['theta_max'] = float((th.max() - th.min()) / 2)
                    row['j0'] = float(abs(bessel_j0(row['theta_max'])))
                    row['fit_rms_rel'] = float(rms / (rf + 1e-12))
                    row['env_resp'] = env_resp_fraction(np.abs(z - cf))
                    rows.append(row)
                    print('.', end='', flush=True)
    print()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'center_removal_table2.csv', index=False, encoding='utf-8')
    print(f'[data] {len(df)} 条记录；分带宽 {df.groupby("bw_GHz").size().to_dict()}')

    print('\n' + '=' * 96)
    print('表 A  记录级零点率（n_deep ≥ 5），与 Table II 同口径')
    print('=' * 96)
    print(f'  {"带宽":>7}{"Table II":>14}' + ''.join(f'{e:>11}' for e in ESTIMATORS))
    for bw in (3, 2.5, 2):
        g = df[df.bw_GHz == bw]
        print(f'  {bw:>6.1f}G{TABLE2_NULL[bw]:>14}'
              + ''.join(f'{np.mean(g[f"n_{e}"] >= NULL_RECORD):>10.1%}'
                        for e in ESTIMATORS))
    print(f'  {"全部":>7}{"39.8%":>14}'
          + ''.join(f'{np.mean(df[f"n_{e}"] >= NULL_RECORD):>10.1%}'
                    for e in ESTIMATORS))

    print('\n' + '=' * 96)
    print('表 B  包络 5% 分位 q05（1.00=不塌陷）')
    print('=' * 96)
    print(f'  {"带宽":>7}' + ''.join(f'{e:>11}' for e in ESTIMATORS))
    for bw in (3, 2.5, 2):
        g = df[df.bw_GHz == bw]
        print(f'  {bw:>6.1f}G' + ''.join(f'{g[f"q05_{e}"].median():>11.3f}'
                                         for e in ESTIMATORS))
    print(f'  {"全部":>7}' + ''.join(f'{df[f"q05_{e}"].median():>11.3f}'
                                     for e in ESTIMATORS))

    print('\n' + '=' * 96)
    print('表 C  残余静止相量 ρ_res = |ĉ(前)-ĉ(后)|/R 与相位误差上界')
    print('=' * 96)
    print(f'  {"带宽":>7}{"中位":>8}{"P75":>8}{"P95":>8}{"ρ>0.5":>8}{"ρ>1":>7}'
          f'{"arcsin 中位":>13}')
    for bw in (3, 2.5, 2):
        g = df[df.bw_GHz == bw]
        print(f'  {bw:>6.1f}G{g.leak.median():>8.2f}{g.leak.quantile(.75):>8.2f}'
              f'{g.leak.quantile(.95):>8.2f}{np.mean(g.leak > .5):>8.1%}'
              f'{np.mean(g.leak > 1):>7.1%}'
              f'{np.degrees(np.arcsin(np.clip(g.leak.median(), 0, 1))):>11.0f}°')
    print(f'  {"全部":>7}{df.leak.median():>8.2f}{df.leak.quantile(.75):>8.2f}'
          f'{df.leak.quantile(.95):>8.2f}{np.mean(df.leak > .5):>8.1%}'
          f'{np.mean(df.leak > 1):>7.1%}'
          f'{np.degrees(np.arcsin(np.clip(df.leak.median(), 0, 1))):>11.0f}°')

    print('\n' + '=' * 96)
    print('表 D  机制量（中位）')
    print('=' * 96)
    print(f'  θ_max = {df.theta_max.median():.2f} rad；|J_0| = {df.j0.median():.3f}；'
          f'实测均值偏差 d_mean = {df.d_mean.median():.3f}')
    print(f'  圆拟合残差/半径 = {df.fit_rms_rel.median():.3f}；'
          f'半径调制 = {df.mod_circle.median():.2f}；'
          f'包络波动落呼吸带 = {df.env_resp.median():.2f}')
    print(f'  |A_s|/R 中位：'
          f'{ {bw: round(df[df.bw_GHz==bw].AsR.median(),2) for bw in (3,2.5,2)} }')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
