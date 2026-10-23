#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2：straddle / scalloping 混杂控制实验。

审稿意见第 4 条
--------------
改带宽不只改变"bin 里混进多少静态杂波"，也改变 bin 网格相对目标的位置
（straddle / scalloping），后者会让 rho 非单调变化。必须排除。

做法
----
1. 用 **16× 零填充**求精细距离像（3 GHz 下网格 3.1 mm），在 ±0.3 m 窗内
   定位目标真实距离 $\\hat r$，分数偏移 $u=(\\hat r-b\\,\\Delta R)/\\Delta R\\in[-0.5,0.5]$。
2. 先看 $\\lvert u\\rvert$ 与 $\\lvert A_s\\rvert/R$、$D$、$n_{\\rm deep}$ 的相关——若强，混杂成立。
3. **控制实验**：把复时间序列改在"目标真实距离"上取值（零填充谱上的插值，
   等价于 DTFT 任意点求值），网格摆放被消除，只剩分辨率差异。
   比较 bin 取值 vs 目标中心取值下，三带宽的 $\\lvert A_s\\rvert/R$ 与零点率是否保持。

同时回答审稿意见 4b（机制叙述与数据方向不一致）：
把 $\\lvert A_s\\rvert$ 与 $\\lvert A_c\\rvert$（=圆拟合半径）分开报，看比值的变化
到底来自分子还是分母。

用法
----
    python exp_straddle.py
"""
from __future__ import annotations

import sys

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from lei_pilot import load, ROOT, R_BIN, FS, bp
from exp_center_removal import circle_geometric, dip_stats

OUT = Path(__file__).resolve().parent / 'exp_out'
R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']
EXPECT = 1200 * 3 * 4 * 200 * 2 * 2
NFINE = 16                     # 零填充倍数
NULL_RECORD = 5


def metrics(z: np.ndarray) -> dict:
    """一个复时间序列的零点侧指标。"""
    a = bp(z)
    c, r, _ = circle_geometric(z)
    D, n = dip_stats(z)
    return {'As_abs': float(abs(c)), 'Ac_abs': float(r),
            'AsR': float(abs(c) / (np.mean(np.abs(a)) + 1e-12)),
            'D': D, 'n_deep': n}


def build() -> pd.DataFrame:
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
                    z = load(fp)
                    # 零填充谱（相干平均 12 通道后逐帧）
                    Zf = np.fft.fft(z, n=200 * NFINE, axis=3).mean(axis=(1, 2))
                    prof = np.mean(np.abs(Zf), axis=0)
                    fine_rb = rb / NFINE
                    for ti, r in enumerate(R_TRUE[pos]):
                        lo = max(1, int((r - 0.30) / fine_rb))
                        hi = min(len(prof) - 2, int((r + 0.30) / fine_rb))
                        seg = prof[lo:hi]
                        if len(seg) < 5:
                            continue
                        r_hat = (lo + int(np.argmax(seg))) * fine_rb
                        b_frac = r_hat / rb
                        b_near = int(round(b_frac))
                        if b_near < 1 or b_near > 199:
                            continue
                        u = b_frac - b_near
                        z_bin = Zf[:, b_near * NFINE]
                        z_tgt = Zf[:, int(round(b_frac * NFINE))]
                        row = {'layout': layout, 'pos': pos, 'bw_GHz': bw,
                               'repeat': k, 'target': ti + 1,
                               'r_true_m': r, 'r_hat_m': round(r_hat, 4),
                               'bin': b_near, 'u': float(u)}
                        m1 = metrics(z_bin)
                        for kk, vv in m1.items():
                            row[f'bin_{kk}'] = vv
                        m2 = metrics(z_tgt)
                        for kk, vv in m2.items():
                            row[f'tgt_{kk}'] = vv
                        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    cache = OUT / 'exp_straddle.csv'
    if cache.exists() and '--rebuild' not in sys.argv:
        df = pd.read_csv(cache)
        print(f'[data] 复用 {cache.name}：{len(df)} 条')
    else:
        df = build()
        df.to_csv(cache, index=False, encoding='utf-8')
        print(f'[data] {len(df)} 条 (记录 × 目标)')

    print('\n--- A. 分数偏移 |u| 的分布（0=目标正落 bin 中心，0.5=落在边缘）---')
    print(f'{"带宽":>8}{"bin(cm)":>9}{"|u|中位":>10}{"|u|>0.25":>10}{"|u|>0.4":>10}')
    for bw in (3, 2.5, 2):
        g = df[df.bw_GHz == bw]
        print(f'{bw:>7.1f}G{R_BIN[bw]*100:>9.1f}{g.u.abs().median():>10.3f}'
              f'{np.mean(g.u.abs() > .25):>10.1%}{np.mean(g.u.abs() > .4):>10.1%}')

    print('\n--- B. |u| 与零点指标的相关（带宽内，剔除带宽本身的混淆）---')
    print(f'{"带宽":>8}{"|u|~|As|/R":>13}{"|u|~D":>10}{"|u|~n_deep":>12}')
    for bw in (3, 2.5, 2):
        g = df[df.bw_GHz == bw]
        if len(g) < 10:
            continue
        print(f'{bw:>7.1f}G'
              f'{spearmanr(g.u.abs(), g.bin_AsR).statistic:>+13.2f}'
              f'{spearmanr(g.u.abs(), g.bin_D).statistic:>10.2f}'
              f'{spearmanr(g.u.abs(), g.bin_n_deep).statistic:>12.2f}')

    print('\n--- C. 带宽效应：bin 取值 vs 目标中心取值（网格摆放被消除）---')
    hdr = (f'{"带宽":>8} | {"|As|":>8}{"|Ac|":>8}{"|As|/R":>8}{"D":>8}'
           f'{"零点率":>8}  →  {"|As|/R":>8}{"D":>8}{"零点率":>8}')
    print(hdr)
    print(' ' * 9 + '|' + ' bin 中心取值'.ljust(41) + '|' + ' 目标中心插值')
    for bw in (3, 2.5, 2):
        g = df[df.bw_GHz == bw]
        print(f'{bw:>7.1f}G | {g.bin_As_abs.median():>8.0f}{g.bin_Ac_abs.median():>8.0f}'
              f'{g.bin_AsR.median():>8.2f}{g.bin_D.median():>8.4f}'
              f'{np.mean(g.bin_n_deep >= NULL_RECORD):>8.1%}  →  '
              f'{g.tgt_AsR.median():>8.2f}{g.tgt_D.median():>8.4f}'
              f'{np.mean(g.tgt_n_deep >= NULL_RECORD):>8.1%}')
    g = df
    print(f'{"全部":>8} | {g.bin_As_abs.median():>8.0f}{g.bin_Ac_abs.median():>8.0f}'
          f'{g.bin_AsR.median():>8.2f}{g.bin_D.median():>8.4f}'
          f'{np.mean(g.bin_n_deep >= NULL_RECORD):>8.1%}  →  '
          f'{g.tgt_AsR.median():>8.2f}{g.tgt_D.median():>8.4f}'
          f'{np.mean(g.tgt_n_deep >= NULL_RECORD):>8.1%}')

    print('\n--- D. 机制方向检验：比值下降来自分子还是分母？---')
    for bw in (3, 2.5, 2):
        g = df[df.bw_GHz == bw]
        print(f'{bw:>7.1f} GHz: 中位 |As| = {g.bin_As_abs.median():>7.0f}，'
              f'|Ac| = {g.bin_Ac_abs.median():>6.0f}，'
              f'比值 = {g.bin_AsR.median():>5.2f}')
    sub = df[df.bw_GHz == 3]
    sub2 = df[df.bw_GHz == 2]
    print(f'  |As| 3.0G vs 2.0G：{sub.bin_As_abs.median():.0f} → '
          f'{sub2.bin_As_abs.median():.0f} '
          f'(×{sub2.bin_As_abs.median()/max(sub.bin_As_abs.median(),1e-9):.2f})')
    print(f'  |Ac| 3.0G vs 2.0G：{sub.bin_Ac_abs.median():.0f} → '
          f'{sub2.bin_Ac_abs.median():.0f} '
          f'(×{sub2.bin_Ac_abs.median()/max(sub.bin_Ac_abs.median(),1e-9):.2f})')

    print('\n--- E. 目标中心插值后，带宽内 |u| 是否还有影响（残差混杂）---')
    for bw in (3, 2.5, 2):
        g = df[df.bw_GHz == bw]
        if len(g) < 10:
            continue
        print(f'{bw:>7.1f} GHz: |u|~tgt_AsR '
              f'{spearmanr(g.u.abs(), g.tgt_AsR).statistic:+.2f}，'
              f'|u|~tgt_n_deep {spearmanr(g.u.abs(), g.tgt_n_deep).statistic:+.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
