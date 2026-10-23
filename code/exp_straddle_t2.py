#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""straddle 控制实验（V2 协议，与 Table II 同口径）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lei_pilot import load, ROOT, R_BIN, bp
from exp_center_removal import circle_geometric, dip_stats, EXPECT

OUT = Path(__file__).resolve().parent / 'exp_out'
R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']
NFINE = 16
NULL_RECORD = 5


def metrics(z):
    a = bp(z)
    c, r, _ = circle_geometric(z)
    D, n = dip_stats(z)
    return float(abs(c) / (np.mean(np.abs(a)) + 1e-12)), D, n


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
                    z = load(fp)
                    zf = np.fft.fft(z, axis=3).mean(axis=(1, 2))
                    Zf = np.fft.fft(z, n=200 * NFINE, axis=3).mean(axis=(1, 2))
                    idx = []
                    for r in R_TRUE[pos]:
                        idx += list(range(max(1, int((r - .25) / rb)),
                                          int((r + .05) / rb) + 1))
                    idx = sorted(set(idx))
                    b = idx[int(np.argmax(np.std(zf[:, idx], axis=0)))]
                    # 最近目标（用**名义距离**，不用幅度峰值——峰值会落在静物上）
                    rt = min(R_TRUE[pos], key=lambda r: abs(r - b * rb))
                    u = b * rb - rt                      # 网格相对目标真实距离的偏移
                    # 诊断：零填充幅度峰（会偏向强静物，仅用于说明选择偏倚）
                    fine_rb = rb / NFINE
                    lo = max(1, int((rt - .30) / fine_rb))
                    hi = min(Zf.shape[1] - 2, int((rt + .30) / fine_rb))
                    prof = np.mean(np.abs(Zf), axis=0)
                    r_pk = (lo + int(np.argmax(prof[lo:hi]))) * fine_rb
                    asr0, D0, n0 = metrics(zf[:, b])
                    asr1, D1, n1 = metrics(Zf[:, int(round(rt / rb * NFINE))])
                    asr2, D2, n2 = metrics(Zf[:, int(round(r_pk / rb * NFINE))])
                    rows.append({'layout': layout, 'pos': pos, 'bw_GHz': bw,
                                 'repeat': k, 'bin': b, 'range_m': round(b * rb, 3),
                                 'r_true': rt, 'r_peak': round(r_pk, 4),
                                 'u': float(u), 'absu': float(abs(u)),
                                 'bin_AsR': asr0, 'bin_D': D0, 'bin_n': n0,
                                 'tgt_AsR': asr1, 'tgt_D': D1, 'tgt_n': n1,
                                 'pk_AsR': asr2, 'pk_n': n2})
                    print('.', end='', flush=True)
    print()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'straddle_table2.csv', index=False, encoding='utf-8')
    print(f'[data] {len(df)} 条记录')

    print('\n' + '=' * 92)
    print('A. 分数偏移分布')
    print('=' * 92)
    print(f'  {"带宽":>7}{"|u|中位":>10}{"|u|>0.25":>11}{"|u|>0.4":>10}')
    for bw in (3, 2.5, 2):
        g = df[df.bw_GHz == bw]
        print(f'  {bw:>6.1f}G{g.absu.median():>10.3f}'
              f'{np.mean(g.absu > .25):>11.1%}{np.mean(g.absu > .4):>10.1%}')

    print('\n' + '=' * 92)
    print('B. |u| 与零点指标的相关（带宽内）')
    print('=' * 92)
    print(f'  {"带宽":>7}{"|u|~|As|/R":>14}{"|u|~D":>10}{"|u|~n_deep":>13}')
    for bw in (3, 2.5, 2):
        g = df[df.bw_GHz == bw]
        print(f'  {bw:>6.1f}G'
              f'{spearmanr(g.absu, g.bin_AsR).statistic:>+14.2f}'
              f'{spearmanr(g.absu, g.bin_D).statistic:>10.2f}'
              f'{spearmanr(g.absu, g.bin_n).statistic:>13.2f}')

    print('\n' + '=' * 92)
    print('C. bin 取值 vs 名义目标距离插值（Table II 同口径）')
    print('=' * 92)
    print(f'  {"带宽":>7}{"|As|/R":>10}{"零点率":>10}{"  →  ":>6}'
          f'{"|As|/R":>10}{"零点率":>10}')
    for bw in (3, 2.5, 2):
        g = df[df.bw_GHz == bw]
        print(f'  {bw:>6.1f}G{g.bin_AsR.median():>10.2f}'
              f'{np.mean(g.bin_n >= NULL_RECORD):>9.1%}{"":>6}'
              f'{g.tgt_AsR.median():>10.2f}'
              f'{np.mean(g.tgt_n >= NULL_RECORD):>9.1%}')
    print(f'  {"全部":>7}{df.bin_AsR.median():>10.2f}'
          f'{np.mean(df.bin_n >= NULL_RECORD):>9.1%}{"":>6}'
          f'{df.tgt_AsR.median():>10.2f}'
          f'{np.mean(df.tgt_n >= NULL_RECORD):>9.1%}')

    print('\n' + '=' * 92)
    print('D. 诊断：若把"窗内幅度峰"当目标位置会怎样（说明选择偏倚）')
    print('=' * 92)
    print(f'  幅度峰距名义距离的中位：{np.median(np.abs(df.r_peak - df.r_true))*100:.0f} cm；'
          f'零填充为 1 cm 网格')
    for bw in (3, 2.5, 2):
        g = df[df.bw_GHz == bw]
        print(f'  {bw:>6.1f}G: 峰值取值 |As|/R {g.pk_AsR.median():>7.2f}，'
              f'零点率 {np.mean(g.pk_n >= NULL_RECORD):>5.1%}')
    print('  → 峰值处 ρ 远高于 bin，零点几乎消失：因为峰落在最强静物上，'
          '不是受试者。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
