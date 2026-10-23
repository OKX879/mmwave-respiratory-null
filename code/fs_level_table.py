#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把"层数边界"写成可查表：采样率 fs → 允许的 L 区间。

两条约定（推导见论文 §4.13(e)）
--------------------------------
约定 A：心搏分量取自近似带 aL，要求 aL 的上边界覆盖心搏带上界
        fs/2^(L+1) >= f_hi   =>   L <= floor(log2(fs/f_hi)) - 1
        可行条件：至少能取 L=1，即 fs >= 4*f_hi。

约定 B：心搏分量取自细节带，要求最深层 dL 的下边界不高于心搏带下界
        fs/2^(L+1) <= f_lo   =>   L >= ceil(log2(fs/f_lo)) - 1

另有工程上限：每层系数不能短于小波支撑的两倍
        N/2^L >= 2*(Lf-1)+1   =>   L <= floor(log2(N/(2*(Lf-1)+1)))
（sym4: Lf=8）
"""
from __future__ import annotations

import math

import pandas as pd

FS_LIST = [10, 12.5, 20, 25, 50, 100, 125, 200, 250, 500]
BANDS = [('展示带', 0.7, 3.0), ('生产带', 1.0, 2.5)]
DUR = 30.0          # s（10 Hz 平台只有 50 s，100 Hz 平台 30–78 s）
LF = 8              # sym4 的滤波器长度


def l_max_of(fs: float, dur: float) -> int:
    n = fs * dur
    return int(math.floor(math.log2(n / (2 * (LF - 1) + 1))))


def row(fs, name, f_lo, f_hi):
    a_max = int(math.floor(math.log2(fs / f_hi))) - 1
    a_ok = a_max >= 1
    b_min = int(math.ceil(math.log2(fs / f_lo))) - 1
    lm = l_max_of(fs, DUR)
    if a_ok and a_max <= lm:
        rec = f'A: L={a_max}'
    elif b_min <= lm:
        rec = f'B: L={b_min}'
    else:
        rec = '不可行'

    notes = []
    if not a_ok:
        notes.append('aL 无法覆盖心搏带，只能改用细节带')
    elif a_max == 1:
        notes.append('A 的 L=1 时 aL 已覆盖到 2×f_hi，近似低通，分解收益很小')
    if b_min > lm:
        notes.append('B 所需层数超过记录长度允许值')
    return {
        'fs_Hz': fs, 'band': name,
        'f_lo': f_lo, 'f_hi': f_hi,
        'A_Lmax': a_max, 'A_feasible': 'yes' if a_ok else 'no',
        'B_Lmin': b_min, 'L_max': lm,
        'recommended': rec,
        'note': '；'.join(notes),
    }


def main():
    rows = [row(fs, n, lo, hi) for fs in FS_LIST for (n, lo, hi) in BANDS]
    df = pd.DataFrame(rows)
    df.to_csv('fs_level_table.csv', index=False, encoding='utf-8')

    print('| fs (Hz) | 心搏带 | 约定 A 的 L 上限 | A 可用 | 约定 B 的 L 下限 | 分解上限 L_max | 推荐 |')
    print('|---|---|---|---|---|---|---|')
    for _, r in df.iterrows():
        band = f"{r.f_lo:g}–{r.f_hi:g}"
        print(f"| {r.fs_Hz:g} | {r.band} {band} | {r.A_Lmax} | {r.A_feasible} | "
              f"{r.B_Lmin} | {r.L_max} | {r.recommended} |")
    print()
    print(df.to_string(index=False))
    print('\nsaved fs_level_table.csv')


if __name__ == '__main__':
    raise SystemExit(main())
