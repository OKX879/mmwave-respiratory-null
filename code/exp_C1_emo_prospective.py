#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C1：在独立前端上做前瞻检验（前半段选格，后半段验证）。
C1: prospective first-half to second-half check on an independent front end."""
from __future__ import annotations

from paths import EMO_MULTI

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.stats import rankdata, spearmanr

FS = 100.0
R_BIN = 0.04505
BIN_LO, BIN_HI = 9, 28          # 0.4-1.2 m
DROP_S = 10.0
DEPTH, PROM = 0.35, 0.15        # the paper's event definition
LEI_THRESHOLD_30S = 5           # rule R1 as calibrated on Lei

HERE = Path(__file__).resolve().parent
RAW = EMO_MULTI
OUT = HERE / "exp_out"


def analyse(fp: Path):
    adc = loadmat(fp)["adcData"]                 # (12, 256 * frames)
    n_ch, n_samp = adc.shape
    n_frame = n_samp // 256
    x = adc.reshape(n_ch, n_frame, 256)
    x = x[:, int(DROP_S * FS):, :]
    for j in (1, 5, 9):                          # official polarity fix
        x[j] *= -1
        x[j + 1] *= -1
    Z = np.fft.fft(x, axis=2)
    z = Z.mean(axis=0)                           # (frames, 256)
    std = np.std(z[:, BIN_LO:BIN_HI], axis=0)
    b = BIN_LO + int(np.argmax(std))
    zc = z[:, b]
    env = np.abs(zc)
    med = float(np.median(env))
    pk, _ = find_peaks(-env, distance=int(0.25 * FS), prominence=PROM * np.std(env))
    pk = pk[env[pk] < DEPTH * med]
    h = len(zc) // 2
    return {'trial': fp.stem, 'frames': len(zc), 'bin': int(b),
            'range_m': round(b * R_BIN, 3),
            'n1': int((pk < h).sum()), 'n2': int((pk >= h).sum()),
            'half_s': len(zc) / FS / 2}


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, int)
    pos, neg = x[y == 1], x[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    return float((rankdata(x)[y == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rows = []
    for fp in sorted(RAW.glob("*.mat")):
        try:
            rows.append(analyse(fp))
            print('.', end='', flush=True)
        except Exception as e:
            print(f"\nfail {fp.name}: {e}")
    print()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'C1_emo_prospective.csv', index=False, encoding='utf-8')

    print('=' * 84)
    print('C1. 独立前端前瞻验证（emo：100 Hz、4.5 cm、38 trial）')
    print('=' * 84)
    print(f'  trial 数 {len(df)}；前半段 {df.half_s.median():.0f} s；'
          f'n_deep 前半中位 {df.n1.median():.0f}，后半中位 {df.n2.median():.0f}')
    print(f'  后半为 0 的 trial：{(df.n2 == 0).sum()}；前半为 0 的：{(df.n1 == 0).sum()}')

    print('\n  以"后半段至少 k 个凹陷"为结局（阈值在 Lei 标定，不重调）：')
    print(f'  {"k":>4}{"正例":>6}{"AUC":>8}{"sens":>8}{"spec":>8}')
    for k in (1, 5, 10):
        y = (df.n2 >= k).astype(int)
        if y.sum() in (0, len(y)):
            print(f'  {k:>4}{int(y.sum()):>6}   退化（单一类别），跳过')
            continue
        pred = df.n1 >= k
        tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
        tn = int((~pred & (y == 0)).sum()); fp_ = int((pred & (y == 0)).sum())
        print(f'  {k:>4}{int(y.sum()):>6}{auc(df.n1, y):>8.2f}'
              f'{tp / max(tp + fn, 1):>8.2f}{tn / max(tn + fp_, 1):>8.2f}')

    r, p = spearmanr(df.n1, df.n2)
    print(f'\n  前后半段计数相关：Spearman {r:+.2f}（p={p:.1e}，n={len(df)}）')
    rate = LEI_THRESHOLD_30S / 30.0
    thr = rate * df.half_s.median()
    print(f'  Lei 的 R1 门槛换算到本段长：{rate:.2f} 事件/s × {df.half_s.median():.0f} s '
          f'= {thr:.1f} 个 → 取 ≥{int(round(thr))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
