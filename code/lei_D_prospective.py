#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D 的前瞻性检验：前 30 s 选格并算 D，后 30 s 数深凹陷。
Prospective test of D: select the bin and compute D on the first 30 s, count deep dips on the last 30 s."""
from __future__ import annotations

from paths import OUT_ROOT

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from lei_pilot import load, ROOT, R_BIN, FS

OUT = OUT_ROOT
R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']
EXPECT = 1200 * 3 * 4 * 200 * 2 * 2


def auc(x, y):
    """rank-based AUC（y 为 0/1）.
Rank-based AUC for binary labels."""
    x = np.asarray(x, float); y = np.asarray(y, int)
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    # Mann-Whitney
    allv = np.concatenate([pos, neg])
    order = allv.argsort().argsort() + 1
    r_pos = order[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def process(bw, fp, pos):
    rb = R_BIN[bw]
    z = load(fp)
    Z = np.fft.fft(z, axis=3)
    zf = Z.mean(axis=(1, 2))

    idx = []
    for r in R_TRUE[pos]:
        a = max(1, int((r - 0.25) / rb)); b = min(zf.shape[1] - 2, int((r + 0.05) / rb))
        idx += list(range(a, b + 1))
    idx = sorted(set(idx))

    half = len(zf) // 2
    z1, z2 = zf[:half], zf[half:]

    # 只用前一半选格（呼吸带复数标准差最大）
    e = np.std(z1[:, idx], axis=0)
    b = idx[int(np.argmax(e))]

    def dipfrac(seg):
        env = np.abs(seg[:, b])
        return float(np.mean(env < 0.3 * np.median(env)))

    D1 = dipfrac(z1)
    # 前一半其余候选格
    D1o = np.asarray([float(np.mean(np.abs(z1[:, bb]) < 0.3 * np.median(np.abs(z1[:, bb]))))
                      for bb in idx])
    D1o = np.delete(D1o, idx.index(b))

    # 后一半的深凹陷事件（与前文同一事件定义）
    env2 = np.abs(z2[:, b]); med2 = np.median(env2)
    pk, _ = find_peaks(-env2, distance=int(0.25 * FS), prominence=0.15 * np.std(env2))
    pk = pk[env2[pk] < 0.35 * med2]
    D2 = float(np.mean(env2 < 0.3 * med2))
    return {
        'bandwidth_GHz': bw, 'position': pos, 'bin': int(b), 'range_m': b * rb,
        'D_first_half': D1, 'D_first_others': float(np.median(D1o)),
        'n_deep_second_half': int(len(pk)), 'D_second_half': D2,
    }


def main():
    rows = []
    for base in BASES:
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            pos = int(d.name.replace('position_ (', '').replace(')', ''))
            for bw in (3, 2.5, 2):
                tag = {3: '3', 2.5: '2_5', 2: '2'}[bw]
                for k in range(1, 7):
                    fp = d / f'adc_{tag}GHZ_position{pos}_ ({k}).bin'
                    if not fp.exists():
                        fp = d / f'adc_{tag}GHZ_positon{pos}_ ({k}).bin'
                    if not fp.exists() or fp.stat().st_size != EXPECT:
                        continue
                    r = process(bw, fp, pos); r['repeat'] = k
                    rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'lei_D_prospective.csv', index=False, encoding='utf-8')

    print('records', len(df))
    for bw, g in df.groupby('bandwidth_GHz'):
        for thr in (1, 5):
            y = (g.n_deep_second_half >= thr).astype(int)
            if y.nunique() > 1:
                print(f'{bw} GHz  n2>={thr}: prev={100*y.mean():.1f}%  '
                      f'AUC(D1)={auc(g.D_first_half, y):.3f}')
        print(f'  Spearman D1 vs n2: '
              f'{g.D_first_half.corr(g.n_deep_second_half, method="spearman"):.3f}')
    # 门限 D=1% 的前瞻表现（合并）
    y1 = (df.n_deep_second_half >= 1).astype(int)
    pred = df.D_first_half > 0.01
    tp = int(((pred == 1) & (y1 == 1)).sum()); fp = int(((pred == 1) & (y1 == 0)).sum())
    fn = int(((pred == 0) & (y1 == 1)).sum()); tn = int(((pred == 0) & (y1 == 0)).sum())
    print(f'D1>1%: TP={tp} FP={fp} FN={fn} TN={tn} '
          f'sens={tp/max(tp+fn,1):.2f} spec={tn/max(tn+fp,1):.2f}')
    print('pooled AUC(D1 -> n2>=1) =', round(auc(df.D_first_half, y1), 3))


if __name__ == '__main__':
    raise SystemExit(main())
