#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D 组：把 AUC 从"自证"里救出来，并给出延迟-性能折中。
Group D: rescue AUC from self-validation and quantify the delay-performance trade-off."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.special import j0
from scipy.stats import rankdata

from lei_pilot import load, ROOT, R_BIN, FS
from exp_center_removal import circle_geometric, EXPECT

OUT = Path(__file__).resolve().parent / 'exp_out'
R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']
WINDOWS = (10.0, 20.0, 30.0, 45.0)      # 预锁定窗长（秒）
DEPTH, PROM = 0.35, 0.15                 # 与论文一致


def dips_of(env):
    """论文定义的深凹陷：0.35×中位、0.25 s 间隔、0.15σ 突出度。
The paper's deep dip: 0.35x median, 0.25 s spacing, 0.15 sigma prominence."""
    med = float(np.median(env))
    pk, _ = find_peaks(-env, distance=int(0.25 * FS), prominence=PROM * np.std(env))
    pk = pk[env[pk] < DEPTH * med]
    return pk, med


def auc(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return np.nan
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    r = rankdata(x)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


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
                    b = idx[int(np.argmax(np.std(zf[:, idx], axis=0)))]
                    z = zf[:, b]
                    env = np.abs(z)
                    med = float(np.median(env))
                    cf, rf, _ = circle_geometric(z)
                    th = np.unwrap(np.angle(z - cf))
                    th_max = float((th.max() - th.min()) / 2)
                    # 全记录统计量（用于 D3）
                    base_row = {'layout': layout, 'pos': pos, 'bw_GHz': bw,
                                'repeat': k, 'bin': b,
                                'range_m': round(b * rb, 3),
                                'th_max': th_max,
                                'j0': float(abs(j0(th_max))),
                                'AsR': float(abs(complex(np.mean(z)))
                                             / (np.median(env) + 1e-12)),
                                'N_frames': len(z)}
                    for T in WINDOWS:
                        n = int(T * FS)
                        if n >= len(z) - 100:
                            continue
                        pre, post = z[:n], z[n:]
                        pk_pre, _ = dips_of(np.abs(pre))
                        pk_post, _ = dips_of(np.abs(post))
                        e_pre = np.abs(pre) / med
                        # 四个预测量：事件计数 / 包络方差 / 包络低分位 / 凹陷比例 D
                        preds = {
                            'n_deep': float(len(pk_pre)),
                            'env_var': float(np.var(e_pre)),
                            'env_q05': float(-np.percentile(e_pre, 5)),
                            'D': float(np.mean(e_pre < 0.3)),
                        }
                        row = dict(base_row)
                        row.update({'T_s': T, 'n_post': len(pk_post),
                                    'post_any': int(len(pk_post) >= 1),
                                    'post_five': int(len(pk_post) >= 5)})
                        row.update(preds)
                        rows.append(row)
    return pd.DataFrame(rows)


def report(df):
    print('=' * 92)
    print('D1. 与非事件型基线比较（预锁定 30 s，预测剩余 30 s 内是否出现深凹陷）')
    print('=' * 92)
    g = df[df.T_s == 30.0]
    for name, lab in [('n_deep', '事件计数 n_deep'), ('env_var', '包络方差'),
                      ('env_q05', '包络 5% 分位（取负）'), ('D', '凹陷比例 D')]:
        a1 = auc(g[name], g.post_any)
        a5 = auc(g[name], g.post_five)
        # 位置级留出（leave-one-position-out）
        lo = []
        for p in g.pos.unique():
            tr, te = g[g.pos != p], g[g.pos == p]
            if len(te) and len(np.unique(te.post_any)) > 1:
                lo.append(auc(te[name], te.post_any))
        print(f'  {lab:>22}: AUC(any)={a1:.3f}  AUC(>=5)={a5:.3f}  '
              f'position-held-out med={np.nanmedian(lo):.3f} ({len(lo)} folds)')

    print()
    print('=' * 92)
    print('D2. 窗长-延迟折中（n_deep 预测剩余段）')
    print('=' * 92)
    print(f'  {"T(s)":>6}{"n":>6}{"AUC(any)":>11}{"AUC(>=5)":>11}{"基础率":>9}')
    for T in WINDOWS:
        h = df[df.T_s == T]
        if not len(h):
            continue
        print(f'  {T:>6.0f}{len(h):>6}{auc(h.n_deep, h.post_any):>11.3f}'
              f'{auc(h.n_deep, h.post_five):>11.3f}{h.post_any.mean():>9.2f}')

    print()
    print('=' * 92)
    print('D3. Â_s 的 Bessel 偏差')
    print('=' * 92)
    print(f'  θ_max 中位 {df.th_max.median():.2f} rad | '
          f'|J0| 中位 {df.j0.median():.3f} (IQR {df.j0.quantile(.25):.3f}-'
          f'{df.j0.quantile(.75):.3f})')
    print(f'  ⇒ 复均值估计的静止相量相对真值系统性偏小约 '
          f'{100*(1-df.j0.median()):.0f}%')

    print()
    print('=' * 92)
    print('D4. 分带宽（30 s 窗，n_deep）')
    print('=' * 92)
    for bw in (3, 2.5, 2):
        h = df[(df.T_s == 30.0) & (df.bw_GHz == bw)]
        if len(h):
            print(f'  {bw:>3.1f} GHz  n={len(h)}  AUC {auc(h.n_deep, h.post_any):.3f}  '
                  f'基础率 {h.post_any.mean():.2f}')


def main() -> int:
    OUT.mkdir(exist_ok=True)
    cache = OUT / 'D_baselines.csv'
    if cache.exists():
        df = pd.read_csv(cache)
        print(f'[cache] {len(df)} 行')
    else:
        df = build()
        df.to_csv(cache, index=False, encoding='utf-8')
        print(f'[data] {len(df)} 行 ({df[["layout", "pos", "repeat"]].drop_duplicates().shape[0]} 条记录 × 窗长)')
    report(df)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
