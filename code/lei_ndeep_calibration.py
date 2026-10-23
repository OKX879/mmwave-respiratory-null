#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n_deep 的校准：给出 R1 里"calibrated threshold"的具体数字。
Calibrate n_deep, giving the numbers behind the calibrated threshold in R1."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import rankdata

from lei_pilot import load, ROOT, R_BIN, FS, bp, slow_mean

OUT = Path(__file__).resolve().parent / 'null_crossdataset'
R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']
EXPECT = 1200 * 3 * 4 * 200 * 2 * 2


def auc(x, y) -> float:
    """Mann-Whitney AUC，**并列值用平均秩**（避免 0 值被任意排序）。
Mann-Whitney AUC with average ranks for ties, so zero values are not ordered arbitrarily."""
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    return float((rankdata(x)[y == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def auc_naive(x, y) -> float:
    """`lei_D_prospective.py` 原来的写法（argsort 名次，**并列值按出现顺序**）。
The original formulation in lei_D_prospective.py: argsort ranks with ties broken by order of appearance."""
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    order = allv.argsort().argsort() + 1
    return float((order[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def deep_count(seg: np.ndarray) -> int:
    env = np.abs(seg)
    med = float(np.median(env))
    pk, _ = find_peaks(-env, distance=int(0.25 * FS), prominence=0.15 * np.std(env))
    return int(np.sum(env[pk] < 0.35 * med))


def dip_frac(seg: np.ndarray) -> float:
    env = np.abs(seg)
    return float(np.mean(env < 0.3 * np.median(env)))


def as_over_r(seg: np.ndarray) -> float:
    a = bp(seg)
    return float(abs(complex(np.mean(seg))) / (np.mean(np.abs(a)) + 1e-12))


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
                    zf = np.fft.fft(load(fp), axis=3).mean(axis=(1, 2))
                    idx = []
                    for r in R_TRUE[pos]:
                        a = max(1, int((r - 0.25) / rb))
                        b = min(zf.shape[1] - 2, int((r + 0.05) / rb))
                        idx += list(range(a, b + 1))
                    idx = sorted(set(idx))
                    h = len(zf) // 2
                    z1, z2 = zf[:h], zf[h:]
                    # 前半段选格（与前瞻脚本同一规则：呼吸带复数标准差最大）
                    e = np.std(z1[:, idx], axis=0)
                    sel = idx[int(np.argmax(e))]
                    for b in idx:
                        rows.append({
                            'layout': layout, 'pos': pos, 'bw_GHz': bw, 'repeat': k,
                            'bin': b, 'range_m': round(b * rb, 3),
                            'is_selected': int(b == sel),
                            'n1': deep_count(z1[:, b]), 'n2': deep_count(z2[:, b]),
                            'n1_20s': deep_count(z1[:int(20 * FS), b]),
                            'D1': dip_frac(z1[:, b]),
                            'logAsR': float(np.log10(max(as_over_r(z1[:, b]), 1e-9))),
                        })
    return pd.DataFrame(rows)


def threshold_table(x, y, ks, label: str):
    npos, nneg = int(y.sum()), int((1 - y).sum())
    print(f'\n  [{label}]  正例 {npos} / 负例 {nneg}  AUC={auc(x, y):.3f}')
    print(f'  {"阈值":>6}{"TP":>5}{"FP":>5}{"FN":>5}{"TN":>5}{"sens":>7}{"spec":>7}'
          f'{"PPV":>7}{"J":>7}')
    best = None
    for k in ks:
        pred = x >= k
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = npos - tp; tn = nneg - fp
        sens = tp / max(npos, 1); spec = tn / max(nneg, 1)
        ppv = tp / max(tp + fp, 1); j = sens + spec - 1
        if best is None or j > best[1]:
            best = (k, j, sens, spec)
        print(f'  {k:>6}{tp:>5}{fp:>5}{fn:>5}{tn:>5}{sens:>7.2f}{spec:>7.2f}'
              f'{ppv:>7.2f}{j:>7.2f}')
    print(f'  → Youden 最优：阈值 {best[0]}（J={best[1]:.2f}, sens {best[2]:.2f}, '
          f'spec {best[3]:.2f}）')
    return best


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(OUT / 'lei_ndeep_calibration.csv', index=False, encoding='utf-8')
    print(f'[data] bins={len(df)} records='
          f'{df.groupby(["layout", "pos", "bw_GHz", "repeat"]).ngroups}')

    # 回归掉 |A_s|/R 的残差（秩不变：OLS 在秩上等价于线性投影）
    b1, b0 = np.polyfit(df.logAsR, df.n1, 1)
    df['n1_res'] = df.n1 - (b0 + b1 * df.logAsR)
    print(f'[residual] n1 = {b0:+.3f} {b1:+.3f}·log10(|As|/R)；'
          f'Spearman(n1, logAsR) = {df.n1.corr(df.logAsR, method="spearman"):+.3f}')

    for thr in (1, 5):
        y = (df.n2 >= thr).astype(int)
        print(f'\n=== 目标：后半段 n2 >= {thr}（基础率 {y.mean():.1%}）===')
        threshold_table(df.n1, y, [1, 2, 3, 4, 5, 6, 8, 10], 'n1（原始事件数）')
        threshold_table(df.n1_res, y, [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                        'n1_res（回归掉 |As|/R 后）')

    # 与 D 的对照：同一批 bin、同一目标
    y1 = (df.n2 >= 1).astype(int)
    print('\n=== 与 D 的对照（同一批 bin）===')
    print(f'  AUC(D1 -> n2>=1) = {auc(df.D1, y1):.3f}；'
          f'AUC(n1 -> n2>=1) = {auc(df.n1, y1):.3f}')
    pred = df.D1 > 0.01
    tp = int((pred & (y1 == 1)).sum()); fp = int((pred & (y1 == 0)).sum())
    npos, nneg = int(y1.sum()), int((1 - y1).sum())
    print(f'  D1>1%: sens={tp/max(npos,1):.2f} spec={(nneg-fp)/max(nneg,1):.2f} '
          f'（同一工作点下 n1 的等价阈值见上表）')

    # 只在前瞻脚本选中的那一格上复核（与论文表内数字同口径）
    sel = df[df.is_selected == 1]
    ys = (sel.n2 >= 1).astype(int)
    print(f'\n[selected bin only] n={len(sel)} 基础率 {ys.mean():.1%}  '
          f'AUC(D1)={auc(sel.D1, ys):.3f}  AUC(n1)={auc(sel.n1, ys):.3f}')

    # ---- AUC 0.87 的口径复原：并列值处理的影响 ----
    print('\n=== AUC「0.87」的口径复原 ===')
    print(f'  同一批 161 格、同一目标 (n2>=1)、同一 D1：')
    print(f'    argsort 名次（原脚本，并列按出现顺序）: {auc_naive(sel.D1, ys):.3f}')
    print(f'    Mann-Whitney 平均秩（正确）          : {auc(sel.D1, ys):.3f}')
    print(f'  分母：{len(sel)} 条记录（每条记录 1 个前半段选出的格），'
          f'正例 {int(ys.sum())} / 负例 {int((1 - ys).sum())}；D1=0 的格 '
          f'{int((sel.D1 == 0).sum())} 个')

    # ---- 记录内 AUC（去掉"哪条记录"这一层混淆）----
    print('\n=== 记录内 AUC（只在同时含正负例的记录上算）===')
    for col, name in (('n1', 'n1'), ('D1', 'D1')):
        vals = []
        for _, g in df.groupby(['layout', 'pos', 'bw_GHz', 'repeat']):
            y = (g.n2 >= 1).astype(int)
            if y.nunique() > 1:
                vals.append(auc(g[col], y))
        v = np.asarray([x for x in vals if np.isfinite(x)])
        print(f'  {name}: n_records={len(v)} 中位 {np.median(v):.3f} '
              f'IQR [{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}] '
              f'|rho|>=0.5 占 {float((np.abs(v - 0.5) >= 0.5).mean()):.0%}')

    # ---- 记录内阈值曲线：每记录自己算 sens/spec 再平均 ----
    print('\n=== 记录内阈值曲线（逐记录算 sens/spec，再对有该类的记录取均值）===')
    for col, ks in (('n1', [1, 2, 3, 4, 5, 6]), ('n1_20s', [1, 2, 3, 4, 5])):
        print(f'  [{col}]  {"k":>3}{"n_rec":>7}{"sens":>7}{"spec":>7}{"J":>7}')
        for k in ks:
            sens, spec = [], []
            for _, g in df.groupby(['layout', 'pos', 'bw_GHz', 'repeat']):
                y = (g.n2 >= 1).to_numpy()
                pred = (g[col] >= k).to_numpy()
                if y.any():
                    sens.append(pred[y].mean())
                if (~y).any():
                    spec.append((~pred[~y]).mean())
            s, p = float(np.mean(sens)), float(np.mean(spec))
            print(f'  {col:>9}{k:>3}{len(sens):>7}{s:>7.2f}{p:>7.2f}{s+p-1:>7.2f}')

    print(f'\n[saved] {OUT / "lei_ndeep_calibration.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
