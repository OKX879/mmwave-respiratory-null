#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""口径审计：① 复现 Table II 的事件定义；② 用记录级口径重算；③ 复核距离轴。

A. 事件定义扫描：在 (深度阈值 × prominence × 最小间隔) 上网格搜索，
   找出能精确复现 7/26/33（=13.0/48.1/62.3%）的那一组。
B. 记录级重算：固定 Table II 的选 bin 协议（每条记录 1 个 bin），
   比较五种减圆心估计 + 目标真实距离插值。
C. 距离轴复核：逐 bin 算"相位呼吸相干性"，含受试者的 bin 应显著高于静物 bin；
   若峰位落在名义 r/ΔR 上，说明 ΔR=c/2B 成立。

用法
----
    python exp_audit_protocol.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from scipy.stats import pearsonr

from lei_pilot import load, ROOT, R_BIN, FS
from exp_center_removal import (dip_stats, residual_series, EXPECT, ESTIMATORS)

OUT = Path(__file__).resolve().parent / 'exp_out'
R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']
NFINE = 16
NULL_RECORD = 5
TABLE2 = {3: (54, 7), 2.5: (54, 26), 2: (53, 33)}
R_LO, R_HI = 0.30, 2.00


def n_deep_variant(seg: np.ndarray, depth: float, prom: float, gap: float) -> int:
    """按参数化的事件定义数深凹陷个数（对应论文 §V）。"""
    env = np.abs(seg)
    med = float(np.median(env))
    if med <= 0:
        return 0
    pk, _ = find_peaks(-env, distance=int(gap * FS), prominence=prom * np.std(env))
    return int(np.sum(env[pk] < depth * med))


def resp_coherence(z: np.ndarray) -> float:
    """相位谱在 0.1–0.6 Hz 的能量占比：受试者所在 bin 高，静物 bin 低。"""
    ph = np.unwrap(np.angle(z))
    f, P = welch(ph - ph.mean(), fs=FS, nperseg=int(min(len(ph), 512)))
    tot = float(P[(f > 0.05) & (f < 2.0)].sum())
    return float(P[(f >= 0.1) & (f <= 0.6)].sum() / tot) if tot > 0 else 0.0


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rows, prof_mag, prof_coh = [], {}, {}
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
                    prof_mag.setdefault((layout, pos, bw), []).append(
                        np.mean(np.abs(zf), axis=0))
                    lo, hi = int(R_LO / rb), min(zf.shape[1] - 2, int(R_HI / rb))
                    prof_coh.setdefault((layout, pos, bw), []).append(
                        np.array([resp_coherence(zf[:, b]) for b in range(lo, hi)]))
                    # Table II 选 bin
                    idx = []
                    for r in R_TRUE[pos]:
                        idx += list(range(max(1, int((r - .25) / rb)),
                                          int((r + .05) / rb) + 1))
                    idx = sorted(set(idx))
                    h = len(zf) // 2
                    b = idx[int(np.argmax(np.std(zf[:h][:, idx], axis=0)))]
                    zs = zf[:, b]
                    rt = min(R_TRUE[pos], key=lambda r: abs(r - b * rb))
                    row = {'layout': layout, 'pos': pos, 'bw_GHz': bw, 'repeat': k,
                           'bin': b, 'range_m': round(b * rb, 3),
                           'r_near': rt,
                           'coh_sel': resp_coherence(zs)}
                    for est in ESTIMATORS:
                        row[f'n_{est}'] = dip_stats(residual_series(zs, est))[1]
                    row['n_tgt'] = dip_stats(Zf[:, int(round(rt / rb * NFINE))])[1]
                    for depth, prom, gap in itertools.product(
                            (0.30, 0.35), (0.10, 0.15, 0.20), (0.20, 0.25, 0.30)):
                        row[f'e_{depth}_{prom}_{gap}'] = n_deep_variant(
                            zs, depth, prom, gap)
                    rows.append(row)
                    print('.', end='', flush=True)
    print()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'audit_protocol.csv', index=False, encoding='utf-8')
    print(f'[data] {len(df)} 条记录；分带宽 {df.groupby("bw_GHz").size().to_dict()} '
          f'（Table II: 54/54/53）')

    print('\n' + '=' * 92)
    print('A. 事件定义扫描  目标 {3: 7, 2.5: 26, 2: 33}')
    print('=' * 92)
    cols = [c for c in df.columns if c.startswith('e_')]
    res = []
    for col in cols:
        cnt = {bw: int((g[col] >= NULL_RECORD).sum()) for bw, g in df.groupby('bw_GHz')}
        err = sum(abs(cnt.get(bw, 0) - TABLE2[bw][1]) for bw in TABLE2)
        res.append((err, col, cnt))
    res.sort(key=lambda x: x[0])
    for err, col, cnt in res[:6]:
        flag = '  ← 精确复现' if err == 0 else ''
        print(f'  {col:>16} → {cnt}  偏差 {err}{flag}')

    print('\n' + '=' * 92)
    print('B. 记录级重算（Table II 口径，每条记录 1 个 bin）')
    print('=' * 92)
    print(f'  {"带宽":>8}' + ''.join(f'{e:>10}' for e in ESTIMATORS)
          + f'{"目标中心":>10}')
    for bw in (3, 2.5, 2):
        g = df[df.bw_GHz == bw]
        print(f'  {bw:>7.1f}G' + ''.join(
            f'{np.mean(g[f"n_{e}"] >= NULL_RECORD):>9.1%}' for e in ESTIMATORS)
            + f'{np.mean(g.n_tgt >= NULL_RECORD):>10.1%}')
    print(f'  {"全部":>8}' + ''.join(
        f'{np.mean(df[f"n_{e}"] >= NULL_RECORD):>9.1%}' for e in ESTIMATORS)
        + f'{np.mean(df.n_tgt >= NULL_RECORD):>10.1%}')
    print(f'\n  选中 bin 的呼吸相干性中位：{df.coh_sel.median():.2f}')

    print('\n' + '=' * 92)
    print('C. 距离轴复核')
    print('=' * 92)

    def coh_peak(bw, pos):
        rb = R_BIN[bw]
        vals = []
        for lay in ('1_AsymmetricalPosition', '2_SymmetricalPosition'):
            if (lay, pos, bw) in prof_coh:
                vals.append(np.mean(prof_coh[(lay, pos, bw)], axis=0))
        if not vals:
            return None
        c = np.mean(vals, axis=0)
        lo = int(R_LO / rb)
        pk, _ = find_peaks(c, distance=4, prominence=0.05)
        if not len(pk):
            return None
        order = pk[np.argsort(c[pk])[::-1][:2]]
        return sorted((int(p + lo), round(float(c[p]), 3)) for p in order)

    for bw in (3, 2.5, 2):
        rb = R_BIN[bw]
        print(f'\n  --- {bw} GHz, ΔR={rb*100:.1f} cm ---')
        print(f'    {"位置":>5}{"名义bin(距离)":>26}{"相干峰bin(值)":>22}{"偏差bin":>10}')
        devs = []
        for pos in sorted(R_TRUE):
            nom = [round(r / rb) for r in R_TRUE[pos]]
            got = coh_peak(bw, pos)
            if got is None:
                continue
            gb = [g[0] for g in got]
            dev = min(min(abs(g - n) for n in nom) for g in gb)
            devs.append(dev)
            print(f'    {pos:>5}{str([(n, round(n*rb,2)) for n in nom]):>26}'
                  f'{str(got):>22}{dev:>10}')
        if devs:
            print(f'    → 峰位与名义 bin 的偏差中位 {np.median(devs):.1f} bin'
                  f'（{np.median(devs)*rb*100:.1f} cm）')

    print('\n  距离像互相关（重采样到 2 cm 物理网格，0.4–2.0 m）：')
    grid = np.arange(0.4, 2.0, 0.02)

    def rs(P, rb):
        r = np.arange(len(P)) * rb
        return np.log(np.interp(grid, r, P / np.median(P)))

    for a, b in ((3, 2.5), (3, 2), (2.5, 2)):
        vals = []
        for pos in R_TRUE:
            for lay in ('1_AsymmetricalPosition', '2_SymmetricalPosition'):
                ka, kb = (lay, pos, a), (lay, pos, b)
                if ka in prof_mag and kb in prof_mag:
                    va = np.mean(prof_mag[ka], axis=0)
                    vb = np.mean(prof_mag[kb], axis=0)
                    vals.append(pearsonr(rs(va, R_BIN[a]), rs(vb, R_BIN[b]))[0])
        if vals:
            print(f'    {a} vs {b} GHz: 相关中位 {np.median(vals):+.2f} '
                  f'（n={len(vals)}）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
