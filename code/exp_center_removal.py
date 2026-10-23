#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1：减圆心（DC offset / null point 修正）对比实验。
E1: comparison of static-centre removal, i.e. DC offset or null-point correction."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.special import j0 as bessel_j0
from scipy.stats import spearmanr

from lei_pilot import load, ROOT, R_BIN, FS, bp, slow_mean

OUT = Path(__file__).resolve().parent / 'exp_out'
R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']
EXPECT = 1200 * 3 * 4 * 200 * 2 * 2
NULL_DEPTH = 0.3
EVENT_DEPTH = 0.35
NULL_RECORD = 5
ESTIMATORS = ('none', 'mean', 'slow', 'circle', 'wcircle')


# --------------------------------------------------------------- 圆拟合
def circle_kasa(z: np.ndarray) -> tuple[complex, float]:
    x, y = z.real.astype(float), z.imag.astype(float)
    A = np.c_[x, y, np.ones_like(x)]
    b = -(x ** 2 + y ** 2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -sol[0] / 2, -sol[1] / 2
    r = float(np.sqrt(max(cx ** 2 + cy ** 2 - sol[2], 0.0)))
    return complex(cx, cy), r


def circle_geometric(z: np.ndarray) -> tuple[complex, float, float]:
    c0, r0 = circle_kasa(z)
    if not np.isfinite(r0) or r0 <= 0:
        return c0, max(r0, 1e-9), float('nan')
    x, y = z.real.astype(float), z.imag.astype(float)

    def resid(p):
        return np.hypot(x - p[0], y - p[1]) - p[2]

    try:
        sol = least_squares(resid, [c0.real, c0.imag, r0], method='lm', max_nfev=200)
        cx, cy, r = sol.x
        if not np.isfinite(r) or r <= 0:
            return c0, r0, float('nan')
        return complex(cx, cy), float(r), float(np.sqrt(np.mean(sol.fun ** 2)))
    except Exception:
        return c0, r0, float('nan')


def circle_windowed(z: np.ndarray, w: float = 10.0, hop: float = 5.0) -> np.ndarray:
    n = len(z)
    nw, nh = int(w * FS), int(hop * FS)
    if nw >= n:
        c, _ = circle_kasa(z)
        return np.full(n, c, dtype=complex)
    centers, mids = [], []
    for s in range(0, n - nw + 1, nh):
        c, _ = circle_kasa(z[s:s + nw])
        centers.append(c)
        mids.append(s + nw / 2)
    centers = np.asarray(centers, dtype=complex)
    mids = np.asarray(mids, dtype=float)
    return (np.interp(np.arange(n), mids, centers.real)
            + 1j * np.interp(np.arange(n), mids, centers.imag))


# --------------------------------------------------------------- 指标
def dip_stats(sig: np.ndarray, ref_scale: float | None = None) -> tuple[float, int]:
    """(凹陷帧占比, 深凹陷事件数)。
Returns the fraction of dip frames and the number of deep-dip events."""
    env = np.abs(sig)
    med = float(np.median(env))
    ref = med if ref_scale is None else float(ref_scale)
    if ref <= 0 or med <= 0:
        return float('nan'), 0
    D = float(np.mean(env < NULL_DEPTH * ref))
    pk, _ = find_peaks(-env, distance=int(0.25 * FS),
                       prominence=0.15 * np.std(env))
    pk = pk[env[pk] < EVENT_DEPTH * ref]
    return D, int(len(pk))


def q05(sig: np.ndarray) -> float:
    """归一化包络的 5% 分位：1.0 = 完全不塌陷（纯圆周），0 = 完全塌陷。
5th percentile of the normalized envelope: 1.0 means no collapse, 0 means full collapse."""
    env = np.abs(sig)
    med = float(np.median(env))
    return float(np.percentile(env, 5) / med) if med > 0 else float('nan')


def radius_modulation(sig: np.ndarray) -> float:
    env = np.abs(sig)
    med = float(np.median(env))
    return float((np.percentile(env, 95) - np.percentile(env, 5)) / med) if med > 0 else float('nan')


def aperture_rad(sig: np.ndarray) -> float:
    ph = np.sort(np.angle(sig))
    if len(ph) < 10:
        return float('nan')
    gaps = np.diff(np.r_[ph, ph[0] + 2 * np.pi])
    return float(2 * np.pi - gaps.max())


def as_over_r_est(z: np.ndarray) -> float:
    a = bp(z)
    return float(abs(complex(np.mean(z))) / (np.mean(np.abs(a)) + 1e-12))


def residual_series(z: np.ndarray, est: str) -> np.ndarray:
    if est == 'none':
        return z
    if est == 'mean':
        return z - complex(np.mean(z))
    if est == 'slow':
        return z - slow_mean(z, 10.0)
    if est == 'circle':
        c, _, _ = circle_geometric(z)
        return z - c
    if est == 'wcircle':
        return z - circle_windowed(z)
    raise ValueError(est)


# --------------------------------------------------------------- 合成验证
def synthetic_check() -> pd.DataFrame:
    rng = np.random.default_rng(20261022)
    t = np.arange(int(60 * FS)) / FS
    rows = []
    for rho in (0.3, 1.0, 3.0, 19.7):
        for pp_mm in (2.0, 4.0, 8.0, 12.0):
            for snr_db in (10, 20, 40):
                d = (pp_mm / 2) * np.sin(2 * np.pi * 0.25 * t)
                theta = 4 * np.pi * (d * 1e-3) / 5e-3
                As = rho * np.exp(1j * rng.uniform(-np.pi, np.pi))
                sig = As + np.exp(1j * theta)
                n0 = np.std(sig) / (10 ** (snr_db / 20))
                sig = sig + n0 * (rng.standard_normal(len(t))
                                  + 1j * rng.standard_normal(len(t))) / np.sqrt(2)
                row = {'rho': rho, 'pp_mm': pp_mm, 'snr_db': snr_db,
                       'theta_max': float(np.max(np.abs(theta))),
                       'j0_pred': float(abs(bessel_j0(np.max(np.abs(theta))))),
                       'bias_mean': float(abs(complex(np.mean(sig)) - As))}
                med_raw = float(np.median(np.abs(sig)))
                for est in ESTIMATORS:
                    res = residual_series(sig, est)
                    D, n = dip_stats(res)
                    Dabs, nabs = dip_stats(res, ref_scale=med_raw)
                    row[f'D_{est}'], row[f'n_{est}'] = D, n
                    row[f'Dabs_{est}'], row[f'nabs_{est}'] = Dabs, nabs
                    row[f'q05_{est}'] = q05(res)
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 真实数据
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
                    e = np.std(zf[:h][:, idx], axis=0)
                    sel = idx[int(np.argmax(e))]
                    key = dict(layout=layout, pos=pos, bw_GHz=bw, repeat=k)
                    for b in idx:
                        z = zf[:, b]
                        z1, z2 = z[:h], z[h:]
                        med_raw = float(np.median(np.abs(z)))
                        row = {**key, 'bin': b, 'range_m': round(b * rb, 3),
                               'is_selected': int(b == sel), 'AsR': as_over_r_est(z)}
                        # 圆心漂移：前后半段各自独立几何拟合
                        c1, r1, _ = circle_geometric(z1)
                        c2, r2, _ = circle_geometric(z2)
                        cfull, rfull, rms = circle_geometric(z)
                        row['ap_raw_deg'] = float(np.degrees(aperture_rad(z - cfull)))
                        row['leak'] = float(abs(c1 - c2) / (rfull + 1e-12))
                        row['fit_rms_rel'] = float(rms / (rfull + 1e-12))
                        # 复均值相对圆拟合圆心的偏差（Bessel 偏差的直接度量）
                        row['d_mean'] = float(abs(complex(np.mean(z)) - cfull)
                                              / (rfull + 1e-12))
                        # 运动相量扫过的半角：theta_max = (max-min)/2
                        th = np.unwrap(np.angle(z - cfull))
                        row['theta_max'] = float((th.max() - th.min()) / 2)
                        row['j0'] = float(abs(bessel_j0(row['theta_max'])))
                        for est in ESTIMATORS:
                            res = residual_series(z, est)
                            D, n = dip_stats(res)
                            Dabs, nabs = dip_stats(res, ref_scale=med_raw)
                            row[f'D_{est}'], row[f'n_{est}'] = D, n
                            row[f'Dabs_{est}'], row[f'nabs_{est}'] = Dabs, nabs
                            row[f'q05_{est}'] = q05(res)
                            row[f'mod_{est}'] = radius_modulation(res)
                        rows.append(row)
    return pd.DataFrame(rows)


def null_rate(x: pd.Series, thr: int = NULL_RECORD) -> float:
    return float(np.mean(x >= thr))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', action='store_true')
    args = ap.parse_args()

    print('=' * 94)
    print('A. 合成验证：模型成立时圆拟合应把零点压到 0（检验审稿人的数学判断）')
    print('=' * 94)
    syn = synthetic_check()
    ok = syn[(syn.rho <= 1.0) & (syn.snr_db >= 20)]
    print(f'  ρ≤1 且 SNR≥20 dB 的配置 {len(ok)} 个：')
    for est in ESTIMATORS:
        print(f'    {est:>8}: 事件数中位 {ok[f"n_{est}"].median():>5.1f}   '
              f'q05 中位 {ok[f"q05_{est}"].median():>5.3f}   '
              f'绝对尺度残余事件中位 {ok[f"nabs_{est}"].median():>5.1f}')
    print(f'  → 圆拟合后事件数中位 {ok["n_circle"].median():.0f}、q05 中位 '
          f'{ok["q05_circle"].median():.2f}（不减时 {ok["n_none"].median():.0f} / '
          f'{ok["q05_none"].median():.2f}）⇒ 理想模型下"减圆心"确实消除零点，'
          '审稿人这条数学上成立。')
    print('\n  同一批合成数据：复均值估计的圆心偏差 vs |J_0(θ_max)|（Bessel 偏差）')
    print(f'    {"pp(mm)":>8}{"θ_max(rad)":>12}{"|J0| 预测":>12}{"实测偏差":>10}')
    for pp in (2.0, 4.0, 8.0, 12.0):
        g = syn[(syn.pp_mm == pp) & (syn.snr_db == 40) & (syn.rho == 1.0)]
        if len(g):
            r0 = g.iloc[0]
            print(f'    {pp:>8.1f}{r0.theta_max:>12.1f}{r0.j0_pred:>12.3f}'
                  f'{r0.bias_mean:>10.3f}')
    if args.synthetic:
        return 0

    print('\n' + '=' * 94)
    print('B. 真实数据：161 条 Lei 记录')
    print('=' * 94)
    df = build()
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / 'exp_center_removal.csv', index=False, encoding='utf-8')
    sel = df[df.is_selected == 1].copy()
    print(f'[data] bins={len(df)}  records={len(sel)}')

    print('\n--- B1. 归一化包络 5% 分位（1.00=不塌陷，0=彻底塌陷），选中 bin 中位 ---')
    print(f'{"带宽":>8}' + ''.join(f'{e:>10}' for e in ESTIMATORS))
    for bw in (3, 2.5, 2):
        g = sel[sel.bw_GHz == bw]
        print(f'{bw:>7.1f}G' + ''.join(f'{g[f"q05_{e}"].median():>10.3f}'
                                       for e in ESTIMATORS))

    print('\n--- B2. 凹陷帧占比 D（自归一化，论文原定义），选中 bin 中位 ---')
    print(f'{"带宽":>8}' + ''.join(f'{e:>10}' for e in ESTIMATORS))
    for bw in (3, 2.5, 2):
        g = sel[sel.bw_GHz == bw]
        print(f'{bw:>7.1f}G' + ''.join(f'{g[f"D_{e}"].median():>10.4f}'
                                       for e in ESTIMATORS))
    print(f'{"全部":>8}' + ''.join(f'{sel[f"D_{e}"].median():>10.4f}'
                                   for e in ESTIMATORS))

    print('\n--- B3. 记录级零点率（n_deep ≥ 5，论文 Table II 的统计量）---')
    print(f'{"带宽":>8}' + ''.join(f'{e:>10}' for e in ESTIMATORS))
    for bw in (3, 2.5, 2):
        g = sel[sel.bw_GHz == bw]
        print(f'{bw:>7.1f}G' + ''.join(f'{null_rate(g[f"n_{e}"]):>9.1%}'
                                       for e in ESTIMATORS))
    print(f'{"全部":>8}' + ''.join(f'{null_rate(sel[f"n_{e}"]):>9.1%}'
                                   for e in ESTIMATORS))

    print('\n--- B4. 残余静止相量 ρ_res = |Â(前半)-Â(后半)| / R（=漂移+估计误差）---')
    print(f'{"带宽":>8}{"中位":>9}{"P75":>8}{"P95":>8}{"ρ>0.5":>9}{"ρ>1":>8}'
          f'{"相位误差上界arcsin中位":>24}')
    for bw in (3, 2.5, 2):
        g = sel[sel.bw_GHz == bw]
        ph = np.degrees(np.arcsin(np.clip(g.leak, 0, 1)))
        print(f'{bw:>7.1f}G{g.leak.median():>9.2f}{g.leak.quantile(.75):>8.2f}'
              f'{g.leak.quantile(.95):>8.2f}{np.mean(g.leak > 0.5):>9.1%}'
              f'{np.mean(g.leak > 1.0):>8.1%}{ph.median():>20.0f}°')
    ph = np.degrees(np.arcsin(np.clip(sel.leak, 0, 1)))
    print(f'{"全部":>8}{sel.leak.median():>9.2f}{sel.leak.quantile(.75):>8.2f}'
          f'{sel.leak.quantile(.95):>8.2f}{np.mean(sel.leak > 0.5):>9.1%}'
          f'{np.mean(sel.leak > 1.0):>8.1%}{ph.median():>20.0f}°')

    print('\n--- B5. 残余塌陷由什么决定（Spearman，仅原本有深凹陷的记录）---')
    sub = sel[sel.n_none >= 1]
    print(f'     n={len(sub)}')
    for est in ('mean', 'circle', 'wcircle'):
        print(f'  [{est}] 残余事件 vs ρ_res: '
              f'{spearmanr(sub.leak, sub[f"n_{est}"]).statistic:+.2f}   '
              f'vs 半径调制: {spearmanr(sub[f"mod_{est}"], sub[f"n_{est}"]).statistic:+.2f}   '
              f'vs 原 |As|/R: {spearmanr(sub.AsR, sub[f"n_{est}"]).statistic:+.2f}   '
              f'vs 弧段覆盖: {spearmanr(sub.ap_raw_deg, sub[f"n_{est}"]).statistic:+.2f}')

    print('\n--- B6. 轨迹离理想圆周有多远（模型违反度，选中 bin 中位）---')
    print(f'  弧段覆盖率 {sel.ap_raw_deg.median():.0f}°；'
          f'θ_max 中位 {sel.theta_max.median():.1f} rad；'
          f'几何拟合残差/半径 {sel.fit_rms_rel.median():.3f}；'
          f'圆拟合后半径调制指数 {sel.mod_circle.median():.3f}')
    print(f'  弧段覆盖 <180° 的记录占 {np.mean(sel.ap_raw_deg < 180):.1%}')

    print('\n--- B7. 残余零点深度是否由"圆心估计不准"定量预测：'
          '观测 q05 vs 预测 |1-ρ_res| ---')
    for est in ('mean', 'circle', 'wcircle'):
        obs = sel[f'q05_{est}'].to_numpy()
        pred = np.abs(1 - sel.leak.to_numpy())
        ok = np.isfinite(obs) & np.isfinite(pred)
        r = spearmanr(pred[ok], obs[ok]).statistic
        print(f'  [{est}] 预测 vs 观测：Spearman {r:+.2f}；'
              f'中位绝对偏差 {np.median(np.abs(pred[ok]-obs[ok])):.3f}')

    print('\n--- B8. 复均值作为圆心估计的 Bessel 偏差（大摆动时系统性偏高）---')
    print(f'  θ_max 中位 {sel.theta_max.median():.2f} rad  ⇒  '
          f'|J_0(θ_max)| 中位 {sel.j0.median():.3f}（= 均值偏差 / 运动相量幅度）')
    r = spearmanr(sel.j0, sel.d_mean).statistic
    print(f'  实测偏差 d_mean = |mean(z)-ĉ|/R 中位 {sel.d_mean.median():.3f}；'
          f'与 |J_0| 的相关 {r:+.2f}')

    print('\n--- B9. 逐 bin 池化（全候选格，n=%d）---' % len(df))
    for est in ESTIMATORS:
        print(f'  {est:>8}: 有深凹陷的 bin 占 {np.mean(df[f"n_{est}"] >= 1):>6.1%}；'
              f'D 中位 {df[f"D_{est}"].median():.4f}；'
              f'q05 中位 {df[f"q05_{est}"].median():.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
