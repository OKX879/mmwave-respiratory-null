#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E 组：四个合成实验，回答"模型是不是唯一解释"这一类质疑。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

OUT = Path(__file__).resolve().parent / 'exp_out'
FS = 20.0
N_FRAMES = 1200                 # 与 Lei 一致：60 s @ 20 Hz
DEPTH, PROM = 0.35, 0.15        # 论文的深凹陷定义
R_TH = 0.3                      # 论文 D 的阈值


# --------------------------------------------------------------- 公共
def detector(env):
    """论文的深凹陷检测器，返回 (事件数, 凹陷比例 D)。"""
    med = float(np.median(env))
    if med <= 0 or not np.isfinite(med):
        return 0, 0.0
    pk, _ = find_peaks(-env, distance=int(0.25 * FS), prominence=PROM * np.std(env))
    pk = pk[env[pk] < DEPTH * med]
    return int(len(pk)), float(np.mean(env < R_TH * med))


def synth(statics, motions, snr_db=30.0, rng=None):
    """statics: [(amp, phase)]；motions: [(amp, theta_max, f, chi)]。"""
    rng = rng or np.random.default_rng(0)
    t = np.arange(N_FRAMES) / FS
    z = np.zeros(N_FRAMES, complex)
    for a, ph in statics:
        z += a * np.exp(1j * ph)
    for a, th_max, f, chi in motions:
        z += a * np.exp(1j * th_max * np.cos(2 * np.pi * f * t + chi))
    if snr_db is not None:
        sig = np.sqrt(np.mean(np.abs(z) ** 2)) / 10 ** (snr_db / 20)
        z = z + sig * (rng.standard_normal(t.size)
                       + 1j * rng.standard_normal(t.size)) / np.sqrt(2)
    return z


def rho_eff(statics, motions):
    """|静止相量和| 与 RMS|运动相量和| 的比值。"""
    s = abs(sum(a * np.exp(1j * ph) for a, ph in statics))
    m2 = sum(a ** 2 for a, *_ in motions)
    return float(np.sqrt(m2) / (s + 1e-15))


# --------------------------------------------------------------- E1
# (静态数, 运动数, 频率关系)：'indep' 各自独立频率，'same' 同频，'harm' 基频+二次谐波
CONFIGS = {
    '1 static + 1 motion (paper)': (1, 1, 'same'),
    '2 statics + 1 motion':        (2, 1, 'same'),
    '1 static + 2 motions, indep': (1, 2, 'indep'),
    '1 static + 2 motions, same f':(1, 2, 'same'),
    '1 static + 2 motions, harmon':(1, 2, 'harm'),
    '2 statics + 2 motions, harmon':(2, 2, 'harm'),
    '1 static + 3 motions, indep': (1, 3, 'indep'),
}


def build_case(n_static, n_motion, fmode, target_ratio, rng):
    """按目标 ρ_eff 造一条记录。静态可有 1 或 2 个，相位随机。"""
    statics = [(1.0, rng.uniform(-np.pi, np.pi))]
    if n_static == 2:
        statics.append((rng.uniform(0.4, 1.0), rng.uniform(-np.pi, np.pi)))
    # 运动分量：总 RMS 幅度按 target 分配
    amps = rng.uniform(0.5, 1.5, n_motion)
    amps = amps / np.sqrt(np.sum(amps ** 2))          # RMS = 1
    s = abs(sum(a * np.exp(1j * ph) for a, ph in statics))
    f0 = rng.uniform(0.18, 0.42)
    motions = []
    for i in range(n_motion):
        a = target_ratio * s * amps[i]
        th = rng.uniform(4.0, 12.0)
        if fmode == 'indep':
            f = rng.uniform(0.18, 0.42)
        elif fmode == 'same':
            f = f0
        else:                                          # 基频 + 二次谐波
            f = f0 if i == 0 else 2 * f0
            th = th if i == 0 else th * rng.uniform(0.15, 0.35)
        chi = rng.uniform(0, 2 * np.pi)
        motions.append((a, th, f, chi))
    return statics, motions


def e1():
    print('=' * 92)
    print('E1. 多相量推广：失败区是否仍在 ρ_eff ≈ 1（同一检测器，SNR 30 dB）')
    print('=' * 92)
    ratios = np.array([0.4, 0.6, 0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 1.8, 2.2])
    rows = []
    for name, (n_static, n_motion, fmode) in CONFIGS.items():
        for ratio in ratios:
            Ds, Ns = [], []
            for seed in range(40):
                rng = np.random.default_rng(hash((name, round(ratio, 3), seed)) % 2 ** 31)
                st, mo = build_case(n_static, n_motion, fmode, ratio, rng)
                z = synth(st, mo, snr_db=30.0, rng=rng)
                n, D = detector(np.abs(z))
                Ds.append(D)
                Ns.append(n)
            rows.append({'config': name, 'n_motion': n_motion,
                         'fmode': fmode, 'rho': ratio,
                         'D_med': float(np.median(Ds)),
                         'D_p90': float(np.percentile(Ds, 90)),
                         'n_med': float(np.median(Ns)),
                         'deep_frac': float(np.mean(np.array(Ns) >= 5))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'E1_polyphasor.csv', index=False, encoding='utf-8')

    print(f'  {"配置":>28}' + ''.join(f'{r:>7.2f}' for r in ratios))
    print(f'  {"":28}' + '  D (median)' .rjust(0) + '')
    for name in CONFIGS:
        g = df[df.config == name].sort_values('rho')
        print(f'  {name:>28}' + ''.join(f'{v:>7.3f}' for v in g.D_med))

    print()
    print('  失败区（D 中位 ≥ 0.10）的 ρ_eff 区间与峰值位置：')
    for name in CONFIGS:
        g = df[df.config == name].sort_values('rho')
        hit = g[g.D_med >= 0.10]
        if len(hit) == 0:
            print(f'  {name:>28}: 无（最大 D = {g.D_med.max():.3f}）')
            continue
        peak = g.loc[g.D_med.idxmax()]
        print(f'  {name:>28}: ρ_eff ∈ [{hit.rho.min():.2f}, {hit.rho.max():.2f}]'
              f'   峰值在 ρ_eff = {peak.rho:.2f} (D={peak.D_med:.3f})')
    return df


# --------------------------------------------------------------- E2
def e2():
    print()
    print('=' * 92)
    print('E2. 固定 ρ，改变距离窗函数与零填充：D 是否只由 ρ 决定')
    print('=' * 92)
    nfft = 200
    t = np.arange(N_FRAMES) / FS
    rows = []
    for rho in (0.7, 0.85, 1.0, 1.15, 1.3, 1.6):
        for win in ('rect', 'hann', 'hamming'):
            for zf in (1, 4):
                Ds, Ns = [], []
                for seed in range(20):
                    rng = np.random.default_rng(hash((rho, win, zf, seed)) % 2 ** 31)
                    # 距离维：静态目标与运动目标相距 ΔR 的一小部分
                    fb_s = rng.uniform(0.25, 0.35)          # bin 分数位置
                    drift = rng.uniform(0.0, 0.4)           # 两者间距（bin 分数）
                    thr = rng.uniform(4.0, 12.0)
                    th = thr * np.cos(2 * np.pi 	* rng.uniform(0.2, 0.4) * t)
                    # 合成两目标的距离像（IFFT 域直接构造）
                    k = np.arange(nfft)
                    if win == 'rect':
                        w = np.ones(nfft)
                    elif win == 'hann':
                        w = np.hanning(nfft)
                    else:
                        w = np.hamming(nfft)
                    # 距离谱：(nfft, nframes)，第 m 个"目标"带自己的相位序列
                    # drift 以「bin 的分数」计 ⇒ 换算成 cycles/sample 需除以 nfft
                    spec = (np.exp(2j * np.pi * fb_s * k)[:, None]
                            + rho * np.exp(2j * np.pi * (fb_s + drift / nfft) * k)[:, None]
                            * np.exp(1j * th)[None, :])
                    spec = spec * w[:, None]
                    Z = np.fft.fft(spec, n=nfft * zf, axis=0)    # (nfft*zf, nframes)
                    nb = nfft * zf
                    c = int(round(fb_s * nb)) % nb      # fb_s 是 bin 分数位置
                    env = np.abs(Z[c, :])
                    n, D = detector(env)
                    Ds.append(D)
                    Ns.append(n)
                rows.append({'rho': rho, 'win': win, 'zf': zf,
                             'D_med': float(np.median(Ds)),
                             'n_med': float(np.median(Ns))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'E2_window.csv', index=False, encoding='utf-8')
    print(f'  {"ρ":>5}' + ''.join(f'{w}{z!s:>2}'.rjust(10) for w in ('rect', 'hann', 'hamming') for z in (1, 4)))
    print(f'  {"":>5}' + ''.join(f'{"D":>10}' for _ in range(6)))
    for rho in sorted(df.rho.unique()):
        g = df[df.rho == rho]
        cells = ''.join(f'{v:>10.3f}' for v in
                        [g[(g.win == w) & (g.zf == z)].D_med.iloc[0]
                         for w in ('rect', 'hann', 'hamming') for z in (1, 4)])
        print(f'  {rho:>5.2f}' + cells)
    spread = (df.groupby('rho').D_med.agg(lambda x: x.max() - x.min()))
    print(f'\n  同一 ρ 下六种窗/零填充组合的 D 极差：中位 {spread.median():.3f}'
          f'（最大 {spread.max():.3f}）⇒ D 主要由 ρ 决定，窗函数不构成独立通路')
    return df


# --------------------------------------------------------------- E3
def e3():
    print()
    print('=' * 92)
    print('E3. 噪声地板：E[min|w|] 与单样本 E|w| 的比值随窗长变化')
    print('=' * 92)
    rng = np.random.default_rng(7)
    print(f'  {"N 帧":>7}{"E[min|w|]/E|w|":>20}{"等效门槛下调":>16}')
    rows = []
    for N in (100, 300, 600, 1200, 2400):
        reps = 20000 if N <= 600 else 8000
        w = (rng.standard_normal((reps, N)) + 1j * rng.standard_normal((reps, N)))
        w /= np.sqrt(2)
        r = np.abs(w).min(axis=1).mean() / np.sqrt(np.pi / 2)
        rows.append({'N': N, 'ratio': float(r), 'db': float(20 * np.log10(r))})
        print(f'  {N:>7}{r:>20.3f}{20*np.log10(r):>15.1f} dB')
    pd.DataFrame(rows).to_csv(OUT / 'E3_noise_floor.csv', index=False, encoding='utf-8')
    print('  ⇒ √(π/2)σ 是单样本参考值；窗内有 N 帧时最小深度按上表下调，'
          '12.4 dB 须注明为单样本理论值。')
    return pd.DataFrame(rows)


# --------------------------------------------------------------- E4
def e4():
    print()
    print('=' * 92)
    print('E4. Â_s 估计误差对 κ 与相位步进的稳健性')
    print('=' * 92)

    def one_world(world, rng):
        t = np.arange(N_FRAMES) / FS
        f = rng.uniform(0.18, 0.40)
        if world == 'geometric':
            rho = rng.uniform(0.85, 1.15)
            th = rng.uniform(4.0, 12.0) * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
            z = 1.0 + rho * np.exp(1j * th)
            A_s_true = 1.0
        else:
            th_max = rng.uniform(1.2, 2.6)
            th = th_max * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
            A_c = np.clip(1.0 - rng.uniform(0.94, 0.99)
                          * 0.5 * (1 + np.cos(2 * np.pi * f * t)), 0.02, None)
            z = 0.06 + A_c * np.exp(1j * th)
            A_s_true = 0.06
        sig = np.sqrt(np.mean(np.abs(z) ** 2)) / 10 ** (25 / 20)
        z = z + sig * (rng.standard_normal(t.size)
                       + 1j * rng.standard_normal(t.size)) / np.sqrt(2)
        return z, A_s_true

    def measures(z, A_s_hat):
        a = z - A_s_hat
        env = np.abs(z)
        med = float(np.median(env))
        pk, _ = find_peaks(-env, distance=int(0.25 * FS), prominence=PROM * np.std(env))
        pk = pk[env[pk] < DEPTH * med]
        w = int(0.2 * FS)
        kap, dph, dep = [], [], []
        for p in pk:
            if p - w < 0 or p + w >= len(z):
                continue
            kappa = np.cos(np.angle(a[p] / (-A_s_hat)))
            kap.append(float(np.real(kappa)))
            ph = np.unwrap(np.angle(a[max(0, p - w):p + w]))
            dph.append(float(np.abs(ph[-1] - ph[0])))
            dep.append(float(1 - env[p] / med))
        return np.array(kap), np.array(dph), np.array(dep)

    def auc(x, y):
        from scipy.stats import rankdata
        x, y = np.asarray(x, float), np.asarray(y, int)
        if len(np.unique(y)) < 2:
            return np.nan
        pos, neg = int((y == 1).sum()), int((y == 0).sum())
        r = rankdata(x)
        return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))

    print(f'  {"扰动 ε":>8}{"AUC(几何 vs 调幅)":>22}{"ρ(深度,|Δφ|) 几何":>22}{"调幅":>9}')
    for eps in (0.0, 0.1, 0.2, 0.3):
        G, A, rg, ra = [], [], [], []
        for seed in range(120):
            rng = np.random.default_rng(1000 + seed)
            zg, As_g = one_world('geometric', rng)
            za, As_a = one_world('am', rng)
            # 扰动：幅度 ±ε、相位随机
            dg = eps * rng.uniform(0.5, 1.0) * np.exp(1j * rng.uniform(0, 2 * np.pi))
            da = eps * rng.uniform(0.5, 1.0) * np.exp(1j * rng.uniform(0, 2 * np.pi))
            kg, dg_, depg = measures(zg, As_g * (1 + dg))
            ka, da_, depa = measures(za, As_a * (1 + da))
            G.extend(dg_)
            A.extend(da_)
            if len(depg) > 5:
                rg.append(np.corrcoef(depg, dg_)[0, 1])
            if len(depa) > 5:
                ra.append(np.corrcoef(depa, da_)[0, 1])
        y = np.r_[np.ones(len(G), int), np.zeros(len(A), int)]
        print(f'  {eps:>8.2f}{auc(np.r_[G, A], y):>22.3f}'
              f'{np.nanmedian(rg):>22.3f}{np.nanmedian(ra):>9.3f}')
    print('  ⇒ 判别力随 Â_s 误差的变化即为该判据的稳健性边界。')


def main() -> int:
    OUT.mkdir(exist_ok=True)
    e1()
    e2()
    e3()
    e4()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
