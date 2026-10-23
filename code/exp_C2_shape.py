#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C2：形状检验，区分几何相消与幅度调制。
C2: shape test separating geometric cancellation from amplitude modulation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from lei_pilot import load, ROOT, R_BIN, FS, bp
from exp_center_removal import circle_geometric, EXPECT

OUT = Path(__file__).resolve().parent / 'exp_out'
FIG = Path(__file__).resolve().parent / 'figs'
R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']

DEPTH = 0.35          # event definition of the paper
PROM = 0.15
DELTA = 0.10          # s, half the inner offset of the shape estimator


# --------------------------------------------------------------- 形状统计量
def dip_shapes(z, pk):
    """计算每个凹陷处的时间指数 p 与相位步进。
The p exponent and the phase step at each dip index in pk."""
    env = np.abs(z)
    med = float(np.median(env))
    e = env / med
    th = np.unwrap(np.angle(z))
    n1 = int(round(DELTA * FS))
    n2 = int(round(2 * DELTA * FS))
    rows = []
    for p in pk:
        lo, hi = p - n2, p + n2 + 1
        if lo < 0 or hi > len(z):
            continue
        e0 = float(e[lo:hi][n2])
        inner = 0.5 * (e[p - n1] + e[p + n1])
        outer = 0.5 * (e[p - n2] + e[p + n2])
        den = outer - e0
        if den <= 0.02:
            continue
        ratio = (inner - e0) / den
        if not (0.02 < ratio < 0.98):
            continue
        p_hat = float(-np.log2(ratio))
        rows.append({'p_hat': p_hat, 'ratio': float(ratio), 'e0': e0,
                     'depth': 1.0 - e0,
                     'dphi': float(abs(th[p + n2] - th[p - n2]))})
    return rows


def detector(z):
    """论文的事件定义，原样复现。
The paper's event definition, verbatim."""
    env = np.abs(z)
    med = float(np.median(env))
    pk, _ = find_peaks(-env, distance=int(0.25 * FS), prominence=PROM * np.std(env))
    return pk[env[pk] < DEPTH * med]


# --------------------------------------------------------------- 真实数据
def real_dips():
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
                    for row in dip_shapes(z, detector(z)):
                        row.update({'world': 'real', 'bw_GHz': bw,
                                    'layout': layout, 'pos': pos, 'repeat': k})
                        rows.append(row)
            print('.', end='', flush=True)
    print()
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 合成世界
def synth_geometric(n_trials=60, snr_db=25.0, rng=None):
    """双相量世界：|z| 扫过或接近原点。
Two-phasor world: |z| passes through or near the origin."""
    rng = rng or np.random.default_rng(0)
    t = np.arange(int(60 * FS)) / FS
    rows = []
    for _ in range(n_trials):
        f = rng.uniform(0.18, 0.40)
        th_max = rng.uniform(4.0, 12.0)
        rho = rng.uniform(0.85, 1.15)          # A_s / A_c
        A_c = 1.0
        th = th_max * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        z = rho * A_c + A_c * np.exp(1j * th)
        sig = np.sqrt(np.mean(np.abs(z) ** 2)) / 10 ** (snr_db / 20)
        z = z + sig * (rng.standard_normal(t.size)
                       + 1j * rng.standard_normal(t.size)) / np.sqrt(2)
        for row in dip_shapes(z, detector(z)):
            row.update({'world': 'geometric'})
            rows.append(row)
    return pd.DataFrame(rows)


def synth_am(n_trials=60, snr_db=25.0, rng=None):
    """幅度调制世界：运动相量衰减，弧段永远到不了反相。
Amplitude-modulated world: the motion phasor fades and the arc never reaches antiphase."""
    rng = rng or np.random.default_rng(1)
    t = np.arange(int(60 * FS)) / FS
    rows = []
    for _ in range(n_trials):
        f = rng.uniform(0.18, 0.40)
        th_max = rng.uniform(1.2, 2.6)          # < pi: no antiphase crossing
        rho = rng.uniform(0.03, 0.10)           # static phasor small but fixed
        m = rng.uniform(0.94, 0.99)
        th = th_max * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        A_c = 1.0 - m * 0.5 * (1 + np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)))
        A_c = np.clip(A_c, 0.02, None)
        z = rho + A_c * np.exp(1j * th)
        sig = np.sqrt(np.mean(np.abs(z) ** 2)) / 10 ** (snr_db / 20)
        z = z + sig * (rng.standard_normal(t.size)
                       + 1j * rng.standard_normal(t.size)) / np.sqrt(2)
        for row in dip_shapes(z, detector(z)):
            row.update({'world': 'am'})
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 报表
BANDS = [(0.0, 0.65, '<0.65'), (0.65, 0.80, '0.65-0.8'),
         (0.80, 0.90, '0.8-0.9'), (0.90, 1.01, '>0.9')]


def summarise(df):
    from scipy.stats import spearmanr
    print('=' * 88)
    print('C2. 形状判据：几何相消 vs 幅度调制')
    print('=' * 88)
    print('  判据：跨凹陷的相位步进 |dphi|。几何相消要求轨迹穿过原点附近，')
    print('        凹陷越深 ⇒ 必须越靠近原点 ⇒ 相位转角越大；')
    print('        幅度调制只是让运动相量变小，没有理由随深度多转。')
    print(f'  {"世界":>10}{"事件数":>8}{"|dphi| 中位":>12}{"P75":>7}'
          f'{"|dphi|>2.0":>11}{"|dphi|>2.5":>11}{"rho(深度,|dphi|)":>18}')
    for w in ('real', 'geometric', 'am'):
        g = df[df.world == w]
        if not len(g):
            continue
        r = spearmanr(g.depth, g.dphi).statistic
        print(f'  {w:>10}{len(g):>8}{g.dphi.median():>12.2f}'
              f'{g.dphi.quantile(.75):>7.2f}'
              f'{np.mean(g.dphi > 2.0):>10.1%}{np.mean(g.dphi > 2.5):>10.1%}'
              f'{r:>+18.3f}')

    print('\n  按凹陷深度分层的 |dphi| 中位（括号为事件数）：')
    print(f'  {"深度":>9}' + ''.join(f'{w:>18}' for w in ('real', 'geometric', 'am')))
    for lo, hi, lab in BANDS:
        cells = []
        for w in ('real', 'geometric', 'am'):
            g = df[(df.world == w) & (df.depth >= lo) & (df.depth < hi)]
            cells.append(f'{g.dphi.median():.2f} ({len(g)})' if len(g) else '--')
        print(f'  {lab:>9}' + ''.join(f'{c:>18}' for c in cells))

    real = df[df.world == 'real']
    print('\n  真实数据按带宽：')
    for bw in (3, 2.5, 2):
        g = real[real.bw_GHz == bw]
        if len(g):
            print(f'    {bw:>3.1f} GHz  n={len(g):>4}  |dphi| 中位 {g.dphi.median():.2f}  '
                  f'rho(深度,|dphi|) {spearmanr(g.depth, g.dphi).statistic:+.2f}')

    print('\n  单事件判别力：用 |dphi| 区分两个合成世界')
    g = df[df.world == 'geometric'].dphi.to_numpy()
    a = df[df.world == 'am'].dphi.to_numpy()
    x = np.r_[g, a]
    y = np.r_[np.ones(len(g), int), np.zeros(len(a), int)]
    order = np.argsort(x)
    ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
    auc = (ranks[y == 1].sum() - len(g) * (len(g) + 1) / 2) / (len(g) * len(a))
    print(f'    AUC = {auc:.2f}')
    print(f'    真实数据中 |dphi|>2.5 的占比 {np.mean(real.dphi > 2.5):.1%}，'
          f'介于几何世界 {np.mean(g > 2.5):.1%} 与 AM 世界 {np.mean(a > 2.5):.1%} 之间')
    return real


def make_figure(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 8.5,
                     'axes.linewidth': 0.7, 'axes.labelsize': 8.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
    colors = {'real': '#4C78A8', 'geometric': '#54A24B', 'am': '#E45756'}
    labels = {'real': 'measured (Lei)', 'geometric': 'two-phasor null',
              'am': 'amplitude modulation'}
    ax = axes[0]
    bins = np.linspace(0, np.pi, 26)
    for w in ('geometric', 'am', 'real'):
        g = df[df.world == w]
        if len(g):
            ax.hist(g.dphi, bins=bins, density=True, histtype='step', lw=1.2,
                    color=colors[w], label=f'{labels[w]} (n={len(g)})')
    ax.axvline(np.pi, color='k', ls=':', lw=0.8)
    ax.set_xlabel(r'phase step across the dip $|\Delta\varphi|$ (rad)')
    ax.set_ylabel('density')
    ax.set_title('(a) phase step at the dip', fontsize=9)
    ax.legend(fontsize=6.0, frameon=False, loc='center right',
              bbox_to_anchor=(1.0, 0.62))
    ax.grid(alpha=0.2)

    ax = axes[1]
    xs = np.arange(len(BANDS))
    for w, off, mk in (('real', -0.18, 'o'), ('geometric', 0.0, 's'),
                       ('am', 0.18, '^')):
        med, lo, hi, pos = [], [], [], []
        for i, (a, b, _) in enumerate(BANDS):
            g = df[(df.world == w) & (df.depth >= a) & (df.depth < b)]
            if len(g) < 20:
                continue
            med.append(g.dphi.median())
            lo.append(g.dphi.median() - g.dphi.quantile(.25))
            hi.append(g.dphi.quantile(.75) - g.dphi.median())
            pos.append(i + off)
        ax.errorbar(pos, med, yerr=[lo, hi], marker=mk, ms=3.5, lw=1.0,
                    capsize=2, color=colors[w], label=labels[w])
    ax.axhline(np.pi, color='k', ls=':', lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[2] for b in BANDS])
    ax.set_xlabel(r'dip depth $1-|z|/\mathrm{median}$')
    ax.set_ylabel(r'$|\Delta\varphi|$ (rad), median and IQR')
    ax.set_title('(b) depth dependence', fontsize=9)
    ax.legend(fontsize=6.0, frameon=False, loc='upper left')
    ax.grid(alpha=0.2)
    fig.tight_layout(w_pad=1.4)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / 'fig_c2_shape.png', dpi=300)
    fig.savefig(FIG / 'fig_c2_shape.pdf')
    plt.close(fig)
    print(f'\n[fig] {FIG / "fig_c2_shape.png"}')


def main() -> int:
    import sys
    OUT.mkdir(exist_ok=True)
    c_real, c_syn = OUT / 'C2_real.csv', OUT / 'C2_synth.csv'
    if c_real.exists():
        real = pd.read_csv(c_real)
    else:
        real = real_dips()
        real.to_csv(c_real, index=False, encoding='utf-8')
    if c_syn.exists() and '--synth' not in sys.argv:
        syn = pd.read_csv(c_syn)
    else:
        syn = pd.concat([synth_geometric(), synth_am()], ignore_index=True)
        syn.to_csv(c_syn, index=False, encoding='utf-8')
    df = pd.concat([real, syn], ignore_index=True)
    summarise(df)
    make_figure(df)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
