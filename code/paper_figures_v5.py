#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5 新图：分辨率效应 + 同位置三带宽轨迹（位置 2，第 2 次重复）。
New figures for v5: the resolution effect and three-bandwidth trajectories at one position, position 2 repeat 2."""
from __future__ import annotations

from paths import FIGS, OUT_ROOT

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from lei_pilot import load, ROOT, R_BIN, bp

OUT = FIGS
CSV = OUT_ROOT / "lei_person_bin.csv"
BWS = [3.0, 2.5, 2.0]
COL = {3.0: '#4C72B0', 2.5: '#DD8452', 2.0: '#55A868'}
EX_DIR = ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data' / 'position_ (2)'
EX_POS, EX_REP = 2, 2


def fig_resolution(df):
    from scipy.stats import spearmanr
    fig, ax = plt.subplots(1, 4, figsize=(7.16, 2.5))
    # (a) fraction with >=5 nulls
    fr = [(df[df.bandwidth_GHz == b]['n_deep'] >= 5).mean() * 100 for b in BWS]
    n = [int((df.bandwidth_GHz == b).sum()) for b in BWS]
    ax[0].bar([f'{b:g}' for b in BWS], fr, color=[COL[b] for b in BWS], width=0.6)
    for i, (f, nn) in enumerate(zip(fr, n)):
        ax[0].text(i, f + 2, f'{f:.1f}%\n({nn})', ha='center', va='bottom', fontsize=6)
    ax[0].set_ylim(0, 100)
    ax[0].set_ylabel('Records with $\\geq$5 nulls (%)', fontsize=6.5)
    ax[0].set_xlabel('Bandwidth (GHz)', fontsize=7)
    ax[0].set_title('(a)', fontsize=9)
    # (b) D selected vs other bins
    data, pos = [], []
    for i, b in enumerate(BWS):
        g = df[df.bandwidth_GHz == b]
        data += [g['D'].values, g['D_others_median'].values]
        pos += [i * 2.6 + 0.8, i * 2.6 + 1.8]
    bp = ax[1].boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False)
    for j, patch in enumerate(bp['boxes']):
        patch.set_facecolor(COL[BWS[j // 2]])
        patch.set_alpha(0.55)
        patch.set_edgecolor('k')
        patch.set_linewidth(0.5)
    for med in bp['medians']:
        med.set_color('k')
        med.set_linewidth(0.8)
    ax[1].set_xticks([i * 2.6 + 1.3 for i in range(3)])
    ax[1].set_xticklabels([f'{b:g}' for b in BWS])
    ax[1].set_xlabel('Bandwidth (GHz)', fontsize=7)
    ax[1].set_ylabel('$D$', fontsize=8)
    ax[1].legend([bp['boxes'][0], bp['boxes'][1]], ['selected', 'others'],
                 fontsize=5, frameon=False, loc='upper left')
    ax[1].set_title('(b)', fontsize=9)
    # (c) |As|/R vs null count
    for b in BWS:
        g = df[df.bandwidth_GHz == b]
        ax[2].scatter(g['As_over_R'], g['n_deep'], s=5, color=COL[b], alpha=0.6,
                      edgecolor='none', label=f'{b:g}')
        rho = spearmanr(np.abs(np.log(g['As_over_R'])), g['n_deep']).statistic
        ax[2].text(0.03, 0.92 - 0.09 * BWS.index(b),
                   f'{b:g} GHz: $\\rho_s$={rho:.2f}', transform=ax[2].transAxes,
                   fontsize=5.2, color=COL[b])
    ax[2].set_xscale('log')
    ax[2].set_xlabel('$|A_s|/R$', fontsize=7)
    ax[2].set_ylabel('nulls per record', fontsize=6.5)
    ax[2].set_title('(c)', fontsize=9)
    # (d) kappa and Rbar among null records
    k, r, x, cnt = [], [], [], []
    for i, b in enumerate(BWS):
        g = df[(df.bandwidth_GHz == b) & (df.n_deep >= 5)]
        cnt.append(len(g))
        k.append(np.nanmedian(g['kappa']) if len(g) else np.nan)
        r.append(np.nanmedian(g['Rbar_drift']) if len(g) else np.nan)
        x.append(i)
    ax[3].plot(x, k, 'o-', color='#C44E52', label='$\\kappa$', ms=3, lw=0.8)
    ax[3].plot(x, r, 's--', color='#8172B2', label='$\\bar R$ (drift)', ms=3, lw=0.8)
    ax[3].set_xticks(range(3))
    ax[3].set_xticklabels([f'{b:g}' for b in BWS])
    ax[3].set_ylim(0, 1.05)
    ax[3].set_xlabel('Bandwidth (GHz)', fontsize=7)
    ax[3].legend(fontsize=5, frameon=False, loc='lower right')
    ax[3].text(0.03, 0.06, f'$n$={cnt[0]}/{cnt[1]}/{cnt[2]}', transform=ax[3].transAxes,
               fontsize=5.2)
    ax[3].set_title('(d)', fontsize=9)
    for a in ax:
        a.tick_params(labelsize=6)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT / 'fig_v5_resolution.png', dpi=300)
    fig.savefig(OUT / 'fig_v5_resolution.pdf')
    plt.close(fig)


def example_series(bw):
    tag = {3.0: '3', 2.5: '2_5', 2.0: '2'}[bw]
    fp = EX_DIR / f'adc_{tag}GHZ_position{EX_POS}_ ({EX_REP}).bin'
    z = load(fp)
    Z = np.fft.fft(z, axis=3)
    zf = Z.mean(axis=(1, 2))
    rb = R_BIN[bw]
    # 候选窗与 Table II 一致：bin 索引 [floor((r-0.25)/dR), floor((r+0.05)/dR)]
    idx = []
    for r in (1.4, 1.4):
        idx += list(range(max(1, int((r - 0.25) / rb)), int((r + 0.05) / rb) + 1))
    idx = sorted(set(idx))
    e = np.std(zf[:, idx], axis=0)
    b = idx[int(np.argmax(e))]
    zc = zf[:, b]
    env = np.abs(zc) / np.median(np.abs(zc))
    return zc, env, b, rb


def fig_example():
    fig, ax = plt.subplots(2, 3, figsize=(7.16, 3.9))
    for j, bw in enumerate(BWS):
        zc, env, b, rb = example_series(bw)
        t = np.arange(len(zc)) / 20.0
        pk, _ = find_peaks(-env, distance=5, prominence=0.15 * np.std(env))
        pk = pk[env[pk] < 0.35]
        # 论文的 |A_s|/R 定义：|mean z| / mean|bandpass(z)|（不是 RMS）
        As = abs(complex(np.mean(zc)))
        R = float(np.mean(np.abs(bp(zc))))
        ax[0, j].plot(zc.real, zc.imag, lw=0.5, color=COL[bw], alpha=0.8)
        ax[0, j].scatter(zc.real[pk], zc.imag[pk], s=6, color='r', zorder=3,
                         edgecolor='none')
        ax[0, j].plot(0, 0, marker='+', color='k', ms=7, mew=1.0)
        ax[0, j].set_aspect('equal', adjustable='datalim')
        ax[0, j].text(0.03, 0.95, f'min $|z|$={env.min():.2f} median',
                      transform=ax[0, j].transAxes, fontsize=5.5, va='top')
        ax[0, j].set_title(
            f'{bw:g} GHz, {rb*100:g} cm/bin\n'
            f'bin {b} ({b*rb:.2f} m), $|A_s|/R$={As/R:.2f}, $n$={len(pk)}',
            fontsize=6.5)
        ax[0, j].tick_params(labelsize=5.5)
        ax[1, j].plot(t, env, lw=0.6, color=COL[bw])
        ax[1, j].scatter(t[pk], env[pk], s=6, color='r', zorder=3, edgecolor='none')
        ax[1, j].axhline(0.35, color='k', ls='--', lw=0.5)
        ax[1, j].set_ylim(0, 1.6)
        ax[1, j].set_xlabel('Time (s)', fontsize=6.5)
        if j == 0:
            ax[1, j].set_ylabel('$|z|$ / median', fontsize=7)
        ax[1, j].tick_params(labelsize=5.5)
        for a in (ax[0, j], ax[1, j]):
            for s in ('top', 'right'):
                a.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / 'fig_v5_example.png', dpi=300)
    fig.savefig(OUT / 'fig_v5_example.pdf')
    plt.close(fig)


if __name__ == '__main__':
    df = pd.read_csv(CSV)
    fig_resolution(df)
    fig_example()
    print('saved fig_v5_resolution and fig_v5_example')
