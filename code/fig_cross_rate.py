#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨采样率复核图：Q(L) 与"固定 0.7 Hz 归属"在低心率下的失效。
Cross-rate recheck: Q(L) and the failure of a fixed 0.7 Hz assignment at low heart rates."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DF = 'parralejo_coverage.csv'
OUT = 'figs/fig_t8_cross_rate.png'


def main():
    from matplotlib import rcParams
    rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 9,
                     'axes.linewidth': 0.7})
    df = pd.read_csv(DF)
    df = df[df.wavelet != 'haar']          # haar 的带外泄放使 Q 虚高（1.50），单独说明
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax = axes[0]
    for rule, c, lab in [('fixed07', '#E45756', 'fixed 0.7 Hz switch'),
                         ('overlap', '#4C78A8', 'cardiac-band overlap')]:
        g = df[df.rule == rule].groupby('level')['Q']
        med = g.median(); q1 = g.quantile(0.25); q3 = g.quantile(0.75)
        ax.plot(med.index, med.values, 'o-', ms=3.5, lw=1.0, color=c, label=lab)
        ax.fill_between(med.index, q1.values, q3.values, color=c, alpha=0.15, lw=0)
    ax.axvline(3, ls=':', lw=0.8, color='#555555')
    ax.text(3.08, 0.08, 'L=3\n(predicted)', fontsize=6.5, color='#555555')
    ax.set_xlabel('decomposition level $L$')
    ax.set_ylabel('cardiac-line gain $Q$')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(1, 7))
    ax.legend(fontsize=6.5, frameon=False, loc='lower right')
    ax.set_title('(a) 10 Hz platform, ECG-defined line', fontsize=8.5)
    ax.grid(alpha=0.2)

    ax = axes[1]
    sub = df[(df.level == 3)]
    sub = sub.assign(hr=sub.f_hr * 60)
    bins = [(0, 75, '<75'), (75, 90, '75–90'), (90, 200, '>90')]
    w = 0.36
    for k, (rule, c, lab) in enumerate([('fixed07', '#E45756', 'fixed 0.7 Hz'),
                                        ('overlap', '#4C78A8', 'band overlap')]):
        xs, ys = [], []
        for i, (lo, hi, _) in enumerate(bins):
            s = sub[(sub.rule == rule) & (sub.hr >= lo) & (sub.hr < hi)]
            xs.append(i + (k - 0.5) * w); ys.append(s.Q.median())
        ax.bar(xs, ys, width=w, color=c, alpha=0.75)
        ax.text(xs[0], ys[0] - 0.03, lab, ha='center', va='top', fontsize=5.5,
                rotation=90, color='white')
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([b[2] for b in bins])
    ax.set_xlabel('ECG heart rate (bpm)')
    ax.set_ylabel('median $Q$ at $L=3$')
    ax.set_ylim(0, 1.1)
    ax.set_title('(b) the fixed switch costs low-rate subjects', fontsize=8.5)
    ax.grid(axis='y', alpha=0.2)

    for a in axes:
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.replace('.png', '.pdf'))
    print('saved', OUT)


if __name__ == '__main__':
    raise SystemExit(main())
