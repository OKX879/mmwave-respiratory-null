#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§4.13 补充材料图：逐受试者 Q 分布，以及它与心率的依赖。

每个点是一个人（4 条记录取中位），不是一条记录。
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = 'parralejo_subject_Q.csv'
OUT = 'figs/fig_supp_subject_Q.png'


def main():
    from matplotlib import rcParams
    rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 9,
                     'axes.linewidth': 0.7})
    d = pd.read_csv(SRC, index_col=0)
    d = d.dropna(subset=['overlap', 'fixed07', 'hr'])
    o = d.sort_values('overlap').reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax = axes[0]
    x = np.arange(len(o))
    ax.plot(x, o.overlap, 'o', ms=2.6, color='#4C78A8', label='band overlap')
    ax.plot(x, o['fixed07'], 'o', ms=2.6, color='#E45756', label='fixed 0.7 Hz')
    ax.axhline(0.9, ls='--', lw=0.7, color='#555555')
    ax.text(2, 0.905, '0.9', fontsize=6, color='#555555', va='bottom')
    ax.set_xlabel('subject rank (sorted by overlap $Q$)')
    ax.set_ylabel('median $Q$ per subject ($L=3$)')
    ax.set_ylim(0, 1.6)
    ax.legend(fontsize=6.5, frameon=False, loc='upper left', ncol=2)
    ax.set_title(f'(a) {len(o)} subjects, 4 records each', fontsize=8.5)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(d.hr, d['fixed07'], 'o', ms=3.0, color='#E45756', alpha=0.75,
            label='fixed 0.7 Hz  ($\\rho=0.83$)')
    ax.plot(d.hr, d.overlap, 'o', ms=3.0, color='#4C78A8', alpha=0.55,
            label='band overlap  ($\\rho=-0.32$)')
    ax.axhline(0.9, ls='--', lw=0.7, color='#555555')
    ax.set_xlabel('median heart rate of the subject (bpm)')
    ax.set_ylabel('median $Q$ per subject')
    ax.set_ylim(0, 1.6)
    ax.legend(fontsize=6.0, frameon=False, loc='lower right')
    ax.set_title('(b) the fixed rule fails at low rate', fontsize=8.5)
    ax.grid(alpha=0.2)

    for a in axes:
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.replace('.png', '.pdf'))
    print('saved', OUT)


if __name__ == '__main__':
    raise SystemExit(main())
