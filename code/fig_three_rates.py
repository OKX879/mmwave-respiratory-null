#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图 T8-C：三个采样率上的覆盖率 Q，与表 T8-B 的预测边界对照。

数据来源
    10 Hz  : parralejo_coverage.csv（B）+ parralejo_Q_A.csv（A），376 条
    20 Hz  : lei_coverage_20hz.csv，322 条
    100 Hz : capsule_Q.csv，18 例
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

OUT = 'figs/fig_t8_three_rates.png'


def load_all():
    q = {}
    # 10 Hz
    ov = pd.read_csv('parralejo_coverage.csv')
    ov = ov[(ov.rule == 'overlap') & (ov.wavelet == 'sym4')]
    b10 = ov.groupby('level').Q.median()
    a10 = pd.read_csv('parralejo_Q_A.csv')
    a10m = a10[[f'Q_A_L{L}' for L in range(1, 7)]].median()
    q[10] = {'A': [a10m[f'Q_A_L{L}'] for L in range(1, 7)],
             'B': [b10.get(L, np.nan) for L in range(1, 7)], 'n': 376}

    # 20 Hz
    d20 = pd.read_csv('lei_coverage_20hz.csv')
    d20 = d20[d20.status == 'ok']
    q[20] = {'A': [d20[f'Q_A_L{L}'].median() for L in range(1, 7)],
             'B': [d20[f'Q_ov_L{L}'].median() for L in range(1, 7)], 'n': len(d20)}

    # 100 Hz：capsule（18 例）+ emo（285 trial）合并
    d100 = pd.concat([pd.read_csv('capsule_Q.csv'),
                      pd.read_csv('emo_Q.csv')], ignore_index=True)
    q[100] = {'A': [d100[f'Q_A_L{L}'].median() for L in range(1, 7)],
              'B': [d100[f'Q_ov_L{L}'].median() for L in range(1, 7)], 'n': len(d100)}
    return q


def main():
    rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 8})
    q = load_all()
    colors = {10: '#6BAED6', 20: '#E8833A', 100: '#2E7D5B'}
    marks = {10: 'o', 20: 's', 100: '^'}
    Ls = np.arange(1, 7)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, key, pred, title in [
            (axes[0], 'A', {10: 1, 20: 1, 100: 4}, '(a) A: cardiac from $a_L$'),
            (axes[1], 'B', {10: 3, 20: 4, 100: 6}, '(b) B: cardiac from details')]:
        for fs in (10, 20, 100):
            ax.axvline(pred[fs], lw=0.8, ls=(0, (2, 2)), color=colors[fs], alpha=0.45)
        for fs in (10, 20, 100):
            ax.plot(Ls, q[fs][key], '-', marker=marks[fs], ms=3.8, lw=1.2,
                    color=colors[fs], label=f'{fs} Hz  (n={q[fs]["n"]})')
        ax.set_xlabel('decomposition level $L$')
        ax.set_xticks(Ls)
        ax.set_ylim(-0.03, 1.12)
        ax.set_xlim(0.7, 6.3)
        ax.axhline(0.95, ls=':', lw=0.7, color='#555555')
        ax.set_title(title, fontsize=8.5, pad=13)
        ax.grid(alpha=0.18)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('median cardiac-line gain $Q$')
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', ncol=3, fontsize=6.6, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.5, columnspacing=1.6)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.88))
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.replace('.png', '.pdf'))
    print('saved', OUT)
    for fs in (10, 20, 100):
        print(fs, 'A', ' '.join(f'{v:.2f}' for v in q[fs]['A']),
              '| B', ' '.join(f'{v:.2f}' for v in q[fs]['B']))


if __name__ == '__main__':
    raise SystemExit(main())
