#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图 T8-L3D v2：两片"覆盖率地形"用网格线绘制，避免实体曲面互相遮挡。
Figure T8-L3D v2: two coverage surfaces drawn as wireframe so they do not occlude each other."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

OUT = 'figs/fig_t8_Lmap_3d.png'
F_LO, F_HI = 0.7, 3.0
BAND = F_HI - F_LO


def cov_A(fs, L):
    up = fs / 2.0 ** (L + 1.0)
    return float(np.clip((min(up, F_HI) - F_LO) / BAND, 0, 1))


def cov_B(fs, L):
    lo = fs / 2.0 ** (L + 1.0)
    return float(np.clip((min(fs / 2.0, F_HI) - max(lo, F_LO)) / BAND, 0, 1))


def main():
    rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 7.5})
    fs_list = [10, 20, 50, 100, 200, 500]
    Ls = np.arange(0, 10.01, 0.25)

    fig = plt.figure(figsize=(4.0, 3.3))
    ax = fig.add_subplot(111, projection='3d')

    for fs in fs_list:
        y = np.full_like(Ls, np.log2(fs), dtype=float)
        za = [cov_A(fs, l) for l in Ls]
        zb = [cov_B(fs, l) for l in Ls]
        ax.plot(Ls, y, za, color='#E8833A', lw=1.0, alpha=0.65)
        ax.plot(Ls, y, zb, color='#3F7EA6', lw=1.0, alpha=0.55, ls=(0, (3.2, 1.6)))

    # 实测：10 Hz 平台 overlap 规则各层中位 Q
    try:
        df = pd.read_csv('parralejo_coverage.csv')
        g = df[(df.rule == 'overlap') & (df.wavelet == 'sym4')]
        med = g.groupby('level').Q.median()
        ax.scatter(med.index, np.full(len(med), np.log2(10)), med.values,
                   s=9, color='#123A52', zorder=12, depthshade=False)
        ax.scatter([3], [np.log2(10)], [med.get(3, 1.0)], s=20,
                   facecolor='white', edgecolor='#123A52', lw=1.1, zorder=13,
                   depthshade=False)
        ax.text(5.0, np.log2(10) - 0.1, 0.05, 'measured $Q$ (10 Hz)',
                fontsize=5.8, color='#123A52', va='bottom')
    except Exception as e:
        print('skip measured:', e)

    # 只用 4 个刻度，避免 3D 轴的标签挤在一起
    ticks = [10, 50, 200, 500]
    ax.set_yticks(np.log2(ticks))
    ax.set_yticklabels([str(v) for v in ticks], fontsize=6.4)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(axis='x', labelsize=6.6, pad=-1)
    ax.tick_params(axis='y', labelsize=6.4, pad=-3)
    ax.set_xlabel('level $L$', labelpad=0, fontsize=7)
    ax.set_ylabel('$f_s$ (Hz)', labelpad=-6, fontsize=7)
    ax.set_zlabel('coverage', labelpad=1, fontsize=6.8)
    ax.set_zlim(0, 1.05)
    ax.set_zticks([0, 0.5, 1.0])
    ax.tick_params(axis='z', labelsize=6.3, pad=0)
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((1.55, 1.2, 0.98))

    from matplotlib.lines import Line2D
    h = [Line2D([], [], color='#E8833A', lw=1.2, label='A: from $a_L$'),
         Line2D([], [], color='#3F7EA6', lw=1.2, ls=(0, (3.2, 1.6)),
                label='B: from details')]
    ax.legend(handles=h, fontsize=5.8, frameon=False, loc='upper left',
              bbox_to_anchor=(-0.06, 0.98), handlelength=1.6,
              labelspacing=0.25, borderpad=0.1)

    fig.savefig(OUT, dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.02)
    print('saved', OUT)


if __name__ == '__main__':
    raise SystemExit(main())
