#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§4.13(e) 图 T8-L：允许的分解层数随采样率的变化。"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = 'figs/fig_t8_Lmap.png'
DUR, LF = 30.0, 8


def main():
    from matplotlib import rcParams
    rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 9,
                     'axes.linewidth': 0.7})

    fs = np.logspace(np.log2(8), np.log2(600), 3000, base=2)
    lmax = np.floor(np.log2(fs * DUR / (2 * (LF - 1) + 1)))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    panels = [('(a) display band 0.7–3.0 Hz', 0.7, 3.0, 12.0),
              ('(b) production band 1.0–2.5 Hz', 1.0, 2.5, 10.0)]

    for ax, (title, f_lo, f_hi, fcrit) in zip(axes, panels):
        la = np.floor(np.log2(fs / f_hi)) - 1
        lb = np.ceil(np.log2(fs / f_lo)) - 1
        a_ok = (la >= 1) & (la <= lmax)
        b_ok = (lb <= lmax)

        ax.fill_between(fs, 1, np.clip(la, 1, lmax), where=a_ok,
                        color='#F58518', alpha=0.18, lw=0, step='mid')
        ax.fill_between(fs, np.clip(lb, 0, lmax), lmax, where=b_ok,
                        color='#4C78A8', alpha=0.16, lw=0, step='mid')
        ax.step(fs, np.clip(la, 0, 10), where='mid', color='#F58518', lw=1.3,
                label='convention A (from $a_L$)')
        ax.step(fs, np.clip(lb, 0, 10), where='mid', color='#4C78A8', lw=1.3,
                label='convention B (from details)')
        ax.plot(fs, lmax, '--', color='#333333', lw=0.9,
                label='record-length limit')

        ax.axvline(fcrit, color='#555555', lw=0.8, ls=':')
        ax.text(fcrit * 0.96, 8.6, f'$4f_{{hi}}$={fcrit:g} Hz', rotation=90,
                fontsize=6.0, color='#555555', va='center', ha='right')

        ax.plot([10], [3], 'o', ms=5, mfc='white', mec='#4C78A8', mew=1.2)
        ax.annotate('10 Hz, B: $L$=3', (10, 3), textcoords='offset points',
                    xytext=(5, -11), fontsize=6.2, color='#2C5F82')
        ax.plot([100], [4], 'o', ms=5, mfc='white', mec='#F58518', mew=1.2)
        ax.annotate('100 Hz, A: $L$=4', (100, 4), textcoords='offset points',
                    xytext=(5, -11), fontsize=6.2, color='#A85A0A')

        ax.set_xscale('log', base=2)
        ax.set_xticks([10, 20, 50, 100, 200, 500])
        ax.set_xticklabels(['10', '20', '50', '100', '200', '500'])
        ax.minorticks_off()
        ax.set_xlim(8, 600)
        ax.set_ylim(0.4, 10.2)
        ax.set_xlabel('sampling rate $f_s$ (Hz)')
        ax.set_title(title, fontsize=8.5, pad=12)
        ax.grid(alpha=0.18)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel('decomposition level $L$')
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', ncol=3, fontsize=6.4,
               frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.5,
               columnspacing=1.4)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.90))
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.replace('.png', '.pdf'))
    print('saved', OUT)


if __name__ == '__main__':
    raise SystemExit(main())
