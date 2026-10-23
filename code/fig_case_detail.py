#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充材料 S2 图：capsule 逐例明细（为什么必须报分布而不是中位数）。
Supplementary figure S2: capsule case detail, showing why a distribution rather than a median is reported."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

HERE = Path(__file__).resolve().parent
RAW = HERE / 'real_matrix' / 'real_matrix_raw.csv'
OUT_PNG = HERE / 'figs' / 'figS2_case_detail.png'
OUT_PDF = HERE / 'figs' / 'figS2_case_detail.pdf'
OUT_CSV = HERE / 'case_detail.csv'
REC = 'dwt|sym4|L4|universal-soft|symmetric|bandpass'
BASE = 'dwt|haar|L2|none|symmetric|bandpass'   # none ⇒ 恒等变换 ⇒ 无小波基线（§4.13）

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def main() -> int:
    d = pd.read_csv(RAW)
    l4 = d[d.level <= 4]
    pool = l4.groupby('sample_id').hr_mae_bpm.agg(['min', 'median', 'max'])
    rec = d[d.candidate_id == REC].set_index('sample_id')
    base = d[d.candidate_id == BASE].set_index('sample_id')['hr_mae_bpm'].rename('no_wavelet')
    case = pool.join(rec[['hr_mae_bpm', 'hr_est', 'gold_hr',
                          'beat_f1', 'beat_prec', 'beat_rec', 'beat_lag_ms']])
    case = case.join(base)
    case['gain_vs_no_wavelet'] = case['no_wavelet'] - case['hr_mae_bpm']
    case = case.sort_values('median')
    case.index.name = 'patient'
    case.to_csv(OUT_CSV, encoding='utf-8')

    spread = (case['max'] - case['min'])
    print(f'例内候选跨度：中位 {spread.median():.3f} BPM，最大 {spread.max():.3f}')
    print(f'病例间跨度：{case["median"].max() - case["median"].min():.2f} BPM')
    print(f'推荐配置误差：中位 {case.hr_mae_bpm.median():.2f}，'
          f'均值 {case.hr_mae_bpm.mean():.2f}')
    print(f'逐拍 F1：中位 {case.beat_f1.median():.3f}，'
          f'可用 {int(case.beat_f1.notna().sum())}/{len(case)} 例')
    g = case.gain_vs_no_wavelet
    print(f'相对无小波基线：中位 {g.median():+.3f} BPM；改善 {int((g > 0).sum())} 例，'
          f'退化 {int((g < 0).sum())} 例，持平 {int((g.abs() < 1e-9).sum())} 例')

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.6, 8.8),
        gridspec_kw={'height_ratios': [3, 2], 'hspace': 0.42})

    y = np.arange(len(case))
    for i, (_, r) in enumerate(case.iterrows()):
        ax1.plot([max(r['min'], 0.05), max(r['max'], 0.05)], [i, i],
                 color='#B4B2A9', lw=6, solid_capstyle='round', alpha=0.85, zorder=1)
    ax1.scatter(case.hr_mae_bpm.clip(lower=0.05), y, s=34, color='#C0392B',
                zorder=3, label='推荐配置 sym4 L4 u-soft symmetric')
    ax1.set_xscale('log')
    ax1.set_yticks(y)
    ax1.set_yticklabels([f'p{int(p):02d}' for p in case.index], fontsize=9)
    ax1.set_ylabel('病例', fontsize=10)
    ax1.set_xlabel('HR MAE（BPM，对数轴）', fontsize=10)
    ax1.set_title('(a) 逐例误差：灰带 = L ≤ 4 候选池范围（多数窄到看不见），圆点 = 推荐配置',
                  fontsize=10.5, pad=8, loc='left')
    ax1.grid(axis='x', ls=':', color='#D0CEC7', lw=0.7)
    ax1.set_xlim(0.03, 300)
    ax1.set_ylim(-0.8, len(case) - 0.2)
    ax1.legend(loc='upper left', fontsize=8.5, frameon=False)
    ax1.text(0.99, 0.10, f'例内跨度中位 {spread.median():.3f} BPM ／ '
                         f'病例间跨度 {case["median"].max() - case["median"].min():.1f} BPM',
             transform=ax1.transAxes, ha='right', va='bottom',
             fontsize=9, color='#55534E')
    ax1.text(0.99, 0.02, f'相对无小波基线：|ΔHR| ≤ {g.abs().max():.3f} BPM，'
                         f'改善 {int((g > 0).sum())} / 退化 {int((g < 0).sum())} 例',
             transform=ax1.transAxes, ha='right', va='bottom',
             fontsize=9, color='#55534E')

    ok = case.beat_f1.notna().to_numpy()
    ax2.scatter(case.loc[ok, 'beat_f1'], y[ok], s=30, color='#1F6FB2',
                label='逐拍 F1（简化检测器）')
    if (~ok).any():
        ax2.scatter(np.full((~ok).sum(), 0.004), y[~ok], s=30, facecolors='none',
                    edgecolors='#8C8A83', label='该例无可用拍序列（画在 0 处）')
    ax2.set_xlim(-0.02, 0.45)
    ax2.set_ylim(-0.8, len(case) - 0.2)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f'p{int(p):02d}' for p in case.index], fontsize=9)
    ax2.set_ylabel('病例', fontsize=10)
    ax2.set_xlabel('逐拍 F1', fontsize=10)
    ax2.set_title('(b) 同一配置的逐拍 F1：15/20 例可用，其余无可用拍序列，'
                  f'中位 {case.beat_f1.median():.2f}', fontsize=10.5, pad=8, loc='left')
    ax2.legend(loc='lower right', fontsize=8.5, frameon=False)
    ax2.grid(axis='x', ls=':', color='#D0CEC7', lw=0.7)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    print(f'[saved] {OUT_PNG.name} / {OUT_PDF.name} / {OUT_CSV.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
