#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图（§4.3 频带覆盖示意）：为什么 L ≤ 4 是安全边界。

100 Hz 采样（f_nyq = 50 Hz）。近似带上界 a_L = f_nyq / 2^L。
心搏展示带 0.7–3.0 Hz。当 L=4 时 a_4 = 3.125 Hz 刚好覆盖心搏带上界；
L=5 时 a_5 = 1.5625 Hz，心搏带 [1.5625, 3.0] 落进被阈值化的细节带 d_5。
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

OUT = 'figs/fig_band_coverage.png'

FNYQ = 50.0
LEVELS = [2, 3, 4, 5]
A_COLOR, A_EDGE = '#1D9E75', '#0F6E56'
D_COLOR, D_EDGE = '#D3D1C7', '#888780'
BAND_COLOR = '#E24B4A'

fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.set_xscale('log')
ax.set_xlim(0.08, 40)
ax.set_ylim(-0.6, len(LEVELS) - 0.4)
ax.set_yticks(range(len(LEVELS)))
ax.set_yticklabels([f'L = {L}' for L in LEVELS], fontsize=11)
ax.set_xlabel('频率 (Hz，对数轴)', fontsize=10)
ax.tick_params(axis='x', labelsize=9)

LO = 0.08  # 对数轴近似 DC 的起点
for row, L in enumerate(LEVELS):
    a_hi = FNYQ / 2 ** L
    y = len(LEVELS) - 1 - row  # L=2 画在最上
    # 近似带 a_L
    ax.barh(y, a_hi - LO, left=LO, height=0.52,
            color=A_COLOR, edgecolor=A_EDGE, lw=0.8, zorder=3)
    # 细节带 d_L（紧邻 a_L 右侧，到 2*a_hi）
    ax.barh(y, a_hi, left=a_hi, height=0.52,
            color=D_COLOR, edgecolor=D_EDGE, lw=0.8, zorder=3)
    # 更高频细节层（浅灰省略示意）
    ax.barh(y, 40 - 2 * a_hi, left=2 * a_hi, height=0.52,
            color='#EEECE2', edgecolor='none', zorder=2, alpha=0.6)
    # 上界标注
    ax.text(a_hi, y + 0.42, f'{a_hi:g} Hz', ha='center', va='bottom',
            fontsize=8.5, color='#085041', fontweight='bold')

# 心搏带 0.7–3.0 Hz
ax.axvspan(0.7, 3.0, color=BAND_COLOR, alpha=0.22, zorder=1)
ax.axvline(3.0, color=BAND_COLOR, ls='--', lw=1.3, zorder=4)
ax.text(3.0, len(LEVELS) - 0.18, '心搏带上界 3.0 Hz', ha='center', va='top',
        fontsize=9.5, color=BAND_COLOR, fontweight='bold')
ax.text(0.72, len(LEVELS) - 0.62, '心搏展示带\n0.7–3.0 Hz', ha='left', va='top',
        fontsize=8.5, color=BAND_COLOR)

# 图例（放在图下方外侧，用文字标注）
ax.text(0.10, -0.52, '■ 近似带 a_L（不被阈值化）', color=A_EDGE, fontsize=9,
        transform=ax.get_yaxis_transform(), ha='left', va='top')
ax.text(0.45, -0.52, '■ 细节带 d_L（被阈值化）', color='#5F5E5A', fontsize=9,
        transform=ax.get_yaxis_transform(), ha='left', va='top')
ax.text(0.78, -0.52, '红色 = 心搏带', color=BAND_COLOR, fontsize=9,
        transform=ax.get_yaxis_transform(), ha='left', va='top')

ax.set_title('L 的覆盖规则：近似带须覆盖心搏带 3.0 Hz（100 Hz 链，f_nyq = 50 Hz）',
             fontsize=11, pad=12)

fig.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print('saved', OUT)
