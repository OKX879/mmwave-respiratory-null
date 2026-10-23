#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图（方法总览）：小波分解参数选择 pipeline（§3 方法部分新增总览图）。
Figure: overview of the wavelet parameter-selection pipeline, new in section 3."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

OUT = 'figs/fig_method_overview.png'

BLUE_F, BLUE_E, BLUE_T = '#E6F1FB', '#185FA5', '#0C447C'
TEAL_F, TEAL_E, TEAL_T = '#E1F5EE', '#0F6E56', '#085041'
GRAY_F, GRAY_E, GRAY_T = '#F1EFE8', '#888780', '#444441'
AMBER_F, AMBER_E, AMBER_T = '#FAEEDA', '#BA7517', '#633806'


def box(ax, x, y, w, h, title, sub, fc, ec, tc, sc):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.12',
                       fc=fc, ec=ec, lw=1.3)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h * 0.64, title, ha='center', va='center',
            fontsize=11, color=tc, fontweight='bold')
    ax.text(x + w / 2, y + h * 0.30, sub, ha='center', va='center',
            fontsize=8.5, color=sc)


def arrow(ax, p1, p2, color='#5F5E5A'):
    a = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=15,
                        lw=1.5, color=color)
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(11.5, 6.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6.4)
ax.axis('off')

# 行 1
box(ax, 0.5, 4.55, 2.3, 1.25, '雷达位移信号', '呼吸 + 心搏 + 体动 + 噪声',
    GRAY_F, GRAY_E, GRAY_T, '#5F5E5A')
box(ax, 3.7, 4.55, 2.7, 1.25, '① 候选矩阵', '15×9×5×3 = 2025 个',
    BLUE_F, BLUE_E, BLUE_T, BLUE_E)
box(ax, 7.3, 4.55, 2.7, 1.25, '② 无参考准则评分', 'PB / RC / SM 共 14 个',
    BLUE_F, BLUE_E, BLUE_T, BLUE_E)

# 行 2（蛇形：从右到左）
box(ax, 7.3, 1.55, 2.7, 1.25, '③ 准则校准（C3）', '合成金标 → 留出闭环',
    TEAL_F, TEAL_E, TEAL_T, TEAL_E)
box(ax, 3.7, 1.55, 2.7, 1.25, '④ 三段式决策', '硬门槛 → Borda → Pareto',
    BLUE_F, BLUE_E, BLUE_T, BLUE_E)
box(ax, 0.5, 1.55, 2.3, 1.25, '输出', '推荐配置 + 停止判据',
    AMBER_F, AMBER_E, AMBER_T, '#854F0B')

# 箭头
arrow(ax, (2.8, 5.175), (3.7, 5.175))
arrow(ax, (6.4, 5.175), (7.3, 5.175))
arrow(ax, (8.65, 4.55), (8.65, 2.8))
arrow(ax, (7.3, 2.175), (6.4, 2.175))
arrow(ax, (3.7, 2.175), (2.8, 2.175))

# 旁注（输入 / 应用场景）
ax.text(10.4, 5.175, '候选空间', ha='center', va='center',
        fontsize=8.5, color='#5F5E5A')
ax.text(8.65, 0.95, '输入：合成数据（有金标）', ha='center', va='center',
        fontsize=8.5, color=TEAL_E)
ax.text(3.7, 0.95, '应用：真实数据（无金标）', ha='center', va='center',
        fontsize=8.5, color='#5F5E5A')

ax.text(6.0, 0.30, '核心逻辑：先在"知道答案"处校准，再到"不知道答案"处使用 —— 规避自证陷阱',
        ha='center', va='center', fontsize=9.5, color=TEAL_E)

fig.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print('saved', OUT)
