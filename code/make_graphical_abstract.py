# -*- coding: utf-8 -*-
"""IEEE graphical abstract: workflow 660x295"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
})

fig = plt.figure(figsize=(6.6, 2.95), dpi=100, facecolor='white')

# ---------- P1: null condition ----------
ax = fig.add_axes([0.005, 0.20, 0.235, 0.58])
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-0.50, 2.12)
ax.set_ylim(-1.30, 1.30)

ax.add_patch(Circle((1, 0), 0.50, fill=False, lw=2.0, ec='#4A90D9', ls=(0, (3, 2))))
ax.add_patch(Circle((1, 0), 1.00, fill=False, lw=2.6, ec='#C0392B'))
ax.plot([0], [0], 'o', ms=6, color='#111111', zorder=5)
ax.annotate('', xy=(1, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', lw=1.3, color='#777777', shrinkA=3, shrinkB=3))
ax.text(0.33, 0.14, r'$A_s$', fontsize=12, color='#555555')
ax.text(0.12, -0.52, r'$|z|=0$', fontsize=12, color='#111111', fontweight='bold')
ax.text(1.00, 1.07, r'$\rho=1$', fontsize=13, color='#C0392B', fontweight='bold',
        ha='center', va='bottom')
ax.text(1.00, -1.14, r'$\rho=0.5$', fontsize=12, color='#4A90D9',
        ha='center', va='top')

# ---------- P2: depth bound ----------
ax2 = fig.add_axes([0.325, 0.26, 0.300, 0.50])
rho = np.linspace(0.30, 2.30, 400)
r = np.abs(rho - 1) / (1 + rho)
lo, hi = 0.7391, 1.3529
mask = (rho >= lo) & (rho <= hi)
ax2.fill_between(rho[mask], 0, 0.15, color='#F5C6C0', alpha=0.9, lw=0)
ax2.plot(rho, r, lw=2.6, color='#1F4E79')
ax2.axhline(0.15, lw=1.2, ls=(0, (4, 3)), color='#888888')

ax2.set_xlim(0.30, 2.30)
ax2.set_ylim(0, 0.72)
ax2.set_xticks([lo, 1.0, hi])
ax2.set_xticklabels(['0.74', '1', '1.35'], fontsize=12)
ax2.set_yticks([0, 0.15])
ax2.set_yticklabels(['0', '0.15'], fontsize=12)
ax2.tick_params(length=2, pad=2, colors='#555555')
for sp in ['top', 'right']:
    ax2.spines[sp].set_visible(False)
ax2.spines['left'].set_color('#999999')
ax2.spines['bottom'].set_color('#999999')
ax2.set_xlabel(r'$\rho$', fontsize=13, labelpad=-1, color='#222222')

ax2.annotate('failure band', xy=(1.02, 0.065), xytext=(1.05, 0.40),
             fontsize=12, color='#C0392B', fontweight='bold',
             ha='center', va='bottom',
             arrowprops=dict(arrowstyle='-|>', lw=1.3, color='#C0392B',
                             connectionstyle='arc3,rad=0'))
ax2.text(2.24, 0.185, '15%', fontsize=12, color='#666666', ha='right', va='bottom')

# ---------- P3: bandwidth control ----------
ax3 = fig.add_axes([0.700, 0.30, 0.275, 0.42])
bins = ['5.0', '6.0', '7.5']
vals = [13.0, 48.1, 62.3]
colors = ['#A8C8E8', '#4A90D9', '#1F4E79']
bars = ax3.bar(bins, vals, color=colors, width=0.60, edgecolor='none')
for b, v in zip(bars, vals):
    ax3.text(b.get_x() + b.get_width() / 2, v + 2.2, '%.1f%%' % v,
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='#111111')
ax3.set_ylim(0, 82)
ax3.set_yticks([])
ax3.set_xlabel('bin width (cm)', fontsize=12, color='#222222', labelpad=4)
ax3.tick_params(axis='x', labelsize=12, length=0, pad=2)
ax3.tick_params(axis='y', length=0)
for sp in ['top', 'right', 'left']:
    ax3.spines[sp].set_visible(False)
ax3.spines['bottom'].set_color('#999999')

# ---------- arrows ----------
for x0, x1 in [(0.248, 0.300), (0.633, 0.688)]:
    fig.patches.append(FancyArrowPatch((x0, 0.50), (x1, 0.50),
                                       transform=fig.transFigure, arrowstyle='-|>',
                                       mutation_scale=13, lw=1.6, color='#555555'))

# ---------- step labels ----------
fig.text(0.122, 0.055, 'null condition', fontsize=12, color='#333333', ha='center', va='bottom')
fig.text(0.475, 0.055, 'depth bound', fontsize=12, color='#333333', ha='center', va='bottom')
fig.text(0.838, 0.055, 'bandwidth control', fontsize=12, color='#333333', ha='center', va='bottom')

# ---------- headline ----------
fig.text(0.5, 0.965, 'Chirp bandwidth decides whether a bin can null',
         fontsize=14, fontweight='bold', ha='center', va='top', color='#111111')

fig.savefig('_ga_raw.png', dpi=100, facecolor='white')

from PIL import Image
import os
im = Image.open('_ga_raw.png')
bg = Image.new('RGB', im.size, (255, 255, 255))
bg.paste(im, mask=im.split()[3] if im.mode == 'RGBA' else None)
bg.save('gagraphic.png', dpi=(300, 300), optimize=True)
print('size:', bg.size, 'mode:', bg.mode, 'bytes:', os.path.getsize('gagraphic.png'))
