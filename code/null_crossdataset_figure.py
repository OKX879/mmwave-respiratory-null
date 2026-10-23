#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨数据集零点分析汇总图（三面板）。

(a) 屏气窗内 vs 窗外的深凹陷率（220 段配对）
(b) 合格记录的 Rbar 经验分布：固定相位 vs 漂移补偿（Parralejo 385 段；胶囊 9 段用竖线标出）
(c) 各条件下锁定比例（Rbar>0.4）：固定 vs 漂移补偿
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "null_crossdataset"
FIG = HERE / "figs"


def ecdf(x):
    x = np.sort(np.asarray(x, dtype=float))
    return x, np.arange(1, len(x) + 1) / len(x)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.linewidth": 0.7,
    })

    bh = pd.read_csv(OUT / "parralejo_summary.csv")
    bh = bh[bh["rate_in"].notna()]
    ext = pd.read_csv(OUT / "parralejo_ext.csv")
    cap = pd.read_csv(OUT / "capsule_ext.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75))

    # (a) 屏气窗内 vs 窗外
    ax = axes[0]
    data = [bh["rate_in"].to_numpy(), bh["rate_out"].to_numpy()]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#333333", lw=1.2))
    for patch, c in zip(bp["boxes"], ["#4C78A8", "#E45756"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for k, d in enumerate(data):
        ax.scatter(np.full(len(d), k + 1) + rng.normal(0, 0.06, len(d)),
                   d, s=4, color=["#4C78A8", "#E45756"][k], alpha=0.35, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["breath hold", "free breathing"])
    ax.set_ylabel("deep nulls per minute")
    ax.set_title("(a) breath-hold control ($n=220$)", fontsize=8.5)
    ax.text(0.5, 0.95,
            f"median {np.median(data[0]):.1f} vs {np.median(data[1]):.1f} /min\n"
            f"$p=1.8\\times10^{{-14}}$ (Wilcoxon)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    # (b) Rbar 分布
    ax = axes[1]
    e = ext[ext["n_deep"] >= 5]
    x, y = ecdf(e["Rbar_fixed"])
    ax.plot(x, y, color="#4C78A8", lw=1.3, label="fixed reference")
    x, y = ecdf(e["Rbar_drift"])
    ax.plot(x, y, color="#E45756", lw=1.3, label="drift-compensated")
    cap_e = cap[cap["n_deep"] >= 5]
    ax.axvline(cap_e["Rbar_fixed"].median(), color="#1F4E79", ls="--", lw=0.9,
               label="capsule, fixed")
    ax.axvline(cap_e["Rbar_drift"].median(), color="#B03A2E", ls=":", lw=0.9,
               label="capsule, drift")
    ax.axvline(0.4, color="#888888", lw=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"resultant length $\bar R$")
    ax.set_ylabel("ECDF")
    ax.set_title("(b) phase concentration", fontsize=8.5)
    ax.legend(fontsize=6.3, frameon=False, loc="lower right")
    ax.grid(alpha=0.25)

    # (c) 各条件锁定比例
    ax = axes[2]
    conds = ["Sitting/Rest", "Sitting/Post-exercise",
             "Lying/Rest", "Lying/Post-exercise"]
    labels = ["sit/rest", "sit/post", "lie/rest", "lie/post"]
    fx, dr = [], []
    for c in conds:
        g = ext[(ext["condition"] == c) & (ext["n_deep"] >= 5)]
        fx.append(float((g["Rbar_fixed"] > 0.4).mean()) * 100)
        dr.append(float((g["Rbar_drift"] > 0.4).mean()) * 100)
    x = np.arange(len(conds))
    ax.bar(x - 0.18, fx, width=0.34, color="#4C78A8", label="fixed")
    ax.bar(x + 0.18, dr, width=0.34, color="#E45756", label="drift-comp.")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("locked records (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(c) locked fraction ($\\bar R>0.4$)", fontsize=8.5)
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    png = FIG / "figS2_crossdataset_null.png"
    fig.savefig(png, dpi=300)
    fig.savefig(FIG / "figS2_crossdataset_null.pdf")
    print("figure ->", png)


if __name__ == "__main__":
    raise SystemExit(main())
