#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""屏气对照图（按被试）：110 名受试者，窗内 vs 窗外深凹陷率。"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "null_crossdataset"
FIG = HERE / "figs"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.linewidth": 0.7,
    })
    g = pd.read_csv(OUT / "bh_subject.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})

    ax = axes[0]
    data = [g["rate_in"].to_numpy(), g["rate_out"].to_numpy()]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#333333", lw=1.2))
    for patch, c in zip(bp["boxes"], ["#4C78A8", "#E45756"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for k, d in enumerate(data):
        ax.scatter(np.full(len(d), k + 1) + rng.normal(0, 0.06, len(d)),
                   d, s=5, color=["#4C78A8", "#E45756"][k], alpha=0.35, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["breath hold", "free breathing"])
    ax.set_ylabel("deep dips per minute")
    ax.set_title("(a) subject-level rates ($n=110$)", fontsize=9)
    ax.text(0.5, 0.95,
            "median 5.4 vs 18.8 /min\nPoisson rate ratio 0.44 (0.35–0.55)\n$p=2.1\\times10^{-12}$",
            transform=ax.transAxes, ha="center", va="top", fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    ratio = g["rate_in"] / g["rate_out"].replace(0, np.nan)
    ratio = ratio[np.isfinite(ratio)]
    ax.hist(ratio, bins=np.linspace(0, 1.5, 31), color="#54A24B",
            edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color="#888888", lw=0.8, ls="--")
    ax.axvline(float(np.median(ratio)), color="#E45756", lw=1.0,
               label=f"median subject ratio {np.median(ratio):.2f}")
    ax.set_xlabel("dip-rate ratio (inside / outside)")
    ax.set_ylabel("subjects")
    ax.set_title("(b) distribution of the ratio", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG / "fig_bh_control.png", dpi=300)
    fig.savefig(FIG / "fig_bh_control.pdf")
    print("figure ->", FIG / "fig_bh_control.png")


if __name__ == "__main__":
    raise SystemExit(main())
