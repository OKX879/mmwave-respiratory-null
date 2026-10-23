#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""失效映射图 v6：摆动轴扩展到实测范围。
Failure map, v6: the swing axis extended to the measured range."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp_out"
FIG = HERE / "figs"

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
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
})

RED = "#E45756"


def r_of(rho, th):
    """半摆幅 th、以 0 为中心的弧段的归一化深度 min|z|/max|z|。
Normalized depth min|z|/max|z| for an arc of half-swing th centred at zero."""
    th_deep = np.where(th >= np.pi, np.pi, th)
    return np.sqrt(1 + rho ** 2 + 2 * rho * np.cos(th_deep)) / (1 + rho)


def panel_a(ax):
    """(rho, theta_max) 平面上的失效集 d<=0.15。
Failure set d <= 0.15 in the (rho, theta_max) plane."""
    rho = np.linspace(0.10, 2.50, 500)
    thm = np.linspace(0.0, 28.0, 500)
    R, T = np.meshgrid(rho, thm)
    Z = r_of(R, T)
    mask = (Z <= 0.15).astype(float)
    ax.contourf(R, T, mask, levels=[0.5, 1.5], colors=["#E45756"], alpha=0.92)
    ax.contour(R, T, Z, levels=[0.15], colors=["#7A1F1F"], linewidths=0.7)

    # grid-calibrated region (143-point noisy sweep, unchanged)
    ax.add_patch(plt.Rectangle((0.98, 2.83), 1.15 - 0.98, 3.20 - 2.83,
                               fill=False, ec="k", lw=0.9, ls=(0, (2, 1.4))))
    ax.annotate("noisy grid", xy=(1.06, 2.88), xytext=(1.55, 0.55),
                fontsize=6.2, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.55, color="k"))

    ax.axhline(np.pi, color="k", ls="--", lw=0.8)
    ax.text(0.13, np.pi + 0.6, r"$\theta_{\max}=\pi$", fontsize=6.4)
    ax.text(0.13, 1.35, r"$2.84$ rad", fontsize=6.2, color="#7A1F1F")

    # measured range of the peak swing
    ax.annotate("", xy=(2.40, 0.97), xytext=(2.40, 27.5),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="#2A6F97"))
    ax.text(2.30, 14.0, "measured records\n(P10--P90)", fontsize=6.0,
            color="#2A6F97", rotation=90, ha="right", va="center")

    ax.text(1.02, 20.0, "swing-\nindependent", fontsize=6.4, color="white",
            ha="center", va="center")
    # numeric bounds of the 15% band, so the figure and the text agree
    rho_lo, rho_hi = (1 - 0.15) / (1 + 0.15), (1 + 0.15) / (1 - 0.15)
    ax.annotate("", xy=(rho_lo, 24.6), xytext=(rho_hi, 24.6),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="white"))
    ax.annotate(r"$\rho\in[0.7391,$" + "\n" + r"$1.3529]$", xy=(rho_hi, 24.6),
                xytext=(1.50, 26.6), fontsize=6.0, color="k",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="k"))
    ax.text(0.16, 11.0, "no failure", fontsize=6.4, ha="left", va="center")
    ax.text(2.02, 20.0, "no failure", fontsize=6.4, ha="right", va="center")

    ax.set_xlim(0.10, 2.62)
    ax.set_ylim(0, 28)
    ax.set_xlabel(r"amplitude ratio $\rho=A_c/A_s$")
    ax.set_ylabel(r"peak swing $\theta_{\max}$ (rad)")
    ax.set_title(r"(a) failure set $r\leq0.15$", fontsize=9)


def panel_b(ax):
    df = pd.read_csv(OUT / "A_records.csv")
    th = df["th_max"].to_numpy()
    th = th[np.isfinite(th)]
    bins = np.arange(0, 28.5, 1.0)
    ax.hist(th, bins=bins, color="#4C78A8", alpha=0.85,
            edgecolor="white", linewidth=0.4)
    ax.axvspan(np.pi, 28, color=RED, alpha=0.07, lw=0)
    ax.axvline(2.84, color=RED, lw=0.9)
    ax.axvline(np.pi, color=RED, lw=0.9, ls="--")
    frac = float(np.mean(th >= np.pi))
    ax.annotate(f"{frac*100:.0f}% beyond $\\pi$\n(median {np.median(th):.1f} rad)",
                xy=(15.0, ax.get_ylim()[1] * 0.42), fontsize=6.4, color="#222222",
                ha="center")
    ax.set_xlim(0, 28)
    ax.set_xlabel(r"measured peak swing $\theta_{\max}$ (rad)")
    ax.set_ylabel("records")
    ax.set_title("(b) measured swing, 161 records", fontsize=9)
    ax.grid(alpha=0.2, axis="y")


def panel_c(ax):
    rr = np.linspace(0.01, 0.99, 400)
    for thm0, c in zip([2.85, 3.0, 3.2], ["#4C78A8", RED, "#54A24B"]):
        D = np.where(2 * np.arccos(rr) < thm0,
                     (2 / np.pi) * np.arccos(2 * np.arccos(rr) / thm0), np.nan)
        ax.plot(rr, D * 100, lw=1.0, color=c, label=rf"$\theta_{{\max}}={thm0}$")
    ax.axhspan(2.0, 6.4, color="#888888", alpha=0.2)
    ax.set_xlabel(r"depth $r$ (relative to maximum)")
    ax.set_ylabel(r"$D$ (%)")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 40)
    ax.set_title(r"(c) $r$–$D$ bridge", fontsize=9)
    ax.legend(fontsize=6.4, frameon=False)
    ax.grid(alpha=0.25)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.62))
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    fig.tight_layout(w_pad=1.2)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_failure_map.png", dpi=300)
    fig.savefig(FIG / "fig_failure_map.pdf")
    plt.close(fig)
    print("[fig]", FIG / "fig_failure_map.png")
