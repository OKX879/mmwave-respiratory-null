#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文 v3 的补充图：理论几何、失效域、r-D 桥、分条件屏气、对齐度、P02 反例、窗敏感性、载频。
Supplementary figures for paper v3: geometry, failure region, r-D bridge, conditional breath-hold, alignment, the P02 counter-example, window sensitivity and carrier frequency."""
from __future__ import annotations

from paths import EMO_MULTI

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, find_peaks, sosfiltfilt

HERE = Path(__file__).resolve().parent
OUT = HERE / "null_crossdataset"
FIG = HERE / "figs"
EMO = EMO_MULTI

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


def r_of(rho, th):
    return np.sqrt(1 + rho ** 2 + 2 * rho * np.cos(th)) / (1 + rho)


def fig_model_depth():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    ax = axes[0]
    As, Ac = 1.0, 1.0
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(As + Ac * np.cos(th), Ac * np.sin(th), lw=1.0, color="#4C78A8")
    ax.annotate("", xy=(As + Ac, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=1.1, color="#333333"))
    ax.text(0.45, -0.16, "$A_s$", fontsize=8)
    ang = 2.3
    ax.annotate("", xy=(As + Ac * np.cos(ang), Ac * np.sin(ang)), xytext=(As, 0),
                arrowprops=dict(arrowstyle="->", lw=1.1, color="#E45756"))
    ax.text(As + 0.2, 0.85, "$A_c$", color="#E45756", fontsize=8)
    ax.scatter([0], [0], marker="+", s=90, color="k", zorder=5)
    ax.text(0.05, 0.1, "$z=0$", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 2.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("(a) two-phasor geometry", fontsize=9)
    ax.grid(alpha=0.2)

    ax = axes[1]
    thm = np.linspace(0.05, np.pi, 400)
    for rho, c in zip([0.5, 1.0, 1.5], ["#4C78A8", "#E45756", "#54A24B"]):
        ax.plot(thm, r_of(rho, thm), lw=1.2, color=c, label=f"$\\rho={rho}$")
    ax.axhline(0.15, color="#888888", lw=0.8, ls="--")
    ax.axvline(2.83, color="#888888", lw=0.8, ls=":")
    ax.set_xlabel(r"peak swing $\theta_{\max}$ (rad)")
    ax.set_ylabel(r"normalized depth $r$")
    ax.set_title("(b) depth vs swing", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig_model_depth.png", dpi=300)
    fig.savefig(FIG / "fig_model_depth.pdf")
    plt.close(fig)


def fig_failure_map():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    ax = axes[0]
    rho = np.linspace(0.1, 2.5, 240)
    thm = np.linspace(0.02, np.pi, 240)
    R, T = np.meshgrid(rho, thm)
    Z = r_of(R, T)
    im = ax.pcolormesh(R, T, Z, shading="auto", cmap="viridis_r", vmin=0, vmax=1)
    cs = ax.contour(R, T, Z, levels=[0.15], colors="white", linewidths=1.0)
    ax.add_patch(plt.Rectangle((0.98, 2.83), 1.15 - 0.98, np.pi - 2.83,
                               fill=False, ec="#E45756", lw=1.2))
    ax.annotate("grid-calibrated\n(noisy)", xy=(1.06, 2.95), xytext=(1.40, 2.30),
                color="#E45756", fontsize=6.5,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#E45756"))
    ax.set_xlabel(r"amplitude ratio $\rho=A_c/A_s$")
    ax.set_ylabel(r"peak swing $\theta_{\max}$ (rad)")
    ax.set_title("(a) analytic depth map", fontsize=9)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$r$", fontsize=7.5)

    ax = axes[1]
    rr = np.linspace(0.01, 0.99, 400)
    for thm0, c in zip([2.85, 3.0, 3.2], ["#4C78A8", "#E45756", "#54A24B"]):
        D = np.where(2 * np.arccos(rr) < thm0,
                     (2 / np.pi) * np.arccos(2 * np.arccos(rr) / thm0), np.nan)
        ax.plot(rr, D * 100, lw=1.2, color=c, label=rf"$\theta_{{\max}}={thm0}$")
    ax.axhspan(2.0, 6.4, color="#888888", alpha=0.2)
    ax.set_xlabel(r"depth $r$ (relative to maximum)")
    ax.set_ylabel(r"$D$ (%)")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 40)
    ax.set_title("(b) $r$–$D$ bridge", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig_failure_map.png", dpi=300)
    fig.savefig(FIG / "fig_failure_map.pdf")
    plt.close(fig)


def fig_bh_conditions():
    df = pd.read_csv(OUT / "parralejo_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharex=True, sharey=True)
    for ax, cond, title in zip(axes, ["Sitting/Rest", "Lying/Rest"],
                               ["(a) sitting, rest", "(b) lying, rest"]):
        g = df[(df["condition"] == cond) & df["bh_dur"].notna()]
        ax.scatter(g["rate_out"], g["rate_in"], s=10, alpha=0.55,
                   color="#4C78A8", edgecolor="none")
        lim = max(g["rate_out"].max(), g["rate_in"].max()) * 1.05
        ax.plot([0, lim], [0, lim], color="#888888", lw=0.8, ls="--")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("free breathing (dips/min)")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)
        ax.text(0.92, 0.94,
                f"median {g['rate_in'].median():.1f} vs {g['rate_out'].median():.1f}\n"
                f"{100*(g['rate_in'] < g['rate_out']).mean():.0f}% lower",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2))
    axes[0].set_ylabel("breath hold (dips/min)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_bh_conditions.png", dpi=300)
    fig.savefig(FIG / "fig_bh_conditions.pdf")
    plt.close(fig)


def fig_alignment_phase():
    ag = pd.read_csv(OUT / "alignment_gate.csv")
    emo = pd.read_csv(OUT / "emo_p02_dissect.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    ax = axes[0]
    for plat, c, lab in [("capsule", "#1F4E79", "capsule (n=9)"),
                         ("emo", "#E45756", "emo (n=37)")]:
        x = np.sort(ag[(ag["platform"] == plat) & (ag["n_deep"] >= 5)]["align"].dropna())
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", lw=1.2, color=c, label=lab)
    ax.axvline(0.6, color="#888888", lw=0.8, ls="--")
    ax.set_xlabel(r"median alignment $\kappa$")
    ax.set_ylabel("ECDF")
    ax.set_xlim(-0.2, 1.0)
    ax.set_title("(a) alignment distribution", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    subj = ["P01", "P02", "P03", "P04", "P05"]
    fixed, drift = [], []
    for s in subj:
        g = emo[(emo["subject"] == s) & (emo["n_deep"] >= 5)]
        fixed.append(100 * (g["Rbar_fixed"] > 0.4).mean())
        drift.append(100 * (g["Rbar_drift"] > 0.4).mean())
    x = np.arange(len(subj))
    ax.bar(x - 0.18, fixed, width=0.34, color="#4C78A8", label="fixed frame")
    ax.bar(x + 0.18, drift, width=0.34, color="#E45756", label="relative")
    ax.text(1, 1.5, "0", ha="center", va="bottom", fontsize=6.5, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(subj)
    ax.set_ylabel("locked trials (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(b) emo per subject", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig_alignment_phase.png", dpi=300)
    fig.savefig(FIG / "fig_alignment_phase.pdf")
    plt.close(fig)


def load_emo_z(fp):
    adc = loadmat(fp)["adcData"]
    n_ch, n_samp = adc.shape
    x = adc.reshape(n_ch, n_samp // 256, 256)[:, 1000:, :]
    for j in (1, 5, 9):
        x[j] *= -1
        x[j + 1] *= -1
    z = np.fft.fft(x, axis=2).mean(axis=0).T
    return z


def fig_p02_p01():
    cases = [("P01_10", "P01, trial 10"), ("P02_00", "P02, trial 00")]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    for i, (stem, title) in enumerate(cases):
        z = load_emo_z(EMO / f"{stem}.mat")
        e = np.std(z[9:28, :], axis=1)
        b = 9 + int(np.argmax(e))
        zc = z[b]
        env = np.abs(zc)
        pk, _ = find_peaks(-env, distance=25, prominence=0.15 * np.std(env))
        pk = pk[env[pk] < 0.35 * np.median(env)]
        ax = axes[i][0]
        ax.plot(zc.real, zc.imag, lw=0.4, color="#4C78A8", alpha=0.5)
        th = np.linspace(0, 2 * np.pi, 300)
        ax.plot(0.35 * np.median(env) * np.cos(th),
                0.35 * np.median(env) * np.sin(th), ls="--", lw=0.6, color="#888888")
        ax.scatter(zc.real[pk], zc.imag[pk], s=8, color="#E45756",
                   edgecolor="k", linewidth=0.2, zorder=4)
        ax.scatter([0], [0], marker="+", s=50, color="k", zorder=5)
        R98 = np.percentile(np.abs(zc), 98)
        ax.set_xlim(-1.05 * R98, 1.05 * R98)
        ax.set_ylim(-1.05 * R98, 1.05 * R98)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}: trajectory (bin {b})", fontsize=8)
        ax.grid(alpha=0.15)

        ax = axes[i][1]
        t = np.arange(len(env)) / 100
        ax.plot(t, env / np.median(env), lw=0.6, color="#4C78A8")
        ax.scatter(pk / 100, env[pk] / np.median(env), s=8, color="#E45756", zorder=4)
        ax.axhline(0.35, ls="--", lw=0.7, color="#888888")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(r"$|z|/$median")
        ax.set_title(f"{title}: envelope, n={len(pk)}", fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_case_p01_p02.png", dpi=300)
    fig.savefig(FIG / "fig_case_p01_p02.pdf")
    plt.close(fig)


def fig_carrier():
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    lam = np.array([5.0, 37.5])
    lo = np.array([5.5, 41.5])
    hi = np.array([7.5, 56.4])
    ax.errorbar(lam, (lo + hi) / 2, yerr=[(lo + hi) / 2 - lo, hi - (lo + hi) / 2],
                fmt="o", ms=6, color="#E45756", capsize=4)
    ax.axhspan(1, 12, color="#54A24B", alpha=0.2)
    ax.text(6, 11, "physiological band", fontsize=7, color="#2E6B2E")
    ax.set_xscale("log")
    ax.set_xticks([5, 37.5])
    ax.set_xticklabels(["5", "37.5"])
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(3.5, 55)
    ax.set_xlabel(r"assumed wavelength $\lambda$ (mm)")
    ax.set_ylabel(r"required $a_{pp}$ (mm)")
    ax.set_title("carrier adjudication", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig_carrier.png", dpi=300)
    fig.savefig(FIG / "fig_carrier.pdf")
    plt.close(fig)


def fig_window():
    df = pd.read_csv(OUT / "window_sensitivity.csv")
    wins = [5, 10, 15, 20]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for plat, c in [("capsule", "#1F4E79"), ("parralejo", "#4C78A8"), ("emo_p01", "#E45756")]:
        g = df[(df["platform"] == plat) & (df["n_deep"] >= 5)]
        frac = [100 * (g[f"Rbar_drift_{w}"] > 0.4).mean() for w in wins]
        ax.plot(wins, frac, marker="o", ms=4, lw=1.1, color=c,
                label={"capsule": "capsule", "parralejo": "110-subject",
                       "emo_p01": "emo (P01)"}[plat])
    ax.set_xlabel("drift window (s)")
    ax.set_ylabel("locked records (%)")
    ax.set_xticks(wins)
    ax.set_ylim(0, 100)
    ax.set_title("window sensitivity", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig_window.png", dpi=300)
    fig.savefig(FIG / "fig_window.pdf")
    plt.close(fig)


def main():
    fig_model_depth()
    fig_failure_map()
    fig_bh_conditions()
    fig_alignment_phase()
    fig_p02_p01()
    fig_carrier()
    fig_window()
    print("figures written to", FIG)


if __name__ == "__main__":
    raise SystemExit(main())
