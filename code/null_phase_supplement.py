#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充材料小图：除正文代表例外，全部合格记录（深凹陷 >= 5 次）的轨迹 + 呼吸相位折叠。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from null_phase_figure import analyse, OUT  # noqa: E402

LOCK_RBAR = 0.4


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
    })

    stems = [f"patient_{i}" for i in range(1, 21)
             if (Path(r"C:\Users\yxh\Desktop\capsule-4604022-data\Minimum_dataset")
                 / f"patient_{i}.mat").exists()]
    rows = [analyse(s) for s in stems]
    elig_all = [r for r in rows if r["n_deep"] >= 5 and np.isfinite(r["Rbar"])]
    # 正文代表例与其重复件不在本图重复展示
    omit = {"patient_15", "patient_16"}
    elig = [r for r in elig_all if r["stem"] not in omit]
    elig.sort(key=lambda r: -r["Rbar"])
    n = len(elig)
    if n == 0:
        raise SystemExit("no eligible records")

    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(7.16, 1.56 * nrow + 0.48))
    gs = fig.add_gridspec(nrow, ncol * 2, wspace=0.24, hspace=0.30,
                          width_ratios=[1.30, 1.0] * ncol,
                          left=0.062, right=0.985, top=0.93, bottom=0.115)

    th = np.linspace(0, 2 * np.pi, 300)
    for k, r in enumerate(elig):
        i, j = divmod(k, ncol)
        z, pk = r["z"], r["peaks"]
        ph = r["phase"]

        ax = fig.add_subplot(gs[i, 2 * j])
        ax.plot(z.real, z.imag, lw=0.35, color="#4C78A8", alpha=0.5)
        ax.plot(r["thr"] * np.cos(th), r["thr"] * np.sin(th),
                ls="--", lw=0.6, color="#8C8C8C")
        ax.scatter(z.real[pk], z.imag[pk], s=9, color="#E45756",
                   edgecolor="black", linewidth=0.25, zorder=5)
        ax.scatter([0], [0], marker="+", s=45, color="black", linewidth=0.9, zorder=6)
        R98 = float(np.percentile(np.abs(z), 98))
        ax.set_xlim(-1.05 * R98, 1.05 * R98)
        ax.set_ylim(-1.05 * R98, 1.05 * R98)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.18, lw=0.3)
        tag = ""
        star = "$\\ast$" if r["Rbar"] > LOCK_RBAR else ""
        ax.set_title(f"{r['stem']}{tag}{star}\n"
                     f"$n$={r['n_deep']}, $\\bar R$={r['Rbar']:.2f}",
                     fontsize=6.8, pad=2.2)
        if j == 0:
            ax.set_ylabel("Im$\\{z\\}$", fontsize=6.8, labelpad=1)
        if i == nrow - 1:
            ax.set_xlabel("Re$\\{z\\}$ (a.u.)", fontsize=6.8, labelpad=1)

        axp = fig.add_subplot(gs[i, 2 * j + 1], projection="polar")
        nb = 12
        cnt, edges = np.histogram(ph, bins=nb, range=(0, 1))
        width = 2 * np.pi / nb
        scale = max(cnt.max(), 1)
        axp.bar(edges[:-1] * 2 * np.pi, cnt / scale, width=width * 0.92,
                color="#54A24B", alpha=0.88, edgecolor="white", linewidth=0.4, zorder=2)
        axp.plot(th, np.full(len(th), cnt.mean() / scale), ls="--", lw=0.6,
                 color="#8C8C8C", zorder=3)
        mv = np.mean(np.exp(1j * 2 * np.pi * ph))
        axp.annotate("", xy=(np.angle(mv), min(abs(mv), 1.0)), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", lw=0.9, color="#E45756"), zorder=4)
        axp.set_theta_zero_location("N")
        axp.set_theta_direction(-1)
        axp.set_ylim(0, 1.02)
        axp.set_yticks([])
        axp.tick_params(axis="y", length=0)
        if i == 0:
            axp.set_xticks(np.arange(0, 2 * np.pi, np.pi / 2))
            axp.set_xticklabels(["0", "", "0.5", ""], fontsize=6)
        else:
            axp.set_xticks([])

    n_lock = sum(1 for r in elig if r["Rbar"] > LOCK_RBAR)
    from scipy.stats import chi2
    # 合并检验覆盖"全部 8 条非重复合格记录"（含正文例 patient_15），不只本图展示的 7 条
    uniq_p = [r["p"] for r in elig_all
              if r["stem"] != "patient_16" and np.isfinite(r["p"])]
    stat = -2.0 * float(np.sum(np.log(uniq_p)))
    p_fisher = float(chi2.sf(stat, 2 * len(uniq_p)))
    n_indiv = sum(1 for r in elig_all
                  if r["stem"] != "patient_16" and np.isfinite(r["p"]) and r["p"] < 0.01)
    fig.text(0.5, 0.006,
             f"Shown: {n} of 20 records with $n\\geq$5 deep nulls "
             f"($0.35\\times$ median $|z|$); locked ($\\bar R>0.4$): {n_lock}/{n}. "
             f"$\\ast$ = locked. patient_15 (main text) and its duplicate "
             f"patient_16 are omitted.\n"
             f"Across all {len(uniq_p)} non-duplicate eligible records: individual "
             f"Rayleigh $p<0.01$ in {n_indiv}; Fisher combined "
             f"$\\chi^2$={stat:.1f}, df={2 * len(uniq_p)}, $p$={p_fisher:.1e} "
             f"(assumes independent records).",
             ha="center", va="bottom", fontsize=6.4, color="#333333",
             linespacing=1.5)

    png = OUT / "figS1_all_eligible_records.png"
    fig.savefig(png, dpi=300)
    fig.savefig(OUT / "figS1_all_eligible_records.pdf")
    plt.close(fig)
    print(f"figure -> {png}")
    print(f"eligible {n}, locked {n_lock}, order: {[r['stem'] for r in elig]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
