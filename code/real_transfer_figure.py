#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T5 迁移对比图：合成 vs 真实（capsule）四联图。

四张面板分别回答 T5 的四个迁移问题：
  (a) 层数曲线：L≤4 安全边界是否在真实信号上复现？（各自归一化到自身 none 水平）
  (b) 因子 η²：因子**次序**是否复现？量级是否缩小？
  (c) 准则 ρ：KEEP 集与反预测器名单是否复现？（真实 vs 合成散点）
  (d) 高带能量占比：解释"为什么基在 L=4 的区分度没有迁移"

用法：python real_transfer_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from decision_presentation import eta_squared, setup_font, KEEP_CRITERIA, ANTI_CRITERIA  # noqa: E402
from criteria_lib import CRITERIA                                                     # noqa: E402


def main() -> int:
    setup_font()
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    syn = pd.read_csv(HERE / "matrix_out_t4" / "matrix_results.csv")
    real = pd.read_csv(HERE / "real_matrix" / "real_matrix_results.csv")
    syn_lv = syn.groupby("level")["hr_mae_bpm"].median()
    real_lv = real.groupby("level")["hr_mae_bpm"].median()
    base_syn = float(syn_lv.loc[2])
    base_real = float(real_lv.loc[2])

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.6))

    # (a) 层数曲线
    ax[0, 0].plot(syn_lv.index, syn_lv / base_syn, "o-", color="#4C78A8", label="合成（2025 候选）")
    ax[0, 0].plot(real_lv.index, real_lv / base_real, "s-", color="#E45756", label="真实 capsule（20 例）")
    ax[0, 0].axvspan(1.6, 4.4, color="#54A24B", alpha=0.10)
    ax[0, 0].axhline(1.0, ls=":", color="gray", lw=1)
    ax[0, 0].set_xlabel("分解层数 L")
    ax[0, 0].set_ylabel("中位 HR MAE（归一化到各自 L=2）")
    ax[0, 0].set_title("(a) 层数悬崖：两侧都在 L=5 处跳起（L≤4 无损伤）")
    ax[0, 0].legend(fontsize=9)
    ax[0, 0].grid(alpha=0.3)

    # (b) η² —— 分析单位为【候选】（与合成侧同口径）；两侧都取主边界 symmetric
    dl = json.loads((HERE / "matrix_out_t4" / "decision_summary.json").read_text(encoding="utf-8"))
    eta_syn = dl["factor_eta2"]
    g = real[(real["level_over_max"] == 0) & (real["boundary"] == "symmetric")]
    y = np.log1p(g["hr_mae_bpm"].to_numpy(float))
    eta_real = {f: eta_squared(y, g[f].to_numpy()) for f in ["level", "threshold", "wavelet", "boundary"]}
    ks = ["level", "threshold", "wavelet", "boundary"]
    zh = ["层数", "阈值", "母小波", "边界"]
    xp = np.arange(len(ks))
    ax[0, 1].bar(xp - 0.2, [eta_syn[k] for k in ks], 0.4, color="#4C78A8", label="合成")
    ax[0, 1].bar(xp + 0.2, [eta_real[k] for k in ks], 0.4, color="#E45756", label="真实")
    for i, k in enumerate(ks):
        ax[0, 1].text(i - 0.2, eta_syn[k] + 0.012, f"{eta_syn[k]:.3f}", ha="center", fontsize=8.5)
        ax[0, 1].text(i + 0.2, eta_real[k] + 0.012, f"{eta_real[k]:.4f}", ha="center", fontsize=8.5)
    ax[0, 1].set_xticks(xp); ax[0, 1].set_xticklabels(zh)
    ax[0, 1].set_ylabel("η²（log1p(HR_MAE)，候选级）")
    ax[0, 1].set_ylim(0, 0.72)
    ax[0, 1].set_title("(b) 因子次序完全复现（层数>阈值>母小波>边界=0），量级可比")
    ax[0, 1].legend(fontsize=9)
    ax[0, 1].grid(axis="y", alpha=0.3)

    # (c) 准则 ρ
    rc = pd.read_csv(HERE / "real_matrix" / "real_criteria_calibration.csv")
    sc = pd.read_csv(HERE / "matrix_out_t4" / "criteria_calibration.csv")
    sc = sc[(sc.scope == "thr:universal-soft") & (sc.gold == "HR_MAE_BPM")]
    srho = {r["criterion"]: float(r["rho"]) for _, r in sc.iterrows()}
    for _, r in rc.iterrows():
        k = r["criterion"]
        xs, ys = srho.get(k, np.nan), r["rho_real_soft"]
        if not (np.isfinite(xs) and np.isfinite(ys)):
            continue
        col = ("#1B7F3B" if k in KEEP_CRITERIA else
               "#B03A2E" if k in ANTI_CRITERIA else "#777777")
        ax[1, 0].scatter([xs], [ys], s=70, color=col, edgecolors="black", zorder=4)
        ax[1, 0].annotate(k, (xs, ys), textcoords="offset points", xytext=(7, 5), fontsize=9)
    lim = 1.05
    ax[1, 0].plot([-lim, lim], [-lim, lim], ls="--", color="gray", lw=1, label="y=x（完美迁移）")
    ax[1, 0].axhline(0, color="black", lw=0.8); ax[1, 0].axvline(0, color="black", lw=0.8)
    ax[1, 0].set_xlim(-lim, lim); ax[1, 0].set_ylim(-lim, lim)
    ax[1, 0].set_xlabel("ρ(准则, HR_MAE)　合成")
    ax[1, 0].set_ylabel("ρ(准则, HR_MAE)　真实")
    ax[1, 0].set_title("(c) 准则迁移：绿=KEEP，红=反预测器；rc1/pb4/rc2 在真实侧 nan")
    ax[1, 0].legend(fontsize=9, loc="lower right")
    ax[1, 0].grid(alpha=0.3)

    # (d) 高带能量占比
    def frac_above(x, f0, fs=100.0):
        f, P = welch(np.asarray(x, float), fs=fs, nperseg=min(2048, len(x)))
        tot = np.trapezoid(P, f)
        m = f > f0
        return float(np.trapezoid(P[m], f[m]) / tot) if tot > 0 else np.nan
    grid = np.load(HERE / "synthetic_data" / "grid.npz")
    syn_frac = [frac_above(grid["card"][i], 3.125) for i in range(12)]
    real_frac = []
    for i in range(1, 21):
        p = HERE / "real_data" / f"patient_{i}.npz"
        if p.exists():
            real_frac.append(frac_above(np.load(p, allow_pickle=True)["wide"], 3.125))
    bp = ax[1, 1].boxplot([np.array(syn_frac), np.array(real_frac)],
                          tick_labels=["合成心搏（高斯脉冲串）", "真实位移（capsule wide）"],
                          vert=True, patch_artist=True,
                          boxprops=dict(facecolor="#4C78A8"),
                          medianprops=dict(color="black"))
    for i, arr in enumerate([syn_frac, real_frac], start=1):
        ax[1, 1].text(i, float(np.median(arr)) + 0.004,
                      f"中位 {np.median(arr):.4f}", ha="center", fontsize=9,
                      color="#1B7F3B" if i == 1 else "#B03A2E")
    ax[1, 1].annotate("", xy=(2, float(np.median(real_frac))), xytext=(1, float(np.median(syn_frac))),
                      arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
    ax[1, 1].text(1.5, (np.median(syn_frac) + np.median(real_frac)) / 2 + 0.006,
                  f"×{np.median(syn_frac)/max(np.median(real_frac),1e-9):.1f}",
                  ha="center", fontsize=10, color="black")
    ax[1, 1].set_ylim(-0.005, max(0.18, np.max(syn_frac) * 1.12))
    ax[1, 1].set_ylabel(">3.125 Hz（d4 带）能量占比")
    ax[1, 1].set_title("(d) 机制：合成心搏的高频含量 ~14× 高于真实位移\n→ 真实侧 L≤4 阈值化够不着心搏，基/阈值失效")
    ax[1, 1].grid(axis="y", alpha=0.3)

    fig.suptitle("图 T5　小波基选择方法的真实数据迁移检验（capsule，20 例，ECG 金标；pilot 非 validation）",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = HERE / "real_matrix" / "figT5_transfer.png"
    fig.savefig(out, dpi=150)
    print(f"[fig] {out}")
    print(f"  高带能量占比 中位：合成 {np.median(syn_frac):.4f}  真实 {np.median(real_frac):.4f}"
          f"（倍数 {np.median(syn_frac)/max(np.median(real_frac),1e-9):.1f}×）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
