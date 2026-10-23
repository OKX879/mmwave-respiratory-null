#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨数据集/跨域汇总：合成 · capsule · emo(vital) · emo(cumsum) 四方对照。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from decision_presentation import eta_squared, setup_font, KEEP_CRITERIA, ANTI_CRITERIA  # noqa: E402
from calibrate_criteria import safe_spearman, expected_sign                            # noqa: E402
from criteria_lib import CRITERIA                                                      # noqa: E402
from closedloop_criteria import parse_candidate                                        # noqa: E402

GOLD_HR = "hr_mae_bpm"
DIR = {c.key: c.direction for c in CRITERIA}

SOURCES = [
    ("合成（2025 候选）", HERE / "matrix_out_t4" / "matrix_results.csv"),
    ("capsule（20 例, ECG）", HERE / "real_matrix" / "real_matrix_results.csv"),
    ("emo vital（60 trial, PPG）", HERE / "emo_matrix" / "emo_results_vital_20.csv"),
    ("emo cumsum（60 trial）", HERE / "emo_matrix" / "emo_results_cumsum_20.csv"),
    ("emo vital 全长（30 trial）", HERE / "emo_matrix" / "emo_results_vital_full.csv"),
]


def load(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if "boundary" not in d.columns:
        d["boundary"] = [parse_candidate(c)["boundary"] for c in d["candidate_id"]]
    d["threshold"] = [parse_candidate(c)["threshold"] for c in d["candidate_id"]]
    return d


def main() -> int:
    setup_font()
    import matplotlib.pyplot as plt

    loaded = [(n, load(p)) for n, p in SOURCES]
    loaded = [(n, d) for n, d in loaded if d is not None]
    if not loaded:
        print("[cmp] 找不到结果文件")
        return 1

    print("\n=== 1) 绝对精度（候选级中位 HR MAE，BPM） ===")
    print("  注：『池中位』随各自的候选池组成而变（合成含 L≥7 / 真实侧只到 L6），")
    print("      **跨数据集比较必须用『L≤4 阈值化中位』**（同一结构条件）。")
    print(f"  {'数据集/域':28s}{'池中位':>9s}{'L≤4中位':>10s}{'min':>8s}{'max':>8s}{'候选数':>7s}")
    acc = {}
    for n, d in loaded:
        g = d[d["level_over_max"] == 0]
        l4 = g[(g["level"] <= 4) & (g["threshold"] != "none")][GOLD_HR].median()
        acc[n] = {"pool_median": float(d[GOLD_HR].median()),
                  "L4_median": float(l4),
                  "min": float(d[GOLD_HR].min()), "max": float(d[GOLD_HR].max()),
                  "n_cand": int(len(d))}
        print(f"  {n:28s}{acc[n]['pool_median']:9.2f}{acc[n]['L4_median']:10.2f}"
              f"{acc[n]['min']:8.2f}{acc[n]['max']:8.2f}{acc[n]['n_cand']:7d}")

    print("\n=== 2) L ≤ 4 结构性边界：配对 Δ(阈值化 − none)，候选级 ===")
    print(f"  {'数据集/域':28s}{'L2':>7s}{'L3':>7s}{'L4':>7s}{'L5':>8s}{'L6':>8s}")
    for n, d in loaded:
        g = d[d["level_over_max"] == 0]
        piv = g.pivot_table(index=["wavelet", "level"], columns="threshold",
                            values=GOLD_HR, aggfunc="median")
        if "none" not in piv.columns:
            continue
        cells = []
        for lv in [2, 3, 4, 5, 6]:
            try:
                sub = piv.xs(lv, level="level")
            except KeyError:
                cells.append("   n/a")
                continue
            ds = [float(np.nanmedian(sub[t] - sub["none"]))
                  for t in ["universal-soft", "sure", "minimax"] if t in sub.columns]
            cells.append(f"{np.median(ds):+.2f}" if ds else "  n/a")
        print(f"  {n:28s}" + "".join(f"{c:>7s}" if i < 3 else f"{c:>8s}"
                                      for i, c in enumerate(cells)))

    print("\n=== 3) 因子 η²（候选级，log1p(HR_MAE)） ===")
    print(f"  {'数据集/域':28s}{'层数':>9s}{'阈值':>9s}{'母小波':>9s}{'边界':>9s}")
    eta_all = {}
    for n, d in loaded:
        g = d[(d["level_over_max"] == 0) & (d["boundary"] == "symmetric")]
        if g.empty:
            g = d[d["level_over_max"] == 0]
        y = np.log1p(g[GOLD_HR].to_numpy(float))
        eta_all[n] = {f: float(eta_squared(y, g[f].to_numpy()))
                      for f in ["level", "threshold", "wavelet", "boundary"]}
        print(f"  {n:28s}" + "".join(f"{eta_all[n][k]:9.4f}"
                                      for k in ["level", "threshold", "wavelet", "boundary"]))

    print("\n=== 4) 边界模式零效应（L=4 配对 Δ vs symmetric，BPM） ===")
    for n, d in loaded:
        l4 = d[(d["level"] == 4) & (d["level_over_max"] == 0)]
        piv = l4.pivot_table(index=["wavelet", "threshold"], columns="boundary",
                             values=GOLD_HR, aggfunc="median")
        if {"zero", "symmetric", "periodic"}.issubset(set(piv.columns)):
            print(f"  {n:28s}Δ(zero)={float(np.nanmedian(piv['zero']-piv['symmetric'])):+.4f}"
                  f"   Δ(periodic)={float(np.nanmedian(piv['periodic']-piv['symmetric'])):+.4f}")
        else:
            print(f"  {n:28s}（无三边界配对）")

    print("\n=== 5) KEEP 准则方向迁移：ρ(准则, HR_MAE)，universal-soft 子集，候选级 ===")
    crit_keys = ["pb1", "pb2", "pb3", "pb4", "pb5", "rc3", "rc4", "sm1", "sm2", "sm4", "sm5"]
    hdr = f"  {'准则':6s}" + "".join(f"{n[:16]:>18s}" for n, _ in loaded)
    print(hdr)
    for k in crit_keys:
        line = f"  {k:6s}"
        for n, d in loaded:
            sub = d[d["threshold"] == "universal-soft"]
            agg = sub.groupby("candidate_id").median(numeric_only=True)
            if k not in agg.columns:
                line += f"{'n/a':>18s}"
                continue
            rho, p, _ = safe_spearman(agg[k], agg[GOLD_HR])
            ok = np.isfinite(rho) and np.sign(rho) == expected_sign(DIR[k], "down")
            line += f"{f'{rho:+.2f}':>12s}{'✓' if ok else '✗':>6s}"
        print(line)
    print("  注：✓=符号与合成侧一致（down 金标 HR_MAE）；✗=符号相反（反预测器）")

    print("\n=== 6) ★ 域对照：emo vital（主）vs emo cumsum（次要对照） ===")
    v = load(HERE / "emo_matrix" / "emo_results_vital_20.csv")
    c = load(HERE / "emo_matrix" / "emo_results_cumsum_20.csv")
    if v is not None and c is not None:
        print(f"  {'':10s}{'池中位':>10s}{'L≤4 中位':>11s}{'L5':>8s}{'L6':>8s}{'L≥5 中位':>11s}")
        for tag, d in [("vital", v), ("cumsum", c)]:
            g = d[d["level_over_max"] == 0]
            l4 = g[(g["level"] <= 4) & (g["threshold"] != "none")][GOLD_HR].median()
            l5 = g[(g["level"] == 5) & (g["threshold"] != "none")][GOLD_HR].median()
            l6 = g[(g["level"] == 6) & (g["threshold"] != "none")][GOLD_HR].median()
            lge = g[(g["level"] >= 5) & (g["threshold"] != "none")][GOLD_HR].median()
            print(f"  {tag:10s}{d[GOLD_HR].median():10.2f}{l4:11.2f}{l5:8.2f}{l6:8.2f}{lge:11.2f}")
        bv = str(v.sort_values(GOLD_HR).iloc[0]["candidate_id"])
        bc = str(c.sort_values(GOLD_HR).iloc[0]["candidate_id"])
        print(f"  vital  最优 = {bv}")
        print(f"  cumsum 最优 = {bc}")
        print(f"  两者 top-1 是否同一 → {bv == bc}")
        # L≤4 池退化程度
        for tag, d in [("vital", v), ("cumsum", c)]:
            l4d = d[(d["level"] <= 4) & (d["boundary"] == "symmetric")]
            print(f"  {tag:10s} L≤4 候选 {len(l4d)}  不同取值 {l4d[GOLD_HR].round(3).nunique()}"
                  f"  最大共享数 {int(l4d[GOLD_HR].round(3).value_counts().iloc[0])}")

    # ---- 图 ----
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.6))
    zh = ["层数", "阈值", "母小波", "边界"]
    ks = ["level", "threshold", "wavelet", "boundary"]
    colors = ["#4C78A8", "#E45756", "#54A24B", "#B279A2", "#F58518"]
    for i, (n, d) in enumerate(loaded):
        lv = d[d["level_over_max"] == 0].groupby("level")[GOLD_HR].median()
        if 2 in lv.index:
            ax[0, 0].plot(lv.index, lv / float(lv.loc[2]), "o-", color=colors[i % len(colors)],
                          label=n, ms=5)
    ax[0, 0].axvspan(1.6, 4.4, color="#54A24B", alpha=0.10)
    ax[0, 0].set_xlabel("分解层数 L")
    ax[0, 0].set_ylabel("中位 HR MAE（归一化到各自 L=2）")
    ax[0, 0].set_title("(a) 层数悬崖：各数据集是否都在 L≥5 跳起")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3)

    xp = np.arange(len(ks))
    w = 0.8 / len(eta_all)
    for i, (n, _) in enumerate(eta_all.items()):
        ax[0, 1].bar(xp + (i - (len(eta_all) - 1) / 2) * w, [eta_all[n][k] for k in ks], w,
                     color=colors[i % len(colors)], label=n[:14])
    ax[0, 1].set_xticks(xp); ax[0, 1].set_xticklabels(zh)
    ax[0, 1].set_ylabel("η²（候选级）")
    ax[0, 1].set_title("(b) 因子次序（层数 >> 阈值 > 基 > 边界）是否普遍")
    ax[0, 1].legend(fontsize=7.5)
    ax[0, 1].grid(axis="y", alpha=0.3)

    # (c) KEEP 准则 ρ：合成 vs emo vital vs capsule
    show = ["pb1", "pb3", "sm1", "sm2", "sm4", "pb2", "rc3", "sm5"]
    xp2 = np.arange(len(show))
    for i, (n, d) in enumerate([x for x in loaded if "emo cumsum" not in x[0] and "全长" not in x[0]]):
        vals = []
        for k in show:
            sub = d[d["threshold"] == "universal-soft"]
            agg = sub.groupby("candidate_id").median(numeric_only=True)
            rho, _, _ = safe_spearman(agg[k], agg[GOLD_HR]) if k in agg.columns else (np.nan, 0, 0)
            vals.append(rho)
        ax[1, 0].bar(xp2 + (i - 1) * 0.27, vals, 0.27, color=colors[i], label=n[:16])
    ax[1, 0].axhline(0, color="black", lw=0.8)
    ax[1, 0].set_xticks(xp2); ax[1, 0].set_xticklabels(show)
    ax[1, 0].set_ylabel("ρ(准则, HR_MAE)")
    ax[1, 0].set_title("(c) 准则方向迁移（KEEP: pb1/pb3/sm1/sm4；反预测: rc3/sm5）")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(axis="y", alpha=0.3)

    # (d) 域对照
    if v is not None and c is not None:
        for tag, d, col in [("vital（一阶差分域，主臂）", v, "#1f77b4"),
                            ("cumsum（位移域，对照）", c, "#d62728")]:
            lv = d[d["level_over_max"] == 0].groupby("level")[GOLD_HR].median()
            ax[1, 1].plot(lv.index, lv.values, "s-", color=col, label=tag, ms=5)
        ax[1, 1].axvspan(1.6, 4.4, color="#54A24B", alpha=0.10)
        ax[1, 1].set_xlabel("分解层数 L")
        ax[1, 1].set_ylabel("中位 HR MAE (BPM，绝对值)")
        ax[1, 1].set_title("(d) ★ 域对照：换成一阶差分域是否改变结论")
        ax[1, 1].legend(fontsize=8)
        ax[1, 1].grid(alpha=0.3)

    fig.suptitle("图 T5-emo　emo_dataset 跨硬件测试（IWR6843，15 受试者，PPG 金标；两变量变化，pilot 非 validation）",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fp = HERE / "emo_matrix" / "figT5_emo.png"
    fig.savefig(fp, dpi=150)
    print(f"\n[cmp] 图 → {fp}")

    pd.DataFrame([{"dataset": n, **acc[n], **eta_all.get(n, {})} for n, _ in loaded]).to_csv(
        HERE / "emo_matrix" / "emo_transfer_summary.csv", index=False, encoding="utf-8")
    print(f"[cmp] 汇总 → {HERE / 'emo_matrix' / 'emo_transfer_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
