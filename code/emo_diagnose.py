#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""emo 侧诚实诊断：把"中位 3.65"拆开，回答三个问题。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from decision_presentation import setup_font  # noqa: E402

G = "hr_mae_bpm"


def diagnose(out_dir: Path, tag: str) -> dict:
    res = pd.read_csv(out_dir / f"emo_results_{tag}.csv")
    raw = pd.read_csv(out_dir / f"emo_raw_{tag}.csv")
    res["threshold"] = [c.split("|")[3] for c in res["candidate_id"]]

    # ---- Q1 无小波基线：threshold=='none' 是恒等变换，等价"不做小波" ----
    # 基线对比**必须在 L≤4 内做**：L≥5 的"提升"是伪的（阈值化打掉了错误的慢波峰，
    # 估计值偶然靠近金标，同一基在相邻层往往是全池最差）。用全池最优会得到
    # 虚假增益（实测 cumsum 臂曾算出 +64.2%，按 L≤4 重算实为 0.0%）。
    none = res[res["threshold"] == "none"]
    base = float(none[G].median())
    l4 = res[(res["level"] <= 4) & (res["level_over_max"] == 0) & (res["threshold"] != "none")]
    best = float(l4[G].min())
    tgt = l4.sort_values(G).iloc[0]

    # ---- Q2 双峰性：取"L≤4 内的最优候选"的逐 trial 误差分布 ----
    best_id = str(tgt["candidate_id"])
    sel = raw[raw["candidate_id"] == best_id]
    q = {p: float(np.percentile(sel[G], p)) for p in (50, 75, 90, 99)}
    frac_bad = float((sel[G] > 5.0).mean())

    # ---- Q3 逐受试者 ----
    ps = sel.groupby("subject")[G].median().sort_values()

    out = {
        "tag": tag, "n_trials": int(res["n_ok"].max()),
        "n_subjects": int(ps.size),
        "pool_median": float(res[G].median()),
        "pool_best": best, "no_wavelet_baseline": base,
        "gain_vs_no_wavelet_bpm": base - best,
        "gain_vs_no_wavelet_pct": 100.0 * (base - best) / base,
        "L4_span_bpm": float(l4[G].max() - l4[G].min()),
        "L4_distinct_values": int(l4[G].round(3).nunique()),
        "L4_n": int(len(l4)),
        "trial_p50": q[50], "trial_p75": q[75], "trial_p90": q[90], "trial_p99": q[99],
        "frac_trials_gt5bpm": frac_bad,
        "subject_median_min": float(ps.min()), "subject_median_max": float(ps.max()),
        "subject_std": float(ps.std()), "subject_mean": float(ps.mean()),
        "subject_table": {k: float(v) for k, v in ps.items()},
        "good_subjects": [k for k, v in ps.items() if v < 5.0],
        "bad_subjects": [k for k, v in ps.items() if v >= 5.0],
    }

    print(f"\n=== {tag}：{out['n_trials']} trial / {out['n_subjects']} 受试者 ===")
    print(f"  Q1 无小波基线（none 阈值 = 恒等变换） 中位 = {base:.3f} BPM")
    print(f"     L≤4 最优 = {best:.3f}（{best_id}）→ 小波环节的净增益 = {base-best:+.3f} BPM "
          f"({out['gain_vs_no_wavelet_pct']:.1f}%)")
    print(f"     L≤4 池内跨度 = {out['L4_span_bpm']:.3f} BPM（{out['L4_n']} 个候选，"
          f"仅 {out['L4_distinct_values']} 个不同取值）")
    print(f"  Q2 双峰性（L≤4 最优候选 {best_id}）")
    print(f"     逐 trial：P50 {q[50]:.2f} | P75 {q[75]:.2f} | P90 {q[90]:.2f} | P99 {q[99]:.2f}")
    print(f"     误差 > 5 BPM 的 trial 占 {frac_bad*100:.0f}%  → "
          f"{'**中位数掩盖了双峰**' if frac_bad > 0.25 else '分布较集中'}")
    print(f"  Q3 逐受试者：范围 [{ps.min():.2f}, {ps.max():.2f}]，"
          f"均值 {ps.mean():.2f}，标准差 {out['subject_std']:.2f}")
    print(f"     良好(<5 BPM)：{', '.join(out['good_subjects']) or '无'}")
    print(f"     失败(≥5 BPM)：{', '.join(out['bad_subjects']) or '无'}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emo 侧诚实诊断（基线/双峰/受试者异质）")
    ap.add_argument("--dir", type=str, default=str(HERE / "emo_matrix"))
    ap.add_argument("--tag", type=str, default="vital_20")
    args = ap.parse_args(argv)

    d = Path(args.dir)
    if not (d / f"emo_results_{args.tag}.csv").exists():
        print(f"[diag] 缺少 {d}/emo_results_{args.tag}.csv")
        return 1
    setup_font()

    all_out = [diagnose(d, args.tag)]
    # 若同目录有 cumsum 臂，一并诊断（域对照）
    if args.tag == "vital_20" and (d / "emo_results_cumsum_20.csv").exists():
        all_out.append(diagnose(d, "cumsum_20"))

    # ---- 图：双峰 + 逐受试者 ----
    import matplotlib.pyplot as plt
    res = pd.read_csv(d / f"emo_results_{args.tag}.csv")
    raw = pd.read_csv(d / f"emo_raw_{args.tag}.csv")
    res["threshold"] = [c.split("|")[3] for c in res["candidate_id"]]
    l4 = res[(res["level"] <= 4) & (res["level_over_max"] == 0) & (res["threshold"] != "none")]
    best_id = str(l4.sort_values(G).iloc[0]["candidate_id"])
    sel = raw[raw["candidate_id"] == best_id]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax[0].hist(sel[G].to_numpy(float), bins=40, color="#4C78A8", edgecolor="white")
    med = float(np.median(sel[G]))
    ax[0].axvline(med, color="#E45756", lw=2, label=f"中位 {med:.2f} BPM")
    ax[0].axvline(5.0, ls="--", color="gray", lw=1.2, label="5 BPM 参考线")
    ax[0].set_xlabel("单 trial HR MAE (BPM)")
    ax[0].set_ylabel("trial 数")
    ax[0].set_title(f"(a) 误差是双峰的：中位 {med:.2f}，但 {100*(sel[G]>5).mean():.0f}% 的 trial > 5 BPM")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.3)

    ps = sel.groupby("subject")[G].median().sort_values()
    colors = ["#54A24B" if v < 5 else "#E45756" for v in ps.values]
    ax[1].bar(range(len(ps)), ps.values, color=colors)
    ax[1].axhline(5.0, ls="--", color="gray", lw=1.2)
    ax[1].set_xticks(range(len(ps)))
    ax[1].set_xticklabels(ps.index, rotation=45, fontsize=8)
    ax[1].set_ylabel("受试者中位 HR MAE (BPM)")
    ax[1].set_title("(b) 受试者高度异质（绿<5 BPM / 红≥5 BPM）")
    ax[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"图 T5-emo-diag　{args.tag}：中位数掩盖的双峰与受试者异质性"
                 f"（{len(sel)} trial / {len(ps)} 人）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fp = d / f"figT5_emo_diag_{args.tag}.png"
    fig.savefig(fp, dpi=150)
    print(f"\n[diag] 图 → {fp}")

    (d / f"emo_diagnosis_{args.tag}.json").write_text(
        json.dumps(all_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[diag] 数字 → {d}/emo_diagnosis_{args.tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
