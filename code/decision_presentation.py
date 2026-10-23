#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T7 决策与呈现：硬门槛 → Borda 秩聚合 → Pareto 前沿；产出表 A / B / C 与四张图。

为什么必须"三段式"而不是线性加权
--------------------------------
项目既有教训：规则堆叠会引入不可测的调参点，一条新规则可毁全链路。
因此禁止给 14 个准则各配权重。本脚本严格按《实验计划》§7 三段式执行：

  1) 硬门槛：先剔除「结构上必然不合格」与「物理上不合格」的候选。
     阈值由生理/数学定，**不由数据拟合**。
  2) 秩聚合：只对**通过校准的准则**做 Borda 计数（无需权重，抗单指标异常）。
     **反预测器（rc1 / rc3 / sm5）必须先禁用**，否则会把基选反（§4 F6 / 口径 9）。
  3) Pareto 前沿：作为结论呈现——"没有任何一个基在所有指标上最优"。

本脚本遵守的硬约束（逐条对应 研究进展记录.md §7）
----------------------------------------------
- 口径 9 ：反预测器 rc1/rc3/sm5 禁用；名单依赖前端（本矩阵为 RHC **关闭**，故用 off 名单）。
- 口径 16：K 折复检一律用**留出协议**（train 选 / test 评），不做"既选又评"。
- 口径 17：秩聚合可能被"输出并列块"带偏 → 本脚本对输出**逐位相同**的候选做结构去重校验，
           并明确其适用边界（小池成立 / 大池不复现）。
- 口径 19：层数结构性安全边界 L ≤ 4（aL 上界 = nyq/2^L 须覆盖心搏带上界 3.0 Hz）→ 硬门槛。
- 口径 20：因子优先级 η²：层数 ≫ 阈值 ≫ 基 > 边界；"基"的区分度只在 L=4 一格。
- 口径 21：**报告名次必须给"值差（gap）"**，不能只给"秩"（T4 上有 870 个候选挤在
           0.005 BPM 窗口内，秩无区分意义）。

产出
----
<table_out>/
  table_A_candidate_matrix.csv   候选矩阵总表（含门槛标记 / Borda 分 / Pareto 标记）
  table_B_factors_config.csv     因子重要性 + 层数边界 + 决策结果
  table_C_criteria_rho.csv       准则校准 ρ 表（soft / all / per-sample + 判定）
  decision_summary.json          全部数字（供论文写作引用）
  figs_T7/fig1_factor_effects.png
  figs_T7/fig2_pareto_front.png
  figs_T7/fig3_criteria_rho_heatmap.png
  figs_T7/fig4_L4_wavelet_threshold_grid.png

用法
----
    python decision_presentation.py --out-dir matrix_out_t4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from criteria_lib import CRITERIA
from calibrate_criteria import RHO_MIN, P_MAX
from closedloop_criteria import level_a_holdout, select_criteria, parse_candidate

# ---------------------------------------------------------------------------
# 常量（全部来自既有结论，**不得据数据再拟合**）
# ---------------------------------------------------------------------------
# 校准保留（T4，§4 F13）：off 条件下通过门槛的准则
KEEP_CRITERIA = ["pb1", "pb3", "sm1", "sm4"]
# 反预测器（§4 F6 / 口径 9，off 名单）：|ρ| 很大但方向相反 → 禁用
ANTI_CRITERIA = ["rc1", "rc3", "sm5"]
# 报告专用（不进评分）
REPORT_ONLY = ["sm3"]

# 硬门槛阈值（生理/数学依据）
LEVEL_MAX_SAFE = 4          # aL 上界 = nyq/2^L 须覆盖心搏带上界 3.0 Hz ⟺ L ≤ 4
IBI_OK_MIN = 0.70           # IBI 落在 333–1500 ms 的比例下限（生理）
RHR_MAX = 1.0               # 呼吸谐波能量不得超过心搏带内其余能量（物理）

CARD_BAND_HI = 3.0          # Hz，心搏展示带上界
NYQ = 50.0                  # Hz，fs=100 → 奈奎斯特

# 金标（合成数据可算）：列名 → (好方向, 展示名)
GOLD = [("hr_mae_bpm", "down", "HR_MAE_BPM"),
        ("sdr_db", "up", "SDR_dB"),
        ("corr_card", "up", "corr_card")]

_FIG = {}

# ---------------------------------------------------------------------------
# 中文字体（避免缺字方块）—— 与项目既有惯例一致
# ---------------------------------------------------------------------------
def setup_font() -> str:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    want = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
            "Source Han Sans SC", "PingFang SC", "WenQuanYi Zen Hei"]
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in want:
        if name in avail:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    matplotlib.rcParams["axes.unicode_minus"] = False
    return "(未找到中文字体)"


# ---------------------------------------------------------------------------
# 基础统计
# ---------------------------------------------------------------------------
def eta_squared(y: np.ndarray, groups: np.ndarray) -> float:
    """单因素 η² = SS_between / SS_total（方差解释比例）。"""
    y = np.asarray(y, dtype=float)
    g = np.asarray(groups)
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0:
        return float("nan")
    ssb = 0.0
    for u in np.unique(g):
        m = y[g == u]
        ssb += len(m) * (m.mean() - y.mean()) ** 2
    return float(ssb / sst)


def compute_factor_eta2(g: pd.DataFrame) -> dict:
    """因子 η²（log1p(HR_MAE)），只统计未超最大层数的候选（避免退化配置污染）。"""
    d = g[g["level_over_max"] == 0]
    y = np.log1p(d["hr_mae_bpm"].to_numpy(dtype=float))
    return {f: eta_squared(y, d[f].to_numpy()) for f in
            ["level", "threshold", "wavelet", "boundary"]}


def borda_score(g: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Borda 秩聚合（与 closedloop_criteria.borda 同口径）：
    按准则方向把秩线性累加，越大越好；NaN 记为该准则最差。"""
    from scipy.stats import rankdata
    d = {c.key: c.direction for c in CRITERIA}
    s = np.zeros(len(g))
    for k in keys:
        v = g[k].to_numpy(dtype=float)
        o = v if d[k] == "up" else -v
        bad = ~np.isfinite(o)
        if bad.any():
            fill = (np.nanmin(o[~bad]) - 1.0) if (~bad).any() else 0.0
            o = np.where(bad, fill, o)
        s += rankdata(o, method="average")
    return s


def pareto_mask(g: pd.DataFrame, obj: list[tuple[str, str]]) -> np.ndarray:
    """Pareto 前沿（非支配集）。obj = [(列名, 'min'|'max'), ...]"""
    A = np.column_stack([
        g[c].to_numpy(dtype=float) * (1.0 if s == "min" else -1.0) for c, s in obj
    ])
    n = len(g)
    m = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.all(np.isfinite(A[i])):
            m[i] = False
            continue
        dom = np.all(A <= A[i], axis=1) & np.any(A < A[i], axis=1)
        if dom.any():
            m[i] = False
    return m


# ---------------------------------------------------------------------------
# 1) 载入
# ---------------------------------------------------------------------------
def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 复杂度：dec_len(基) × 层数（数据点级运算量的近似）
    import pywt
    def _dec_len(w):
        try:
            return int(pywt.Wavelet(w).dec_len)
        except Exception:
            return -1
    df["dec_len"] = df["wavelet"].map(_dec_len)
    df["complexity"] = (df["dec_len"] * df["level"]).astype(float)
    return df


# ---------------------------------------------------------------------------
# 2) 硬门槛
# ---------------------------------------------------------------------------
def apply_hard_gates(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """三段式第一段：剔除结构/物理上必然不合格的候选（阈值不由数据拟合）。"""
    h1 = (df["level"] <= LEVEL_MAX_SAFE) & (df["level_over_max"] == 0)
    h2 = df["pb3"] >= IBI_OK_MIN
    h3 = df["pb2"] < RHR_MAX
    g = df.copy()
    g["gate_H1_structural"] = h1
    g["gate_H2_ibi"] = h2
    g["gate_H3_rhr"] = h3
    g["in_pool"] = h1 & h2 & h3
    info = {
        "n_total": int(len(g)),
        "H1_structural": {"rule": f"level<={LEVEL_MAX_SAFE} AND level_over_max==0",
                          "n_drop": int((~h1).sum()),
                          "n_drop_level_ge5": int((g["level"] > LEVEL_MAX_SAFE).sum()),
                          "n_drop_over_max": int((g["level_over_max"] == 1).sum()),
                          "n_drop_within_H1": 0},
        "H2_ibi": {"rule": f"pb3 >= {IBI_OK_MIN}", "n_drop": int((~h2).sum()),
                   "n_drop_within_H1": int((h1 & ~h2).sum())},
        "H3_rhr": {"rule": f"pb2 < {RHR_MAX}", "n_drop": int((~h3).sum()),
                   "n_drop_within_H1": int((h1 & ~h3).sum())},
        "n_pool": int(g["in_pool"].sum()),
        "rejected_plan_gate": {
            "rule": "跨通道相干 rc1 > tau3（《实验计划》§7 原案）",
            "status": "REJECTED_BY_CALIBRATION",
            "reason": ("rc1 在本矩阵（RHC 关闭）上是反预测器：rho(rc1, sm3)=-0.769（§4 F6 / 口径 9）；"
                       "合成生成器令 8 通道共享同一条真实位移，过度收缩后只剩共模 → 相干仍高（B7）。"
                       "因此该门槛不进入最终决策，如实记录为『计划假设被数据推翻』。"),
        },
    }
    if verbose:
        print("[T7-1 硬门槛]")
        print(f"  全池 {info['n_total']} 候选")
        print(f"  H1 结构安全  level<={LEVEL_MAX_SAFE} 且未超最大层数 → 剔除 {info['H1_structural']['n_drop']}"
              f"（level>{LEVEL_MAX_SAFE}: {info['H1_structural']['n_drop_level_ge5']}，"
              f"超最大层数: {info['H1_structural']['n_drop_over_max']}）")
        print(f"  H2 生理门槛  pb3(IBI合格率) >= {IBI_OK_MIN} → 全池剔除 {info['H2_ibi']['n_drop']}；"
              f"其中落在 H1 安全区内的 {info['H2_ibi']['n_drop_within_H1']} 个（即安全区内全数通过）")
        print(f"  H3 物理门槛  pb2(呼吸谐波残余比) < {RHR_MAX} → 全池剔除 {info['H3_rhr']['n_drop']}；"
              f"其中落在 H1 安全区内的 {info['H3_rhr']['n_drop_within_H1']} 个")
        print(f"  → 决策池 {info['n_pool']} 候选")
        print("  注：生理/物理门槛主要剔除的是 L≥5 的退化候选；在 H1 安全区内它们近于饱和"
              "（pb3 中位 0.976、pb2 中位 0.282）——如实记录。")
        print(f"  [计划原案第三道门槛 rc1>τ3 被校准否决] {info['rejected_plan_gate']['reason']}")
    return g, info


# ---------------------------------------------------------------------------
# 3) 决策链：Borda + Pareto
# ---------------------------------------------------------------------------
def decide(g: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """三段式第二、三段：对决策池做 Borda 秩聚合 → 选 top-1；再取 Pareto 前沿。"""
    pool = g[g["in_pool"]].copy()
    pool["borda"] = borda_score(pool, KEEP_CRITERIA)
    obj = [("hr_mae_bpm", "min"), ("sdr_db", "max"), ("complexity", "min")]
    pool["pareto"] = pareto_mask(pool, obj)
    pool = pool.sort_values("borda", ascending=False)
    pick = pool.iloc[0]
    best = pool.sort_values("hr_mae_bpm").iloc[0]

    # 推荐 = 落在 Pareto 前沿上的 Borda 第一名（若 top-1 不在前沿，如实标注）
    front = pool[pool["pareto"]].sort_values("borda", ascending=False)
    recommend = front.iloc[0] if len(front) else pick

    n_better = int((pool["hr_mae_bpm"] < pick["hr_mae_bpm"] - 1e-9).sum())
    hr_rank = pool.set_index("candidate_id")["hr_mae_bpm"].rank()
    # 值差（口径 21）：必须给 gap，不能只给秩
    res = {
        "pool_size": int(len(pool)),
        "keep_criteria": list(KEEP_CRITERIA),
        "anti_criteria_disabled": list(ANTI_CRITERIA),
        "borda_pick": {
            "candidate_id": str(pick["candidate_id"]),
            "hr_mae_bpm": float(pick["hr_mae_bpm"]),
            "sdr_db": float(pick["sdr_db"]),
            "corr_card": float(pick["corr_card"]),
            "rr_mae_bpm": float(pick["rr_mae_bpm"]),
            "complexity": float(pick["complexity"]),
            "on_pareto_front": bool(pick["pareto"]),
        },
        "gold_best": {
            "candidate_id": str(best["candidate_id"]),
            "hr_mae_bpm": float(best["hr_mae_bpm"]),
            "sdr_db": float(best["sdr_db"]),
            "complexity": float(best["complexity"]),
        },
        "gap_vs_best_bpm": float(pick["hr_mae_bpm"] - best["hr_mae_bpm"]),
        "n_candidates_strictly_better": n_better,
        "gold_rank_of_pick": int(hr_rank.loc[pick["candidate_id"]]),
        "recommend": {
            "candidate_id": str(recommend["candidate_id"]),
            "hr_mae_bpm": float(recommend["hr_mae_bpm"]),
            "sdr_db": float(recommend["sdr_db"]),
            "complexity": float(recommend["complexity"]),
        },
        "pareto": {
            "n_front": int(pool["pareto"].sum()),
            "members": pool[pool["pareto"]]
                .sort_values("hr_mae_bpm")[["candidate_id", "hr_mae_bpm", "sdr_db", "complexity"]]
                .to_dict(orient="records"),
        },
        "pool_hr_quantiles": {str(q): float(pool["hr_mae_bpm"].quantile(q))
                              for q in (0.0, 0.25, 0.5, 0.75, 1.0)},
    }
    if verbose:
        print("\n[T7-2 Borda 秩聚合]")
        print(f"  评分准则（校准 KEEP）: {'+'.join(KEEP_CRITERIA)}")
        print(f"  已禁用（反预测器）    : {', '.join(ANTI_CRITERIA)}")
        print(f"  准则选出 → {res['borda_pick']['candidate_id']}")
        print(f"     HR_MAE={res['borda_pick']['hr_mae_bpm']:.4f}  SDR={res['borda_pick']['sdr_db']:.3f}"
              f"  RR_MAE={res['borda_pick']['rr_mae_bpm']:.3f}  复杂度={res['borda_pick']['complexity']:.0f}")
        print(f"  金标最优 → {res['gold_best']['candidate_id']}")
        print(f"     HR_MAE={res['gold_best']['hr_mae_bpm']:.4f}  SDR={res['gold_best']['sdr_db']:.3f}")
        print(f"  ★ 值差(gap) = {res['gap_vs_best_bpm']:+.4f} BPM ；"
              f"严格更优的候选数 = {n_better}/{len(pool)} （**秩无区分意义，以值差为准**，口径 21）")
        print(f"  选出者是否在 Pareto 前沿: {res['borda_pick']['on_pareto_front']}")
        print("\n[T7-3 Pareto 前沿]（目标：HR_MAE↓ / SDR↑ / 复杂度↓）")
        print(f"  前沿成员 {res['pareto']['n_front']} 个；前沿上 Borda 最优 = {res['recommend']['candidate_id']}")
        print(f"     HR_MAE={res['recommend']['hr_mae_bpm']:.4f}  SDR={res['recommend']['sdr_db']:.3f}"
              f"  复杂度={res['recommend']['complexity']:.0f}")
        print("  说明：没有任何一个基在全部指标上最优（这本身是结论）。")
    return pool, res


# ---------------------------------------------------------------------------
# 4) K 折复检（留出协议；硬门槛前后对照）
# ---------------------------------------------------------------------------
def kfold_recheck(g: pd.DataFrame, raw_path: Path, folds: int = 5,
                  verbose: bool = True) -> dict:
    """决策流程 vs 裸 Borda：硬门槛是否修复了 top-1 跳坏候选（T4 的 F12 发现 2）。"""
    raw = pd.read_csv(raw_path)
    raw = raw.copy()
    raw["threshold"] = [parse_candidate(c)["threshold"] for c in raw["candidate_id"]]

    def _run(df, tag, per_fold_selection: bool):
        """per_fold_selection=True 复现 T4 口径（每折在 train 上重新筛准则）；
        False = 直接用已校准的固定 KEEP 集。两者都遵守留出协议。"""
        ids = sorted(df["sample_id"].unique().tolist())
        K = folds
        fl = [ids[i::K] for i in range(K)]
        rows = []
        for f in range(K):
            test_ids = fl[f]
            train_ids = [s for gi in range(K) if gi != f for s in fl[gi]]
            if per_fold_selection:
                kept, _ = select_criteria(df, train_ids, "thr:universal-soft", RHO_MIN, P_MAX)
            else:
                kept = list(KEEP_CRITERIA)
            if not kept:
                rows.append({"fold": f, "kept": "", "pick_id": None, "pick_hr": float("nan"),
                             "best_hr": float("nan"), "gap": float("nan"),
                             "rho": float("nan"), "n_cand": 0, "gold_rank": -1})
                continue
            la = level_a_holdout(df, train_ids, test_ids, "all", kept)
            rows.append({"fold": f, "kept": "+".join(kept), "pick_id": la["pick_id"],
                         "pick_hr": la["pick_gold"]["hr_mae_bpm"],
                         "best_hr": la["best_gold"]["hr_mae_bpm"], "gap": la["pick_hr_gap_vs_best"],
                         "rho": la["rho_borda_vs_hr"], "n_cand": la["n_candidates"],
                         "gold_rank": la["gold_rank_of_pick"]})
        r = pd.DataFrame(rows)
        ok = r[np.isfinite(r["gap"].to_numpy(dtype=float))]
        return {
            "tag": tag, "per_fold_selection": per_fold_selection,
            "n_candidates": int(r["n_cand"].iloc[0]) if len(r) else 0,
            "rho_mean": float(ok["rho"].mean()) if len(ok) else float("nan"),
            "rho_std": float(ok["rho"].std(ddof=0)) if len(ok) else float("nan"),
            "gap_mean": float(ok["gap"].mean()) if len(ok) else float("nan"),
            "gap_std": float(ok["gap"].std(ddof=0)) if len(ok) else float("nan"),
            "gap_max": float(ok["gap"].max()) if len(ok) else float("nan"),
            "zero_gap_folds": int((ok["gap"].abs() < 1e-6).sum()) if len(ok) else 0,
            "folds": rows,
        }

    gated = raw[raw["candidate_id"].isin(set(g[g["in_pool"]]["candidate_id"]))]
    runs = [
        _run(raw, "A 全池 2025 + 冻结 KEEP", False),
        _run(gated, "B 决策池 675 + 冻结 KEEP（= 决策流程）", False),
        _run(gated, "C 决策池 675 + 逐折重筛准则", True),
    ]
    if verbose:
        print(f"\n[T7-4 K 折复检（留出协议，K={folds}；train 选 / test 评）]")
        for o in runs:
            print(f"  {o['tag']:34s} 候选{o['n_candidates']:5d}  "
                  f"ρ={o['rho_mean']:+.3f}±{o['rho_std']:.3f}  "
                  f"值差 mean {o['gap_mean']:+.2f}±{o['gap_std']:.2f} (max {o['gap_max']:.2f})")
            for r in o["folds"]:
                print(f"      折{r['fold']} [{r['kept']:26s}] {str(r['pick_id']):40s}"
                      f" pick={r['pick_hr']:7.3f} best={r['best_hr']:7.3f} gap={r['gap']:+7.3f}")
        print("  【对照】T4 原口径（全池 + 逐折重筛准则）：ρ=-0.659±0.109、值差 +7.45±7.45（max 20.01）")
        print("  读取（如实，勿过度解读）：")
        print("   · A vs B：在本 5 折切分下，**选出的配置与值差逐折完全相同**（都是 L4 候选）——"
              "硬门槛的收益在此切分下是『预防性』的（杜绝 top-1 落到 L≥5），而非可观测改善。")
        print("   · 残留的大值差（折2 +6.8、折4 +10.4）并非坏基/坏层造成，而是 F9 的**逐样本双峰误差**"
              "（12 样本的折中位可能落进坏模态）→ 硬门槛修不了，只有空间前端（T5）能修。")
        print("   · C 说明『在子池上重筛准则』不可取：它会筛出 rc1/rc3/sm5（全池上的反预测器）、"
              "并有 2/5 折筛不出任何准则 → 必须冻结已校准的准则集（口径 12）。")
    return {"A_full_pool_frozen_keep": runs[0], "B_gated_frozen_keep": runs[1],
            "C_gated_refilter": runs[2]}


# ---------------------------------------------------------------------------
# 5) 表 A / B / C
# ---------------------------------------------------------------------------
def build_table_A(g: pd.DataFrame, out_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """表 A：候选矩阵总表（含门槛标记 / Borda 分 / Pareto 标记）。
    注：beat F1 与 IBI RMSE(TP) 需逐拍金标，本合成矩阵不产出（T5 真实数据才有），故不伪造该两列。"""
    cols = ["candidate_id", "wavelet", "level", "threshold", "boundary", "dec_len", "complexity",
            "hr_mae_bpm", "rr_mae_bpm", "sdr_db", "corr_card",
            "pb1", "pb2", "pb3", "sm1", "sm2", "sm4", "rc1", "rc3", "sm5", "sm3",
            "gate_H1_structural", "gate_H2_ibi", "gate_H3_rhr", "in_pool"]
    A = g[cols].copy()
    if "borda" in g.columns:
        A = A.join(g[["borda", "pareto"]])
    A = A.sort_values("hr_mae_bpm")
    A.to_csv(out_dir / "table_A_candidate_matrix.csv", index=False, encoding="utf-8")
    if verbose:
        print(f"\n[表 A] 候选矩阵总表 → table_A_candidate_matrix.csv（{len(A)} 行）")
        print("  决策池内 HR_MAE 最优前 12：")
        print(f"  {'候选':44s}{'HR_MAE':>9s}{'RR_MAE':>9s}{'SDR':>8s}{'corr':>8s}{'复杂度':>7s}{'Pareto':>8s}")
        sub = A[A["in_pool"]].head(12)
        for _, r in sub.iterrows():
            print(f"  {r['candidate_id']:44s}{r['hr_mae_bpm']:9.4f}{r['rr_mae_bpm']:9.3f}"
                  f"{r['sdr_db']:8.3f}{r['corr_card']:8.3f}{r['complexity']:7.0f}"
                  f"{(str(bool(r['pareto'])) if 'pareto' in A.columns else '-'):>8s}")
        print("  注：beat F1 / IBI RMSE(TP) 不在本矩阵（需真实逐拍金标，T5 产出）——不伪造。")
    return A


def build_table_B(g: pd.DataFrame, eta: dict, decision: dict,
                  out_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """表 B：因子重要性 + 层数边界 + 决策结果（含最优配置）。"""
    # 层数安全边界表：该层**全部候选**（含 none）的中位数 —— 与 F13 口径一致
    lv = []
    for L in sorted(g["level"].unique()):
        aL = NYQ / (2 ** L)
        cov = bool(aL >= CARD_BAND_HI)
        med = float(g[g["level"] == L]["hr_mae_bpm"].median())
        lv.append({"level": int(L), "aL_upper_Hz": round(aL, 4), "covers_cardiac_band": cov,
                   "median_HR_MAE": round(med, 4)})

    # L=4 单格：跨 5 阈值 × 3 边界 的最优 HR_MAE（按基）
    l4 = g[(g["level"] == LEVEL_MAX_SAFE) & (g["level_over_max"] == 0)]
    per_base = l4.groupby("wavelet")["hr_mae_bpm"].min().sort_values()
    benefit = per_base[per_base < 2.7]
    no_benefit = per_base[per_base >= 2.7]

    rows = []
    for f, v in eta.items():
        rows.append({"section": "factor_eta2", "key": f, "value": round(v, 4)})
    for r in lv:
        rows.append({"section": "level_boundary", "key": f"L{r['level']}",
                     "value": r["median_HR_MAE"],
                     "note": f"aL<= {r['aL_upper_Hz']} Hz, covers={r['covers_cardiac_band']}"})
    for w, v in per_base.items():
        rows.append({"section": "L4_per_wavelet_best", "key": w, "value": round(float(v), 4),
                     "note": "benefit" if w in benefit.index else "no_benefit"})
    for k, v in [("pool_size", decision["pool_size"]),
                 ("gap_vs_best_bpm", decision["gap_vs_best_bpm"]),
                 ("n_strictly_better", decision["n_candidates_strictly_better"]),
                 ("pareto_n_front", decision["pareto"]["n_front"])]:
        rows.append({"section": "decision", "key": k, "value": v})
    B = pd.DataFrame(rows)
    B.to_csv(out_dir / "table_B_factors_config.csv", index=False, encoding="utf-8")

    if verbose:
        print(f"\n[表 B] 因子与最优配置 → table_B_factors_config.csv")
        print("  ★ 因子重要性 η²（log1p(HR_MAE)，剔除超层数候选）  [口径 20：不要把重点放在选基]")
        for f in ["level", "threshold", "wavelet", "boundary"]:
            print(f"    {f:10s} {eta[f]:.4f}")
        print(f"  ★ 层数结构性安全边界（口径 19：aL 上界须覆盖心搏带上界 {CARD_BAND_HI} Hz）")
        print(f"    {'层数':>4s}{'aL上界/Hz':>12s}{'覆盖心搏带':>11s}{'该层中位HR_MAE':>16s}")
        for r in lv:
            print(f"    {r['level']:>4d}{r['aL_upper_Hz']:>12.4f}"
                  f"{('是' if r['covers_cardiac_band'] else '否'):>11s}"
                  f"{r['median_HR_MAE']:>16.3f}")
        print("    （该层全部候选含 none 的中位；L≤4 的 ≈2.80 即 none 块水平，L≥5 落进阈值化区）")
        print(f"  ★ L=4 单格：换基能否拿到增益（口径 20）")
        print(f"    受益基({len(benefit)}): " + ", ".join(f"{w} {v:.3f}" for w, v in benefit.items()))
        print(f"    不受益({len(no_benefit)}): " + ", ".join(f"{w} {v:.3f}" for w, v in no_benefit.items()))
        print(f"  ★ 决策结果")
        print(f"    决策池 {decision['pool_size']}；Borda 选出 {decision['borda_pick']['candidate_id']}")
        print(f"    值差 {decision['gap_vs_best_bpm']:+.4f} BPM；严格更优候选 {decision['n_candidates_strictly_better']} 个")
        print(f"    推荐（Pareto ∩ Borda 最优）= {decision['recommend']['candidate_id']}")
    return B


def build_table_C(calib_csv: Path, out_dir: Path, verbose: bool = True,
                  calib_json: Path | None = None) -> pd.DataFrame:
    """表 C：准则校准 ρ 表（soft / all / per-sample + 判定）。"""
    t = pd.read_csv(calib_csv)
    official = {}
    if calib_json is not None and Path(calib_json).exists():
        official = {v["criterion"]: v["verdict"]
                    for v in json.loads(Path(calib_json).read_text(encoding="utf-8"))["verdicts"]}
    GOLDN = [g[2] for g in GOLD]
    recs = []
    for c in CRITERIA:
        row = {"criterion": c.key, "group": c.group, "direction": c.direction,
               "report_only": c.report_only}
        for scope, tag in [("thr:universal-soft", "soft"), ("all", "all"), ("per-sample", "ps")]:
            for gn in GOLDN:
                r = t[(t.criterion == c.key) & (t.scope == scope) & (t.gold == gn)]
                v = float(r.iloc[0]["rho"]) if len(r) else float("nan")
                row[f"rho_{tag}_{gn}"] = v
        # 判定（来自 criteria_calibration.json 更权威，这里用 CSV 的 pass 列近似汇总）
        prim = t[(t.criterion == c.key) & (t.scope == "thr:universal-soft")]
        full = t[(t.criterion == c.key) & (t.scope == "all")]
        if c.report_only or c.direction == "n/a":
            verdict = "report_only"
        elif bool(prim["pass"].any()):
            verdict = "KEEP"
        elif bool(full["pass"].any()):
            verdict = "WATCH"
        else:
            fin = prim[np.isfinite(prim["rho"].to_numpy(dtype=float))]
            flipped = fin[(~fin["sign_ok"].astype(bool)) & (fin["rho"].abs() >= 0.4)]
            verdict = "DROP(anti-predictor)" if len(flipped) else "DROP"
        row["verdict"] = verdict
        row["verdict_official"] = official.get(c.key, verdict)
        row["role_in_decision"] = ("KEEP(进 Borda)" if c.key in KEEP_CRITERIA else
                                   "DISABLED(反预测器)" if c.key in ANTI_CRITERIA else
                                   "REPORT_ONLY" if c.key in REPORT_ONLY else "not_used")
        recs.append(row)
    C = pd.DataFrame(recs)
    C.to_csv(out_dir / "table_C_criteria_rho.csv", index=False, encoding="utf-8")
    if verbose:
        print(f"\n[表 C] 准则校准 ρ 表 → table_C_criteria_rho.csv（主判据 = universal-soft）")
        hdr = (f"  {'准则':6s}{'组':4s}{'向':5s}{'ρHR':>8s}{'ρSDR':>8s}{'ρcorr':>8s}"
               f"{'ρHR_all':>9s}  {'判定(官方)':12s}{'决策角色':18s}")
        print(hdr)
        for _, r in C.iterrows():
            print(f"  {r['criterion']:6s}{r['group']:4s}{r['direction']:5s}"
                  f"{r['rho_soft_HR_MAE_BPM']:+8.2f}{r['rho_soft_SDR_dB']:+8.2f}{r['rho_soft_corr_card']:+8.2f}"
                  f"{r['rho_all_HR_MAE_BPM']:+9.2f}  {r['verdict_official']:12s}{r['role_in_decision']:18s}")
        print("  反预测器（禁用）：" + ", ".join(ANTI_CRITERIA) + "   ← 口径 9")
    return C


# ---------------------------------------------------------------------------
# 6) 四张图
# ---------------------------------------------------------------------------
def make_figs(g: pd.DataFrame, eta: dict, decision: dict, table_C: pd.DataFrame,
              fig_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []

    # ---- 图 1：因子效应（η² bar + 层数悬崖）----
    zh = {"level": "层数", "threshold": "阈值规则", "wavelet": "母小波", "boundary": "边界模式"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ks = list(eta.keys())
    ax[0].barh([zh[k] for k in ks][::-1], [eta[k] for k in ks][::-1],
               color=["#4C78A8", "#F58518", "#54A24B", "#BAB0AC"][::-1])
    for i, k in enumerate(ks[::-1]):
        ax[0].text(eta[k] + 0.01, i, f"{eta[k]:.3f}", va="center", fontsize=10)
    ax[0].set_xlim(0, max(eta.values()) * 1.25)
    ax[0].set_xlabel("η²（log1p(HR_MAE) 的方差解释比例）")
    ax[0].set_title("(a) 因子重要性：层数 >> 阈值 >> 母小波 > 边界")
    ax[0].grid(axis="x", alpha=0.3)

    med = g[g["threshold"] != "none"].groupby("level")["hr_mae_bpm"].median()
    ax[1].plot(med.index, med.values, "o-", color="#4C78A8", lw=2, ms=6, label="阈值化候选 中位 HR_MAE")
    ax[1].axhline(2.8037, ls="--", color="gray", lw=1.2, label="none 块（恒等变换）2.804")
    ax[1].axvspan(1.6, LEVEL_MAX_SAFE + 0.4, color="#54A24B", alpha=0.12)
    ax[1].text((1.6 + LEVEL_MAX_SAFE + 0.4) / 2, 12, "安全区 L ≤ 4\nal 上界覆盖心搏带",
               ha="center", color="#2E6B3E", fontsize=9)
    ax[1].text(7.5, 12, "危险区 L ≥ 5\n心搏带落入被阈值化的细节层", ha="center", color="#B03A2E", fontsize=9)
    ax[1].set_xlabel("分解层数 L")
    ax[1].set_ylabel("中位 HR_MAE (BPM)")
    ax[1].set_title("(b) 层数悬崖：L≤4 无损伤，L≥5 必然伤害心搏")
    ax[1].legend(fontsize=9, loc="upper left")
    ax[1].grid(alpha=0.3)
    fig.suptitle("图 1　小波基系统的因子效应（2025 候选 / 60 样本）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = fig_dir / "fig1_factor_effects.png"
    fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))

    # ---- 图 2：Pareto 前沿（复杂度 vs HR_MAE，颜色=SDR）----
    pool = g[g["in_pool"]]
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    sc = ax.scatter(pool["complexity"], pool["hr_mae_bpm"], c=pool["sdr_db"],
                    s=26, cmap="viridis", alpha=0.75, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("SDR (dB)　越大越好")
    fr = pool[pool["pareto"]].sort_values("complexity")
    ax.plot(fr["complexity"], fr["hr_mae_bpm"], "-", color="#E45756", lw=1.4, alpha=0.9,
            label=f"Pareto 前沿（{len(fr)} 个）")
    bp = decision["borda_pick"]
    ax.scatter([bp["complexity"]], [bp["hr_mae_bpm"]], marker="*", s=320, color="#E45756",
               edgecolors="black", zorder=5, label="Borda 选出（KEEP 准则）")
    gb = decision["gold_best"]
    ax.scatter([gb["complexity"]], [gb["hr_mae_bpm"]], marker="P", s=180, color="white",
               edgecolors="black", zorder=5, label="金标最优（HR_MAE）")
    ax.annotate(f"值差仅 {decision['gap_vs_best_bpm']:+.4f} BPM\n（秩 {decision['gold_rank_of_pick']}/"
                f"{decision['pool_size']} 无区分意义）",
                xy=(bp["complexity"], bp["hr_mae_bpm"]), xytext=(bp["complexity"] + 6, bp["hr_mae_bpm"] + 0.16),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax.set_xlabel("复杂度（dec_len × 层数，越小越快）")
    ax.set_ylabel("HR MAE (BPM)　越小越好")
    ax.set_title("图 2　精度–分离度–复杂度 Pareto 前沿（决策池 675 候选）")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "fig2_pareto_front.png"
    fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))

    # ---- 图 3：ρ 热图（准则 × 金标，soft / all 两面板）----
    order = [c.key for c in CRITERIA]
    glabel = [f"{c.key} {c.group}" for c in CRITERIA]
    # 行标签着色：绿 = 校准 KEEP；红 = 反预测器（禁用）；灰 = 其它
    ycol = ["#1B7F3B" if c in KEEP_CRITERIA else
            "#B03A2E" if c in ANTI_CRITERIA else "#777777" for c in order]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 6.2), constrained_layout=True)
    for k, (tag, ttl) in enumerate([("soft", "(a) universal-soft 子集（主判据）"),
                                    ("all", "(b) 全集（对照）")]):
        M = np.array([[table_C[table_C.criterion == ck][f"rho_{tag}_{gn}"].iloc[0]
                       for gn in [g[2] for g in GOLD]] for ck in order], dtype=float)
        im = ax[k].imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax[k].set_xticks(range(3)); ax[k].set_xticklabels(["HR MAE", "SDR", "corr"], fontsize=9)
        ax[k].set_yticks(range(len(order)))
        ax[k].set_yticklabels(glabel, fontsize=8.5)
        ax[k].set_title(ttl, fontsize=10)
        for i in range(len(order)):
            ax[k].get_yticklabels()[i].set_color(ycol[i])
            for j in range(3):
                v = M[i, j]
                if np.isfinite(v):
                    ax[k].text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                               color="white" if abs(v) > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Spearman ρ")
    fig.suptitle("图 3　无参考准则校准 ρ（绿字 = 校准 KEEP ／ 红字 = 反预测器·禁用 ／ 灰字 = 其它）",
                 fontsize=12)
    p = fig_dir / "fig3_criteria_rho_heatmap.png"
    fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))

    # ---- 图 4：L=4 单格的 基 × 阈值 热图（"基只在 L4 有区分度"）----
    l4 = g[(g["level"] == LEVEL_MAX_SAFE) & (g["level_over_max"] == 0)]
    piv = l4.pivot_table(index="wavelet", columns="threshold", values="hr_mae_bpm", aggfunc="min")
    piv["L2 最优参照"] = (g[(g["level"] == 2) & (g["level_over_max"] == 0)]
                          .groupby("wavelet")["hr_mae_bpm"].min())
    piv = piv.sort_values("L2 最优参照")
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap="RdYlGn_r", aspect="auto",
                   vmin=2.5, vmax=3.0)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=20, fontsize=9)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=9)
    for i, w in enumerate(piv.index):
        for j, c in enumerate(piv.columns):
            v = piv.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="HR MAE (BPM)")
    ax.set_title("图 4　L=4 单格（基 × 阈值）：换基能拿到增益，其它层数则不能\n"
                 "（绿色 = 优于 none 块 2.804；只有 8 个基在此格受益）", fontsize=11)
    fig.tight_layout()
    p = fig_dir / "fig4_L4_wavelet_threshold_grid.png"
    fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))
    return made


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="T7 决策与呈现（硬门槛 + Borda + Pareto + 表A/B/C + 四图）",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out-dir", type=str, default="matrix_out_t4")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    results_csv = out / "matrix_results.csv"
    raw_csv = out / "matrix_raw.csv"
    calib_csv = out / "criteria_calibration.csv"
    for p in (results_csv, raw_csv, calib_csv):
        if not p.exists():
            print(f"[T7] 缺少输入 {p}")
            return 1

    print(f"[T7] 字体: {setup_font()}")
    print(f"[T7] 数据: {results_csv}")

    g = load_results(results_csv)
    g, gate_info = apply_hard_gates(g)
    pool, decision = decide(g)
    kf = kfold_recheck(g, raw_csv, folds=args.folds)

    eta = compute_factor_eta2(g)

    # 把 borda/pareto 写回主表，供表 A
    g = g.merge(pool[["candidate_id", "borda", "pareto"]], on="candidate_id", how="left")
    A = build_table_A(g, out)
    B = build_table_B(g, eta, decision, out)
    C = build_table_C(calib_csv, out, calib_json=out / "criteria_calibration.json")

    figs = []
    if not args.no_fig:
        figs = make_figs(g, eta, decision, C, out / "figs_T7")
        print("\n[T7-5 图]")
        for p in figs:
            print(f"  {Path(p).name}")

    summary = {"gate_info": gate_info, "decision": decision, "kfold": kf,
               "factor_eta2": eta, "figures": [str(x) for x in figs]}
    (out / "decision_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[T7] 输出目录：{out}（表 A/B/C + decision_summary.json + figs_T7/）")
    print("[T7] 完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
