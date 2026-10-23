#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""b 闭环验证：只用"通过校准的无参考准则"选基，能否选到金标最优？
Closed-loop check b: selecting a basis only from calibrated reference-free criteria, does it recover the gold optimum?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from calibrate_criteria import (
    GOLD, P_MAX, RHO_MIN, expected_sign, safe_spearman,
)
from criteria_lib import CRITERIA

_DIR = {c.key: c.direction for c in CRITERIA}
_NAME = {c.key: c.name for c in CRITERIA}
_GOLD_KEYS = [g[0] for g in GOLD]          # hr_mae_bpm / sdr_db / corr_card
_GOLD_NAMES = {g[0]: g[2] for g in GOLD}


# ---------------------------------------------------------------------------
def parse_candidate(cid: str) -> dict:
    parts = cid.split("|")
    if parts[0] == "dwt":
        return {"kind": "dwt", "wavelet": parts[1], "level": parts[2].lstrip("L"),
                "threshold": parts[3], "boundary": parts[4] if len(parts) > 4 else "?"}
    return {"kind": "cwt", "wavelet": parts[1], "level": "-1",
            "threshold": "none", "boundary": "n/a"}


def scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "all":
        return pd.Series(True, index=df.index)
    if scope.startswith("thr:"):
        return df["threshold"] == scope[4:]
    raise ValueError(f"未知 scope: {scope}")


def borda(cand: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Borda 秩聚合：按准则方向把秩线性累加（越大越好）。NaN 记为该准则最差。
Borda rank aggregation: accumulate ranks by criterion direction, higher is better; NaN counts as worst."""
    s = np.zeros(len(cand))
    for k in keys:
        v = cand[k].to_numpy(dtype=float)
        o = v if _DIR[k] == "up" else -v
        bad = ~np.isfinite(o)
        if bad.any():
            fill = (np.nanmin(o[~bad]) - 1.0) if (~bad).any() else 0.0
            o = np.where(bad, fill, o)
        s += rankdata(o, method="average")
    return s


def select_criteria(df: pd.DataFrame, sample_ids: list[int], scope: str,
                    rho_min: float, p_max: float) -> tuple[list[str], list[dict]]:
    """在给定样本集上，用候选级 Spearman ρ（对 3 个金标）筛出通过的准则。
Screen criteria on a sample set by candidate-level Spearman rho against the three gold standards."""
    sub = df[df["sample_id"].isin(sample_ids) & scope_mask(df, scope)]
    agg = sub.groupby("candidate_id").median(numeric_only=True)
    keys = [c.key for c in CRITERIA if c.direction != "n/a" and not c.report_only]
    detail, kept = [], []
    for k in keys:
        passed_on = []
        for gk, gdir, gname in GOLD:
            rho, p, n = safe_spearman(agg[k], agg[gk])
            es = expected_sign(_DIR[k], gdir)
            if (np.isfinite(rho) and np.sign(rho) == es
                    and np.isfinite(p) and p < p_max and abs(rho) >= rho_min):
                passed_on.append(gname)
        detail.append({"criterion": k, "passed_on": passed_on, "n_candidates": int(len(agg))})
        if passed_on:
            kept.append(k)
    return kept, detail


# ---------------------------------------------------------------------------
def level_a_global(df: pd.DataFrame, sample_ids: list[int], scope: str,
                   keys: list[str]) -> dict:
    """全局选基：候选级 Borda → 第一名 vs 金标最优。
Global basis selection: candidate-level Borda, then compare the winner against the gold optimum."""
    sub = df[df["sample_id"].isin(sample_ids) & scope_mask(df, scope)]
    agg = sub.groupby("candidate_id").median(numeric_only=True)
    agg["meta"] = [parse_candidate(c) for c in agg.index]
    agg["borda"] = borda(agg, keys)
    pick = agg.sort_values("borda", ascending=False).iloc[0]
    best = agg.sort_values("hr_mae_bpm").iloc[0]
    rho_hr, p_hr, _ = safe_spearman(agg["borda"], agg["hr_mae_bpm"])
    rho_sdr, p_sdr, _ = safe_spearman(agg["borda"], agg["sdr_db"])
    return {
        "scope": scope, "n_candidates": int(len(agg)), "n_samples": len(sample_ids),
        "rho_borda_vs_hr": rho_hr, "p_hr": p_hr,
        "rho_borda_vs_sdr": rho_sdr, "p_sdr": p_sdr,
        "pick_id": pick.name,
        "pick_gold": {g: float(pick[g]) for g in _GOLD_KEYS},
        "best_id": best.name,
        "best_gold": {g: float(best[g]) for g in _GOLD_KEYS},
        "pick_is_best": bool(pick.name == best.name),
        "pick_hr_gap_vs_best": float(pick["hr_mae_bpm"] - best["hr_mae_bpm"]),
        "gold_rank_of_pick": int(agg["hr_mae_bpm"].rank().loc[pick.name]),
    }


def level_b_persample(df: pd.DataFrame, sample_ids: list[int], scope: str,
                      keys: list[str], global_best_id: str, fixed_id: str) -> dict:
    """逐样本选基：每样本内 Borda → top1；与 oracle / 全局最优 / 全体中位 / 最差比。
Per-sample selection: Borda within each sample, compared with oracle, global best, median and worst."""
    sub = df[df["sample_id"].isin(sample_ids) & scope_mask(df, scope)]
    rows = []
    for sid, g in sub.groupby("sample_id"):
        g = g.copy()
        g["borda"] = borda(g, keys)
        pick = g.sort_values("borda", ascending=False).iloc[0]
        oracle = g.sort_values("hr_mae_bpm").iloc[0]
        gb = g[g["candidate_id"] == global_best_id]
        fx = g[g["candidate_id"] == fixed_id]
        rows.append({
            "sample_id": int(sid),
            "pick_id": pick["candidate_id"], "pick_hr": float(pick["hr_mae_bpm"]),
            "pick_sdr": float(pick["sdr_db"]),
            "oracle_id": oracle["candidate_id"], "oracle_hr": float(oracle["hr_mae_bpm"]),
            "gb_hr": float(gb["hr_mae_bpm"].iloc[0]) if len(gb) else float("nan"),
            "fx_hr": float(fx["hr_mae_bpm"].iloc[0]) if len(fx) else float("nan"),
            "mean_hr": float(g["hr_mae_bpm"].median()),
            "worst_hr": float(g["hr_mae_bpm"].max()),
            "hit_oracle": bool(pick["candidate_id"] == oracle["candidate_id"]),
            "near_oracle": bool(abs(pick["hr_mae_bpm"] - oracle["hr_mae_bpm"]) <= 1.0),
        })
    r = pd.DataFrame(rows)
    def med(c): return float(np.nanmedian(r[c]))
    return {
        "n_samples": len(r), "fixed_id": fixed_id,
        "strategies_median_hr": {
            "criteria_pick(persample)": med("pick_hr"),
            "criteria_fixed(Level A 单一配置)": med("fx_hr"),
            "oracle_per_sample": med("oracle_hr"),
            "global_best(金标知情上界)": med("gb_hr"),
            "no_selection(median_over_cands)": med("mean_hr"),
            "worst_candidate": med("worst_hr"),
        },
        "strategies_median_sdr": {"criteria_pick": med("pick_sdr")},
        "hit_oracle_rate": float(r["hit_oracle"].mean()),
        "near_oracle_rate(<=1BPM)": float(r["near_oracle"].mean()),
        "pick_worse_than_no_selection_rate": float((r["pick_hr"] > r["mean_hr"]).mean()),
        "pick_worse_than_fixed_rate": float((r["pick_hr"] > r["fx_hr"]).mean()),
        "samples": r.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
def creds_agg(df: pd.DataFrame, ids: list[int], scope: str, keys: list[str]) -> pd.DataFrame:
    """在给定样本集上算候选级（跨样本中位数）指标 + Borda 分。
Candidate-level metrics (median across samples) plus the Borda score on a given sample set."""
    sub = df[df["sample_id"].isin(ids) & scope_mask(df, scope)]
    if sub.empty:
        return pd.DataFrame()
    agg = sub.groupby("candidate_id").median(numeric_only=True)
    agg["meta"] = [parse_candidate(c) for c in agg.index]
    agg["borda"] = borda(agg, keys)
    return agg


def level_a_holdout(df: pd.DataFrame, train_ids: list[int], test_ids: list[int],
                    scope: str, keys: list[str]) -> dict:
    """留出协议下的全局选基：**在 train 上按准则选，在 test 上评**（无信息泄漏）。
Held-out global selection: select on train, evaluate on test, so no information leaks."""
    tr = creds_agg(df, train_ids, scope, keys)
    te = creds_agg(df, test_ids, scope, keys)
    nan = float("nan")
    if tr.empty or te.empty or te["hr_mae_bpm"].isna().all():
        return {"scope": scope, "n_candidates": int(len(te)), "n_train": len(train_ids),
                "n_test": len(test_ids), "rho_borda_vs_hr": nan, "p_hr": nan,
                "pick_id": None, "pick_gold": {g: nan for g in _GOLD_KEYS},
                "pick_train_gold": {g: nan for g in _GOLD_KEYS},
                "best_id": None, "best_gold": {g: nan for g in _GOLD_KEYS},
                "pick_is_best": False, "pick_hr_gap_vs_best": nan,
                "gold_rank_of_pick": -1}
    pick_id = str(tr["borda"].idxmax())
    if pick_id not in te.index:                     # 该候选在留出折无数据
        pick_id = str(te["borda"].idxmax())
    best_id = str(te["hr_mae_bpm"].idxmin())
    rho, p, _ = safe_spearman(te["borda"], te["hr_mae_bpm"])
    rho_sdr, p_sdr, _ = safe_spearman(te["borda"], te["sdr_db"])
    return {
        "scope": scope, "n_candidates": int(len(te)),
        "n_train": len(train_ids), "n_test": len(test_ids),
        "rho_borda_vs_hr": rho, "p_hr": p,
        "rho_borda_vs_sdr": rho_sdr, "p_sdr": p_sdr,
        "pick_id": pick_id,
        "pick_gold": {g: float(te.loc[pick_id, g]) for g in _GOLD_KEYS},
        "pick_train_gold": {g: float(tr.loc[pick_id, g]) for g in _GOLD_KEYS},
        "best_id": best_id,
        "best_gold": {g: float(te.loc[best_id, g]) for g in _GOLD_KEYS},
        "pick_is_best": pick_id == best_id,
        "pick_hr_gap_vs_best": float(te.loc[pick_id, "hr_mae_bpm"]
                                     - te.loc[best_id, "hr_mae_bpm"]),
        "gold_rank_of_pick": int(te["hr_mae_bpm"].rank().loc[pick_id]),
    }


def run_kfold(df: pd.DataFrame, args, out_dir: Path) -> int:
    """K 折重复闭环：每折在 K-1 折上筛准则、在留出折上评闭环，报告均值±标准差。
Repeated K-fold loop: screen criteria on K-1 folds, evaluate on the held-out fold; report mean and sd."""
    all_ids = sorted(df["sample_id"].unique().tolist())
    K = args.folds
    # 交错分折：按索引取模，保证每折都覆盖 (SNR,HR,RR) 网格（数据集按块排列）
    folds = [all_ids[i::K] for i in range(K)]
    rows, crit_counts = [], {}
    for f in range(K):
        test_ids = folds[f]
        train_ids = [s for g in range(K) if g != f for s in folds[g]]
        if args.criteria:
            kept = list(args.criteria)
        else:
            kept, _ = select_criteria(df, train_ids, args.select_scope, args.rho_min, args.p_max)
        for k in kept:
            crit_counts[k] = crit_counts.get(k, 0) + 1
        if not kept:
            rows.append({"fold": f, "n_train": len(train_ids), "n_test": len(test_ids),
                         "kept": "", "rho": float("nan"), "pick_hr": float("nan"),
                         "best_hr": float("nan"), "gap": float("nan"), "gold_rank": -1,
                         "lb_persample": float("nan"), "lb_fixed": float("nan")})
            continue
        la = level_a_holdout(df, train_ids, test_ids, args.eval_scope, kept)
        lb = level_b_persample(df, test_ids, args.eval_scope, kept, la["best_id"], la["pick_id"])
        rows.append({
            "fold": f, "n_train": len(train_ids), "n_test": len(test_ids),
            "kept": "+".join(kept),
            "rho": la["rho_borda_vs_hr"], "pick_hr": la["pick_gold"]["hr_mae_bpm"],
            "best_hr": la["best_gold"]["hr_mae_bpm"], "gap": la["pick_hr_gap_vs_best"],
            "gold_rank": la["gold_rank_of_pick"], "n_cand": la["n_candidates"],
            "pick_id": la["pick_id"],
            "lb_persample": lb["strategies_median_hr"]["criteria_pick(persample)"],
            "lb_fixed": lb["strategies_median_hr"]["criteria_fixed(Level A 单一配置)"],
            "lb_oracle": lb["strategies_median_hr"]["oracle_per_sample"],
            "lb_hit": lb["hit_oracle_rate"], "lb_near": lb["near_oracle_rate(<=1BPM)"],
        })

    r = pd.DataFrame(rows)
    print(f"\n[loop] ===== {K} 折重复闭环（eval_scope={args.eval_scope}）=====")
    print(f"  每折：train {r.n_train.iloc[0]} / test {r.n_test.iloc[0]}")
    print(f"  {'折':>2s}  {'筛出准则':28s}{'ρ(Borda,HR)':>12s}{'选出HR':>8s}{'最优HR':>8s}{'差距':>7s}{'金标秩':>7s}"
          f"{'逐样本':>7s}{'单一配置':>9s}")
    for _, x in r.iterrows():
        nc = int(x["n_cand"]) if "n_cand" in x and np.isfinite(float(x["n_cand"])) else 0
        print(f"  {int(x['fold']):>2d}  {str(x['kept'])[:28]:28s}{x['rho']:>12.2f}"
              f"{x['pick_hr']:>8.2f}{x['best_hr']:>8.2f}{x['gap']:>+7.2f}"
              f"{(str(int(x['gold_rank']))+'/'+str(nc)):>7s}"
              f"{x['lb_persample']:>7.2f}{x['lb_fixed']:>9.2f}")
    ok = r[np.isfinite(r["rho"].to_numpy(dtype=float))]
    if len(ok):
        print(f"\n  汇总（有效折 {len(ok)}/{K}）：")
        print(f"    ρ(Borda,HR_MAE)  mean {ok['rho'].mean():+.3f}  std {ok['rho'].std(ddof=0):.3f}"
              f"   范围 [{ok['rho'].min():+.2f}, {ok['rho'].max():+.2f}]")
        print(f"    差距(选出 vs 最优)  mean {ok['gap'].mean():+.2f}  std {ok['gap'].std(ddof=0):.2f}"
              f"   max {ok['gap'].max():+.2f} BPM    零差距折数 {int((ok['gap'].abs()<1e-6).sum())}/{len(ok)}")
        print(f"    金标秩 中位 {ok['gold_rank'].median():.0f}  最好 {int(ok['gold_rank'].min())}  最差 {int(ok['gold_rank'].max())}")
        print(f"    Level B 逐样本 {ok['lb_persample'].median():.2f} vs 单一配置 {ok['lb_fixed'].median():.2f}"
              f"  → 比值中位 {(ok['lb_persample']/ok['lb_fixed'].replace(0,np.nan)).median():.1f}×")
        print(f"    命中 oracle {ok['lb_hit'].mean()*100:.0f}%   近命中 {ok['lb_near'].mean()*100:.0f}%")
    print(f"\n  准则筛选稳定性（被选中的折数 / {K}）：")
    for c in CRITERIA:
        print(f"    {c.key:5s} {crit_counts.get(c.key, 0)}/{K}"
              f"{'   ← 稳定' if crit_counts.get(c.key, 0) == K else ''}")

    r.to_csv(out_dir / "closedloop_kfold.csv", index=False, encoding="utf-8")
    summary = {
        "mode": "kfold", "K": K, "raw": str(args.raw),
        "select_scope": args.select_scope, "eval_scope": args.eval_scope,
        "folds": rows, "criterion_selected_in_folds": crit_counts,
        "aggregate": {
            "rho_mean": float(ok["rho"].mean()) if len(ok) else None,
            "rho_std": float(ok["rho"].std(ddof=0)) if len(ok) else None,
            "gap_mean": float(ok["gap"].mean()) if len(ok) else None,
            "gap_std": float(ok["gap"].std(ddof=0)) if len(ok) else None,
            "gap_max": float(ok["gap"].max()) if len(ok) else None,
            "zero_gap_folds": int((ok["gap"].abs() < 1e-6).sum()) if len(ok) else 0,
            "gold_rank_median": float(ok["gold_rank"].median()) if len(ok) else None,
            "gold_rank_best": int(ok["gold_rank"].min()) if len(ok) else None,
            "gold_rank_worst": int(ok["gold_rank"].max()) if len(ok) else None,
            "lb_persample_median": float(ok["lb_persample"].median()) if len(ok) else None,
            "lb_fixed_median": float(ok["lb_fixed"].median()) if len(ok) else None,
        },
    }
    (out_dir / f"closedloop_kfold_{args.eval_scope.replace(':', '_')}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="T2b 闭环验证（准则选基 vs 金标最优）",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--raw", type=str, default="matrix_out/matrix_raw.csv")
    p.add_argument("--out-dir", type=str, default="matrix_out")
    p.add_argument("--select-scope", type=str, default="thr:universal-soft",
                   help="准则筛选所在的 scope（与 T1 主判据一致）")
    p.add_argument("--eval-scope", type=str, default="thr:universal-soft",
                   help="闭环评估所在的 scope")
    p.add_argument("--criteria", type=str, nargs="*", default=None,
                   help="手动指定准则（覆盖 train 筛选），如 --criteria pb1 sm4")
    p.add_argument("--split", choices=["interleave", "none"], default="interleave",
                   help="interleave=交错切分 train/test；none=不切分（仅作乐观对照）")
    p.add_argument("--folds", type=int, default=0,
                   help="K≥2 时启用 K 折重复闭环（报告均值±标准差与准则筛选稳定性）；"
                        "0=关闭，用单次切分（原始 T2b 口径）")
    p.add_argument("--exclude-none", action="store_true",
                   help="剔除 threshold=='none' 的候选（结构去重：F1 已证其在各基/层/边界下"
                        "输出逐位相同，整个 none 块只贡献 1 个等效候选）——用于检验"
                        "『秩聚合在输出并列块上失效』（§4 F12 发现 2）")
    p.add_argument("--rho-min", type=float, default=RHO_MIN)
    p.add_argument("--p-max", type=float, default=P_MAX)
    args = p.parse_args(argv)

    df = pd.read_csv(args.raw)
    df = df.copy()
    df["threshold"] = [parse_candidate(c)["threshold"] for c in df["candidate_id"]]
    if args.exclude_none:
        df = df[df["threshold"] != "none"].copy()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.folds and args.folds >= 2:
        return run_kfold(df, args, out_dir)

    all_ids = sorted(df["sample_id"].unique().tolist())
    if args.split == "interleave":
        train_ids = all_ids[0::2]
        test_ids = all_ids[1::2]
    else:
        train_ids = all_ids
        test_ids = all_ids

    # ---- 1) 准则筛选：train 半（诚实） vs 全量（T1 的乐观值） ----
    if args.criteria:
        kept_train = list(args.criteria)
        detail_train = [{"criterion": k, "passed_on": ["<manual>"], "n_candidates": -1}
                        for k in kept_train]
    else:
        kept_train, detail_train = select_criteria(df, train_ids, args.select_scope,
                                                   args.rho_min, args.p_max)
    kept_full, detail_full = select_criteria(df, all_ids, args.select_scope,
                                             args.rho_min, args.p_max)

    print(f"[loop] 数据 {Path(args.raw).name}  样本 {len(all_ids)} "
          f"(train {len(train_ids)} / test {len(test_ids)})  切分={args.split}")
    print(f"[loop] 筛选 scope={args.select_scope}  评估 scope={args.eval_scope}")
    print(f"[loop] train 筛出 {len(kept_train)} 个准则：{kept_train}")
    print(f"[loop] 全量筛出 {len(kept_full)} 个准则：{kept_full}"
          f"  （差集 = {sorted(set(kept_full) - set(kept_train))}）")
    if not kept_train:
        print("[loop] train 未筛出任何准则 → 闭环无法进行")
        return 1

    # ---- 2) 闭环 ----
    # 诚实口径（留出）：在 train 上按准则选基 → 在 test 上评（无信息泄漏）
    # 乐观对照（自选自评）：在 train 上选也在 train 上评 —— 保留它只为量化泄漏的量级
    results = {}
    la_h = level_a_holdout(df, train_ids, test_ids, args.eval_scope, kept_train)
    lb_h = level_b_persample(df, test_ids, args.eval_scope, kept_train,
                             la_h["best_id"], la_h["pick_id"])
    results["test(诚实·留出)"] = {"level_a": la_h, "level_b": lb_h}

    la_t = level_a_global(df, train_ids, args.eval_scope, kept_train)
    lb_t = level_b_persample(df, train_ids, args.eval_scope, kept_train,
                             la_t["best_id"], la_t["pick_id"])
    results["train(自选自评·乐观)"] = {"level_a": la_t, "level_b": lb_t}

    # ---- 3) 呈现 ----
    for tag, res in results.items():
        la, lb = res["level_a"], res["level_b"]
        print(f"\n===== {tag}：评估 scope={la['scope']}，候选 {la['n_candidates']} =====")
        print(f"[Level A 全局选基] Borda vs HR_MAE ρ={la['rho_borda_vs_hr']:+.2f} (p={la['p_hr']:.3g})"
              f"  vs SDR ρ={la['rho_borda_vs_sdr']:+.2f} (p={la['p_sdr']:.3g})")
        print(f"  准则选出 → {la['pick_id']}")
        print(f"     金标 HR_MAE={la['pick_gold']['hr_mae_bpm']:.2f} / SDR={la['pick_gold']['sdr_db']:.2f}");
        print(f"  金标最优 → {la['best_id']}")
        print(f"     金标 HR_MAE={la['best_gold']['hr_mae_bpm']:.2f} / SDR={la['best_gold']['sdr_db']:.2f}")
        print(f"  是否同一候选 = {la['pick_is_best']}；差 {la['pick_hr_gap_vs_best']:+.2f} BPM；"
              f"准则选的候选在金标里的秩 = {la['gold_rank_of_pick']}/{la['n_candidates']}")
        sm = lb["strategies_median_hr"]
        print(f"[Level B 逐样本选基] 各策略 HR_MAE 中位数：")
        print(f"  逐样本准则选 = {sm['criteria_pick(persample)']:.2f}"
              f"   准则选出的单一配置(可行) = {sm['criteria_fixed(Level A 单一配置)']:.2f}"
              f"   oracle逐样本 = {sm['oracle_per_sample']:.2f}")
        print(f"  金标知情上界 = {sm['global_best(金标知情上界)']:.2f}"
              f"   不选(全体中位) = {sm['no_selection(median_over_cands)']:.2f}"
              f"   最差 = {sm['worst_candidate']:.2f}")
        print(f"  命中 oracle = {lb['hit_oracle_rate']*100:.0f}%   "
              f"近命中(≤1BPM) = {lb['near_oracle_rate(<=1BPM)']*100:.0f}%   "
              f"比'不选'更差 = {lb['pick_worse_than_no_selection_rate']*100:.0f}%   "
              f"比'单一配置'更差 = {lb['pick_worse_than_fixed_rate']*100:.0f}%")

    # ---- 4) 鲁棒性：单准则 vs 四准则组合（仅在 test 上做 Level A） ----
    robustness = []
    print(f"\n[鲁棒性] Level A 选基：单准则 vs 组合（在 train 上选、在 test 上评，无泄漏）")
    print(f"   {'准则组合':22s}{'选出':16s}{'金标HR':>8s}{'金标秩':>8s}{'距最优':>9s}{'ρ(Borda,HR)':>13s}")
    combos = [[k] for k in kept_train] + ([list(kept_train)] if len(kept_train) > 1 else [])
    for ks in combos:
        la = level_a_holdout(df, train_ids, test_ids, args.eval_scope, ks)
        short = la["pick_id"].split("|")[1] + "|" + la["pick_id"].split("|")[2]
        print(f"   {'+'.join(ks):22s}{short:16s}{la['pick_gold']['hr_mae_bpm']:8.2f}"
              f"{str(la['gold_rank_of_pick'])+'/'+str(la['n_candidates']):>8s}"
              f"{la['pick_hr_gap_vs_best']:+9.2f}{la['rho_borda_vs_hr']:+13.2f}")
        robustness.append({"criteria": ks, "pick_id": la["pick_id"],
                           "pick_hr": la["pick_gold"]["hr_mae_bpm"],
                           "gold_rank": la["gold_rank_of_pick"],
                           "gap_vs_best": la["pick_hr_gap_vs_best"],
                           "rho_borda_hr": la["rho_borda_vs_hr"]})

    summary = {
        "raw": str(args.raw), "split": args.split,
        "select_scope": args.select_scope, "eval_scope": args.eval_scope,
        "n_samples": len(all_ids), "n_train": len(train_ids), "n_test": len(test_ids),
        "kept_train": kept_train, "kept_full": kept_full,
        "select_detail_train": detail_train, "select_detail_full": detail_full,
        "results": results, "robustness_level_a_test": robustness,
    }
    (out_dir / "closedloop_criteria.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for tag, res in results.items():
        la, lb = res["level_a"], res["level_b"]
        rows.append({
            "set": tag, "scope": la["scope"], "n_cand": la["n_candidates"],
            "L_A_rho_borda_hr": la["rho_borda_vs_hr"], "L_A_rho_borda_sdr": la["rho_borda_vs_sdr"],
            "L_A_pick_id": la["pick_id"], "L_A_pick_hr": la["pick_gold"]["hr_mae_bpm"],
            "L_A_pick_sdr": la["pick_gold"]["sdr_db"],
            "L_A_best_id": la["best_id"], "L_A_best_hr": la["best_gold"]["hr_mae_bpm"],
            "L_A_pick_is_best": la["pick_is_best"], "L_A_pick_hr_gap": la["pick_hr_gap_vs_best"],
            "L_A_gold_rank_of_pick": la["gold_rank_of_pick"],
            **{f"L_B_{k}": v for k, v in lb["strategies_median_hr"].items()},
            "L_B_hit_oracle": lb["hit_oracle_rate"],
            "L_B_near_oracle": lb["near_oracle_rate(<=1BPM)"],
            "L_B_worse_than_no_sel": lb["pick_worse_than_no_selection_rate"],
        })
    pd.DataFrame(rows).to_csv(out_dir / "closedloop_criteria.csv", index=False, encoding="utf-8")
    print(f"\n[loop] 输出：closedloop_criteria.csv / closedloop_criteria.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
