#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实数据上的准则校准 + 闭环：合成结论能否迁移？"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from calibrate_criteria import GOLD, RHO_MIN, P_MAX, calibrate, safe_spearman, expected_sign  # noqa: E402
from closedloop_criteria import borda, select_criteria, level_a_holdout, parse_candidate     # noqa: E402
from criteria_lib import CRITERIA                                                           # noqa: E402
from decision_presentation import eta_squared, KEEP_CRITERIA, ANTI_CRITERIA                  # noqa: E402

GOLD_HR = "hr_mae_bpm"
_DIR = {c.key: c.direction for c in CRITERIA}


def load_real(path: Path, main_boundary_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["threshold"] = [parse_candidate(c)["threshold"] for c in df["candidate_id"]]
    if main_boundary_only:
        df = df[df["boundary"] == "symmetric"].copy()
    # 真实数据无干净源 → 合成侧的 sdr_db / corr_card 不存在。补 NaN 列，
    # 使 calibrate_criteria / closedloop_criteria 的既有接口可直接复用
    # （safe_spearman 对 NaN 返回 nan → 该金标自动不参与判定，不会伪造数据）。
    for col in ("sdr_db", "corr_card"):
        if col not in df.columns:
            df[col] = np.nan
    return df


def rho_table(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """候选级 ρ(准则, HR_MAE)：scope='all' 或 'thr:xxx'。"""
    sub = df if scope == "all" else df[df["threshold"] == scope[4:]]
    agg = sub.groupby("candidate_id").median(numeric_only=True)
    rows = []
    for c in CRITERIA:
        if c.key not in agg.columns:
            continue
        rho, p, n = safe_spearman(agg[c.key], agg[GOLD_HR])
        es = expected_sign(c.direction, "down")      # 金标 HR_MAE 是 down
        ok = bool(np.isfinite(rho) and np.sign(rho) == es)
        passed = bool(ok and np.isfinite(p) and p < P_MAX and abs(rho) >= RHO_MIN)
        rows.append({"criterion": c.key, "group": c.group, "direction": c.direction,
                     "rho_real": rho, "p_real": p, "n_cand": n,
                     "sign_ok": ok, "pass_real": passed,
                     "verdict_real": ("report_only" if (c.report_only or c.direction == "n/a")
                                      else "KEEP" if passed
                                      else "DROP" if not ok and np.isfinite(rho) and abs(rho) >= RHO_MIN
                                      else "WATCH/weak")})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="T5-3 真实数据校准 + 闭环")
    ap.add_argument("--real", type=str, default=str(HERE / "real_matrix" / "real_matrix_raw.csv"))
    ap.add_argument("--syn-raw", type=str, default=str(HERE / "matrix_out_t4" / "matrix_raw.csv"))
    ap.add_argument("--group-col", type=str, default=None,
                    help="留出时按该列分组（如 emo 的 'subject'）→ 做 leave-one-SUBJECT-out，"
                         "避免把同一受试者的多个 trial 当独立样本（伪重复）")
    ap.add_argument("--out", type=str, default=str(HERE / "real_matrix"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_real(Path(args.real))
    n_cases = df["sample_id"].nunique()
    n_cand = df["candidate_id"].nunique()
    print(f"[T5-3] 真实矩阵：{n_cases} 例 × {n_cand} 候选（主边界 symmetric）")
    print(f"       金标 = HR MAE vs ECG；SDR/corr_card 在真实数据上不可得（无干净源）")

    # ---------------- Q1 层数边界（含配对检验） ----------------
    print("\n=== Q1 结构性安全边界 L ≤ 4（真实数据） ===")
    g = df[df["level_over_max"] == 0]
    print(f"  {'L':>3s}{'aL上界/Hz':>11s}{'覆盖心搏带':>11s}{'中位HR_MAE':>12s}")
    for lv in sorted(g["level"].unique()):
        aL = (100.0 / 2) / 2 ** lv
        med = g[(g["level"] == lv) & (g["threshold"] != "none")]["hr_mae_bpm"].median()
        print(f"  {lv:>3d}{aL:>11.4f}{('是' if aL >= 3.0 else '否'):>11s}{med:>12.2f}")
    piv = (g.pivot_table(index=["wavelet", "level"], columns="threshold",
                         values="hr_mae_bpm", aggfunc="median"))
    if "none" in piv.columns:
        print("  配对 Δ（阈值化 − none，中位）：")
        for lv in sorted(g["level"].unique()):
            sub = piv.xs(lv, level="level")
            ds = {t: float(np.nanmedian(sub[t] - sub["none"]))
                  for t in ["universal-soft", "sure", "minimax"] if t in sub.columns}
            print(f"    L{lv:>2d}  " + "  ".join(f"Δ({t})={v:+7.2f}" for t, v in ds.items()))

    # ---------------- 附加：边界模式配对检验 ----------------
    dfb = pd.read_csv(args.real)
    dfb["threshold"] = [parse_candidate(c)["threshold"] for c in dfb["candidate_id"]]
    l4 = dfb[(dfb["level"] == 4) & (dfb["level_over_max"] == 0)]
    pivb = l4.pivot_table(index=["wavelet", "threshold"], columns="boundary",
                          values="hr_mae_bpm", aggfunc="median")
    print("\n=== 附加：边界模式配对检验（L=4，真实数据） ===")
    if {"zero", "symmetric", "periodic"}.issubset(set(pivb.columns)):
        for b in ["zero", "periodic"]:
            d = float(np.nanmedian(pivb[b] - pivb["symmetric"]))
            print(f"  Δ({b} − symmetric) 中位 = {d:+.4f} BPM")
        print("  （合成侧结论：边界零效应，Δ 中位 0.000–0.005 → 真实侧是否复现见上）")
    # η² 的分析单位必须是【候选】（与合成侧 matrix_results.csv 同口径）：
    # 若直接拿候选×样本的 raw 行去算，20 例之间的巨大难度差异会淹没候选间差异，
    # η² 会被稀释到 0.02 量级（实测），与合成侧不可比。
    cand = g.groupby("candidate_id").agg({
        "level": "first", "threshold": "first", "wavelet": "first", "boundary": "first",
        "hr_mae_bpm": "median"})
    y = np.log1p(cand["hr_mae_bpm"].to_numpy(float))
    etab = {f: eta_squared(y, cand[f].to_numpy())
            for f in ["level", "threshold", "wavelet", "boundary"]}
    print("  因子 η²（真实，候选级，主边界内，n=%d）: " % len(cand) +
          "  ".join(f"{k}={v:.4f}" for k, v in etab.items()))
    print("  合成侧对照: level=0.6167  threshold=0.1909  wavelet=0.0304  boundary=0.0000"
          "  → 看**次序**是否复现，量级只作参考")

    # ---------------- Q2/Q3 准则校准 ----------------
    print("\n=== Q2/Q3 无参考准则校准（真实 vs 合成） ===")
    rt = rho_table(df, "thr:universal-soft")
    ra = rho_table(df, "all")
    syn = pd.read_csv(HERE / "matrix_out_t4" / "criteria_calibration.csv")
    syn_soft = syn[(syn.scope == "thr:universal-soft") & (syn.gold == "HR_MAE_BPM")]
    syn_rho = {r["criterion"]: float(r["rho"]) for _, r in syn_soft.iterrows()}
    print(f"  {'准则':6s}{'组':4s}{'向':5s}{'ρ_HR 真实(soft)':>16s}{'ρ_HR 合成(soft)':>16s}"
          f"{'真实判定':>12s}{'合成判定':>12s}")
    merged = []
    for _, r in rt.iterrows():
        k = r["criterion"]
        s_rho = syn_rho.get(k, float("nan"))
        s_row = syn_soft[syn_soft.criterion == k]
        s_v = str(s_row.iloc[0]["pass"]) if len(s_row) else ""
        s_verdict = "KEEP" if s_v == "True" else ("WATCH/weak")
        print(f"  {k:6s}{r['group']:4s}{r['direction']:5s}"
              f"{r['rho_real']:>+16.2f}{s_rho:>+16.2f}{r['verdict_real']:>12s}{s_verdict:>12s}")
        merged.append({"criterion": k, "group": r["group"],
                       "rho_real_soft": r["rho_real"], "rho_syn_soft": s_rho,
                       "verdict_real": r["verdict_real"],
                       "role_in_synthetic_decision": (
                           "KEEP" if k in KEEP_CRITERIA else
                           "ANTI(disabled)" if k in ANTI_CRITERIA else "not_used")})
    pd.DataFrame(merged).to_csv(out / "real_criteria_calibration.csv",
                                index=False, encoding="utf-8")
    rho_hr_all = ra.set_index("criterion")["rho_real"].to_dict()
    print("  全集口径 ρ_real（对照）：" +
          "  ".join(f"{k}={v:+.2f}" for k, v in list(rho_hr_all.items())[:14]))

    # ---------------- Q4 推荐配置在真实数据上的位置 ----------------
    print("\n=== Q4 合成侧推荐配置在真实数据上的位置 ===")
    agg = df.groupby("candidate_id").median(numeric_only=True)
    best_id = str(agg[GOLD_HR].idxmin())
    print(f"  真实金标最优 = {best_id}  HR_MAE={agg.loc[best_id, GOLD_HR]:.3f}"
          f"  beat_F1={agg.loc[best_id].get('beat_f1', float('nan')):.3f}")
    pins = ["dwt|bior3.5|L4|sure|symmetric|bandpass",       # 合成侧金标最优
            "dwt|sym4|L4|universal-soft|symmetric|bandpass",  # 合成侧 T7 推荐
            "dwt|haar|L5|universal-soft|symmetric|bandpass",  # 合成侧危险区（L>=5）
            "dwt|haar|L4|none|symmetric|bandpass"]            # 恒等变换对照
    for pid in pins:
        if pid in agg.index:
            rank = float(agg[GOLD_HR].rank().loc[pid])
            print(f"    {pid:44s} HR_MAE={agg.loc[pid, GOLD_HR]:7.3f}"
                  f"  值差={agg.loc[pid, GOLD_HR]-agg.loc[best_id, GOLD_HR]:+7.3f}"
                  f"  秩={rank:.0f}/{len(agg)}")
        else:
            print(f"    {pid:44s} （不在真实池中）")

    # ---------------- Level A：冻结 KEEP vs 重筛（一次性，无留出） ----------------
    print("\n=== Level A（Borda 选基，候选级；一次性口径=乐观，仅看次序） ===")
    for tag, keys in [("冻结 KEEP（合成侧校准）", list(KEEP_CRITERIA)),
                      ("真实数据重筛", None)]:
        if keys is None:
            ids = sorted(df["sample_id"].unique().tolist())
            keys, _ = select_criteria(df, ids, "thr:universal-soft", RHO_MIN, P_MAX)
        if not keys:
            print(f"  {tag:26s} 未筛出任何准则")
            continue
        agg2 = df.groupby("candidate_id").median(numeric_only=True)
        agg2["borda"] = borda(agg2, keys)
        pick = str(agg2["borda"].idxmax())
        gap = float(agg2.loc[pick, GOLD_HR] - agg2.loc[best_id, GOLD_HR])
        rho, p, _ = safe_spearman(agg2["borda"], agg2[GOLD_HR])
        print(f"  {tag:26s} 准则={'+'.join(keys):28s} → {pick[:44]:44s} "
              f"HR_MAE={agg2.loc[pick, GOLD_HR]:7.3f}  值差={gap:+7.3f}  "
              f"ρ(Borda,HR)={rho:+.2f}")

    # ---------------- 留出（LOO / leave-one-group-out） ----------------
    ids = sorted(df["sample_id"].unique().tolist())
    if args.group_col and args.group_col in df.columns:
        s2g = df.groupby("sample_id")[args.group_col].first().to_dict()
        groups = sorted({s2g[s] for s in ids})
        folds = [[s for s in ids if s2g[s] == g] for g in groups]
        unit = f"leave-one-{args.group_col}-out（{len(groups)} 组）"
    else:
        folds = [[s] for s in ids]
        unit = "LOO（每折 1 例 → 高方差，仅作次序证据）"
    print(f"\n=== 留出复检：{unit} ===")
    for tag, keys in [("冻结 KEEP", list(KEEP_CRITERIA)), ("逐折重筛", None)]:
        rows = []
        for fold in folds:
            tr = [s for s in ids if s not in fold]
            if keys is None:
                kk, _ = select_criteria(df, tr, "thr:universal-soft", RHO_MIN, P_MAX)
            else:
                kk = keys
            if not kk:
                continue
            la = level_a_holdout(df, tr, fold, "all", kk)
            rows.append({"held": str(fold[:2]), "pick": la["pick_id"],
                         "gap": la["pick_hr_gap_vs_best"],
                         "rho": la["rho_borda_vs_hr"]})
        r = pd.DataFrame(rows)
        ok = r[np.isfinite(r["gap"].to_numpy(float))]
        if len(ok):
            print(f"  {tag:12s} 有效折 {len(ok)}/{len(folds)}  "
                  f"值差 中位 {ok['gap'].median():+.3f}  mean {ok['gap'].mean():+.3f}  "
                  f"(max {ok['gap'].max():.3f})")
    print("\n[T5-3] 输出：real_criteria_calibration.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
