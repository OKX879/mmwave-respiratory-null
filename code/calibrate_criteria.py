#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""无参考准则校准（对应《小波基系统选择_实验计划.md》§6 / 待办 T1）。

要回答的问题
------------
候选矩阵给出的是【金标排名】（合成数据有干净源，可算 hr_mae / sdr / corr）。
真实场景没有金标，只能用无参考准则去选基。那么在**知道答案**的合成数据上，
无参考准则能不能预测出金标排名？——这就是本脚本要检验的东西，
也是把工作从"填表"升级为"方法学贡献"、并规避"自证陷阱"的唯一办法。

做法
----
对每个准则 c 与每个金标指标 g，算 Spearman 秩相关 rho(c, g)：

- **候选级（scope=all / thr:*）**：先把每个候选在 60 个样本上的中位数聚合成
  一行（36 个候选 → 36 个点），再算 rho。回答"这个准则能否跨候选排序"。
- **样本级（scope=per-sample）**：在每个样本内部，跨 36 个候选算一次 rho，
  再对所有样本取均值。回答"对某一条具体信号，这个准则能否挑出更好的基"。

方向判定（关键，防止"只看绝对值"）
----------------------------------
每个准则声明 direction（up/down），每个金标指标也有自己的"好"方向：
    rho 期望符号 = +1（同向）  或  -1（反向）
    up   vs up   → 期望 +
    up   vs down → 期望 -
    down vs up   → 期望 -
    down vs down → 期望 +
只有【符号正确 + |rho| 显著（p<0.05）+ |rho|>=0.4】才算"通过"。
符号不对 = 该准则会把基选反，无论 |rho| 多大都要淘汰。

为什么要分 scope
----------------
F1（已确立结论）指出：不阈值时正交小波的 wavedec→waverec 是恒等变换，
18 个 `none` 候选结果**逐位相同**，基完全不起作用。若在全集上算 rho，
这 18 个并列点会稀释真实的"选基"信号。因此把 **universal-soft 子集
（18 候选 = 6 基 × 3 层，此时基才通过阈值收缩真正起作用）作为主判据**，
全集结果作为对照一并输出。

输出
----
<out>/criteria_calibration.csv    长表：criterion × scope × gold → rho / p / 判定
<out>/criteria_calibration.json   汇总（保留 / 淘汰清单及理由）

用法
----
    python calibrate_criteria.py --raw matrix_out/matrix_raw.csv --out-dir matrix_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from criteria_lib import CRITERIA

# 金标指标：(列名, 好方向, 展示名)
GOLD = [
    ("hr_mae_bpm", "down", "HR_MAE_BPM"),
    ("sdr_db", "up", "SDR_dB"),
    ("corr_card", "up", "corr_card"),
]

RHO_MIN = 0.40      # |rho| 门槛
P_MAX = 0.05        # 显著性门槛

_NAN = float("nan")


# ---------------------------------------------------------------------------
def expected_sign(crit_dir: str, gold_dir: str) -> int:
    """准则方向与金标好方向相同 → rho 期望为正，否则为负。"""
    return 1 if crit_dir == gold_dir else -1


def safe_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """过滤非有限值后算 Spearman；退化情况返回 (nan, nan, n)。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < 4:
        return _NAN, _NAN, n
    va, vb = a[m], b[m]
    if np.ptp(va) == 0 or np.ptp(vb) == 0:      # 常量 → 秩相关无定义
        return _NAN, _NAN, n
    rho, p = spearmanr(va, vb)
    if not np.isfinite(rho):
        return _NAN, _NAN, n
    return float(rho), float(p), n


def parse_candidate(cid: str) -> dict:
    """从候选 id 还原元数据（用于分 scope）。"""
    parts = cid.split("|")
    if parts[0] == "dwt":
        return {"kind": "dwt", "wavelet": parts[1], "level": parts[2].lstrip("L"),
                "threshold": parts[3], "boundary": parts[4]}
    return {"kind": "cwt", "wavelet": parts[1], "level": "-1",
            "threshold": "none", "boundary": "n/a"}


# ---------------------------------------------------------------------------
def calibrate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    crits = [c for c in CRITERIA]
    crit_keys = [c.key for c in crits]
    crit_dir = {c.key: c.direction for c in crits}

    meta = pd.DataFrame([parse_candidate(c) for c in df["candidate_id"].unique()],
                        index=df["candidate_id"].unique())

    # ---- 候选级聚合：每候选一行（跨样本中位数）----
    agg = df.groupby("candidate_id")[crit_keys + [g[0] for g in GOLD]].median()
    agg = agg.join(meta)

    scopes: dict[str, pd.DataFrame] = {"all": agg}
    # 按阈值规则分子集（基只在收缩时才起作用；none 组是常量，单独标注）
    for thr in sorted(agg["threshold"].unique()):
        sub = agg[agg["threshold"] == thr]
        scopes[f"thr:{thr}"] = sub

    rows: list[dict] = []

    # ---- 候选级 rho ----
    for scope, sub in scopes.items():
        for ck in crit_keys:
            for gk, gdir, gname in GOLD:
                rho, p, n = safe_spearman(sub[ck], sub[gk])
                es = expected_sign(crit_dir[ck], gdir)
                sign_ok = bool(np.isfinite(rho) and np.sign(rho) == es)
                passed = bool(sign_ok and np.isfinite(p) and p < P_MAX and abs(rho) >= RHO_MIN)
                rows.append({"criterion": ck, "group": _group_of(ck),
                             "direction": crit_dir[ck], "scope": scope, "gold": gname,
                             "gold_dir": gdir, "expected_sign": es, "n": n,
                             "rho": rho, "p": p, "sign_ok": sign_ok, "pass": passed})

    # ---- 样本级 rho（跨候选，逐样本）----
    per_scope_stats: list[dict] = []
    by_sample = df.groupby("sample_id")
    for ck in crit_keys:
        for gk, gdir, gname in GOLD:
            es = expected_sign(crit_dir[ck], gdir)
            rhos, sign_oks, passed = [], [], []
            for _, gdf in by_sample:
                rho, p, n = safe_spearman(gdf[ck], gdf[gk])
                if not np.isfinite(rho):
                    continue
                rhos.append(rho)
                so = (np.sign(rho) == es)
                sign_oks.append(float(so))
                passed.append(float(so and np.isfinite(p) and p < P_MAX and abs(rho) >= RHO_MIN))
            if rhos:
                per_scope_stats.append({
                    "criterion": ck, "group": _group_of(ck), "direction": crit_dir[ck],
                    "scope": "per-sample", "gold": gname, "gold_dir": gdir,
                    "expected_sign": es, "n": len(rhos),
                    "rho": float(np.mean(rhos)),          # 均值 rho
                    "p": _NAN,                            # 逐样本 p 不合并
                    "sign_ok": bool(np.mean(sign_oks) >= 0.5),
                    "pass": bool(np.mean(passed) >= 0.5),
                    "rho_median": float(np.median(rhos)),
                    "sign_ok_frac": float(np.mean(sign_oks)),
                    "pass_frac": float(np.mean(passed)),
                })
    rows.extend(per_scope_stats)
    out = pd.DataFrame(rows)

    # ---- 判定：主判据 = thr:universal-soft（基真正起作用）+ 候选级 ----
    primary_scope = "thr:universal-soft"
    verdict_rows: list[dict] = []
    for c in crits:
        prim = out[(out.criterion == c.key) & (out.scope == primary_scope)]
        full = out[(out.criterion == c.key) & (out.scope == "all")]
        ps = out[(out.criterion == c.key) & (out.scope == "per-sample")]
        prim_pass = prim[prim["pass"]]["gold"].tolist() if not prim.empty else []
        full_pass = full[full["pass"]]["gold"].tolist() if not full.empty else []
        ps_pass = ps[ps["pass"]]["gold"].tolist() if not ps.empty else []
        if c.report_only or c.direction == "n/a":
            verdict = "report_only"
            reason = "非单调，禁止进评分（设计约定）"
        elif prim_pass:
            verdict = "KEEP"
            reason = f"主判据(universal-soft)通过：{', '.join(prim_pass)}"
        elif full_pass or ps_pass:
            verdict = "WATCH"
            reason = ("仅对照组通过：" +
                      ("全集 " + ", ".join(full_pass) + "；" if full_pass else "") +
                      ("逐样本 " + ", ".join(ps_pass) if ps_pass else ""))
        else:
            fin = prim[np.isfinite(prim["rho"].to_numpy(dtype=float))] if not prim.empty else prim
            if fin.empty:
                verdict = "DROP"
                reason = "本矩阵未提供该准则所需输入（rho 恒为 nan）"
            else:
                flipped = fin[(~fin["sign_ok"]) & (fin["rho"].abs() >= RHO_MIN)]
                if len(flipped):
                    verdict = "DROP"
                    reason = ("符号相反 → 反预测器："
                              + ", ".join(f"{g}(rho={r:+.2f})" for g, r in
                                          zip(flipped["gold"], flipped["rho"])))
                else:
                    worst = float(fin["rho"].abs().max())
                    verdict = "DROP"
                    reason = f"主判据无一通过（最大 |rho|={worst:.2f} < {RHO_MIN} 或符号不符/不显著）"
        # 记录主判据三个金标上的 rho，便于人工复核
        detail = {}
        for gk, gdir, gname in GOLD:
            r_prim = prim[prim.gold == gname]["rho"]
            r_full = full[full.gold == gname]["rho"]
            detail[gname] = {
                "rho_universal_soft": float(r_prim.iloc[0]) if len(r_prim) else _NAN,
                "rho_all": float(r_full.iloc[0]) if len(r_full) else _NAN,
            }
        verdict_rows.append({"criterion": c.key, "name": c.name, "group": c.group,
                             "direction": c.direction, "verdict": verdict,
                             "reason": reason, "detail": detail})

    summary = {
        "raw_csv": None,
        "rho_min": RHO_MIN, "p_max": P_MAX,
        "primary_scope": primary_scope,
        "n_candidates_all": int(len(agg)),
        "n_candidates_universal_soft": int(len(agg[agg.threshold == "universal-soft"])),
        "verdicts": verdict_rows,
    }
    return out, summary


def _group_of(key: str) -> str:
    for c in CRITERIA:
        if c.key == key:
            return c.group
    return "?"


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="无参考准则校准（Spearman rho vs 金标）",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--raw", type=str, default="matrix_out/matrix_raw.csv")
    p.add_argument("--out-dir", type=str, default="matrix_out")
    args = p.parse_args(argv)

    raw = Path(args.raw)
    if not raw.exists():
        print(f"[calib] 找不到 {raw}，请先运行 matrix_eval.py")
        return 1
    df = pd.read_csv(raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table, summary = calibrate(df)
    summary["raw_csv"] = str(raw)

    csv_path = out_dir / "criteria_calibration.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8")
    json_path = out_dir / "criteria_calibration.json"
    json.dump(summary, open(json_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- 控制台呈现：表 C（候选级，主判据 universal-soft 与全集对照）----
    print(f"[calib] 数据：{raw.name}  候选={summary['n_candidates_all']}  "
          f"universal-soft 子集={summary['n_candidates_universal_soft']}")
    print(f"[calib] 判定门槛：|rho|>={RHO_MIN} 且 p<{P_MAX} 且 符号正确；主判据 scope={summary['primary_scope']}\n")

    prim = table[(table.scope == "thr:universal-soft") & (table.gold.isin([g[2] for g in GOLD]))]
    full = table[(table.scope == "all") & (table.gold.isin([g[2] for g in GOLD]))]
    ps = table[table.scope == "per-sample"]

    print("表 C　无参考准则校准（rho；* = 符号正确，+ = 通过门槛）")
    hdr = (f"{'准则':6s}{'组':4s}{'向':5s}"
           f"{'ρ_HR(soft)':>12s}{'ρ_SDR(soft)':>13s}{'ρ_corr(soft)':>13s}"
           f"{'ρ_HR(all)':>11s}{'ρ_SDR(all)':>12s}{'ρ_corr(all)':>12s}"
           f"   判定")
    print(hdr)
    print("-" * len(hdr))

    def _cell(sub, ck, gname):
        r = sub[(sub.criterion == ck) & (sub.gold == gname)]
        if not len(r):
            return "     —"
        rho = float(r.iloc[0]["rho"])
        if not np.isfinite(rho):
            return "     —"
        ok = bool(r.iloc[0]["sign_ok"])
        pas = bool(r.iloc[0]["pass"])
        mark = "+" if pas else ("*" if ok else " ")
        return f"{rho:+8.2f}{mark}"

    vmap = {v["criterion"]: v for v in summary["verdicts"]}
    for c in CRITERIA:
        ck = c.key
        cells = "".join(_cell(prim, ck, g[2]) for g in GOLD)
        cells_full = "".join(_cell(full, ck, g[2]) for g in GOLD)
        v = vmap[ck]["verdict"]
        print(f"{ck:6s}{c.group:4s}{c.direction:5s}{cells}{cells_full}   {v}")

    print("\n逐样本（跨候选，均值 rho，符号正确率 / 通过率）")
    for c in CRITERIA:
        r = ps[ps.criterion == c.key]
        if r.empty:
            continue
        s = "  ".join(
            f"{x['gold']}={float(x['rho']):+.2f}"
            f"({float(x['sign_ok_frac'])*100:.0f}%/{float(x['pass_frac'])*100:.0f}%)"
            for _, x in r.iterrows()
        )
        print(f"  {c.key:6s}{s}")

    print("\n判定汇总")
    for v in summary["verdicts"]:
        print(f"  [{v['verdict']:11s}] {v['criterion']:5s} {v['name']:12s} — {v['reason']}")

    anti = [v["criterion"] for v in summary["verdicts"] if "反预测器" in v["reason"]]
    if anti:
        print(f"\n[calib] 反预测器（|rho| 很大但方向相反，用它选基会把基选反）：{', '.join(anti)}")
        print("[calib] 这些准则的物理假设需要重新审视，或在论文中如实记录为'校准失败'。")

    n_keep = sum(1 for v in summary["verdicts"] if v["verdict"] == "KEEP")
    print(f"\n[calib] 保留 {n_keep} 个准则（主判据通过）；输出：{csv_path.name} / {json_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
