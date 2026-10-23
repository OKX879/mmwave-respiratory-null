#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rc2（跨距离格心搏相干）在真实数据上的首次校准。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import detrend

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from criteria_lib import evaluate_all, estimate_hr_psd, DISPLAY_BAND   # noqa: E402
from matrix_eval import decompose_dwt                                  # noqa: E402
from real_eval import BASES, LEVELS, THRESHOLDS, FS                    # noqa: E402

DATA = Path(r"C:\Users\yxh\Desktop\capsule-4604022-data\Minimum_dataset")
REAL = HERE / "real_data"
G = "hr_mae_bpm"


# ---------------------------------------------------------------------------
def load_cube(stem: str) -> np.ndarray:
    """返回 (channels, angles, bins, frames) 的复数立方体。"""
    with h5py.File(DATA / f"{stem}.mat", "r") as f:
        raw = f[f["data_all"][0, 0]][:]
    return (raw["real"] + 1j * raw["imag"]).astype(np.complex128)


def bin_displacement(z: np.ndarray, ch_axis_bin: int, angle: int,
                     t0: int, n: int) -> np.ndarray:
    """单距离格的位移序列（与生产链同族但不等价的融合，见文件头）。"""
    seg = z[:, angle, ch_axis_bin, :].copy()           # (ch, T)
    seg = seg - seg.mean(axis=1, keepdims=True)        # MTI：逐通道去静态
    wgt = np.abs(seg).mean(axis=1, keepdims=True)      # 幅度权重
    fused = (seg * wgt).sum(axis=0)
    ph = np.unwrap(np.angle(fused))
    ph = detrend(ph)
    seg_out = ph[t0:t0 + n]
    if len(seg_out) < n:                               # 越界保护
        seg_out = np.pad(seg_out, (0, n - len(seg_out)), mode="edge")
    return seg_out


def build_pool() -> list[dict]:
    out = []
    for b in BASES:
        for lv in LEVELS:
            for th in THRESHOLDS:
                out.append({"kind": "dwt", "wavelet": b, "level": lv,
                            "threshold": th, "boundary": "symmetric",
                            "separation": "bandpass",
                            "id": f"dwt|{b}|L{lv}|{th}|symmetric|bandpass"})
    return out


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="rc2 真实数据校准（capsule，中心格±1）")
    ap.add_argument("--out", type=str, default=str(HERE / "real_rc2"))
    ap.add_argument("--only", type=int, nargs="*", default=None)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    npzs = sorted(REAL.glob("patient_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if args.only:
        keep = {f"patient_{i}" for i in args.only}
        npzs = [p for p in npzs if p.stem in keep]

    # ---- 逐例构造三 bin 位移 ----
    cases = []
    for p in npzs:
        d = np.load(p, allow_pickle=True)
        c = int(d["center_bin"]); a = int(d["angle"]); t0 = int(d["t0_wide"])
        n = len(d["wide"]); gold = float(d["gold_hr"])
        if c - 1 < 0 or c + 1 >= 30 or not np.isfinite(gold):
            print(f"[rc2] 跳过 {p.stem}（bin 越界或金标无效）")
            continue
        z = load_cube(p.stem)
        if z.shape[1] <= a:
            print(f"[rc2] 跳过 {p.stem}（angle 越界）")
            continue
        bins = [bin_displacement(z, c + o, a, t0, n) for o in (-1, 0, 1)]
        cases.append({"sid": int(p.stem.split("_")[1]), "bins": bins,
                      "gold": gold, "center_bin": c,
                      "corr_mid_vs_wide": float(np.corrcoef(
                          (lambda v: (v - v.mean()) / (v.std() + 1e-12))(bins[1]),
                          (lambda v: (v - v.mean()) / (v.std() + 1e-12))(
                              np.asarray(d["wide"], float)))[0, 1])})
    print(f"[rc2] {len(cases)} 例可用；中位 bin 与 wide 相关 = "
          f"{np.median([c['corr_mid_vs_wide'] for c in cases]):.3f}")

    pool = build_pool()
    print(f"[rc2] 候选 {len(pool)}（symmetric 主边界）；评估量 {len(pool)*len(cases)*3} 次分解")

    rows, t0w = [], time.time()
    for ci, cand in enumerate(pool):
        for rec in cases:
            cards = []
            ok = True
            for j, x in enumerate(rec["bins"]):
                dd = decompose_dwt(x, cand["wavelet"], cand["level"], cand["threshold"],
                                   cand["boundary"], FS, separation="bandpass",
                                   need_full=False)
                if dd is None:
                    ok = False
                    break
                cards.append(dd["card"])
            if not ok:
                continue
            crit = evaluate_all({"card": cards[1], "card_bins": cards, "mix": rec["bins"][1]},
                                FS)
            hr = estimate_hr_psd(cards[1], FS, band=DISPLAY_BAND)
            rows.append({"candidate_id": cand["id"], "sample_id": rec["sid"],
                         "rc2": crit.get("rc2", float("nan")),
                         G: abs(hr - rec["gold"]) if np.isfinite(hr) else float("nan"),
                         "wavelet": cand["wavelet"], "level": cand["level"],
                         "threshold": cand["threshold"]})
        if (ci + 1) % 50 == 0 or ci == 0:
            el = time.time() - t0w
            print(f"  [{ci+1}/{len(pool)}] rows={len(rows)} ({el:.0f}s, "
                  f"~{el/(ci+1)*(len(pool)-ci-1):.0f}s left)")

    raw = pd.DataFrame(rows)
    if raw.empty:
        print("[rc2] 无有效结果")
        return 1
    raw.to_csv(out / "rc2_raw.csv", index=False, encoding="utf-8")

    meta_cols = ["wavelet", "level", "threshold"]
    agg = raw.groupby("candidate_id")[["rc2", G]].median()
    agg = agg.join(raw.groupby("candidate_id")[meta_cols].first())
    agg.reset_index().to_csv(out / "rc2_results.csv", index=False, encoding="utf-8")

    # ---- 校准：两种聚合方式都必须报 ----
    # ① 候选级（跨例中位数）—— 与其它准则的主口径一致
    # ② 逐例内（每例跨 375 候选算 ρ，再统计）—— rc2 的绝对值随病例而变
    #    （跨例差异 0.725 远大于候选间差异），候选级中位数会把它压平，
    #    所以只看①会误判；②是 rc2 这类"病例相关"准则更合适的口径。
    from calibrate_criteria import safe_spearman, expected_sign
    es = expected_sign("up", "down")

    rho, p, n = safe_spearman(agg["rc2"], agg[G])
    sign_ok = bool(np.isfinite(rho) and np.sign(rho) == es)
    passed = bool(sign_ok and np.isfinite(p) and p < 0.05 and abs(rho) >= 0.40)

    per_case = []
    for sid, g in raw.groupby("sample_id"):
        r1, p1, n1 = safe_spearman(g["rc2"], g[G])
        per_case.append({"sample_id": int(sid), "rho": r1, "p": p1, "n": n1,
                         "sign_ok": bool(np.isfinite(r1) and np.sign(r1) == es)})
    pc = pd.DataFrame(per_case)
    pc_ok = pc[np.isfinite(pc["rho"])]
    pc_mean = float(pc_ok["rho"].mean())
    pc_median = float(pc_ok["rho"].median())
    pc_sign_frac = float(pc_ok["sign_ok"].mean())
    pc_strong = int((pc_ok["rho"].abs() >= 0.40).sum())

    print(f"\n=== rc2 校准（capsule {len(cases)} 例，候选 {n} 个） ===")
    print(f"  ① 候选级（跨例中位）：ρ = {rho:+.3f}  p = {p:.3g}  "
          f"方向正确 = {sign_ok}  → {'KEEP' if passed else 'DROP'}")
    print(f"  ② 逐例内（每例跨 {int(pc_ok['n'].median())} 候选）："
          f"mean ρ = {pc_mean:+.3f}  中位 {pc_median:+.3f}  std {pc_ok['rho'].std():.3f}")
    print(f"     符号正确 {pc_sign_frac*100:.0f}%（{int(pc_ok['sign_ok'].sum())}/{len(pc_ok)}）；"
          f"|ρ|≥0.4 的例数 {pc_strong}/{len(pc_ok)}")
    print(f"  rc2 取值：[{agg['rc2'].min():.3f}, {agg['rc2'].max():.3f}]，"
          f"中位 {agg['rc2'].median():.3f}；候选级 IQR = "
          f"{agg['rc2'].quantile(.75)-agg['rc2'].quantile(.25):.4f}")

    # 饱和诊断：跨例差异 vs 例内差异
    pc_spread = raw.groupby("sample_id")["rc2"].agg(lambda v: v.max() - v.min())
    case_med = raw.groupby("sample_id")["rc2"].median()
    print(f"  饱和诊断：例内跨度 中位 {pc_spread.median():.3f}"
          f"  vs  跨例跨度 {case_med.max()-case_med.min():.3f}"
          f"  → {'病例差异压过候选差异（需逐例口径）' if case_med.max()-case_med.min() > pc_spread.median() else '候选差异为主'}")

    # 三分位对比
    q1, q3 = agg[G].quantile(1 / 3), agg[G].quantile(2 / 3)
    good = agg[agg[G] <= q1]["rc2"]
    bad = agg[agg[G] >= q3]["rc2"]
    print(f"  好候选（HR_MAE ≤ {q1:.2f}，n={len(good)}）rc2 中位 = {good.median():.3f}"
          f"  vs 坏候选（≥ {q3:.2f}）{bad.median():.3f}"
          f"  → 差 {abs(good.median()-bad.median()):.3f}")

    import json
    (out / "rc2_calibration.json").write_text(json.dumps({
        "method": "center_bin +/-1, amplitude-weighted complex fusion per bin, "
                  "MTI, unwrap, detrend; NOT the production chain's dynamic tracking",
        "n_cases": len(cases), "n_candidates": int(n),
        "median_corr_mid_vs_wide": float(np.median([c["corr_mid_vs_wide"] for c in cases])),
        "rho_vs_hr_mae": float(rho), "p": float(p), "sign_ok": sign_ok,
        "verdict": "KEEP" if passed else "DROP",
        "per_case": {"rho_mean": pc_mean, "rho_median": pc_median,
                     "rho_std": float(pc_ok["rho"].std()),
                     "sign_correct_frac": pc_sign_frac,
                     "n_cases_with_abs_rho_ge_0.4": pc_strong,
                     "n_cases": int(len(pc_ok)), "detail": per_case},
        "saturation": {"within_case_spread_median": float(pc_spread.median()),
                       "between_case_spread": float(case_med.max() - case_med.min())},
        "rc2_range": [float(agg["rc2"].min()), float(agg["rc2"].max())],
        "rc2_median": float(agg["rc2"].median()),
        "rc2_iqr": float(agg["rc2"].quantile(.75) - agg["rc2"].quantile(.25)),
        "good_tercile_rc2_median": float(good.median()),
        "bad_tercile_rc2_median": float(bad.median()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[rc2] 输出：{out}/rc2_raw.csv / rc2_results.csv / rc2_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
