#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充分析：漂移补偿相位 + ACC 运动控制。"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from null_crossdataset import (  # noqa: E402
    OUT, PAR_ROOT, R_BIN_PAR, bp_complex, deep_nulls, first_frame_time,
    load_capsule, load_parralejo, parse_bh, phase_B, rayleigh,
)


def slow_mean(z, fs, win_s=10.0):
    w = int(win_s * fs)
    k = np.ones(w) / w
    pad = w // 2
    zr = np.convolve(np.pad(z.real, pad, mode="reflect"), k, "valid")[:len(z)]
    zi = np.convolve(np.pad(z.imag, pad, mode="reflect"), k, "valid")[:len(z)]
    return zr + 1j * zi


def phase_drift(z, fs, win_s=10.0):
    """a(t) 相对 -A_s(t) 的夹角，折叠到 [0,1)。"""
    a = bp_complex(z, fs)
    As = slow_mean(z, fs, win_s)
    psi = np.unwrap(np.angle(a)) - (np.unwrap(np.angle(As)) + np.pi)
    return (psi / (2 * np.pi)) % 1.0


def core(z, fs, bin_lo=0, bin_hi=None):
    zm = z[bin_lo:bin_hi]
    b = bin_lo + int(np.argmax(np.std(zm, axis=1)))
    env = np.abs(z[b])
    pk = deep_nulls(env, fs)
    ph = phase_B(z[b], fs)
    phd = phase_drift(z[b], fs)
    rf, _ = rayleigh(ph[pk])
    rd, _ = rayleigh(phd[pk])
    return b, len(pk), rf, rd, pk, env


def acc_stats(path, fs_radar, i0, i1, n):
    """返回 (窗内去重力 std, 窗外 std)；ACC 与雷达按各自时间轴对齐。"""
    try:
        a = pd.read_csv(path)
    except Exception:
        return np.nan, np.nan
    cols = [c for c in a.columns if c.lower().startswith(("x", "y", "z"))]
    if not cols:
        cols = list(a.columns)[1:4]
    mag = np.sqrt(np.sum(a[cols].to_numpy(dtype=float) ** 2, axis=1))
    mag = mag - np.mean(mag)
    t = np.arange(len(mag)) / (len(mag) / (n / fs_radar))
    inside = mag[(t >= i0 / fs_radar) & (t < i1 / fs_radar)]
    outside = np.concatenate([mag[t < i0 / fs_radar], mag[t >= i1 / fs_radar]])
    return float(np.std(inside)) if len(inside) else np.nan, \
        float(np.std(outside)) if len(outside) else np.nan


def main():
    rows = []
    for i in range(1, 21):
        stem = f"patient_{i}"
        try:
            z, fs, _ = load_capsule(stem)
            b, n, rf, rd, _, _ = core(z, fs)
            rows.append({"record": stem, "bin": b, "n_deep": n,
                         "Rbar_fixed": rf, "Rbar_drift": rd})
        except Exception as e:
            print("cap fail", stem, e)
    pd.DataFrame(rows).to_csv(OUT / "capsule_ext.csv", index=False, encoding="utf-8")

    rows = []
    for i in range(1, 111):
        sid = f"P{i:03d}"
        for cond in ["Sitting/Rest", "Sitting/Post-exercise",
                     "Lying/Rest", "Lying/Post-exercise"]:
            try:
                z, fs = load_parralejo(sid, cond)
                b, n, rf, rd, pk, env = core(z, fs, 1, 7)
                d = PAR_ROOT / sid / cond
                row = {"subject": sid, "condition": cond, "bin": b,
                       "range_m": b * R_BIN_PAR, "n_deep": n,
                       "Rbar_fixed": rf, "Rbar_drift": rd,
                       "acc_in": np.nan, "acc_out": np.nan}
                bhf = d / "non_breathing_ts.csv"
                if bhf.exists():
                    bb, ee = parse_bh(bhf)
                    t0 = first_frame_time(d / "radar_timestamps.csv")
                    i0 = int(round((bb - t0).total_seconds() * fs))
                    i1 = int(round((ee - t0).total_seconds() * fs))
                    i0, i1 = max(0, i0), min(len(env), i1)
                    row["acc_in"], row["acc_out"] = acc_stats(
                        d / "movesense_acc.csv", fs, i0, i1, len(env))
                rows.append(row)
            except Exception as e:
                print("par fail", sid, cond, e)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "parralejo_ext.csv", index=False, encoding="utf-8")

    def agg(d):
        e = d[d["n_deep"] >= 5]
        return {"n": int(len(d)), "eligible": int(len(e)),
                "locked_fixed": int((e["Rbar_fixed"] > 0.4).sum()),
                "locked_drift": int((e["Rbar_drift"] > 0.4).sum()),
                "median_Rfixed": float(e["Rbar_fixed"].median()) if len(e) else np.nan,
                "median_Rdrift": float(e["Rbar_drift"].median()) if len(e) else np.nan}

    summary = {
        "capsule": agg(pd.DataFrame(pd.read_csv(OUT / "capsule_ext.csv"))
                       .rename(columns={"Rbar_fixed": "Rbar_fixed",
                                        "Rbar_drift": "Rbar_drift"})),
        "parralejo": agg(df),
        "parralejo_by_condition": {
            c: agg(g) for c, g in df.groupby("condition")},
    }
    bhd = df[df["acc_in"].notna()]
    if len(bhd):
        summary["acc"] = {
            "n": int(len(bhd)),
            "median_acc_in": float(bhd["acc_in"].median()),
            "median_acc_out": float(bhd["acc_out"].median()),
            "median_ratio": float((bhd["acc_in"] / bhd["acc_out"]).median()),
        }
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(bhd["acc_in"], bhd["acc_out"])
        summary["acc"]["wilcoxon_p"] = float(p)
    (OUT / "summary_ext.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
