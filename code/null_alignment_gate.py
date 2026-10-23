#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对齐度门槛：只把“几何零点”纳入相位锁定统计。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from null_crossdataset import (  # noqa: E402
    OUT, bp_complex, deep_nulls, load_capsule, load_parralejo, phase_B, rayleigh,
)

ALIGN_THR = 0.6
WIN = 10.0


def slow_mean(z, fs, win_s=WIN):
    w = int(win_s * fs)
    k = np.ones(w) / w
    pad = w // 2
    zr = np.convolve(np.pad(z.real, pad, mode="reflect"), k, "valid")[:len(z)]
    zi = np.convolve(np.pad(z.imag, pad, mode="reflect"), k, "valid")[:len(z)]
    return zr + 1j * zi


def analyse(z, fs, bin_lo=0, bin_hi=None):
    zm = z[bin_lo:bin_hi]
    b = bin_lo + int(np.argmax(np.std(zm, axis=1)))
    zc = z[b]
    env = np.abs(zc)
    pk = deep_nulls(env, fs)
    a = bp_complex(zc, fs)
    As = slow_mean(zc, fs)
    if len(pk):
        align = float(np.median(np.cos(np.angle(a[pk]) - (np.angle(As[pk]) + np.pi))))
    else:
        align = np.nan
    Rf, _ = rayleigh(phase_B(zc, fs)[pk])
    phd = (np.unwrap(np.angle(a)) - (np.unwrap(np.angle(As)) + np.pi)) / (2 * np.pi) % 1.0
    Rd, _ = rayleigh(phd[pk])
    return {"bin": b, "n_deep": int(len(pk)), "align": align,
            "Rbar_fixed": Rf, "Rbar_drift": Rd}


def main():
    rows = []
    for i in range(1, 21):
        z, fs, _ = load_capsule(f"patient_{i}")
        rows.append({"platform": "capsule", "id": f"patient_{i}",
                     **analyse(z, fs)})
    for i in range(1, 111):
        sid = f"P{i:03d}"
        for cond in ["Sitting/Rest", "Sitting/Post-exercise",
                     "Lying/Rest", "Lying/Post-exercise"]:
            z, fs = load_parralejo(sid, cond)
            rows.append({"platform": "parralejo", "id": f"{sid}_{cond}",
                         **analyse(z, fs, 1, 7)})
    emo = pd.read_csv(OUT / "emo_p02_dissect.csv")
    for _, r in emo.iterrows():
        rows.append({"platform": "emo", "id": r["trial"],
                     "bin": r["bin"], "n_deep": r["n_deep"],
                     "align": r["align_null"], "Rbar_fixed": r["Rbar_fixed"],
                     "Rbar_drift": r["Rbar_drift"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "alignment_gate.csv", index=False, encoding="utf-8")

    summary = {}
    for plat, g in df.groupby("platform"):
        el = g[g["n_deep"] >= 5]
        gate = el[el["align"] > ALIGN_THR]
        summary[plat] = {
            "records": int(len(g)),
            "eligible": int(len(el)),
            "median_align_eligible": float(el["align"].median()),
            "gate_pass": int(len(gate)),
            "gate_pass_frac_of_eligible": float(len(gate) / max(len(el), 1)),
            "ungated_locked_fixed": int((el["Rbar_fixed"] > 0.4).sum()),
            "ungated_locked_drift": int((el["Rbar_drift"] > 0.4).sum()),
            "gated_locked_fixed": int((gate["Rbar_fixed"] > 0.4).sum()),
            "gated_locked_drift": int((gate["Rbar_drift"] > 0.4).sum()),
            "gated_median_Rfixed": float(gate["Rbar_fixed"].median()) if len(gate) else np.nan,
            "gated_median_Rdrift": float(gate["Rbar_drift"].median()) if len(gate) else np.nan,
        }
    (OUT / "alignment_gate.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
