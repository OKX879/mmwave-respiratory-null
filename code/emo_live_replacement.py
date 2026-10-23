#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 emo 数据替代活体采集：D 分布 + 载频裁决所需的表观摆幅。
Use emo data in place of live capture: the D distribution and the apparent swing needed for the carrier-frequency argument."""
from __future__ import annotations

from paths import EMO_MULTI, OUT_ROOT

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

FS = 100.0
BIN_LO, BIN_HI = 9, 28
DROP = 1000
DIR = EMO_MULTI
OUT = OUT_ROOT


def bp(x, lo=0.1, hi=0.6):
    sos = butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x.real) + 1j * sosfiltfilt(sos, x.imag)


def analyse(fp):
    adc = loadmat(fp)["adcData"]
    n_ch, n_samp = adc.shape
    x = adc.reshape(n_ch, n_samp // 256, 256)[:, DROP:, :]
    for j in (1, 5, 9):
        x[j] *= -1
        x[j + 1] *= -1
    z = np.fft.fft(x, axis=2).mean(axis=0).T       # (bin, T)
    e = np.std(z[BIN_LO:BIN_HI], axis=1)
    b = BIN_LO + int(np.argmax(e))
    zc = z[b]
    env = np.abs(zc)
    med = float(np.median(env))
    D = float(np.mean(env < 0.3 * med))
    # 单周期呼吸摆幅：去趋势后的相位半峰峰值
    ph = np.unwrap(np.angle(bp(zc)))
    t = np.arange(len(ph))
    coef = np.polyfit(t, ph, 1)
    ph_d = ph - np.polyval(coef, t)
    th_max = float((np.percentile(ph_d, 99) - np.percentile(ph_d, 1)) / 2)
    # 滑动中位 D（10 s 窗），更接近实际判据
    w = int(10 * FS)
    roll = pd.Series(env).rolling(w, center=True, min_periods=w // 2).median().to_numpy()
    D_roll = float(np.nanmean(env < 0.3 * roll))
    # 所有候选 bin 的 D（全局中位）
    Dall = []
    Droll_all = []
    for bb in range(BIN_LO, BIN_HI):
        e2 = np.abs(z[bb])
        Dall.append(float(np.mean(e2 < 0.3 * np.median(e2))))
        r2 = pd.Series(e2).rolling(w, center=True, min_periods=w // 2).median().to_numpy()
        Droll_all.append(float(np.nanmean(e2 < 0.3 * r2)))
    Dall = np.asarray(Dall)
    Droll_all = np.asarray(Droll_all)
    others = np.delete(Dall, b - BIN_LO)
    others_r = np.delete(Droll_all, b - BIN_LO)
    return {
        "trial": fp.stem, "bin": b, "range_m": b * 0.04505,
        "D_selected": D, "D_others_median": float(np.median(others)),
        "D_others_p90": float(np.percentile(others, 90)),
        "D_roll_selected": D_roll, "D_roll_others_median": float(np.median(others_r)),
        "theta_max_rad": th_max,
        "app_pp_5mm": th_max * 5.0 / (2 * np.pi),
        "app_pp_37mm": th_max * 37.5 / (2 * np.pi),
    }


def main():
    rows = [analyse(p) for p in sorted(DIR.glob("*.mat"))]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "emo_live_replacement.csv", index=False, encoding="utf-8")
    summary = {
        "trials": int(len(df)),
        "D_selected_median": float(df["D_selected"].median()),
        "D_selected_range": [float(df["D_selected"].min()), float(df["D_selected"].max())],
        "D_others_median": float(df["D_others_median"].median()),
        "D_others_p90_median": float(df["D_others_p90"].median()),
        "frac_selected_above_1pct": float((df["D_selected"] > 0.01).mean()),
        "D_roll_selected_median": float(df["D_roll_selected"].median()),
        "D_roll_others_median": float(df["D_roll_others_median"].median()),
        "frac_roll_selected_above_1pct": float((df["D_roll_selected"] > 0.01).mean()),
        "theta_max_median": float(df["theta_max_rad"].median()),
        "theta_max_range": [float(df["theta_max_rad"].min()), float(df["theta_max_rad"].max())],
        "app_pp_5mm_median": float(df["app_pp_5mm"].median()),
        "app_pp_5mm_range": [float(df["app_pp_5mm"].min()), float(df["app_pp_5mm"].max())],
        "app_pp_37mm_median": float(df["app_pp_37mm"].median()),
        "frac_37mm_outside_1_12mm": float((df["app_pp_37mm"] > 12).mean()),
        "frac_5mm_inside_1_12mm": float(((df["app_pp_5mm"] >= 1) & (df["app_pp_5mm"] <= 12)).mean()),
    }
    (OUT / "emo_live_replacement.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
