#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稳健性验证 1：漂移补偿窗口敏感性 + 漂移幅度与固定相位锁定的关系。

对三个平台（capsule 20 例、Parralejo 440 段、emo P01 18 trial）分别计算：
  Rbar_fixed        固定坐标系呼吸相位
  Rbar_drift(W)     a(t) 相对 -A_s(t) 的夹角，A_s 用 W 秒滑动平均（W=5/10/15/20）
  drift_deg(W)      A_s 方向的圆标准差（度），衡量背景漂移大小

输出：null_crossdataset/window_sensitivity.csv / .json
"""
from __future__ import annotations

import json
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, find_peaks, sosfiltfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from null_crossdataset import (  # noqa: E402
    OUT, load_capsule, load_parralejo, bp_complex, deep_nulls, phase_B, rayleigh,
)

WINDOWS = (5.0, 10.0, 15.0, 20.0)
EMO = Path(r"C:\Users\yxh\Desktop\emo_dataset\raw\_pilot")
EMO_FS = 100.0


def slow_mean(z, fs, win_s):
    w = int(win_s * fs)
    k = np.ones(w) / w
    pad = w // 2
    zr = np.convolve(np.pad(z.real, pad, mode="reflect"), k, "valid")[:len(z)]
    zi = np.convolve(np.pad(z.imag, pad, mode="reflect"), k, "valid")[:len(z)]
    return zr + 1j * zi


def circ_std_deg(theta):
    """圆标准差（度）。"""
    r = abs(np.mean(np.exp(1j * theta)))
    return float(np.degrees(np.sqrt(max(-2 * np.log(max(r, 1e-12)), 0.0))))


def drift_phase(z, fs, win_s):
    a = bp_complex(z, fs)
    As = slow_mean(z, fs, win_s)
    psi = np.unwrap(np.angle(a)) - (np.unwrap(np.angle(As)) + np.pi)
    return (psi / (2 * np.pi)) % 1.0, circ_std_deg(np.angle(As))


def load_emo(fp):
    adc = loadmat(fp)["adcData"]
    n_ch, n_samp = adc.shape
    n_frame = n_samp // 256
    x = adc.reshape(n_ch, n_frame, 256)[:, int(10 * EMO_FS):, :]
    for j in (1, 5, 9):
        x[j] *= -1
        x[j + 1] *= -1
    z = np.fft.fft(x, axis=2).mean(axis=0)
    return z.T, EMO_FS          # (bin, T)，与其它平台一致


def analyse_record(z, fs, bin_lo=0, bin_hi=None):
    zm = z[bin_lo:bin_hi]
    b = bin_lo + int(np.argmax(np.std(zm, axis=1)))
    zc = z[b]
    env = np.abs(zc)
    pk = deep_nulls(env, fs)
    ph = phase_B(zc, fs)
    fixed, _ = rayleigh(ph[pk])
    out = {"bin": b, "n_deep": int(len(pk)), "Rbar_fixed": fixed}
    for w in WINDOWS:
        phd, drift = drift_phase(zc, fs, w)
        out[f"Rbar_drift_{int(w)}"] = rayleigh(phd[pk])[0]
        out[f"drift_deg_{int(w)}"] = drift
    return out


def main():
    rows = []
    for i in range(1, 21):
        z, fs, _ = load_capsule(f"patient_{i}")
        r = analyse_record(z, fs)
        rows.append({"platform": "capsule", "id": f"patient_{i}", **r})
    for i in range(1, 111):
        sid = f"P{i:03d}"
        for cond in ["Sitting/Rest", "Sitting/Post-exercise",
                     "Lying/Rest", "Lying/Post-exercise"]:
            z, fs = load_parralejo(sid, cond)
            r = analyse_record(z, fs, 1, 7)
            rows.append({"platform": "parralejo", "id": f"{sid}_{cond}", **r})
    for fp in sorted(EMO.glob("*.mat")):
        z, fs = load_emo(fp)
        r = analyse_record(z, fs, 9, 28)
        rows.append({"platform": "emo_p01", "id": fp.stem, **r})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "window_sensitivity.csv", index=False, encoding="utf-8")

    summary = {}
    for plat, g in df.groupby("platform"):
        el = g[g["n_deep"] >= 5]
        s = {"records": int(len(g)), "eligible": int(len(el)),
             "locked_fixed": int((el["Rbar_fixed"] > 0.4).sum()),
             "median_Rfixed": float(el["Rbar_fixed"].median()) if len(el) else np.nan}
        for w in WINDOWS:
            k = f"Rbar_drift_{int(w)}"
            s[f"locked_{int(w)}s"] = int((el[k] > 0.4).sum())
            s[f"median_{int(w)}s"] = float(el[k].median()) if len(el) else np.nan
            s[f"median_driftdeg_{int(w)}s"] = float(el[f"drift_deg_{int(w)}"].median())
        summary[plat] = s

    # 机制检验：漂移幅度 vs 固定相位 Rbar（Spearman）
    from scipy.stats import spearmanr
    corr = {}
    for plat, g in df[df["n_deep"] >= 5].groupby("platform"):
        for w in WINDOWS:
            rho, p = spearmanr(g[f"drift_deg_{int(w)}"], g["Rbar_fixed"],
                               nan_policy="omit")
            corr[f"{plat}_{int(w)}s"] = {"rho": float(rho), "p": float(p),
                                         "n": int(len(g))}
    summary["drift_vs_fixed"] = corr

    (OUT / "window_sensitivity.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
