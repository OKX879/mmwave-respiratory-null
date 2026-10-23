#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""emo 数据集零点试点（P01，17 个 trial）。
emo null pilot, P01, 17 trials."""
from __future__ import annotations

from paths import EMO_PILOT, OUT_ROOT

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, find_peaks, sosfiltfilt

FS = 100.0
R_BIN = 0.04505
BIN_LO, BIN_HI = 9, 28          # 0.4-1.2 m
DROP_S = 10.0
PILOT = (Path(sys.argv[1]) if len(sys.argv) > 1
         else EMO_PILOT)
OUT = OUT_ROOT / f"emo_{PILOT.name}"


def bp_complex(x, lo=0.1, hi=0.6):
    sos = butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x.real) + 1j * sosfiltfilt(sos, x.imag)


def phase_B(z):
    zb = bp_complex(z)
    return (np.unwrap(np.angle(zb)) / (2 * np.pi)) % 1.0


def slow_mean(z, win_s=10.0):
    w = int(win_s * FS)
    k = np.ones(w) / w
    pad = w // 2
    zr = np.convolve(np.pad(z.real, pad, mode="reflect"), k, "valid")[:len(z)]
    zi = np.convolve(np.pad(z.imag, pad, mode="reflect"), k, "valid")[:len(z)]
    return zr + 1j * zi


def phase_drift(z):
    a = bp_complex(z)
    As = slow_mean(z)
    psi = np.unwrap(np.angle(a)) - (np.unwrap(np.angle(As)) + np.pi)
    return (psi / (2 * np.pi)) % 1.0


def rayleigh(ph):
    if len(ph) < 5:
        return np.nan, np.nan
    r = float(abs(np.mean(np.exp(1j * 2 * np.pi * ph))))
    return r, float(np.exp(-len(ph) * r * r))


def kasa_R(z):
    x, y = z.real, z.imag
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0], sol[1]
    return float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0)))


def peak_freq(x, lo, hi):
    v = x - np.mean(x)
    F = np.abs(np.fft.rfft(v))
    f = np.fft.rfftfreq(len(v), 1 / FS)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(F[m])]) if m.any() else np.nan


def analyse(fp: Path):
    adc = loadmat(fp)["adcData"]                     # (12, 256*F)
    n_ch, n_samp = adc.shape
    n_frame = n_samp // 256
    x = adc.reshape(n_ch, n_frame, 256)
    x = x[:, int(DROP_S * FS):, :]
    # 官方极性校正：MATLAB j=2:4:10 -> 0-based 1,5,9；j+1 -> 2,6,10
    for j in (1, 5, 9):
        x[j] *= -1
        x[j + 1] *= -1
    Z = np.fft.fft(x, axis=2)                         # (12, F, 256)
    z = Z.mean(axis=0)                                # (F, 256) 复数通道平均
    n = z.shape[0]
    std = np.std(z[:, BIN_LO:BIN_HI], axis=0)
    b = BIN_LO + int(np.argmax(std))
    zc = z[:, b]
    env = np.abs(zc)
    pk_all, _ = find_peaks(-env, distance=int(0.25 * FS),
                           prominence=0.15 * np.std(env))
    med = float(np.median(env))
    pk = pk_all[env[pk_all] < 0.35 * med]
    ph = phase_B(zc)
    R, p = rayleigh(ph[pk])
    phd = phase_drift(zc)
    Rd, _ = rayleigh(phd[pk])
    sens = {}
    for frac in (0.20, 0.50):
        q = pk_all[env[pk_all] < frac * med]
        sens[frac] = (len(q), rayleigh(ph[q])[0])
    As = abs(complex(np.mean(zc)))
    Rad = kasa_R(zc - np.mean(zc))
    f_resp = peak_freq(bp_complex(zc).real, 0.1, 0.6)
    f_dip = peak_freq(env, 0.1, 2.0)
    return {
        "trial": fp.stem, "frames_used": n, "bin": b, "range_m": b * R_BIN,
        "n_deep": int(len(pk)), "rate_per_min": len(pk) / (n / FS) * 60,
        "Rbar": R, "p": p, "Rbar_drift": Rd,
        "n020": sens[0.20][0], "R020": sens[0.20][1],
        "n050": sens[0.50][0], "R050": sens[0.50][1], "As_over_R": As / Rad if np.isfinite(Rad) and Rad > 0 else np.nan,
        "f_resp": f_resp, "f_dip": f_dip,
        "rel_err": abs(f_dip - f_resp) / f_resp if np.isfinite(f_resp) and f_resp > 0 else np.nan,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fp in sorted(PILOT.glob("*.mat")):
        try:
            r = analyse(fp)
            rows.append(r)
            print("%s bin=%2d range=%.2fm n=%2d Rbar=%s p=%s" % (
                r["trial"], r["bin"], r["range_m"], r["n_deep"],
                ("%.2f" % r["Rbar"]) if np.isfinite(r["Rbar"]) else " - ",
                ("%.1e" % r["p"]) if np.isfinite(r["p"]) else " - "))
        except Exception as e:
            print("fail", fp.name, e)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "emo_P01_pilot.csv", index=False, encoding="utf-8")
    el = df[df["n_deep"] >= 5]
    summary = {
        "trials": int(len(df)),
        "median_n_deep": float(df["n_deep"].median()),
        "median_rate_per_min": float(df["rate_per_min"].median()),
        "eligible": int(len(el)),
        "locked": int((el["Rbar"] > 0.4).sum()),
        "locked_drift": int((el["Rbar_drift"] > 0.4).sum()),
        "locked_020": int(((el["n020"] >= 5) & (el["R020"] > 0.4)).sum()),
        "locked_050": int(((el["n050"] >= 5) & (el["R050"] > 0.4)).sum()),
        "median_Rbar": float(el["Rbar"].median()) if len(el) else np.nan,
        "median_Rdrift": float(el["Rbar_drift"].median()) if len(el) else np.nan,
        "median_AsR": float(df["As_over_R"].median()),
        "frac_dip_at_resp": float((df["rel_err"] < 0.2).mean()),
        "median_bin": int(df["bin"].median()),
    }
    (OUT / "emo_P01_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
