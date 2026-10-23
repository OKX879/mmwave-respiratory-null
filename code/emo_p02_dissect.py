#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解剖 emo P02 反例：与其它被试对比信号几何、漂移速度、相位质量。

每个 trial 计算：
  energy_top1/top3    运动能量集中在最强 1/3 个 bin 的比例（多反射体 -> 分散）
  coh_neighbor        最强 bin 与相邻 bin 的复数相关（多反射体/泄漏 -> 特征改变）
  kasa_resid          呼吸带轨迹的圆拟合残差 / R（非圆轨迹 -> 大）
  AsR                 |A_s| / R
  drift_std_deg       10 s 慢变相量方向的圆标准差
  drift_rate_deg_s    慢变相量方向的平均角速度
  f_out_band          瞬时呼吸频率落在 0.08-0.65 Hz 之外的比例（相位参考失效 -> 高）
  slip_rate           相位突跳（|Δφ|>0.2 rad/样本）比例
  gain_at_null        null 时刻 |A_s(t)| / 中位 |A_s|（增益塌陷 -> 明显 <1）
  Rbar_fixed / Rbar_drift(10s) / rel_err(f_dip vs f_resp)
输出：null_crossdataset/emo_p02_dissect.csv/.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

FS = 100.0
R_BIN = 0.04505
BIN_LO, BIN_HI = 9, 28
DROP_S = 10.0
DIR = Path(r"C:\Users\yxh\Desktop\emo_dataset\raw\_multi")
OUT = Path(r"C:\Users\yxh\Desktop\2\runtime\小波基研究\null_crossdataset")


def bp(x, lo=0.1, hi=0.6):
    sos = butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x.real) + 1j * sosfiltfilt(sos, x.imag)


def slow_mean(z, win_s=10.0):
    w = int(win_s * FS)
    k = np.ones(w) / w
    pad = w // 2
    zr = np.convolve(np.pad(z.real, pad, mode="reflect"), k, "valid")[:len(z)]
    zi = np.convolve(np.pad(z.imag, pad, mode="reflect"), k, "valid")[:len(z)]
    return zr + 1j * zi


def rayleigh(ph):
    if len(ph) < 5:
        return np.nan
    return float(abs(np.mean(np.exp(1j * 2 * np.pi * ph))))


def kasa_center_R(z):
    x, y = z.real, z.imag
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0], sol[1]
    R = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0)))
    return cx, cy, R


def circ_std_deg(theta):
    r = abs(np.mean(np.exp(1j * theta)))
    return float(np.degrees(np.sqrt(max(-2 * np.log(max(r, 1e-12)), 0.0))))


def analyse(fp):
    adc = loadmat(fp)["adcData"]
    n_ch, n_samp = adc.shape
    x = adc.reshape(n_ch, n_samp // 256, 256)[:, int(DROP_S * FS):, :]
    for j in (1, 5, 9):
        x[j] *= -1
        x[j + 1] *= -1
    z = np.fft.fft(x, axis=2).mean(axis=0).T       # (bin, T)
    e = np.std(z, axis=1)
    e_win = e[BIN_LO:BIN_HI]
    b = BIN_LO + int(np.argmax(e_win))
    zc = z[b]
    env = np.abs(zc)

    # 能量集中度
    et = np.sort(e_win)[::-1]
    top1 = float(et[:1].sum() / et.sum())
    top3 = float(et[:3].sum() / et.sum())

    # 邻 bin 复相关
    def coh(z1, z2):
        a1, a2 = z1 - z1.mean(), z2 - z2.mean()
        den = np.std(a1) * np.std(a2)
        return float(abs(np.mean(a1 * np.conj(a2))) / den) if den > 0 else np.nan
    coh_l = coh(zc, z[max(b - 1, 0)]) if b - 1 >= 0 else np.nan
    coh_r = coh(zc, z[min(b + 1, z.shape[0] - 1)])

    # 呼吸带几何
    a = bp(zc)
    As = slow_mean(zc)
    cx, cy, R = kasa_center_R(a)
    resid = float(np.std(np.abs(a - (cx + 1j * cy))) / max(R, 1e-12))
    As_med = float(np.median(np.abs(As)))
    AsR = float(abs(complex(np.mean(zc)))) / max(R, 1e-12)

    # 漂移
    ang = np.unwrap(np.angle(As))
    drift_std = circ_std_deg(np.angle(As))
    drift_rate = float(np.median(np.abs(np.diff(ang))) * FS * 180 / np.pi)

    # 相位质量
    ph = np.unwrap(np.angle(a))
    inst = np.diff(ph) / (2 * np.pi) * FS
    f_out = float(np.mean((inst < 0.08) | (inst > 0.65)))
    slip = float(np.mean(np.abs(np.diff(ph)) > 0.2))

    # 深凹陷与增益
    pk, _ = find_peaks(-env, distance=int(0.25 * FS), prominence=0.15 * np.std(env))
    pk = pk[env[pk] < 0.35 * np.median(env)]
    gain = float(np.median(np.abs(As[pk]) / As_med)) if len(pk) else np.nan

    # 几何对齐度：零点要求 a(t) 指向 -A_s(t)；align=1 才是几何零点
    if len(pk):
        align = np.cos(np.angle(a[pk]) - (np.angle(As[pk]) + np.pi))
        align_med = float(np.median(align))
        R_est = float(np.median(np.abs(a)))
        d_pred = abs(R_est - As_med)
        depth_ratio = float(np.median(env[pk]) / d_pred) if d_pred > 1e-12 else np.nan
    else:
        align_med = depth_ratio = np.nan
    phB = (np.unwrap(np.angle(bp(zc))) / (2 * np.pi)) % 1.0
    phD = (np.unwrap(np.angle(a)) - (np.unwrap(np.angle(As)) + np.pi)) / (2 * np.pi) % 1.0
    Rf, Rd = rayleigh(phB[pk]), rayleigh(phD[pk])

    f_resp = np.nan
    F = np.abs(np.fft.rfft(bp(zc).real - bp(zc).real.mean()))
    f = np.fft.rfftfreq(len(zc), 1 / FS)
    m = (f >= 0.1) & (f <= 0.6)
    f_resp = float(f[m][np.argmax(F[m])])
    Fd = np.abs(np.fft.rfft(env - env.mean()))
    md = (f >= 0.1) & (f <= 2.0)
    f_dip = float(f[md][np.argmax(Fd[md])])

    return {
        "trial": fp.stem, "subject": fp.stem[:3], "bin": b, "range_m": b * R_BIN,
        "n_deep": int(len(pk)), "top1": top1, "top3": top3,
        "coh_left": coh_l, "coh_right": coh_r,
        "kasa_resid": resid, "AsR": AsR,
        "drift_std_deg": drift_std, "drift_rate_deg_s": drift_rate,
        "f_out_band": f_out, "slip_rate": slip, "gain_at_null": gain,
        "align_null": align_med, "depth_ratio": depth_ratio,
        "Rbar_fixed": Rf, "Rbar_drift": Rd,
        "f_resp": f_resp, "f_dip": f_dip,
        "rel_err": abs(f_dip - f_resp) / f_resp if f_resp > 0 else np.nan,
    }


def main():
    rows = [analyse(fp) for fp in sorted(DIR.glob("*.mat"))]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "emo_p02_dissect.csv", index=False, encoding="utf-8")
    keys = ["top3", "coh_right", "kasa_resid", "AsR", "drift_std_deg",
            "drift_rate_deg_s", "f_out_band", "slip_rate", "gain_at_null",
            "align_null", "depth_ratio",
            "Rbar_fixed", "Rbar_drift", "rel_err"]
    summary = {}
    for s, g in df.groupby("subject"):
        summary[s] = {k: float(g[k].median()) for k in keys}
        summary[s]["trials"] = int(len(g))
        summary[s]["n_deep_median"] = float(g["n_deep"].median())
    (OUT / "emo_p02_dissect.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
