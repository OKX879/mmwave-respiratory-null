#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lei 2024 双人数据试点：同一位置、三种带宽，检验零点几何/相位与 D 的区分度。
Lei 2024 dual-subject pilot: one position, three bandwidths; null geometry, phase and the discriminability of D."""
from __future__ import annotations

from paths import FMCW_DUAL, OUT_ROOT

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

ROOT = FMCW_DUAL
POS = ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data' / 'position_ (1)'
FS = 20.0
R_BIN = {3: 0.05, 2.5: 0.06, 2: 0.075}
BIN_MIN, BIN_MAX = 8, 32          # 0.4-1.6 m（3GHz 基准；按 rb 换算见下）
OUT = OUT_ROOT


def load(fp):
    x = np.fromfile(fp, dtype='<i2')[:1200 * 3 * 4 * 200 * 2]
    x = x.reshape(1200, 3, 4, 200, 2)
    return x[..., 0].astype(np.float32) + 1j * x[..., 1].astype(np.float32)


def bp(x, lo=0.1, hi=0.6):
    sos = butter(4, [lo, hi], btype='bandpass', fs=FS, output='sos')
    return sosfiltfilt(sos, x.real) + 1j * sosfiltfilt(sos, x.imag)


def slow_mean(z, w=10.0):
    n = int(w * FS)
    k = np.ones(n) / n
    pad = n // 2
    zr = np.convolve(np.pad(z.real, pad, mode='reflect'), k, 'valid')[:len(z)]
    zi = np.convolve(np.pad(z.imag, pad, mode='reflect'), k, 'valid')[:len(z)]
    return zr + 1j * zi


def rayleigh(ph):
    if len(ph) < 5:
        return np.nan
    return float(abs(np.mean(np.exp(1j * 2 * np.pi * ph))))


def analyse(bw, fp=None):
    tag = {3: '3', 2.5: '2_5', 2: '2'}[bw]
    fp = fp or (POS / f'adc_{tag}GHZ_position1_ (1).bin')
    rb = R_BIN[bw]
    z = load(fp)
    Z = np.fft.fft(z, axis=3)                 # (frame, tx, rx, bin)
    zf = Z.mean(axis=(1, 2))                  # (frame, bin) 12 通道复数平均
    lo = int(0.4 / rb); hi = int(1.6 / rb)
    e = np.std(zf[:, lo:hi], axis=0)
    b = lo + int(np.argmax(e))
    zc = zf[:, b]
    env = np.abs(zc)
    med = float(np.median(env))
    D = float(np.mean(env < 0.3 * med))
    # 其他候选 bin 的 D
    Ds = []
    for bb in range(lo, hi):
        e2 = np.abs(zf[:, bb])
        Ds.append(float(np.mean(e2 < 0.3 * np.median(e2))))
    Ds = np.asarray(Ds)
    D_others = np.delete(Ds, b - lo)
    # 深凹陷与相位
    pk, _ = find_peaks(-env, distance=int(0.25 * FS),
                       prominence=0.15 * np.std(env))
    pk = pk[env[pk] < 0.35 * med]
    a = bp(zc)
    As = slow_mean(zc)
    kappa = float(np.median(np.cos(np.angle(a[pk]) - (np.angle(As[pk]) + np.pi)))) if len(pk) else np.nan
    phf = (np.unwrap(np.angle(a)) / (2 * np.pi)) % 1.0
    phd = (np.unwrap(np.angle(a)) - (np.unwrap(np.angle(As)) + np.pi)) / (2 * np.pi) % 1.0
    Rf, Rd = rayleigh(phf[pk]), rayleigh(phd[pk])
    f_resp = np.nan
    F = np.abs(np.fft.rfft(a.real - a.real.mean()))
    f = np.fft.rfftfreq(len(zc), 1 / FS)
    m = (f >= 0.1) & (f <= 0.6)
    f_resp = float(f[m][np.argmax(F[m])])
    Fd = np.abs(np.fft.rfft(env - env.mean()))
    md = (f >= 0.1) & (f <= 2.0)
    f_dip = float(f[md][np.argmax(Fd[md])])
    return {
        'bandwidth_GHz': bw, 'range_bin_m': rb, 'bin': int(b), 'range_m': b * rb,
        'n_deep': int(len(pk)), 'D': D,
        'D_others_median': float(np.median(D_others)),
        'D_others_p90': float(np.percentile(D_others, 90)),
        'As_over_R': float(abs(complex(np.mean(zc))) / (np.mean(np.abs(a)) + 1e-12)),
        'kappa': kappa, 'Rbar_fixed': Rf, 'Rbar_drift': Rd,
        'f_resp': f_resp, 'f_dip': f_dip,
        'rel_err': abs(f_dip - f_resp) / f_resp if f_resp > 0 else np.nan,
    }


def main():
    rows = [analyse(bw) for bw in (3, 2.5, 2)]
    for r in rows:
        print('{bandwidth_GHz} GHz: bin {bin} ({range_m:.2f} m), n={n_deep}, '
              'D={D:.3f} vs others {D_others_median:.3f}, kappa={kappa:.2f}, '
              'Rf={Rbar_fixed:.2f}, Rd={Rbar_drift:.2f}, dip/resp={rel_err:.2f}'.format(**r))
    OUT.mkdir(exist_ok=True)
    (OUT / 'lei_pilot.json').write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
