#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lei 20 Hz 平台的覆盖率 Q —— 先做单记录探针。

预测（表 T8-B，展示带 0.7–3.0 Hz）：
    约定 A：L <= floor(log2(20/3)) - 1 = 1  → L=1 时 a1 上界 5 Hz，覆盖满
    约定 B：L >= ceil(log2(20/0.7)) - 1 = 4 → L=4 时 d4 下界 0.625 Hz，覆盖满
因此 20 Hz 是"刚好跨过临界 4f_hi=12 Hz"的点：A 恢复可用（但只有 L=1），B 需要 4 层。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

from lei_pilot import load, ROOT, R_BIN, FS
from parralejo_coverage import cardiac_recon, amp_at, recon_approx

R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
FS_LOG = 125.0


def ecg_hr(v, fs=FS_LOG):
    v = np.asarray(v, dtype=float)
    v = v - np.mean(v)
    sos = butter(3, [8, 30], btype='bandpass', fs=fs, output='sos')
    e = sosfiltfilt(sos, v)
    d1 = np.diff(e, prepend=e[0])
    env = d1 ** 2
    k = max(3, int(0.04 * fs))
    env = np.convolve(env, np.ones(k) / k, mode='same')
    thr = 0.35 * np.percentile(env, 99)
    pk, _ = find_peaks(env, distance=int(0.33 * fs), height=thr)
    if len(pk) < 10:
        return np.nan, np.nan, len(pk)
    rr = np.diff(pk) / fs
    rr = rr[(rr > 0.33) & (rr < 1.6)]
    if len(rr) < 8:
        return np.nan, np.nan, len(pk)
    med = np.median(rr)
    rr2 = rr[np.abs(rr - med) < 0.30 * med]
    if len(rr2) < 6:
        return np.nan, np.nan, len(pk)
    return 60.0 / np.median(rr2), float(np.std(rr2) / np.mean(rr2)), len(pk)


def pick_bin_and_phase(bw, fp, pos):
    rb = R_BIN[bw]
    z = load(fp)
    zf = np.fft.fft(z, axis=3).mean(axis=(1, 2))
    idx = []
    for r in R_TRUE[pos]:
        a = max(1, int((r - 0.25) / rb))
        b = min(zf.shape[1] - 2, int((r + 0.05) / rb))
        idx += list(range(a, b + 1))
    idx = sorted(set(idx))
    e = np.std(zf[:, idx], axis=0)
    b = idx[int(np.argmax(e))]
    zc = zf[:, b]
    ph = np.unwrap(np.angle(zc))
    return ph, b, b * rb


def detrend(x, fs, win=10.0):
    n = int(win * fs)
    if n % 2 == 0:
        n += 1
    k = np.ones(n) / n
    pad = n // 2
    base = np.convolve(np.pad(x, pad, mode='reflect'), k, 'valid')[:len(x)]
    return x - base


def main():
    POS = ROOT / '1_AsymmetricalPosition'
    for bw, tag, k in [(2.5, '2_5', 1), (3, '3', 1)]:
        radar = POS / '1_Radar_Raw_Data' / 'position_ (1)' / f'adc_{tag}GHZ_position1_ ({k}).bin'
        if not radar.exists():
            print('missing radar', radar.name); continue
        ph, b, rng = pick_bin_and_phase(bw, radar, 1)
        x = detrend(ph, FS)
        print(f'\n=== {bw} GHz, repeat {k}: bin {b} ({rng:.2f} m) ===')
        for tgt in ('Target1', 'Target2'):
            log = POS / '2_Log_data' / tgt / 'position_ (1)' / f'log_{tgt}_{tag}GHZ_position1_ ({k}).csv'
            if not log.exists():
                print('  missing', log.name); continue
            df = pd.read_csv(log, skiprows=1)
            hr, cv, n = ecg_hr(-df.iloc[:, 0].to_numpy())
            print(f'  {tgt}: HR={hr:.1f} bpm (cv={cv:.3f}, n={n})')
            if not (60 <= hr <= 150):
                print('    out of band, skip'); continue
            f0 = hr / 60.0
            ref = amp_at(x, f0, FS)
            qa = [amp_at(recon_approx(x, 'sym4', L, FS), f0, FS) / ref for L in range(1, 7)]
            print('    %-8s Q(L=1..6) = %s' % ('A:aL', ' '.join(f'{q:.2f}' for q in qa)))
            for rule in ('fixed07', 'overlap'):
                qs = []
                for L in range(1, 7):
                    xc, kept = cardiac_recon(x, 'sym4', L, FS, rule)
                    qs.append(0.0 if xc is None else amp_at(xc, f0, FS) / ref)
                print('    %-8s Q(L=1..6) = %s' % (rule, ' '.join(f'{q:.2f}' for q in qs)))


if __name__ == '__main__':
    raise SystemExit(main())
