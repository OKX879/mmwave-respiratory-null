#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parralejo 前端 v2：用距离像选格（而非心搏带功率），并校验 ECG 金标。
Parralejo front end v2: select the bin from the range profile rather than beat-band power, and validate the ECG gold standard."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

from parralejo_probe import load_record, bin_phase, detrend_phase, hr_from_series

TESTS = [('Sitting', 'Rest'), ('Sitting', 'Post-exercise'),
         ('Lying', 'Rest'), ('Lying', 'Post-exercise')]


def ecg_hr_robust(d):
    """ECG 金标：8–30 Hz 带通 → 平方增强 → 峰检 → RR 稳健中位 + 质量标记。
ECG gold standard: 8-30 Hz bandpass, squaring, peak detection, robust median RR and a quality flag."""
    ecg = pd.read_csv(d / 'movesense_ecg.csv')
    t = pd.to_datetime(ecg.Timestamp).values.astype('datetime64[ms]').astype(np.int64) / 1000.0
    fs = 1.0 / np.median(np.diff(t))
    v = ecg.mV.to_numpy(dtype=float)
    sos = butter(3, [8, 30], btype='bandpass', fs=fs, output='sos')
    e = sosfiltfilt(sos, v - v.mean())
    d1 = np.diff(e, prepend=e[0])
    env = d1 ** 2
    k = max(3, int(0.04 * fs))
    env = np.convolve(env, np.ones(k) / k, mode='same')
    thr = 0.35 * np.percentile(env, 99)
    pk, _ = find_peaks(env, distance=int(0.33 * fs), height=thr)
    if len(pk) < 10:
        return np.nan, np.nan, 0, np.nan
    rr = np.diff(t[pk])
    rr = rr[(rr > 0.33) & (rr < 1.6)]
    if len(rr) < 8:
        return np.nan, np.nan, len(pk), np.nan
    med = np.median(rr)
    keep = np.abs(rr - med) < 0.30 * med
    rr2 = rr[keep]
    hr = 60.0 / np.median(rr2) if len(rr2) >= 6 else np.nan
    cv = float(np.std(rr2) / np.mean(rr2)) if len(rr2) >= 6 else np.nan
    return hr, med, len(pk), cv


def pick_bin(cube, cfg, b_lo=1, b_hi=6):
    prof = np.abs(cube).mean(axis=(0, 1))
    seg = prof[b_lo:b_hi + 1]
    return b_lo + int(np.argmax(seg))


def probe_record(subj, posture, test):
    cube, cfg, t, fs, d = load_record(subj, posture, test)
    hr_ecg, period, n_ecg, cv = ecg_hr_robust(d)
    b = pick_bin(cube, cfg)
    ph = bin_phase(cube, b)
    x = detrend_phase(ph, fs)
    return {
        'subj': subj, 'posture': posture, 'test': test,
        'bin': b, 'range_m': b * cfg['R_BIN'], 'fs': fs,
        'hr_ecg': hr_ecg, 'rr_cv': cv, 'n_rpeak': n_ecg,
        'hr_card': hr_from_series(x, fs, (1.0, 2.5)),
        'hr_wide': hr_from_series(x, fs, (0.7, 3.0)),
        'hr_narrow': hr_from_series(x, fs, (0.9, 2.2)),
    }


def main(n_subj=8):
    subs = [f'P{i:03d}' for i in range(1, 111)][:n_subj]
    rows = []
    for s in subs:
        for posture, test in TESTS:
            d = None
            try:
                rows.append(probe_record(s, posture, test))
            except Exception as e:
                print('fail', s, posture, test, e)
    df = pd.DataFrame(rows)
    for c in ['hr_card', 'hr_wide', 'hr_narrow']:
        df['err_' + c] = df[c] - df.hr_ecg
    pd.set_option('display.width', 200)
    print(df[['subj', 'posture', 'test', 'bin', 'range_m', 'rr_cv',
              'hr_ecg', 'hr_card', 'hr_wide', 'hr_narrow']].to_string(index=False))
    print()
    for c in ['hr_card', 'hr_wide', 'hr_narrow']:
        e = df['err_' + c].abs().dropna()
        print(f'{c}: n={len(e)} |err| median {e.median():.1f}  mean {e.mean():.1f}  '
              f'OK<=3 {(e <= 3).sum()}/{len(e)}')
    df.to_csv('parralejo_probe_v2.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
