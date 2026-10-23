#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parralejo 110 人数据集的前端可行性探针（10 Hz、31.2 cm 距离格）。"""
from __future__ import annotations

import pickle
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

ROOT = Path(r'C:\Users\yxh\Desktop\db_records')
TESTS = [('Sitting', 'Rest'), ('Sitting', 'Post-exercise'),
         ('Lying', 'Rest'), ('Lying', 'Post-exercise')]


# ---------------------------------------------------------------- 数据读取
def load_record(subj: str, posture: str, test: str):
    d = ROOT / subj / posture / test
    cfg = pd.read_json(d / 'radar_chirpConfig.json', typ='series').to_dict()
    cube = pickle.loads(zlib.decompress((d / 'radar_rFFTs.zlib').read_bytes()))[0]
    ts = pd.read_csv(d / 'radar_timestamps.csv').iloc[:, 0]
    t = pd.to_datetime(ts).values.astype('datetime64[ms]').astype(np.int64) / 1000.0
    fs = 1.0 / np.median(np.diff(t))
    return cube, cfg, t, fs, d


def ecg_hr(d: Path):
    """ECG 金标：5–25 Hz 带通 → 自相关定周期 → 峰值精修。"""
    ecg = pd.read_csv(d / 'movesense_ecg.csv')
    t = pd.to_datetime(ecg.Timestamp).values.astype('datetime64[ms]').astype(np.int64) / 1000.0
    fs = 1.0 / np.median(np.diff(t))
    v = ecg.mV.to_numpy(dtype=float)
    sos = butter(3, [5, 25], btype='bandpass', fs=fs, output='sos')
    e = sosfiltfilt(sos, v - v.mean())
    # 自相关定周期（0.4–1.6 s）
    ac = np.correlate(e, e, 'full')[len(e) - 1:]
    ac = ac / ac[0]
    lo, hi = int(0.40 * fs), int(1.60 * fs)
    lag = lo + int(np.argmax(ac[lo:hi]))
    period = lag / fs
    # 以此周期做峰值检测，再取 RR 中位
    pk, _ = find_peaks(e, distance=max(1, int(0.6 * period * fs)),
                       height=0.5 * np.percentile(e, 95))
    if len(pk) < 8:
        return 60.0 / period, period, len(pk), t, e, fs
    rr = np.diff(t[pk])
    rr = rr[(rr > 0.4) & (rr < 1.6)]
    return (60.0 / np.median(rr) if len(rr) else 60.0 / period), period, len(pk), t, e, fs


# ---------------------------------------------------------------- 相位提取
def bin_phase(cube, b):
    z = cube[:, :, b].mean(axis=1)          # 8 通道复数平均
    return np.unwrap(np.angle(z))


def detrend_phase(ph, fs, win=10.0):
    """滑动中值去慢漂移（呼吸以外的大尺度），再减均值。"""
    n = int(win * fs)
    if n % 2 == 0:
        n += 1
    k = np.ones(n) / n
    pad = n // 2
    base = np.convolve(np.pad(ph, pad, mode='reflect'), k, 'valid')[:len(ph)]
    return ph - base


def hr_from_series(x, fs, band):
    sos = butter(4, band, btype='bandpass', fs=fs, output='sos')
    y = sosfiltfilt(sos, x - x.mean())
    y = y * np.hanning(len(y))
    f = np.fft.rfftfreq(len(y), 1 / fs)
    S = np.abs(np.fft.rfft(y))
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        return np.nan
    return float(f[m][int(np.argmax(S[m]))] * 60)


def cardiac_power(x, fs, band=(1.0, 2.5)):
    sos = butter(4, band, btype='bandpass', fs=fs, output='sos')
    y = sosfiltfilt(sos, x - x.mean())
    return float(np.var(y))


# ---------------------------------------------------------------- 主流程
def probe(subj, posture, test, bins=range(1, 12)):
    cube, cfg, t, fs, d = load_record(subj, posture, test)
    hr_ecg, period, n_ecg, _, _, _ = ecg_hr(d)
    rows = []
    for b in bins:
        if b >= cube.shape[2]:
            break
        ph = bin_phase(cube, b)
        x = detrend_phase(ph, fs)
        p = cardiac_power(x, fs)
        rows.append({
            'subj': subj, 'posture': posture, 'test': test, 'bin': b,
            'range_m': b * cfg['R_BIN'],
            'card_pow': p,
            'hr_1_2p5': hr_from_series(x, fs, (1.0, 2.5)),
            'hr_0p7_3': hr_from_series(x, fs, (0.7, 3.0)),
            'hr_0p9_2p2': hr_from_series(x, fs, (0.9, 2.2)),
        })
    df = pd.DataFrame(rows)
    df['hr_ecg'] = hr_ecg
    df['fs'] = fs
    return df


def main(n_subj=6):
    subs = [f'P{i:03d}' for i in range(1, 111)][:n_subj]
    out = []
    for s in subs:
        for posture, test in TESTS[:2]:
            d = ROOT / s / posture / test
            if not d.exists():
                continue
            try:
                out.append(probe(s, posture, test))
            except Exception as e:
                print('fail', s, posture, test, e)
    df = pd.concat(out, ignore_index=True)
    print(df.groupby(['subj', 'posture']).size().to_string())
    # 每个记录：选心搏功率最大的 bin，看误差
    best = df.loc[df.groupby(['subj', 'posture', 'test'])['card_pow'].idxmax()].copy()
    for c in ['hr_1_2p5', 'hr_0p7_3', 'hr_0p9_2p2']:
        best[f'err_{c}'] = best[c] - best.hr_ecg
    print()
    print(best[['subj', 'posture', 'bin', 'range_m', 'hr_ecg',
                'hr_1_2p5', 'hr_0p7_3', 'hr_0p9_2p2']].to_string(index=False))
    print()
    for c in ['hr_1_2p5', 'hr_0p7_3', 'hr_0p9_2p2']:
        e = best[f'err_{c}'].abs()
        print(f'{c}: |err| median {e.median():.1f}  mean {e.mean():.1f}  '
              f'OK<=3bpm {(e <= 3).sum()}/{len(e)}')
    df.to_csv('parralejo_probe.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
