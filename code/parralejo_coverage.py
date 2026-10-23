#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨采样率复核：DWT 层数的"边界"是频带覆盖规则，不是数字 4。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, sosfiltfilt

from parralejo_probe import load_record, bin_phase, detrend_phase
from parralejo_probe_v2 import ecg_hr_robust, pick_bin, TESTS

CARD = (1.0, 2.5)
WAVELETS = ['haar', 'db4', 'db8', 'sym4', 'sym8', 'coif4', 'bior3.5']
LEVELS = [1, 2, 3, 4, 5, 6]
OUT = 'parralejo_coverage.csv'


def level_bands(level, fs):
    nyq = fs / 2.0
    out = {j: (nyq / 2 ** j, nyq / 2 ** (j - 1)) for j in range(1, level + 1)}
    out['a'] = (0.0, nyq / 2 ** level)
    return out


def assign(band, rule='fixed07'):
    """层归属。"""
    lo, hi = band
    if rule == 'overlap':
        return 'card' if (min(hi, CARD[1]) - max(lo, CARD[0])) > 0 else 'resp'
    if hi <= CARD[0]:
        return 'resp'
    if lo >= CARD[0]:
        return 'card'
    return 'resp'


def recon_approx(x, w, L, fs):
    """约定 A 的忠实实现：只保留近似带 aL，细节全部置零，再重构。"""
    c = pywt.wavedec(x, w, level=L, mode='symmetric')
    coeffs = [c[0]] + [np.zeros_like(cc) for cc in c[1:]]
    xr = pywt.waverec(coeffs, w, mode='symmetric')
    xr = xr[:len(x)] if len(xr) >= len(x) else np.pad(xr, (0, len(x) - len(xr)))
    best, best_lag = -np.inf, 0
    for lag in range(-8, 9):
        r = float(np.corrcoef(np.roll(xr, lag), x)[0, 1])
        if r > best:
            best, best_lag = r, lag
    return np.roll(xr, best_lag)


def cardiac_recon(x, w, L, fs, rule='fixed07'):
    c = pywt.wavedec(x, w, level=L, mode='symmetric')
    # wavedec 返回 [cA_L, cD_L, cD_{L-1}, ..., cD_1]；
    # 第 j 层细节是 c[L-j+1]，容易写反（写反会把 d1 的频带套到 cD_L 上）。
    bands = level_bands(L, fs)
    kept = [assign(bands[j], rule) == 'card' for j in range(1, L + 1)]
    if not any(kept):
        return None, kept
    # c[i]（i=1..L）对应细节层 j = L-i+1；顺序必须保持 wavedec 的原序。
    coeffs = [np.zeros_like(c[0])]
    for i in range(1, L + 1):
        j = L - i + 1
        coeffs.append(c[i] if kept[j - 1] else np.zeros_like(c[i]))
    xr = pywt.waverec(coeffs, w, mode='symmetric')
    # 对齐：waverec 的边界补零会让输出比输入多 1–2 点，且偏移随层数变化。
    # 若不对齐，f0 处的相位误差会随层数跳变，Q 的层间比较就变成伪影。
    xr = xr[:len(x)] if len(xr) >= len(x) else np.pad(xr, (0, len(x) - len(xr)))
    best, best_lag = -np.inf, 0
    for lag in range(-8, 9):
        cand = np.roll(xr, lag)
        r = float(np.corrcoef(cand, x)[0, 1])
        if r > best:
            best, best_lag = r, lag
    return np.roll(xr, best_lag), kept


def amp_at(x, f0, fs, half=0.12):
    """f0±half Hz 内的峰值幅度（加 Hann 窗抑制泄漏）。"""
    y = (x - np.mean(x)) * np.hanning(len(x))
    f = np.fft.rfftfreq(len(x), 1 / fs)
    P = np.abs(np.fft.rfft(y))
    m = (f >= f0 - half) & (f <= f0 + half)
    return float(P[m].max()) if m.any() else np.nan


def one(subj, posture, test):
    cube, cfg, t, fs, d = load_record(subj, posture, test)
    hr, _, n_pk, cv = ecg_hr_robust(d)
    if not np.isfinite(hr):
        return []
    f0 = hr / 60.0
    if not (CARD[0] <= f0 <= CARD[1]):
        return []
    b = pick_bin(cube, cfg)
    x = detrend_phase(bin_phase(cube, b), fs)
    ref = amp_at(x, f0, fs)
    if not np.isfinite(ref) or ref <= 0:
        return []
    rows = []
    for w in WAVELETS:
        for L in LEVELS:
            for rule in ('fixed07', 'overlap'):
                try:
                    xc, kept = cardiac_recon(x, w, L, fs, rule)
                except Exception:
                    continue
                if xc is None:
                    q = 0.0
                else:
                    q = amp_at(xc, f0, fs) / ref
                rows.append({'subj': subj, 'posture': posture, 'test': test,
                             'wavelet': w, 'level': L, 'rule': rule,
                             'fs': fs, 'f_hr': f0,
                             'Q': q, 'n_kept': int(sum(kept))})
    return rows


def main(n_subj=40):
    import sys
    if len(sys.argv) > 1:
        n_subj = int(sys.argv[1])
    subs = [f'P{i:03d}' for i in range(1, 111)][:n_subj]
    rows = []
    for i, s in enumerate(subs, 1):
        for posture, test in TESTS:
            try:
                rows += one(s, posture, test)
            except Exception as e:
                print('fail', s, posture, test, e)
        if i % 10 == 0:
            print('...', i, 'subjects', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding='utf-8')
    print('records', df[['subj', 'posture', 'test']].drop_duplicates().shape[0],
          'rows', len(df))
    for rule in ('fixed07', 'overlap'):
        g = df[df.rule == rule]
        print(f'\n=== rule={rule} ===')
        print(g.groupby('level')['Q'].agg(['median', 'mean']).round(2).to_string())
        piv = g.groupby(['wavelet', 'level'])['Q'].median().unstack()
        print('\nmedian Q (rows=wavelet, cols=L)')
        print(piv.round(2).to_string())
    # 分层：低频心搏（<1.25 Hz）与高频心搏在两条规则下的差别
    df['low_hr'] = df.f_hr < 1.25
    print('\nQ by rule x low_hr (median, L=3, sym4)')
    sub = df[(df.level == 3) & (df.wavelet == 'sym4')]
    print(sub.groupby(['rule', 'low_hr'])['Q'].agg(['median', 'size']).round(2).to_string())


if __name__ == '__main__':
    raise SystemExit(main())
