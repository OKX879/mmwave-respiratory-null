#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按数据集给出的真实距离约束选 bin，复核分辨率效应。
Select bins under the dataset's true range constraints and recheck the resolution effect."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lei_pilot import load, ROOT, R_BIN, FS, bp, slow_mean, rayleigh
from scipy.signal import find_peaks

OUT = Path(r'C:\Users\yxh\Desktop\2\runtime\小波基研究\null_crossdataset')
R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']


def analyse_bin(bw, fp, pos):
    rb = R_BIN[bw]
    z = load(fp)
    Z = np.fft.fft(z, axis=3)
    zf = Z.mean(axis=(1, 2))
    lo = max(1, int((min(R_TRUE[pos]) - 0.25) / rb))
    hi = min(zf.shape[1] - 2, int((max(R_TRUE[pos]) + 0.05) / rb))
    idx = []
    for r in R_TRUE[pos]:
        a = max(1, int((r - 0.25) / rb)); b = min(zf.shape[1] - 2, int((r + 0.05) / rb))
        idx += list(range(a, b + 1))
    idx = sorted(set(idx))
    e = np.std(zf[:, idx], axis=0)
    b = idx[int(np.argmax(e))]
    zc = zf[:, b]
    env = np.abs(zc)
    med = float(np.median(env))
    D = float(np.mean(env < 0.3 * med))
    Ds = np.asarray([float(np.mean(np.abs(zf[:, bb]) < 0.3 * np.median(np.abs(zf[:, bb]))))
                     for bb in idx])
    D_others = np.delete(Ds, idx.index(b))
    pk, _ = find_peaks(-env, distance=int(0.25 * FS), prominence=0.15 * np.std(env))
    pk = pk[env[pk] < 0.35 * med]
    a = bp(zc)
    As = slow_mean(zc)
    kappa = float(np.median(np.cos(np.angle(a[pk]) - (np.angle(As[pk]) + np.pi)))) if len(pk) else np.nan
    phf = (np.unwrap(np.angle(a)) / (2 * np.pi)) % 1.0
    phd = (np.unwrap(np.angle(a)) - (np.unwrap(np.angle(As)) + np.pi)) / (2 * np.pi) % 1.0
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
        'As_over_R': float(abs(complex(np.mean(zc))) / (np.mean(np.abs(a)) + 1e-12)),
        'kappa': kappa, 'Rbar_fixed': rayleigh(phf[pk]), 'Rbar_drift': rayleigh(phd[pk]),
        'f_resp': f_resp, 'f_dip': f_dip,
        'rel_err': abs(f_dip - f_resp) / f_resp if f_resp > 0 else np.nan,
        'position': pos,
    }


def main():
    rows = []
    for base in BASES:
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            pos = int(d.name.replace('position_ (', '').replace(')', ''))
            for bw in (3, 2.5, 2):
                tag = {3: '3', 2.5: '2_5', 2: '2'}[bw]
                for k in range(1, 7):
                    fp = d / f'adc_{tag}GHZ_position{pos}_ ({k}).bin'
                    if not fp.exists():
                        fp = d / f'adc_{tag}GHZ_positon{pos}_ ({k}).bin'
                    if not fp.exists() or fp.stat().st_size != 1200 * 3 * 4 * 200 * 2 * 2:
                        continue
                    try:
                        r = analyse_bin(bw, fp, pos)
                        r['repeat'] = k
                        rows.append(r)
                    except Exception as e:
                        print('fail', fp.name, e)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'lei_person_bin.csv', index=False, encoding='utf-8')
    for bw, g in df.groupby('bandwidth_GHz'):
        gl = g[g.n_deep >= 5]
        print(f'{bw} GHz n={len(g)} frac>=5={100*(g.n_deep>=5).mean():.1f}% '
              f'medN={g.n_deep.median():.0f} D={g.D.median():.4f} '
              f'Dother={g.D_others_median.median():.4f} As/R={g.As_over_R.median():.2f} '
              f'kappa={gl.kappa.median():.2f} R={gl.Rbar_drift.median():.2f} '
              f'range={g.range_m.median():.2f}')


if __name__ == '__main__':
    raise SystemExit(main())
