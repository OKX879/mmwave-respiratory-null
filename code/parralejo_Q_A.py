#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补测 10 Hz 平台的"约定 A"（心搏取自近似带 aL）。
Convention A on the 10 Hz platform, taking the beat from the approximation band aL."""
from __future__ import annotations

import numpy as np
import pandas as pd

from parralejo_probe import load_record, bin_phase, detrend_phase
from parralejo_probe_v2 import ecg_hr_robust, pick_bin, TESTS
from parralejo_coverage import amp_at, recon_approx

OUT = 'parralejo_Q_A.csv'


def main():
    rows = []
    n = 0
    for i in range(1, 111):
        s = f'P{i:03d}'
        for posture, test in TESTS:
            try:
                cube, cfg, _, fs, d = load_record(s, posture, test)
                hr, _, n_pk, cv = ecg_hr_robust(d)
            except Exception as e:
                print('skip', s, posture, test, e)
                continue
            if not np.isfinite(hr) or not (60 <= hr <= 150):
                continue
            b = pick_bin(cube, cfg)
            x = detrend_phase(bin_phase(cube, b), fs)
            f0 = hr / 60.0
            ref = amp_at(x, f0, fs)
            if not np.isfinite(ref) or ref <= 0:
                continue
            row = {'subj': s, 'posture': posture, 'test': test,
                   'hr_bpm': round(hr, 2), 'f_hr_hz': round(f0, 4)}
            for L in range(1, 7):
                row[f'Q_A_L{L}'] = amp_at(recon_approx(x, 'sym4', L, fs), f0, fs) / ref
            rows.append(row)
            n += 1
            if n % 40 == 0:
                print('...', n, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding='utf-8')
    print('records', len(df), 'subjects', df.subj.nunique())
    print('\nQ_A 中位:'); print(df[[f'Q_A_L{L}' for L in range(1, 7)]].median().round(3).to_string())
    print('\n达标率 (Q>=0.95):')
    print(' '.join(f'L{L}:{100*float((df[f"Q_A_L{L}"]>=0.95).mean()):.0f}%'
                   for L in range(1, 7)))


if __name__ == '__main__':
    raise SystemExit(main())
