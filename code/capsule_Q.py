#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""capsule（100 Hz）平台的覆盖率 Q —— 三点验证的最后一点。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from parralejo_coverage import amp_at, cardiac_recon, recon_approx

FS = 100.0
DATA = Path('real_data')
OUT = 'capsule_Q.csv'


def main():
    rows = []
    for fp in sorted(DATA.glob('patient_*.npz')):
        d = np.load(fp, allow_pickle=True)
        x = np.asarray(d['wide'], dtype=float)
        hr = float(d['gold_hr'])
        if not np.isfinite(hr) or not (60 <= hr <= 150):
            print('skip', fp.stem, hr); continue
        x = np.nan_to_num(x - np.mean(x))
        f0 = hr / 60.0
        ref = amp_at(x, f0, FS)
        if not np.isfinite(ref) or ref <= 0:
            print('skip (no line)', fp.stem); continue
        row = {'subj': fp.stem, 'hr_bpm': round(hr, 2), 'n': len(x), 'fs_hz': FS}
        for L in range(1, 7):
            row[f'Q_A_L{L}'] = amp_at(recon_approx(x, 'sym4', L, FS), f0, FS) / ref
            xc, _ = cardiac_recon(x, 'sym4', L, FS, 'overlap')
            row[f'Q_ov_L{L}'] = 0.0 if xc is None else amp_at(xc, f0, FS) / ref
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding='utf-8')
    print('records', len(df), 'HR median %.1f' % df.hr_bpm.median())
    print('\nQ_A 中位:'); print(df[[f'Q_A_L{L}' for L in range(1, 7)]].median().round(3).to_string())
    print('\nQ_ov 中位:'); print(df[[f'Q_ov_L{L}' for L in range(1, 7)]].median().round(3).to_string())
    print('\n达标率 (Q>=0.95)')
    print('A  ' + ' '.join(f'L{L}:{100*float((df[f"Q_A_L{L}"]>=0.95).mean()):.0f}%' for L in range(1, 7)))
    print('ov ' + ' '.join(f'L{L}:{100*float((df[f"Q_ov_L{L}"]>=0.95).mean()):.0f}%' for L in range(1, 7)))


if __name__ == '__main__':
    raise SystemExit(main())
