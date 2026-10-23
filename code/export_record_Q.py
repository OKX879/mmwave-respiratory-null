#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 §4.13 的逐记录 Q 明细（376 行，一条记录一行）。
Export the per-record Q detail of section 4.13: 376 rows, one per record."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from parralejo_probe import load_record
from parralejo_probe_v2 import ecg_hr_robust, pick_bin, TESTS

SRC = 'parralejo_coverage.csv'
OUT = 'parralejo_record_Q.csv'


def main():
    cov = pd.read_csv(SRC)
    cov = cov[(cov.wavelet == 'sym4')]
    wide = cov.pivot_table(index=['subj', 'posture', 'test'],
                           columns=['rule', 'level'], values='Q').reset_index()
    wide.columns = ['subj', 'posture', 'test'] + [
        f'Q_{r}_L{int(l)}' for r, l in wide.columns[3:]]

    rows = []
    keys = wide[['subj', 'posture', 'test']].drop_duplicates()
    for _, k in keys.iterrows():
        s, po, te = k.subj, k.posture, k.test
        cube, cfg, _, fs, d = load_record(s, po, te)
        hr, _, n_pk, cv = ecg_hr_robust(d)
        b = pick_bin(cube, cfg)
        rows.append({'subj': s, 'posture': po, 'test': te,
                     'bin': b, 'range_m': round(b * cfg['R_BIN'], 3),
                     'fs_hz': round(float(fs), 4),
                     'hr_ecg_bpm': round(float(hr), 2),
                     'f_hr_hz': round(float(hr) / 60.0, 4),
                     'rr_cv': round(float(cv), 4),
                     'n_rpeak': int(n_pk)})
    meta = pd.DataFrame(rows)
    out = meta.merge(wide, on=['subj', 'posture', 'test'], how='left')

    # 列顺序：元数据 → L3（正文用）→ 其余层
    order = (['subj', 'posture', 'test', 'bin', 'range_m', 'fs_hz',
              'hr_ecg_bpm', 'f_hr_hz', 'rr_cv', 'n_rpeak',
              'Q_overlap_L3', 'Q_fixed07_L3']
             + [f'Q_overlap_L{L}' for L in (1, 2, 4, 5, 6)]
             + [f'Q_fixed07_L{L}' for L in (1, 2, 4, 5, 6)])
    out = out[[c for c in order if c in out.columns]]
    out.to_csv(OUT, index=False, encoding='utf-8')

    print('rows', len(out), 'subjects', out.subj.nunique())
    print('columns', len(out.columns))
    print(out.head(5).round(3).to_string(index=False))
    print('\nQ_overlap_L3  median %.3f  IQR %.3f-%.3f' % (
        out.Q_overlap_L3.median(), out.Q_overlap_L3.quantile(.25),
        out.Q_overlap_L3.quantile(.75)))
    print('Q_fixed07_L3  median %.3f  IQR %.3f-%.3f' % (
        out.Q_fixed07_L3.median(), out.Q_fixed07_L3.quantile(.25),
        out.Q_fixed07_L3.quantile(.75)))
    print('\n缺失值:', int(out.isna().sum().sum()))


if __name__ == '__main__':
    raise SystemExit(main())
