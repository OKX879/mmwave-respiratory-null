#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""emo（100 Hz）平台补测覆盖率 Q —— 给 100 Hz 那个点加样本。

为什么可以补
------------
Q 是"某频率处的幅度保留率"，只取决于 DWT 的频带划分，与输入是速度域还是位移域无关
（速度域只多乘一个 |jω|，比值里约掉）。emo 的 `vital` 列是速度域位移，正好可用。
金标取同步 PPG 的逐拍中位心率（与 emo_eval 同参数）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from emo_eval import list_trials, load_trial
from parralejo_coverage import amp_at, cardiac_recon, recon_approx

FS = 100.0
OUT = 'emo_Q.csv'


def main():
    trials = list_trials(10_000)
    print('trials found', len(trials))
    rows = []
    for i, fp in enumerate(trials, 1):
        rec = None
        for win in ('full', 'win'):
            try:
                rec = load_trial(fp, 'vital', win)
            except Exception as e:
                rec = None
            if rec:
                break
        if not rec:
            continue
        hr = rec['gold_hr']
        if not np.isfinite(hr) or not (60 <= hr <= 150):
            continue
        x = np.asarray(rec['wide'], dtype=float)
        x = np.nan_to_num(x - np.mean(x))
        f0 = hr / 60.0
        ref = amp_at(x, f0, FS)
        if not np.isfinite(ref) or ref <= 0:
            continue
        row = {'trial': fp.stem, 'subject': rec['subject'],
               'hr_bpm': round(hr, 2), 'n': len(x)}
        for L in range(1, 8):
            row[f'Q_A_L{L}'] = amp_at(recon_approx(x, 'sym4', L, FS), f0, FS) / ref
            xc, _ = cardiac_recon(x, 'sym4', L, FS, 'overlap')
            row[f'Q_ov_L{L}'] = 0.0 if xc is None else amp_at(xc, f0, FS) / ref
        rows.append(row)
        if i % 40 == 0:
            print('...', i, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding='utf-8')
    print('\nrecords', len(df), 'subjects', df.subject.nunique(),
          'HR median %.1f' % df.hr_bpm.median())
    print('\nQ_A 中位:')
    print(df[[f'Q_A_L{L}' for L in range(1, 8)]].median().round(3).to_string())
    print('\nQ_ov 中位:')
    print(df[[f'Q_ov_L{L}' for L in range(1, 8)]].median().round(3).to_string())
    print('\n达标率 (Q>=0.95)')
    print('A  ' + ' '.join(f'L{L}:{100*float((df[f"Q_A_L{L}"]>=0.95).mean()):.0f}%'
                           for L in range(1, 8)))
    print('ov ' + ' '.join(f'L{L}:{100*float((df[f"Q_ov_L{L}"]>=0.95).mean()):.0f}%'
                           for L in range(1, 8)))


if __name__ == '__main__':
    raise SystemExit(main())
