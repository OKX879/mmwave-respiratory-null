#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lei 20 Hz 平台的覆盖率 Q 全量扫描（第三个采样率点）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lei_pilot import load, ROOT, R_BIN, FS
from parralejo_coverage import cardiac_recon, amp_at, recon_approx
from lei_coverage_probe import ecg_hr, pick_bin_and_phase, detrend, R_TRUE

OUT = Path('lei_coverage_20hz.csv')
EXPSIZE = 1200 * 3 * 4 * 200 * 2 * 2
TAGS = {3: '3', 2.5: '2_5', 2: '2'}
POS_BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
             ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']
LOG_BASES = [ROOT / '1_AsymmetricalPosition' / '2_Log_data',
             ROOT / '2_SymmetricalPosition' / '2_Log_data']


def find_log(logbase, tgt, tag, pos, k):
    for name in (f'log_{tgt}_{tag}GHZ_position{pos}_ ({k}).csv',
                 f'log_{tgt}_{tag}GHZ_positon{pos}_ ({k}).csv'):
        fp = logbase / tgt / f'position_ ({pos})' / name
        if fp.exists():
            return fp
    return None


def main():
    rows = []
    for pb, lb in zip(POS_BASES, LOG_BASES):
        if not pb.exists():
            continue
        for d in sorted(pb.iterdir()):
            if not d.is_dir():
                continue
            pos = int(d.name.replace('position_ (', '').replace(')', ''))
            for bw in (3, 2.5, 2):
                tag = TAGS[bw]
                for k in range(1, 7):
                    fp = d / f'adc_{tag}GHZ_position{pos}_ ({k}).bin'
                    if not fp.exists():
                        fp = d / f'adc_{tag}GHZ_positon{pos}_ ({k}).bin'
                    if not fp.exists() or fp.stat().st_size != EXPSIZE:
                        continue
                    try:
                        ph, b, rng = pick_bin_and_phase(bw, fp, pos)
                        x = detrend(ph, FS)
                    except Exception as e:
                        print('radar fail', fp.name, e)
                        continue

                    for tgt in ('Target1', 'Target2'):
                        log = find_log(lb, tgt, tag, pos, k)
                        if log is None:
                            continue
                        try:
                            ecg = -pd.read_csv(log, skiprows=1).iloc[:, 0].to_numpy()
                            hr, cv, n = ecg_hr(ecg)
                        except Exception as e:
                            print('ecg fail', log.name, e)
                            continue
                        row = {'subj': tgt, 'position': pos, 'bw_GHz': bw,
                               'repeat': k, 'bin': b, 'range_m': round(rng, 3),
                               'hr_bpm': hr, 'rr_cv': cv, 'n_rpeak': n,
                               'fs_hz': FS}
                        if not (60 <= hr <= 150):
                            row['status'] = 'out_of_band'
                            rows.append(row)
                            continue
                        row['status'] = 'ok'
                        f0 = hr / 60.0
                        ref = amp_at(x, f0, FS)
                        if ref <= 0:
                            row['status'] = 'no_line'
                            rows.append(row)
                            continue
                        for L in range(1, 7):
                            row[f'Q_A_L{L}'] = amp_at(
                                recon_approx(x, 'sym4', L, FS), f0, FS) / ref
                            xc, _ = cardiac_recon(x, 'sym4', L, FS, 'overlap')
                            row[f'Q_ov_L{L}'] = 0.0 if xc is None else \
                                amp_at(xc, f0, FS) / ref
                        rows.append(row)
                    print(f'  pos{pos} {bw}GHz #{k}: bin {b} ({rng:.2f} m) ok', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding='utf-8')
    ok = df[df.status == 'ok']
    print(f'\nrecords {len(df)}  ok {len(ok)}  out_of_band '
          f'{(df.status == "out_of_band").sum()}')
    print('\n--- Q_A (aL) 中位 ---')
    print(ok[[f'Q_A_L{L}' for L in range(1, 7)]].median().round(3).to_string())
    print('\n--- Q_overlap (details) 中位 ---')
    print(ok[[f'Q_ov_L{L}' for L in range(1, 7)]].median().round(3).to_string())
    print('\n--- 达标率 (Q>=0.95) ---')
    for pref, name in ((('Q_A_L'), 'A'), (('Q_ov_L'), 'ov')):
        s = {L: float((ok[f'{pref}{L}'] >= 0.95).mean()) for L in range(1, 7)}
        print(name, ' '.join(f'L{L}:{100*v:.0f}%' for L, v in s.items()))
    print('\nHR 中位 %.1f bpm  (P10/P90 %.1f/%.1f)'
          % (ok.hr_bpm.median(), ok.hr_bpm.quantile(.1), ok.hr_bpm.quantile(.9)))


if __name__ == '__main__':
    raise SystemExit(main())
