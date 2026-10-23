#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用同步 ECG/PCG 验证 Lei 数据选中 bin 的生理性。
Check the physiological plausibility of the selected Lei bin against synchronous ECG and PCG."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from lei_pilot import analyse, ROOT

LOG = ROOT / '1_AsymmetricalPosition' / '2_Log_data' / 'Target1' / 'position_ (1)'
FS_LOG = 125.0
DUR = 3500 / FS_LOG


def physio(fp):
    df = pd.read_csv(fp, skiprows=1)
    ecg = -df.iloc[:, 0].to_numpy()
    pcg = df.iloc[:, 1].to_numpy()
    ecg = ecg[-3500:]
    pcg = pcg[-3500:]
    # 呼吸：低通后找峰
    k = np.ones(25) / 25
    pcg_s = np.convolve(pcg, k, 'same')
    rp, _ = find_peaks(pcg_s, distance=int(2.0 * FS_LOG))
    hp, _ = find_peaks(ecg, height=20, distance=int(0.3 * FS_LOG))
    return {
        'resp_bpm': len(rp) / DUR * 60,
        'hr_bpm': len(hp) / DUR * 60,
        'ecg_n': len(hp), 'pcg_n': len(rp),
    }


def main():
    for bw in (3, 2.5, 2):
        tag = {3: '3', 2.5: '2_5', 2: '2'}[bw]
        radar_fp = ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data' / 'position_ (1)' / f'adc_{tag}GHZ_position1_ (1).bin'
        log_fp = LOG / f'log_Target1_{tag}GHZ_position1_ (1).csv'
        if not radar_fp.exists() or not log_fp.exists():
            print('missing', bw); continue
        r = analyse(bw, radar_fp)
        p = physio(log_fp)
        print(f"{bw} GHz: radar f_resp={r['f_resp']*60:.1f} bpm, f_dip={r['f_dip']*60:.1f} bpm | "
              f"PCG resp={p['resp_bpm']:.1f} bpm, ECG HR={p['hr_bpm']:.1f} bpm")


if __name__ == '__main__':
    raise SystemExit(main())
