#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lei 数据集全扫描：9 个位置 × 6 次重复 × 3 带宽。
Full Lei sweep: 9 positions by 6 repeats by 3 bandwidths."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lei_pilot import analyse, ROOT

OUT = Path(r'C:\Users\yxh\Desktop\2\runtime\小波基研究\null_crossdataset')
RAW = ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data'
RAW2 = ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data'
EXPECT = 1200 * 3 * 4 * 200 * 2 * 2   # bytes（int16 × I/Q）


def pos_dirs():
    out = []
    for base in (RAW, RAW2):
        if base.exists():
            out += sorted([d for d in base.iterdir() if d.is_dir()])
    return out


def main():
    rows = []
    for d in pos_dirs():
        pos = d.name.replace('position_ (', '').replace(')', '')
        for bw in (3, 2.5, 2):
            tag = str(bw).replace('.', '_')
            for k in range(1, 7):
                fp = d / f'adc_{tag}GHZ_position{pos}_ ({k}).bin'
                if not fp.exists():
                    fp = d / f'adc_{tag}GHZ_positon{pos}_ ({k}).bin'  # 数据集内 position8 拼写错误
                if not fp.exists():
                    continue
                if fp.stat().st_size != EXPECT:
                    print('skip truncated', fp.name, fp.stat().st_size)
                    continue
                try:
                    r = analyse(bw, fp)
                    r.update({'position': int(pos), 'repeat': k})
                    rows.append(r)
                except Exception as e:
                    print('fail', fp.name, e)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'lei_all_positions.csv', index=False, encoding='utf-8')
    summary = {}
    for bw, g in df.groupby('bandwidth_GHz'):
        summary[str(bw)] = {
            'n': int(len(g)),
            'frac_n_ge5': float((g['n_deep'] >= 5).mean()),
            'median_n_deep': float(g['n_deep'].median()),
            'median_D': float(g['D'].median()),
            'median_D_others': float(g['D_others_median'].median()),
            'median_kappa': float(g['kappa'].median()),
            'median_Rbar_drift': float(g['Rbar_drift'].median()),
        }
    (OUT / 'lei_all_positions.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    raise SystemExit(main())
