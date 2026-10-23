#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lei 数据集：位置 1 的 6 次重复 × 3 带宽扫描。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lei_pilot import analyse, POS

OUT = Path(r'C:\Users\yxh\Desktop\2\runtime\小波基研究\null_crossdataset')


def main():
    rows = []
    for bw in (3, 2.5, 2):
        tag = str(bw).replace('.', '_')
        for k in range(1, 7):
            fp = POS / f'adc_{tag}GHZ_position1_ ({k}).bin'
            if not fp.exists():
                print('missing', fp.name); continue
            r = analyse(bw, fp)
            r['repeat'] = k
            rows.append(r)
            print(f"{bw:>4} GHz #{k}: bin {r['bin']:>2} ({r['range_m']:.2f}m) "
                  f"n={r['n_deep']:>3} D={r['D']:.3f}/{r['D_others_median']:.3f} "
                  f"kappa={r['kappa']:.2f} Rf={r['Rbar_fixed']:.2f} Rd={r['Rbar_drift']:.2f}")
    (OUT / 'lei_position1_scan.json').write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    for bw in (3, 2.5, 2):
        g = [r for r in rows if r['bandwidth_GHz'] == bw]
        print(f'{bw} GHz median: n={np.median([x["n_deep"] for x in g]):.0f} '
              f'D={np.median([x["D"] for x in g]):.3f} '
              f'D_other={np.median([x["D_others_median"] for x in g]):.3f} '
              f'kappa={np.nanmedian([x["kappa"] for x in g]):.2f}')


if __name__ == '__main__':
    raise SystemExit(main())
