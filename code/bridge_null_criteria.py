#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""桥接实验：小波准则库能否充当多径零点论文里的 bin 级判据？
Bridge experiment: can the wavelet criteria library serve as a bin-level criterion for the null paper?"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from scipy.stats import spearmanr

from lei_pilot import load, ROOT, R_BIN, FS
from lei_coverage_probe import detrend, ecg_hr, R_TRUE
from matrix_eval import decompose_dwt
from criteria_lib import evaluate_all, estimate_hr_psd

OUT_CSV = Path(__file__).resolve().parent / 'bridge_null_criteria.csv'
OUT_RHO = Path(__file__).resolve().parent / 'bridge_null_per_record_rho.csv'
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']
EXPECT = 1200 * 3 * 4 * 200 * 2 * 2
TAGS = {3: '3', 2.5: '2_5', 2: '2'}
VITAL = (0.1, 2.5)          # 生理频带（呼吸 + 心搏），与两篇论文的"生理带"一致
CRIT_COLS = ['pb1', 'pb2', 'pb3', 'pb4', 'pb5', 'sm2', 'sm4', 'sm5', 'rc3', 'rc4']


# ---------------------------------------------------------------------------
def bin_metrics(zc: np.ndarray) -> tuple[float, int]:
    """多径零点侧指标（定义与 lei_person_bin / 论文 v5 一致）。
Null-side metrics (definitions identical to lei_person_bin and paper v5)."""
    env = np.abs(zc)
    med = float(np.median(env))
    D = float(np.mean(env < 0.3 * med))
    pk, _ = find_peaks(-env, distance=int(0.25 * FS), prominence=0.15 * np.std(env))
    pk = pk[env[pk] < 0.35 * med]
    return D, int(len(pk))


def band_ratio_fourier(x: np.ndarray, fs: float = FS, band: tuple = VITAL) -> float:
    """Zhu 式：生理频带能量 / 非直流总能量（傅里叶域，不做任何分解）。
Zhu-style: physiological-band power over non-DC power, in the Fourier domain, no decomposition."""
    nper = int(min(len(x), 512))
    f, P = welch(x, fs=fs, nperseg=nper)
    m_tot = (f >= 0.05) & (f <= fs / 2.0)
    m_v = (f >= band[0]) & (f <= band[1])
    tot = float(P[m_tot].sum())
    return float(P[m_v].sum() / tot) if tot > 0 else float('nan')


def band_ratio_wavelet(coeffs, L: int, fs: float = FS, band: tuple = VITAL) -> tuple[float, float, float]:
    """小波子带能量评分：与 band_ratio_fourier 同一频带，但在倍频程子带上算。
Wavelet subband energy score: same band as band_ratio_fourier, computed on octave subbands."""
    bands = [(0.0, fs / 2 ** (L + 1))] + [(fs / 2 ** (j + 1), fs / 2 ** j)
                                          for j in range(1, L + 1)]
    e = np.array([float(np.sum(np.asarray(c) ** 2)) for c in coeffs])
    ov = np.array([min(b[1], band[1]) - max(b[0], band[0]) > 0 for b in bands])
    tot = float(e.sum())
    if tot <= 0:
        return float('nan'), float('nan'), float('nan')
    nodc = ov.copy()
    nodc[0] = False
    return (float(e[ov].sum() / tot), float(e[nodc].sum() / tot), float(e[0] / tot))


def gold_hrs(base: Path, pos: int, tag: str, k: int) -> dict:
    """两目标 ECG 心率（逐记录）。log 目录与雷达数据目录同级。
Two-target ECG heart rate per record; the log directory sits beside the radar data directory."""
    log_root = base.parent / '2_Log_data'
    out = {}
    for tgt in ('Target1', 'Target2'):
        fp = (log_root / tgt / f'position_ ({pos})'
              / f'log_{tgt}_{tag}GHZ_position{pos}_ ({k}).csv')
        if not fp.exists():
            continue
        v = pd.read_csv(fp, skiprows=1).iloc[:, 0].to_numpy()
        hr, _, _ = ecg_hr(-np.asarray(v, dtype=float))
        out[tgt] = float(hr)
    return out


# ---------------------------------------------------------------------------
def build_rows(only: set | None) -> pd.DataFrame:
    rows = []
    for base in BASES:
        if not base.exists():
            continue
        layout = base.parent.name
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            pos = int(d.name.replace('position_ (', '').replace(')', ''))
            if only and pos not in only:
                continue
            r1, r2 = R_TRUE[pos]
            separable = abs(r1 - r2) >= 0.2
            for bw in (3, 2.5, 2):
                tag = TAGS[bw]
                rb = R_BIN[bw]
                for k in range(1, 7):
                    fp = d / f'adc_{tag}GHZ_position{pos}_ ({k}).bin'
                    if not fp.exists():
                        fp = d / f'adc_{tag}GHZ_positon{pos}_ ({k}).bin'
                    if not fp.exists() or fp.stat().st_size != EXPECT:
                        continue
                    try:
                        z = load(fp)
                        zf = np.fft.fft(z, axis=3).mean(axis=(1, 2))
                    except Exception as e:                       # noqa: BLE001
                        print('load fail', fp.name, e)
                        continue
                    idx = []
                    for r in R_TRUE[pos]:
                        a = max(1, int((r - 0.25) / rb))
                        b = min(zf.shape[1] - 2, int((r + 0.05) / rb))
                        idx += list(range(a, b + 1))
                    idx = sorted(set(idx))
                    gh = gold_hrs(base, pos, tag, k) if separable else {}
                    for b in idx:
                        zc = zf[:, b]
                        D, n_deep = bin_metrics(zc)
                        x = detrend(np.unwrap(np.angle(zc)), FS)
                        res = decompose_dwt(x, 'sym4', 4, 'none', 'symmetric', FS)
                        if res is None:
                            continue
                        res['mix'] = x
                        crit = evaluate_all(res, FS)
                        rng = b * rb
                        if separable:
                            tgt = 'Target1' if abs(rng - r1) <= abs(rng - r2) else 'Target2'
                            hr_gold = gh.get(tgt, float('nan'))
                        else:
                            tgt, hr_gold = '', float('nan')
                        rW, rW_nodc, a_share = band_ratio_wavelet(res['coeffs'], 4)
                        row = {
                            'layout': layout, 'pos': pos, 'bw_GHz': bw, 'repeat': k,
                            'bin': b, 'range_m': round(rng, 3), 'target': tgt,
                            'D': D, 'n_deep': n_deep,
                            'As_over_R': float(abs(complex(np.mean(zc)))
                                               / (np.mean(np.abs(res['resp'])) + 1e-12)),
                            'rho_F': band_ratio_fourier(x),
                            'rho_F_low': band_ratio_fourier(x, FS, (0.1, FS / 32)),
                            'rho_W': rW, 'rho_W_nodc': rW_nodc, 'a_share': a_share,
                            'hr_bin': estimate_hr_psd(x, FS),
                            'hr_gold': hr_gold,
                        }
                        for kk, vv in crit.items():
                            if isinstance(vv, (int, float)) and np.isfinite(vv):
                                row[kk] = float(vv)
                        rows.append(row)
                    print(f'  {layout[:3]} pos{pos} {bw}GHz #{k}: {len(idx)} bins', flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def per_record_rho(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """逐记录 Spearman，并给出「控制 |A_s|/R 后的偏相关」。
Per-record Spearman, plus the partial correlation controlling for |A_s|/R."""
    recs = []
    keys = ['layout', 'pos', 'bw_GHz', 'repeat']
    for key, g in df.groupby(keys):
        if len(g) < 4:
            continue
        rec = dict(zip(keys, key))
        rec['n_bins'] = len(g)
        rec['rho_DA'] = (spearmanr(g.D, g.As_over_R)[0]
                         if g.D.nunique() > 1 and g.As_over_R.nunique() > 1 else np.nan)
        rec['rho_nA'] = (spearmanr(g.n_deep, g.As_over_R)[0]
                         if g.n_deep.nunique() > 1 and g.As_over_R.nunique() > 1 else np.nan)
        for c in cols + ['rho_F', 'rho_F_low', 'rho_W']:
            s = g[[c, 'D', 'n_deep', 'As_over_R']].dropna()
            for tg, pref in (('D', 'rhoD'), ('n_deep', 'rhoN'), ('As_over_R', 'rhoA')):
                rec[f'{pref}_{c}'] = np.nan
            rec[f'pN_{c}'] = np.nan
            if len(s) < 4 or s[c].nunique() < 2:
                continue
            for tg, pref in (('D', 'rhoD'), ('n_deep', 'rhoN'), ('As_over_R', 'rhoA')):
                if s[tg].nunique() > 1:
                    rec[f'{pref}_{c}'] = spearmanr(s[c], s[tg])[0]
            # 控制 |As|/R 后的偏相关（用 n_deep 作零点指标：D 在 70% 的记录里恒为 0）
            r_ca, r_na = rec[f'rhoA_{c}'], rec.get('rho_nA')
            if (np.isfinite(r_ca) and r_na is not None and np.isfinite(r_na)
                    and np.isfinite(rec[f'rhoN_{c}'])):
                den = np.sqrt(max((1 - r_ca ** 2) * (1 - r_na ** 2), 1e-12))
                rec[f'pN_{c}'] = (rec[f'rhoN_{c}'] - r_ca * r_na) / den
        recs.append(rec)
    return pd.DataFrame(recs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='小波准则库 × 多径零点判据 桥接实验')
    ap.add_argument('--only', type=int, nargs='*', default=None)
    args = ap.parse_args(argv)

    df = build_rows(set(args.only) if args.only else None)
    if df.empty:
        print('no data')
        return 1
    df.to_csv(OUT_CSV, index=False, encoding='utf-8')
    print(f'\n[data] bins={len(df)} records={df.groupby(["layout","pos","bw_GHz","repeat"]).ngroups}')
    print(f'[data] 可归属两目标的位置（|r1-r2|>=0.2m）: '
          f'hr_gold 非空 {int(df.hr_gold.notna().sum())} bins')

    cols = [c for c in CRIT_COLS if c in df.columns]

    # ---- Q1：逐记录相关（控制几何）----
    rho = per_record_rho(df, cols)
    rho.to_csv(OUT_RHO, index=False, encoding='utf-8')
    sDA = rho['rho_DA'].dropna()
    sNA = rho['rho_nA'].dropna()
    print(f'\n=== Q1 逐记录 Spearman（n_records={len(rho)}），控制几何/带宽/受试者 ===')
    print(f'[基准] D 有方差的记录 {len(sDA)}/{len(rho)}（中位 D 恒为 0 → D 在单记录内多半退化），'
          f'Spearman(D,|As|/R) 中位 {sDA.median() if len(sDA) else np.nan:+.2f}')
    print(f'[基准] Spearman(n_deep,|As|/R) 中位 {sNA.median() if len(sNA) else np.nan:+.2f} '
          f'(IQR [{np.percentile(sNA,25):+.2f},{np.percentile(sNA,75):+.2f}]) '
          f'→ 零点指标与信号强弱本身同向，必须看偏相关列')
    print(f'{"crit":9s} {"rhoN":>6s} {"IQR":>14s} {"|rhoN|>.5":>10s} '
          f'{"rhoA":>6s} {"偏相关pN":>8s} {"|pN|>.5":>8s} {"rhoD(n)":>9s}')
    for c in cols + ['rho_F', 'rho_F_low', 'rho_W']:
        key = f'rhoN_{c}'
        if key not in rho.columns:
            continue
        s = rho[key].dropna()
        if s.empty:
            continue
        q1, q3 = np.percentile(s, [25, 75])
        a = rho[f'rhoA_{c}'].dropna()
        p = rho[f'pN_{c}'].dropna()
        dd = rho[f'rhoD_{c}'].dropna()
        print(f'{c:9s} {s.median():6.2f} [{q1:6.2f},{q3:6.2f}] '
              f'{float((s.abs()>=0.5).mean()):10.0%} '
              f'{a.median() if len(a) else np.nan:6.2f} '
              f'{p.median() if len(p) else np.nan:8.2f} '
              f'{float((p.abs()>=0.5).mean()) if len(p) else np.nan:8.0%} '
              f'{dd.median() if len(dd) else np.nan:+.2f}({len(dd)})')

    # ---- Q2：Zhu 式傅里叶带能量比 vs 小波子带能量比 ----
    s = df[['rho_F', 'rho_W']].dropna()
    r_pool = spearmanr(s.rho_F, s.rho_W)[0] if len(s) > 3 else np.nan
    per, picks, per_low = [], [], []
    for key, g in df.groupby(['layout', 'pos', 'bw_GHz', 'repeat']):
        gg = g[['rho_F', 'rho_W']].dropna()
        if len(gg) >= 4 and gg.rho_F.nunique() > 1 and gg.rho_W.nunique() > 1:
            per.append(spearmanr(gg.rho_F, gg.rho_W)[0])
        gl = g[['rho_F_low', 'rho_W']].dropna()
        if len(gl) >= 4 and gl.rho_F_low.nunique() > 1 and gl.rho_W.nunique() > 1:
            per_low.append(spearmanr(gl.rho_F_low, gl.rho_W)[0])
        if g.rho_F.notna().any() and g.rho_W.notna().any():
            picks.append((int(g.loc[g.rho_F.idxmax(), 'bin']),
                          int(g.loc[g.rho_W.idxmax(), 'bin'])))
    per = np.asarray(per)
    per_low = np.asarray(per_low)
    pk = np.asarray(picks)
    print('\n=== Q2 Zhu(傅里叶带能量比) vs 小波子带能量比 ===')
    print(f'分布：rho_F 中位 {s.rho_F.median():.4f} (p5 {s.rho_F.quantile(.05):.4f}, '
          f'p95 {s.rho_F.quantile(.95):.4f})；rho_W 中位 {s.rho_W.median():.4f} '
          f'(p5 {s.rho_W.quantile(.05):.4f}, p95 {s.rho_W.quantile(.95):.4f})')
    print(f'pooled Spearman = {r_pool:.3f} (n={len(s)})；逐记录中位 = '
          f'{np.median(per) if len(per) else np.nan:.3f} '
          f'(n={len(per)}, |rho|>=0.5 占 {float((np.abs(per)>=0.5).mean()) if len(per) else np.nan:.0%})')
    print(f'诊断（把小波最低子带 a4=[0,{FS/32:.3f}]Hz 换成同频带傅里叶比 rho_F_low）：'
          f'逐记录中位 = {np.median(per_low) if len(per_low) else np.nan:.3f} '
          f'(n={len(per_low)}, |rho|>=0.5 占 '
          f'{float((np.abs(per_low)>=0.5).mean()) if len(per_low) else np.nan:.0%})')
    s2 = df[['rho_F', 'rho_W_nodc']].dropna()
    print(f'诊断（把近似带从 rho_W 里去掉再与 rho_F 比）：pooled Spearman = '
          f'{spearmanr(s2.rho_F, s2.rho_W_nodc)[0]:.3f} (n={len(s2)})；'
          f'近似带占系数能量 中位 {df.a_share.median():.3f} '
          f'(IQR {df.a_share.quantile(.25):.3f}–{df.a_share.quantile(.75):.3f})')
    if len(pk):
        print(f'两者各自选出的 bin：完全相同 {float((pk[:,0]==pk[:,1]).mean()):.0%}，'
              f'相差≤1 {float((np.abs(pk[:,0]-pk[:,1])<=1).mean()):.0%} (n={len(pk)})')

    # ---- Q3：零点 bin 的 HR 代价 ----
    d3 = df.dropna(subset=['hr_gold', 'hr_bin']).copy()
    if len(d3) > 20:
        d3['err'] = (d3.hr_bin - d3.hr_gold).abs()
        print(f'\n=== Q3 零点代价（n={len(d3)} bins, {d3.groupby(["layout","pos","bw_GHz","repeat"]).ngroups} 条记录）===')
        print(f'[基线] |ΔHR| 中位 {d3.err.median():.1f} BPM，'
              f'≤3 BPM 占 {float((d3.err<=3).mean()):.0%}（20 Hz 单 bin PSD 本底，见 §4.13(f)）')
        rr = []
        for key, g in d3.groupby(['layout', 'pos', 'bw_GHz', 'repeat']):
            if len(g) < 5 or g.err.nunique() < 2:
                continue
            rr.append({
                'rho_nD': (spearmanr(g.n_deep, g.err)[0] if g.n_deep.nunique() > 1 else np.nan),
                'rho_D': (spearmanr(g.D, g.err)[0] if g.D.nunique() > 1 else np.nan),
                'rho_A': (spearmanr(g.As_over_R, g.err)[0] if g.As_over_R.nunique() > 1 else np.nan),
            })
        rr = pd.DataFrame(rr)
        print(f'逐记录 Spearman(·, |ΔHR|)，n={len(rr)}：'
              f'n_deep {rr.rho_nD.median():+.2f} (|rho|>=.5 占 '
              f'{float((rr.rho_nD.abs()>=.5).mean()):.0%})；'
              f'D {rr.rho_D.median():+.2f} ({float((rr.rho_D.abs()>=.5).mean()):.0%})；'
              f'|As|/R {rr.rho_A.median():+.2f} ({float((rr.rho_A.abs()>=.5).mean()):.0%})')
        d3['nb'] = pd.cut(d3.n_deep, [-1, 0, 4, 15, 1000],
                          labels=['0', '1-4', '5-15', '>15'])
        t = d3.groupby('nb', observed=True).agg(
            n=('err', 'size'), hr_err_med=('err', 'median'),
            within3=('err', lambda x: float((x <= 3).mean())),
            As_med=('As_over_R', 'median'))
        print(t.round(3).to_string())
    else:
        print('\n=== Q3 跳过：可归属 bin 太少 ===')

    print(f'\n[saved] {OUT_CSV.name}, {OUT_RHO.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
