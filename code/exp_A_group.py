#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A 组：网格外推 / θ0 推广 / 留出验证 / 稳健性扫描。
Group A: grid extrapolation, theta0 generalization, held-out test and robustness sweep."""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import rankdata, spearmanr

from lei_pilot import load, ROOT, R_BIN, FS, bp
from exp_center_removal import circle_geometric, EXPECT

OUT = Path(__file__).resolve().parent / 'exp_out'
FIG = Path(__file__).resolve().parent / 'figs'
R_TRUE = {1: (1.0, 1.0), 2: (1.4, 1.4), 3: (1.4, 1.0), 4: (1.0, 0.6), 5: (1.4, 1.0),
          6: (0.6, 1.0), 7: (0.6, 0.6), 8: (1.0, 1.0), 9: (1.4, 1.4)}
BASES = [ROOT / '1_AsymmetricalPosition' / '1_Radar_Raw_Data',
         ROOT / '2_SymmetricalPosition' / '1_Radar_Raw_Data']
DEPTHS = (0.25, 0.30, 0.35, 0.40)
PROMS = (0.10, 0.15, 0.20)
NULL_RECORD = 5
RHO_LO, RHO_HI = 0.7391, 1.3529


# --------------------------------------------------------------- 解析模型
def depth_model(rho, th_max, th0=0.0):
    """归一化深度 d = min|z|/max|z|，弧段 [θ0-θmax, θ0+θmax]。
Normalized depth d = min|z|/max|z| over the arc [theta0-theta_max, theta0+theta_max]."""
    rho = np.asarray(rho, float)
    th_max = np.asarray(th_max, float)
    # 弧段到 π 的角距离
    lo, hi = th0 - th_max, th0 + th_max
    d_lo = np.abs(np.mod(lo - np.pi + np.pi, 2 * np.pi) - np.pi)
    inner = (lo <= np.pi) & (np.pi <= hi)
    # 覆盖 π 时取 |ρ-1|；否则取端点最近者
    d_in = np.sqrt(rho ** 2 + 1 + 2 * rho * np.cos(th_max))   # θ0=π 时的端点
    d_out = np.abs(rho - 1)
    return np.where(inner, d_out / (rho + 1), d_in / (rho + 1))


def coverage(th_max, th0=0.0):
    """弧段 [θ0-θmax, θ0+θmax] 是否覆盖 π。
Whether the arc [theta0-theta_max, theta0+theta_max] covers pi."""
    lo, hi = th0 - th_max, th0 + th_max
    return (lo <= np.pi) & (np.pi <= hi)


# --------------------------------------------------------------- 数据
def build():
    rows, dips = [], []
    for base in BASES:
        if not base.exists():
            continue
        layout = base.parent.name
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            pos = int(d.name.replace('position_ (', '').replace(')', ''))
            for bw in (3, 2.5, 2):
                tag = {3: '3', 2.5: '2_5', 2: '2'}[bw]
                rb = R_BIN[bw]
                for k in range(1, 7):
                    fp = d / f'adc_{tag}GHZ_position{pos}_ ({k}).bin'
                    if not fp.exists():
                        fp = d / f'adc_{tag}GHZ_positon{pos}_ ({k}).bin'
                    if not fp.exists() or fp.stat().st_size != EXPECT:
                        continue
                    zf = np.fft.fft(load(fp), axis=3).mean(axis=(1, 2))
                    # V2：整段 std 最大（Table II 协议）
                    idx = []
                    for r in R_TRUE[pos]:
                        idx += list(range(max(1, int((r - .25) / rb)),
                                          int((r + .05) / rb) + 1))
                    idx = sorted(set(idx))
                    b = idx[int(np.argmax(np.std(zf[:, idx], axis=0)))]
                    z = zf[:, b]
                    h = len(z) // 2
                    env = np.abs(z)
                    med = float(np.median(env))
                    cf, rf, _ = circle_geometric(z)
                    a = bp(z)
                    row = {'layout': layout, 'pos': pos, 'bw_GHz': bw, 'repeat': k,
                           'bin': b, 'range_m': round(b * rb, 3),
                           'AsR': float(abs(complex(np.mean(z)))
                                        / (np.mean(np.abs(a)) + 1e-12)),
                           'rho_fit': float(abs(cf) / (rf + 1e-12))}
                    # 弧段参数（相对拟合圆心）
                    th = np.unwrap(np.angle(z - cf))
                    row['th_max'] = float((th.max() - th.min()) / 2)
                    row['th_center'] = float(np.angle(np.mean(np.exp(1j * th))))
                    # 弧段自动覆盖整圈 ⇔ θ_max ≥ π（与弧中心无关）
                    row['covered'] = bool(row['th_max'] >= np.pi)
                    # 稳健性网格
                    for dep in DEPTHS:
                        for pr in PROMS:
                            pk, _ = find_peaks(-env, distance=int(0.25 * FS),
                                               prominence=pr * np.std(env))
                            ok = env[pk] < dep * med
                            row[f'n_{dep}_{pr}'] = int(ok.sum())
                            row[f'n1_{dep}_{pr}'] = int((pk[ok] < h).sum())
                            row[f'n2_{dep}_{pr}'] = int((pk[ok] >= h).sum())
                    # 逐凹陷：观测深度 vs 模型预测深度
                    pk, _ = find_peaks(-env, distance=int(0.25 * FS),
                                       prominence=0.15 * np.std(env))
                    pk = pk[env[pk] < 0.35 * med]
                    w = int(2.0 * FS)
                    for p in pk:
                        s, e = max(0, p - w), min(len(z), p + w)
                        cw, rw, _ = circle_geometric(z[s:e])
                        if rw <= 0:
                            continue
                        rho_l = abs(cw) / rw
                        d_obs = abs(z[p]) / (abs(cw) + rw + 1e-12)
                        d_pred = abs(rho_l - 1) / (rho_l + 1)
                        dips.append({'layout': layout, 'pos': pos, 'bw_GHz': bw,
                                     'repeat': k, 'rho_local': float(rho_l),
                                     'd_obs': float(d_obs), 'd_pred': float(d_pred)})
                    # 基线：随机时刻的局部 ρ 落在失败带内的比例（同样的 4 s 窗）
                    # 确定性种子：hash() 在 Python 中受 PYTHONHASHSEED 影响，每次运行不同
                    seed = zlib.crc32(f'{layout}|{pos}|{bw}|{k}'.encode()) % (2 ** 31)
                    rng = np.random.default_rng(seed)
                    band = []
                    for p in rng.choice(np.arange(w, len(z) - w), 20, replace=False):
                        cw, rw, _ = circle_geometric(z[p - w:p + w])
                        if rw > 0:
                            band.append(RHO_LO <= abs(cw) / rw <= RHO_HI)
                    row['rand_band'] = float(np.mean(band)) if band else np.nan
                    rows.append(row)
                    print('.', end='', flush=True)
    print()
    return pd.DataFrame(rows), pd.DataFrame(dips)


# --------------------------------------------------------------- A1/A2
def section_A1A2(df, dips):
    print('\n' + '=' * 94)
    print('A1. 失败区外推：θ_max ≥ π 后与 θ 无关，退化为 ρ 的窄带')
    print('=' * 94)
    frac = float(df.covered.mean())
    print(f'  实测 θ_max：中位 {df.th_max.median():.2f} rad，'
          f'P10 {df.th_max.quantile(.1):.2f}，P90 {df.th_max.quantile(.9):.2f}')
    print(f'  θ_max ≥ π（弧段必然覆盖整圈、含反相方向）的记录占 {frac:.1%}；'
          f'θ_max ≥ 2.84 rad（15% 深度判据）的占 '
          f'{(df.th_max >= 2.84).mean():.1%}')

    grid_r = np.linspace(0.1, 3.0, 400)
    for thm in (1.0, 2.0, 2.84, np.pi, 6.0, 12.0):
        d = depth_model(grid_r, np.full_like(grid_r, thm))
        ok = grid_r[d <= 0.15]
        print(f'    θ_max={thm:>5.2f} rad → 失败区 ρ ∈ '
              f'[{ok.min():.3f}, {ok.max():.3f}]' if len(ok)
              else f'    θ_max={thm:>5.2f} rad → 失败区为空')
    print(f'  ⇒ 全覆盖时（θ_max ≥ π）失败区恒为 ρ ∈ [{RHO_LO:.3f}, {RHO_HI:.3f}]，'
          f'与 θ_max 无关。')

    print('\n' + '=' * 94)
    print('A2. 逐凹陷检验：观测深度 vs 模型预测 |ρ_local-1|/(ρ_local+1)')
    print('=' * 94)
    ok = np.isfinite(dips.d_pred) & np.isfinite(dips.d_obs)
    dd = dips[ok]
    r = spearmanr(dd.d_pred, dd.d_obs).statistic
    err = float(np.median(np.abs(dd.d_pred - dd.d_obs)))
    print(f'  凹陷事件 {len(dd)} 个；Spearman {r:+.2f}；'
          f'中位绝对误差 {err:.3f}（深度为 0–1 归一化）')
    for bw in (3, 2.5, 2):
        g = dd[dd.bw_GHz == bw]
        print(f'    {bw:>3.1f} GHz: n={len(g):>5}  ρ_local 中位 {g.rho_local.median():>5.2f}  '
              f'd_pred {g.d_pred.median():.2f} vs d_obs {g.d_obs.median():.2f}')
    print(f'  逐记录 ρ_fit 落在失败带 [{RHO_LO:.2f},{RHO_HI:.2f}] 内的记录：'
          f'{np.mean((df.rho_fit >= RHO_LO) & (df.rho_fit <= RHO_HI)):.1%}')
    dip_band = float(np.mean((dd.rho_local >= RHO_LO) & (dd.rho_local <= RHO_HI)))
    base_band = float(df.rand_band.mean())
    print(f'  ρ_local 落在失败带内：凹陷时刻 {dip_band:.1%} vs '
          f'随机时刻 {base_band:.1%} → 富集 {dip_band / max(base_band, 1e-9):.1f}×')


# --------------------------------------------------------------- A3
def auc(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float((rankdata(x)[y == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def section_A3(df):
    print('\n' + '=' * 94)
    print('A3. 留出验证：按位置分组的 leave-one-position-out')
    print('=' * 94)
    d = df.copy()
    d['x'] = d['n_0.35_0.15']
    d['x'] = d['n1_0.35_0.15']           # 前半段计数值
    d['y'] = (d['n2_0.35_0.15'] >= 1).astype(int)
    print(f'  样本 {len(d)} 条；正例（后半段仍有凹陷）{d.y.mean():.1%}')
    preds, groups = [], []
    print(f'  {"留出位置":>10}{"n":>5}{"阈值":>7}{"AUC":>8}{"sens":>8}{"spec":>8}')
    for (lay, pos), g in d.groupby(['layout', 'pos']):
        tr = d.drop(g.index)
        best = None
        for k in sorted(tr.x.unique()):
            p = tr.x >= k
            if p.sum() == 0 or (~p).sum() == 0:
                continue
            j = (p[tr.y == 1].mean()) + ((~p)[tr.y == 0].mean()) - 1
            if best is None or j > best[1]:
                best = (k, j)
        thr = best[0] if best else 1
        a = auc(g.x, g.y)
        pred = g.x >= thr
        sens = pred[g.y == 1].mean() if (g.y == 1).any() else np.nan
        spec = (~pred)[g.y == 0].mean() if (g.y == 0).any() else np.nan
        print(f'  {str((lay[:4], pos)):>10}{len(g):>5}{thr:>7}{a if a == a else float("nan"):>8.2f}'
              f'{sens:>8.2f}{spec:>8.2f}' if a == a else
              f'  {str((lay[:4], pos)):>10}{len(g):>5}{thr:>7}{"n/a":>8}{sens:>8.2f}{spec:>8.2f}')
        if a == a:
            preds.append(a)
            groups.append((lay, pos))
    print(f'  → 留出 AUC 中位 {np.median(preds):.2f}，'
          f'范围 {min(preds):.2f}–{max(preds):.2f}（{len(preds)} 折）')
    print(f'  → 合并（in-sample，供对照）AUC {auc(d.x, d.y):.2f}')


# --------------------------------------------------------------- A4
def section_A4(df):
    print('\n' + '=' * 94)
    print('A4. 稳健性：零点率随深度阈值 / prominence / 零点门限变化')
    print('=' * 94)
    print(f'  {"深度":>6}{"prom":>6}{"零点门限":>9}   '
          f'{"5.0cm":>8}{"6.0cm":>8}{"7.5cm":>8}{"单调?":>7}')
    for dep in DEPTHS:
        for pr in PROMS:
            for nthr in (3, 5, 8):
                vals = [float((df[df.bw_GHz == bw][f'n_{dep}_{pr}'] >= nthr).mean())
                        for bw in (3, 2.5, 2)]
                mono = 'Y' if vals[0] < vals[1] < vals[2] else 'n'
                print(f'  {dep:>6.2f}{pr:>6.2f}{nthr:>9}   '
                      f'{vals[0]:>7.1%}{vals[1]:>7.1%}{vals[2]:>7.1%}{mono:>7}')
    print('\n  换成"前 30 s 计数"作为判据（与 R1 的预锁窗一致）：')
    for dep in DEPTHS:
        for nthr in (3, 5):
            vals = [float((df[df.bw_GHz == bw][f'n1_{dep}_0.15'] >= nthr).mean())
                    for bw in (3, 2.5, 2)]
            print(f'    深度 {dep:.2f} 门限 {nthr}: '
                  f'{vals[0]:.1%} / {vals[1]:.1%} / {vals[2]:.1%}')


def make_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    rho = np.linspace(0.1, 3.0, 300)
    thm = np.linspace(0.02, 16.0, 300)
    R, T = np.meshgrid(rho, thm)
    D = depth_model(R, T, np.zeros_like(R))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.pcolormesh(R, T, D, cmap='viridis_r', shading='auto')
    ax.contour(R, T, D, levels=[0.15], colors='w', linewidths=1.6)
    ax.axhline(np.pi, color='#E45756', ls='--', lw=1.2)
    ax.text(0.12, np.pi + 0.35, r'$\theta_{\max}=\pi$', color='#B02A2A',
            fontsize=9, ha='left')
    ax.annotate(r'$\theta_{\max}<2.84$ rad：不可能出现 15% 深度凹陷',
                xy=(1.45, 1.30), fontsize=8.5, color='#222222', ha='left')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('归一化深度 $d=\\min|z|/\\max|z|$')
    ax.set_xlabel(r'静止/运动相量比 $\rho=|A_s|/|A_c|$')
    ax.set_ylabel(r'峰值摆动 $\theta_{\max}$ (rad)')
    ax.set_title('失败区（白线内 $d\\leq15\\%$）：$\\theta_{\\max}\\geq\\pi$ 后与摆动无关',
                 fontsize=10)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / 'fig_failure_map_ext.png', dpi=300)
    print(f'\n[fig] {FIG / "fig_failure_map_ext.png"}')
    print(f'  失败带（d≤15%，全覆盖）：ρ ∈ [{RHO_LO:.3f}, {RHO_HI:.3f}]')


def main():
    OUT.mkdir(exist_ok=True)
    cache_r, cache_d = OUT / 'A_records.csv', OUT / 'A_dips.csv'
    if '--from-csv' in sys.argv and cache_r.exists():
        df, dips = pd.read_csv(cache_r), pd.read_csv(cache_d)
        print(f'[data] 复用缓存 {len(df)} 条 / {len(dips)} 个凹陷')
    else:
        df, dips = build()
        df.to_csv(cache_r, index=False, encoding='utf-8')
        dips.to_csv(cache_d, index=False, encoding='utf-8')
        print(f'[data] {len(df)} 条记录，{len(dips)} 个凹陷事件')
    section_A1A2(df, dips)
    section_A3(df)
    section_A4(df)
    make_figure()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
