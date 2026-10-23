#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐搏零点检验：在 cardiac 带（0.8–2.5 Hz）内看复平面轨迹与幅度凹陷。
Beat-level null test: complex-plane trajectory and amplitude dips in the cardiac band (0.8-2.5 Hz)."""
from __future__ import annotations

from paths import CAPSULE

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATA = CAPSULE
OUT = HERE / "null_probe_out"
FS = 100.0
CARD = (0.8, 2.5)      # 心搏带（与准则库 CARD_BAND 一致）


# ---------------------------------------------------------------------------
def load(stem: str):
    with h5py.File(DATA / f"{stem}.mat", "r") as f:
        raw = f[f["data_all"][0, 0]][:]
        lbl = np.asarray(f[f["data_all"][1, 0]][:]).ravel()
    z = (raw["real"] + 1j * raw["imag"]).astype(np.complex128)
    return z, lbl


def bp_complex(x: np.ndarray, band=CARD, order: int = 4) -> np.ndarray:
    """复数带通：对实部/虚部分别做零相位滤波（线性算子，等价于复数滤波）。
Complex bandpass: zero-phase filtering of the real and imaginary parts separately, equivalent to complex filtering."""
    sos = butter(order, [band[0], band[1]], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x.real) + 1j * sosfiltfilt(sos, x.imag)


def kasa(z: np.ndarray):
    x, y = z.real, z.imag
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0], sol[1]
    R = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0)))
    return float(cx), float(cy), R


def envelope_dips(env: np.ndarray, fs=FS, min_sep_s=0.25):
    """幅度包络的局部极小（凹陷），返回索引与深度比。
Local minima of the amplitude envelope (dips), returning indices and depth ratios."""
    inv = -env
    pk, _ = find_peaks(inv, distance=int(min_sep_s * fs),
                       prominence=0.15 * np.std(env))
    if len(pk) == 0:
        return np.array([], dtype=int), np.array([])
    depth = env[pk] / max(float(np.percentile(env, 90)), 1e-12)
    return pk, depth


def rayleigh(phases: np.ndarray) -> tuple[float, float]:
    """返回 (R̄, Rayleigh p)。phases 为 [0,1) 的循环相位。
Returns (mean R, Rayleigh p) for circular phases on [0,1)."""
    n = len(phases)
    if n < 5:
        return float("nan"), float("nan")
    Rbar = float(abs(np.mean(np.exp(1j * 2 * np.pi * phases))))
    p = float(np.exp(-n * Rbar ** 2))
    return Rbar, p


# ---------------------------------------------------------------------------
def analyse(stem: str) -> dict:
    z, lbl = load(stem)
    # 通道/角度：用通道均值融合（复数相加），选"心搏带能量"最强的 bin
    zm = z[:, 0, :, :].mean(axis=0)                     # (bin, T) 复数融合
    card = np.stack([bp_complex(zm[b]) for b in range(zm.shape[0])])   # (bin, T)
    b = int(np.argmax(np.std(card, axis=1)))
    zc = card[b]

    env = np.abs(zc)
    pk, depth = envelope_dips(env)
    # 凹陷频率（对包络做 FFT）
    v = env - env.mean()
    F = np.abs(np.fft.rfft(v))
    f = np.fft.rfftfreq(len(v), 1 / FS)
    m = (f >= 0.5) & (f <= 3.0)
    f_dip = float(f[m][np.argmax(F[m])]) if m.any() else float("nan")

    gold = (60000.0 / np.median(np.diff(lbl) / FS * 1000.0)) if len(lbl) > 1 else float("nan")
    hr_hz = gold / 60.0 if np.isfinite(gold) else float("nan")
    rel = abs(f_dip - hr_hz) / hr_hz if np.isfinite(hr_hz) else float("nan")

    # Q2：把凹陷时刻折叠到心搏相位
    beat_s = lbl / FS
    phases = []
    for i in pk:
        t = i / FS
        j = int(np.searchsorted(beat_s, t))
        if j <= 0 or j >= len(beat_s):
            continue
        t0, t1 = beat_s[j - 1], beat_s[j]
        phases.append((t - t0) / max(t1 - t0, 1e-9))
    phases = np.asarray(phases)
    Rbar, pval = rayleigh(phases)

    # --- 对照：折叠到【呼吸相位】（用 0.1–0.6 Hz 分带的瞬时相位）---
    from scipy.signal import hilbert
    sos_r = butter(4, [0.1, 0.6], btype="bandpass", fs=FS, output="sos")
    resp_b = sosfiltfilt(sos_r, zm[b].real) + 1j * sosfiltfilt(sos_r, zm[b].imag)
    resp_ph = np.unwrap(np.angle(hilbert(resp_b.real)))
    # 归一化到 [0,1)：按 2π 取模
    resp_ph01 = (resp_ph / (2 * np.pi)) % 1.0
    rph = resp_ph01[pk] if len(pk) else np.array([])
    Rbar_resp, p_resp = rayleigh(rph)

    # 几何：心搏带轨迹的圆心/半径（BPF 后 DC 已去除，看是否仍偏离）
    cx, cy, R = kasa(zc)
    cen_ratio = float(np.hypot(cx, cy) / max(R, 1e-12))

    # 相位速率（零点加速）
    ph = np.unwrap(np.angle(zc))
    d = np.abs(np.diff(ph))
    base = float(np.median(d))
    if len(pk):
        kern = int(0.1 * FS)
        near = np.concatenate([d[max(0, i - kern):i + kern] for i in pk])
        accel = float(np.max(near) / max(base, 1e-12)) if len(near) else float("nan")
    else:
        accel = float("nan")

    return {"stem": stem, "bin": b, "gold_hr": gold, "dip_freq_hz": f_dip,
            "dip_freq_is_hr": bool(np.isfinite(rel) and rel < 0.10),
            "rel_err_vs_hr": rel, "n_dips": int(len(pk)),
            "dip_depth_median": float(np.median(depth)) if len(depth) else float("nan"),
            "env_min_over_max": float(env.min() / max(env.max(), 1e-12)),
            "Rbar": Rbar, "rayleigh_p": pval, "per_beat_locked": bool(
                np.isfinite(Rbar) and Rbar > 0.4 and np.isfinite(pval) and pval < 0.01),
            "center_over_R": cen_ratio, "phase_accel_max": accel,
            "Rbar_resp": Rbar_resp, "p_resp": p_resp,
            "resp_locked": bool(np.isfinite(Rbar_resp) and Rbar_resp > 0.4
                                and np.isfinite(p_resp) and p_resp < 0.01),
            "_env": env, "_ph": ph, "_pk": pk, "_phases": phases,
            "_zc": zc, "_lbl": lbl}


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="逐搏零点检验（心搏带）")
    ap.add_argument("--patient", type=int, default=0, help=">0 则只跑该例")
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.patient:
        r = analyse(f"patient_{args.patient}")
        print(f"=== patient_{args.patient} (bin {r['bin']}) ===")
        print(f"  金标 HR {r['gold_hr']:.1f} BPM；凹陷频率 {r['dip_freq_hz']:.3f} Hz "
              f"→ {'与心率同频' if r['dip_freq_is_hr'] else '不与心率同频'}（相对误差 {r['rel_err_vs_hr']*100:.0f}%）")
        print(f"  凹陷个数 {r['n_dips']}，深度中位 {r['dip_depth_median']:.2f}，"
              f"包络 min/max {r['env_min_over_max']:.3f}")
        print(f"  ★ 逐搏锁定：R̄={r['Rbar']:.3f}，Rayleigh p={r['rayleigh_p']:.2e} → "
              f"{'**锁定**' if r['per_beat_locked'] else '未锁定'}")
        print(f"  心搏带轨迹：|中心|/R = {r['center_over_R']:.3f}；零点相位加速 {r['phase_accel_max']:.1f}×")
        return 0

    stems = [f"patient_{i}" for i in range(1, 21)
             if (DATA / f"patient_{i}.mat").exists()]
    rows = [analyse(s) for s in stems]
    import pandas as pd
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in rows])
    df.to_csv(out / "cardiac_null_summary.csv", index=False, encoding="utf-8")

    print(f"=== 心搏带逐搏零点检验：{len(df)} 例 ===")
    print(f"  {'pat':>4s}{'bin':>5s}{'金标HR':>8s}{'凹陷频':>8s}{'同HR':>6s}"
          f"{'R̄_beat':>8s}{'p_beat':>10s}{'逐搏':>6s}{'R̄_resp':>8s}{'呼吸锁定':>9s}")
    for _, r in df.iterrows():
        print(f"  {r['stem'][-2:]:>4s}{r['bin']:>5d}{r['gold_hr']:>8.1f}{r['dip_freq_hz']:>8.3f}"
              f"{('是' if r['dip_freq_is_hr'] else '否'):>6s}{r['Rbar']:>8.2f}"
              f"{r['rayleigh_p']:>10.1e}{('★' if r['per_beat_locked'] else '否'):>6s}"
              f"{r['Rbar_resp']:>8.2f}{('★' if r['resp_locked'] else '否'):>9s}")

    n_hr = int(df["dip_freq_is_hr"].sum())
    n_lock = int(df["per_beat_locked"].sum())
    n_resp = int(df["resp_locked"].sum())
    print(f"\n  → 凹陷频率与心率同频：{n_hr}/{len(df)}；**逐搏相位锁定**：{n_lock}/{len(df)}")
    print(f"  → **呼吸相位锁定**（对照）：{n_resp}/{len(df)}")
    print(f"  → R̄_beat 中位 {df['Rbar'].median():.2f}；R̄_resp 中位 {df['Rbar_resp'].median():.2f}；"
          f"相位加速中位 {df['phase_accel_max'].median():.1f}×")

    # ---- 代表例三张图 ----
    best = df.sort_values("Rbar", ascending=False).iloc[0]
    r = [x for x in rows if x["stem"] == best["stem"]][0]
    import matplotlib.pyplot as plt
    from decision_presentation import setup_font
    setup_font()
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax[0].plot(r["_zc"].real, r["_zc"].imag, lw=0.6, color="#4C78A8", alpha=0.85)
    ax[0].scatter([0], [0], marker="+", s=160, color="#E45756", zorder=5, label="原点（零点）")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("Re"); ax[0].set_ylabel("Im")
    ax[0].set_title(f"(a) 心搏带复平面轨迹（{r['stem']}, bin {r['bin']}）")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    t = np.arange(len(r["_env"])) / FS
    ax[1].plot(t, r["_env"], color="#4C78A8", lw=0.9)
    ax[1].scatter(r["_pk"] / FS, r["_env"][r["_pk"]], s=22, color="#E45756", zorder=4, label="包络凹陷")
    for bs in r["_lbl"] / FS:
        ax[1].axvline(bs, color="gray", lw=0.5, alpha=0.45)
    ax[1].set_xlabel("时间 (s)"); ax[1].set_ylabel("|z|（心搏带）")
    ax[1].set_title("(b) 幅度包络与 ECG R 峰（灰线）")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    ph = r["_phases"]
    if len(ph) >= 5:
        ax[2].hist(ph, bins=10, range=(0, 1), color="#54A24B", edgecolor="white")
        ax[2].set_xlabel("心搏相位 φ = (t_min − t_R)/(t_R+1 − t_R)")
        ax[2].set_ylabel("凹陷个数")
        ax[2].set_title(f"(c) 凹陷的心搏相位分布：R̄={r['Rbar']:.2f}, p={r['rayleigh_p']:.1e}")
        ax[2].grid(axis="y", alpha=0.3)
    fig.suptitle("图 N1　逐搏零点检验（心搏带 0.8–2.5 Hz；Zhang 等公开数据）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fp = out / "figN1_per_beat_null.png"
    fig.savefig(fp, dpi=150)
    print(f"\n  图 → {fp}")
    print(f"  汇总 → {out / 'cardiac_null_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
