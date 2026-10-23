#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RHC 谐波消除后的逐搏零点复检（复数域陷波，不破坏复平面几何）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cardiac_null_probe import (DATA, FS, CARD, load, bp_complex,          # noqa: E402
                                envelope_dips, rayleigh)
from rhc_frontend import make_rhc_frontend                                # noqa: E402

BW = 0.04          # 陷波半宽（Hz）
PROTECT_HZ = 0.06  # 与心率相距小于此值 → 不陷波（保护心搏）


def notch_complex(z: np.ndarray, freqs: list[float], bw: float = BW) -> np.ndarray:
    """在复数频谱上对给定频率做窄带陷波（零相位、保持共轭对称）。"""
    n = len(z)
    Z = np.fft.fft(z)
    f = np.fft.fftfreq(n, 1 / FS)
    m = np.ones(n, dtype=bool)
    for f0 in freqs:
        m &= np.abs(f - f0) > bw
    return np.fft.ifft(Z * m)


def analyse(zc: np.ndarray, lbl: np.ndarray) -> dict:
    env = np.abs(zc)
    pk, depth = envelope_dips(env)
    v = env - env.mean()
    F = np.abs(np.fft.rfft(v))
    f = np.fft.rfftfreq(len(v), 1 / FS)
    m = (f >= 0.5) & (f <= 3.0)
    f_dip = float(f[m][np.argmax(F[m])]) if m.any() else float("nan")

    beat_s = lbl / FS
    ph = []
    for i in pk:
        t = i / FS
        j = int(np.searchsorted(beat_s, t))
        if j <= 0 or j >= len(beat_s):
            continue
        ph.append((t - beat_s[j - 1]) / max(beat_s[j] - beat_s[j - 1], 1e-9))
    Rbar, pval = rayleigh(np.asarray(ph))

    d = np.abs(np.diff(np.unwrap(np.angle(zc))))
    base = float(np.median(d))
    acc = float("nan")
    if len(pk):
        k = int(0.1 * FS)
        near = np.concatenate([d[max(0, i - k):i + k] for i in pk])
        acc = float(np.max(near) / max(base, 1e-12)) if len(near) else float("nan")
    return {"n_dips": int(len(pk)), "dip_depth": float(np.median(depth)) if len(depth) else float("nan"),
            "dip_freq": f_dip, "Rbar": Rbar, "p": pval, "accel": acc,
            "card_energy": float(np.std(bp_complex(zc)))}


def run_one(stem: str, rng: np.random.Generator, n_placebo: int = 30) -> dict:
    z, lbl = load(stem)
    zm = z[:, 0, :, :].mean(axis=0)                     # (bin, T) 复数融合
    card = np.stack([bp_complex(zm[b]) for b in range(zm.shape[0])])
    b = int(np.argmax(np.std(card, axis=1)))
    zc0 = zm[b]

    gold = 60000.0 / np.median(np.diff(lbl) / FS * 1000.0) if len(lbl) > 1 else np.nan
    hr_hz = gold / 60.0

    # 1) RHC 的输入必须是**位移/相位**，不能是 Re(z)（实测 Re 会让 f_resp 错 2.8×）
    from scipy.signal import detrend
    from scipy.signal import butter, sosfiltfilt
    front = make_rhc_frontend(FS)
    out = front(detrend(np.unwrap(np.angle(zc0))))
    trust = float(out["rhc_trust"])
    f_rhc = float(out["resp_track"].f_resp_mean)

    # 2) trust 低 → 用直接谱峰回退（RHC 自己承认不可信时不采信它）
    sos = butter(4, [0.08, 0.6], btype="bandpass", fs=FS, output="sos")
    xr = sosfiltfilt(sos, np.real(zc0))
    F = np.abs(np.fft.rfft(xr - xr.mean()))
    fq = np.fft.rfftfreq(len(xr), 1 / FS)
    mm = (fq >= 0.08) & (fq <= 0.6)
    f_spec = float(fq[mm][np.argmax(F[mm])]) if mm.any() else float("nan")
    f_resp = f_rhc if (trust >= 0.5 and 0.08 <= f_rhc <= 0.6) else f_spec
    src = "RHC" if f_resp == f_rhc else "谱峰回退"

    # 受保护谐波：本检验的目的是**尽量压掉呼吸谐波**，故只保护"与金标心率重合"的那一条。
    # （RHC 自带的 protected_harmonics 面向在线部署、保护面很宽，会把大部分谐波都保护掉，
    #   实测导致每例只剩 1–4 条陷波、谐波几乎没被压——测试强度不足。）
    protect = []
    _ = out["reg_result"]                   # 仅保留 RHC 的 f_resp/trust 用途，见下

    # 3) 陷波频率：k·f_resp ∈ [0.7, 3.0]，避开心率与受保护谐波
    fs_notch = []
    for k in range(2, 12):
        fk = k * f_resp
        if 0.7 <= fk <= 3.0 and np.isfinite(hr_hz):
            if abs(fk - hr_hz) > PROTECT_HZ and all(abs(fk - p_) > PROTECT_HZ for p_ in protect):
                fs_notch.append(fk)
    z_rhc = notch_complex(zc0, fs_notch)

    # 4) placebo：同样数量、同样带宽的随机窄带，抽样 n_placebo 次取 R̄ 分布
    rbar_pbo = []
    for _ in range(n_placebo):
        fs_p = []
        for _ in range(len(fs_notch)):
            for _try in range(50):
                cand = float(rng.uniform(0.7, 3.0))
                if abs(cand - hr_hz) > PROTECT_HZ:
                    fs_p.append(cand)
                    break
        rbar_pbo.append(analyse(notch_complex(zc0, fs_p), lbl)["Rbar"])
    rbar_pbo = np.asarray([x for x in rbar_pbo if np.isfinite(x)])

    a0, a1 = analyse(zc0, lbl), analyse(z_rhc, lbl)
    p95 = float(np.percentile(rbar_pbo, 95)) if len(rbar_pbo) else float("nan")
    locked = bool(np.isfinite(a1["Rbar"]) and a1["Rbar"] > 0.4
                  and np.isfinite(a1["p"]) and a1["p"] < 0.01
                  and np.isfinite(p95) and a1["Rbar"] > p95)
    return {"stem": stem, "bin": b, "gold_hr": gold,
            "f_resp": f_resp, "f_rhc": f_rhc, "f_spec": f_spec, "trust": trust, "f_src": src,
            "n_notch": len(fs_notch), "notch_hz": [round(x, 3) for x in fs_notch],
            "card_e_before": a0["card_energy"], "card_e_after": a1["card_energy"],
            "Rbar_base": a0["Rbar"], "Rbar_rhc": a1["Rbar"],
            "Rbar_pbo_med": float(np.median(rbar_pbo)) if len(rbar_pbo) else float("nan"),
            "Rbar_pbo_p95": p95,
            "p_rhc": a1["p"], "accel_base": a0["accel"], "accel_rhc": a1["accel"],
            "dip_freq_rhc": a1["dip_freq"], "locked": locked}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RHC 谐波消除后的逐搏零点复检")
    ap.add_argument("--patient", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(HERE / "null_probe_out"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261021)

    if args.patient:
        r = run_one(f"patient_{args.patient}", rng)
        print(f"=== patient_{args.patient} (bin {r['bin']}) ===")
        print(f"  金标 HR {r['gold_hr']:.1f} BPM")
        print(f"  f_resp：RHC={r['f_rhc']:.4f}（trust={r['trust']:.2f}）/ 谱峰={r['f_spec']:.4f} "
              f"→ 采用 {r['f_resp']:.4f} Hz（{r['f_src']}）")
        print(f"  陷波 {r['n_notch']} 条 {r['notch_hz']}")
        print(f"  心搏带能量 去谐波前 {r['card_e_before']:.3f} → 后 {r['card_e_after']:.3f}"
              f"（保留 {r['card_e_after']/max(r['card_e_before'],1e-12)*100:.0f}%）")
        print(f"  R̄：base {r['Rbar_base']:.2f} → RHC {r['Rbar_rhc']:.2f}"
              f"（placebo 中位 {r['Rbar_pbo_med']:.2f} / p95 {r['Rbar_pbo_p95']:.2f}）")
        print(f"  锁定：{'★ 是' if r['locked'] else '否'}（需同时满足 R̄>0.4、p<0.01、R̄>placebo p95）")
        return 0

    import pandas as pd
    stems = [f"patient_{i}" for i in range(1, 21) if (DATA / f"patient_{i}.mat").exists()]
    rows = [run_one(s, rng) for s in stems]
    df = pd.DataFrame(rows)
    df.to_csv(out / "rhc_null_summary.csv", index=False, encoding="utf-8")

    print(f"=== RHC 谐波消除后逐搏零点复检（{len(df)} 例）===")
    print(f"  {'pat':>4s}{'bin':>4s}{'HR':>6s}{'f_resp':>8s}{'来源':>10s}{'trust':>6s}{'n':>3s}"
          f"{'R̄_base':>8s}{'R̄_RHC':>8s}{'p95_pbo':>8s}{'锁定':>6s}{'加速':>7s}")
    for _, r in df.iterrows():
        print(f"  {r['stem'][-2:]:>4s}{r['bin']:>4d}{r['gold_hr']:>6.1f}{r['f_resp']:>8.4f}"
              f"{r['f_src']:>10s}{r['trust']:>6.2f}{r['n_notch']:>3d}"
              f"{r['Rbar_base']:>8.2f}{r['Rbar_rhc']:>8.2f}{r['Rbar_pbo_p95']:>8.2f}"
              f"{('★' if r['locked'] else '否'):>6s}{r['accel_rhc']:>7.1f}")

    n_lock = int(df["locked"].sum())
    print(f"\n  → **RHC 后逐搏锁定：{n_lock}/{len(df)}**（判据：R̄>0.4 且 p<0.01 且 R̄>placebo p95）")
    print(f"  → R̄ 中位：base {df['Rbar_base'].median():.2f} / RHC {df['Rbar_rhc'].median():.2f}"
          f" / placebo p95 中位 {df['Rbar_pbo_p95'].median():.2f}")
    print(f"  → 心搏带能量保留率 中位 {np.nanmedian(df['card_e_after']/df['card_e_before']):.2f}")
    print(f"\n  汇总 → {out / 'rhc_null_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
