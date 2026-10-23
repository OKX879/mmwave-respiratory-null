#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""《多径零点》双面板代表图：复平面轨迹 + 呼吸相位折叠。"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

HERE = Path(__file__).resolve().parent
DATA = Path(r"C:\Users\yxh\Desktop\capsule-4604022-data\Minimum_dataset")
OUT = HERE / "figs"
FS = 100.0
DEEP_FRAC = 0.35      # 深凹陷阈值：0.35 x 中位包络
MIN_SEP_S = 0.25      # 事件最小间隔
PROM_FRAC = 0.15      # prominence = 0.15 sigma
RESP_BAND = (0.1, 0.6)


def load(stem: str):
    with h5py.File(DATA / f"{stem}.mat", "r") as f:
        raw = f[f["data_all"][0, 0]][:]
        lbl = np.asarray(f[f["data_all"][1, 0]][:]).ravel()
    z = (raw["real"] + 1j * raw["imag"]).astype(np.complex128)   # (ch, ang, bin, T)
    return z, lbl


def bp_complex(x: np.ndarray, band, order: int = 4) -> np.ndarray:
    sos = butter(order, list(band), btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x.real) + 1j * sosfiltfilt(sos, x.imag)


def deep_nulls(env: np.ndarray):
    pk, _ = find_peaks(-env, distance=int(MIN_SEP_S * FS),
                      prominence=PROM_FRAC * np.std(env))
    thr = DEEP_FRAC * float(np.median(env))
    return pk[env[pk] < thr], thr


def resp_phase(z: np.ndarray) -> np.ndarray:
    rb = bp_complex(z, RESP_BAND)
    return (np.unwrap(np.angle(rb)) / (2 * np.pi)) % 1.0


def rayleigh(ph: np.ndarray):
    n = len(ph)
    if n < 5:
        return float("nan"), float("nan")
    Rbar = float(abs(np.mean(np.exp(1j * 2 * np.pi * ph))))
    return Rbar, float(np.exp(-n * Rbar ** 2))


def analyse(stem: str) -> dict:
    z, lbl = load(stem)
    zm = z[:, 0, :, :].mean(axis=0)                  # 8 通道复数平均 -> (bin, T)
    b = int(np.argmax(np.std(zm, axis=1)))           # 宽带运动能量最强的 bin
    zc = zm[b]
    env = np.abs(zc)
    pk, thr = deep_nulls(env)
    ph = resp_phase(zc)
    rph = ph[pk]
    Rbar, p = rayleigh(rph)
    return {"stem": stem, "bin": b, "n_deep": int(len(pk)),
            "thr": thr, "Rbar": Rbar, "p": p,
            "gold_hr": float(
                60000.0 / np.median(np.diff(lbl) / FS * 1000.0) if len(lbl) > 1 else np.nan),
            "z": zc, "env": env, "peaks": pk, "phase": rph, "lbl": lbl}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stems = [f"patient_{i}" for i in range(1, 21)
             if (DATA / f"patient_{i}.mat").exists()]
    rows = [analyse(s) for s in stems]

    print(f"{'record':>11s}{'bin':>4s}{'n_deep':>7s}{'Rbar':>6s}{'p':>9s}")
    for r in sorted(rows, key=lambda x: -x["Rbar"]):
        print(f"{r['stem']:>11s}{r['bin']:>4d}{r['n_deep']:>7d}"
              f"{r['Rbar']:>6.2f}{r['p']:>9.1e}")

    elig = [r for r in rows if r["n_deep"] >= 5 and np.isfinite(r["Rbar"])]
    if not elig:
        raise SystemExit("no record with >=5 deep nulls")
    # 正文代表例：非重复、Rbar>0.4 中单例证据最强（Rayleigh p 最小）
    cand = [r for r in elig if r["stem"] != "patient_16" and r["Rbar"] > 0.4]
    if not cand:
        cand = elig
    best = min(cand, key=lambda x: x["p"])
    print(f"\nrepresentative: {best['stem']}  bin={best['bin']}  "
          f"n={best['n_deep']}  Rbar={best['Rbar']:.3f}  p={best['p']:.2e}")

    # ---------------- 绘图 ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.linewidth": 0.7,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    zc, env, pk, ph = best["z"], best["env"], best["peaks"], best["phase"]
    fig = plt.figure(figsize=(7.0, 3.15))

    # (a) 复平面轨迹（原始 z；虚线圆 = 深凹陷阈值）
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(zc.real, zc.imag, lw=0.5, color="#4C78A8", alpha=0.45, zorder=2)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(best["thr"] * np.cos(th), best["thr"] * np.sin(th),
            ls="--", lw=0.8, color="#8C8C8C", zorder=3)
    ax.scatter(zc.real[pk], zc.imag[pk], s=26, color="#E45756",
               edgecolor="black", linewidth=0.4, zorder=5)
    ax.scatter([0], [0], marker="+", s=110, color="black", linewidth=1.2, zorder=6)
    ax.set_aspect("equal", adjustable="box")
    R98 = float(np.percentile(np.abs(zc), 98))
    ax.set_xlim(-1.05 * R98, 1.05 * R98)
    ax.set_ylim(-1.05 * R98, 1.05 * R98)
    ax.set_xlabel("Re$\\{z\\}$ (a.u.)")
    ax.set_ylabel("Im$\\{z\\}$ (a.u.)")
    ax.set_title(f"(a) complex trajectory, {best['stem']}, bin {best['bin']}\n"
                 f"30 s, {best['gold_hr']:.1f} bpm", fontsize=9)
    ax.grid(alpha=0.22, lw=0.4)
    ax.legend([ax.lines[1], ax.collections[0], ax.collections[1]],
              ["deep-null threshold\n($0.35\\times$ median $|z|$)",
               f"deep nulls ($n={len(pk)}$)", "origin ($z=0$)"],
              loc="upper left", fontsize=6.5, frameon=True, framealpha=0.9,
              edgecolor="none", handlelength=1.4, labelspacing=0.3)

    # 近原点放大插图：展示"反复穿过原点"的细节
    axi = ax.inset_axes([0.64, 0.07, 0.33, 0.33])
    axi.plot(zc.real, zc.imag, lw=0.45, color="#4C78A8", alpha=0.5, zorder=2)
    axi.plot(best["thr"] * np.cos(th), best["thr"] * np.sin(th),
             ls="--", lw=0.8, color="#8C8C8C", zorder=3)
    axi.scatter(zc.real[pk], zc.imag[pk], s=18, color="#E45756",
                edgecolor="black", linewidth=0.3, zorder=5)
    axi.scatter([0], [0], marker="+", s=70, color="black", linewidth=1.0, zorder=6)
    lim = 1.6 * best["thr"]
    axi.set_xlim(-lim, lim)
    axi.set_ylim(-lim, lim)
    axi.set_aspect("equal", adjustable="box")
    axi.set_xticks([])
    axi.set_yticks([])
    axi.tick_params(length=0)
    axi.set_facecolor("white")
    axi.grid(alpha=0.2, lw=0.3)
    axi.text(0.04, 0.93, "zoom", transform=axi.transAxes, fontsize=6.5,
             va="top", color="#333333")

    # (b) 呼吸相位折叠（玫瑰图）
    axp = fig.add_subplot(1, 2, 2, projection="polar")
    nb = 12
    cnt, edges = np.histogram(ph, bins=nb, range=(0, 1))
    width = 2 * np.pi / nb
    axp.bar(edges[:-1] * 2 * np.pi, cnt, width=width * 0.92,
            color="#54A24B", alpha=0.85, edgecolor="white", linewidth=0.5, zorder=2)
    axp.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, cnt.mean()),
             ls="--", lw=0.8, color="#8C8C8C", zorder=3, label="uniform level")
    mean_vec = np.mean(np.exp(1j * 2 * np.pi * ph))
    axp.annotate("", xy=(np.angle(mean_vec), abs(mean_vec) * max(cnt.max(), 1) * 0.95),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", lw=1.2, color="#E45756"), zorder=4)
    axp.set_theta_zero_location("N")
    axp.set_theta_direction(-1)
    axp.set_xticks(np.arange(0, 2 * np.pi, np.pi / 2))
    axp.set_xticklabels(["0", "0.25", "0.5", "0.75"], fontsize=7.5)
    axp.set_yticklabels([])
    axp.tick_params(axis="y", length=0)
    axp.set_title("(b) folded onto respiratory phase", fontsize=9, pad=12)
    axp.text(0.5, -0.16,
             f"$\\bar{{R}}={best['Rbar']:.2f}$,  Rayleigh $p={best['p']:.1e}$,  $n={len(pk)}$",
             transform=axp.transAxes, ha="center", fontsize=7.5)

    fig.tight_layout()
    png = OUT / "fig_null_phase.png"
    fig.savefig(png, dpi=300)
    fig.savefig(OUT / "fig_null_phase.pdf")
    plt.close(fig)

    meta = {k: (v if not isinstance(v, np.ndarray) else None) for k, v in best.items()}
    meta["all_records"] = [{k: (v if not isinstance(v, np.ndarray) else None)
                            for k, v in r.items()} for r in rows]
    (OUT / "fig_null_phase.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(f"\nfigure -> {png}")
    print(f"meta   -> {OUT / 'fig_null_phase.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
