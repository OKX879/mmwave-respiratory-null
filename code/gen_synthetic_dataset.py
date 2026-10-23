#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成雷达位移域生命体征数据集生成器。"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal

# ---------------------------------------------------------------------------
# 生理频段常量（与生产链保持一致）
# ---------------------------------------------------------------------------
RESP_BAND = (0.08, 0.6)          # 呼吸带
CARD_DISPLAY_BAND = (0.7, 3.0)   # 心搏波形展示带
CARD_HR_BAND = (1.0, 2.5)        # 心率估计带（同时作为 SNR 参考带）
SNR_REF_BAND = CARD_HR_BAND


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def bandpower(x: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    """零相位带通后的平均功率。"""
    sos = signal.butter(4, [band[0], band[1]], btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, np.asarray(x, dtype=float))
    return float(np.mean(y ** 2))


@lru_cache(maxsize=8)
def noise_band_factor(fs: float, band: tuple[float, float]) -> float:
    """单位方差白噪声经 bandpower() 实测的【带内增益因子】。"""
    rng = np.random.default_rng(0)
    w = rng.normal(0.0, 1.0, 1 << 18)
    return bandpower(w, fs, band)


def noise_sigma_for_cardiac_snr(
    card_clean: np.ndarray, fs: float, snr_db: float, band: tuple[float, float] = SNR_REF_BAND
) -> tuple[float, float]:
    """求白噪声 sigma，使【实测带内噪声功率】= 心搏带内功率 / 10^(SNR/10)。"""
    p_card = bandpower(card_clean, fs, band)
    p_noise_target = p_card / (10.0 ** (snr_db / 10.0))
    factor = noise_band_factor(fs, band)
    sigma = math.sqrt(max(p_noise_target / max(factor, 1e-18), 0.0))
    return sigma, p_card


# ---------------------------------------------------------------------------
# 各分量合成
# ---------------------------------------------------------------------------
def make_respiration(
    t: np.ndarray, f_resp: float, a_resp: float, rng: np.random.Generator,
    n_harm: int = 6, decay: float = 3.0,
) -> np.ndarray:
    """呼吸：基频 + 多次谐波（模拟非正弦的真实呼吸波形）。"""
    x = np.zeros_like(t)
    for k in range(1, n_harm + 1):
        amp = (a_resp / (k ** decay)) * float(rng.uniform(0.7, 1.3))
        ph = rng.uniform(0.0, 2.0 * np.pi)
        x += amp * np.sin(2.0 * np.pi * k * f_resp * t + ph)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1e-12:
        x = x / peak * a_resp
    return x


def make_cardiac(
    t: np.ndarray,
    fs: float,
    hr_bpm: float,
    a_card: float,
    sigma_c: float,
    f_resp: float,
    rng: np.random.Generator,
    rsa: float = 0.03,
    riav: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, float]:
    """心跳：非均匀高斯脉冲串（含 RSA + RIAV）。"""
    T = float(t[-1])
    ph_rsa = rng.uniform(0.0, 2.0 * np.pi)
    ph_riav = rng.uniform(0.0, 2.0 * np.pi)

    beats: list[float] = []
    tt = 0.0
    while True:
        inst_hr = hr_bpm * (1.0 + rsa * math.sin(2.0 * np.pi * f_resp * tt + ph_rsa))
        inst_hr = max(inst_hr, 30.0)
        tt += 60.0 / inst_hr
        if tt >= T:
            break
        beats.append(tt)

    beats_arr = np.asarray(beats, dtype=float)
    x = np.zeros_like(t)
    for tb in beats_arr:
        amp = a_card * (1.0 + riav * math.sin(2.0 * np.pi * f_resp * tb + ph_riav))
        x += amp * np.exp(-((t - tb) ** 2) / (2.0 * sigma_c ** 2))

    if beats_arr.size >= 2:
        hr_mean = float(60.0 / np.mean(np.diff(beats_arr)))
    else:
        hr_mean = float("nan")
    return x, beats_arr, hr_mean


def make_motion(
    t: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    n_bursts: tuple[int, int] = (2, 5),
    amp_mm: tuple[float, float] = (1.0, 10.0),
    dur_s: tuple[float, float] = (0.5, 2.0),
    scale: float = 1.0,
) -> tuple[np.ndarray, int]:
    """运动：稀疏宽带突发（带通白噪声）。"""
    x = np.zeros_like(t)
    if scale <= 0:
        return x, 0
    nb = int(rng.integers(n_bursts[0], n_bursts[1] + 1))
    sos = signal.butter(4, [0.5 / (fs / 2), min(10.0 / (fs / 2), 0.95)], btype="bandpass", output="sos")
    for _ in range(nb):
        d = float(rng.uniform(*dur_s))
        n = max(8, int(round(d * fs)))
        if n >= len(t):
            n = len(t)
        start = int(rng.integers(0, max(1, len(t) - n)))
        seg = signal.sosfilt(sos, rng.normal(0.0, 1.0, n))
        seg = seg / (np.std(seg) + 1e-12) * float(rng.uniform(*amp_mm)) * scale
        x[start:start + n] += seg
    return x, nb


def make_drift(t: np.ndarray, fs: float, rng: np.random.Generator, amp_mm: float,
               scale: float = 1.0) -> np.ndarray:
    """基线漂移：白噪声经极低频低通（约 1/f 形）。scale=0 时不生成。"""
    if scale <= 0 or amp_mm <= 0:
        return np.zeros_like(t)
    sos = signal.butter(2, 0.05 / (fs / 2), btype="low", output="sos")
    y = signal.sosfiltfilt(sos, rng.normal(0.0, 1.0, len(t)))
    y = y / (np.std(y) + 1e-12) * amp_mm * scale
    return y


# ---------------------------------------------------------------------------
# 单样本生成
# ---------------------------------------------------------------------------
def make_one_sample(cfg: dict, combo: tuple[float, float, float], k: int, seed: int) -> dict:
    """按 (snr_db, hr_bpm, rr_bpm) 与第 k 个重复，生成一条样本。"""
    fs = float(cfg["fs"])
    duration = float(cfg["duration_s"])
    n = int(round(duration * fs))
    t = np.arange(n, dtype=float) / fs

    snr_db, hr_bpm, rr_bpm = combo
    rng = np.random.default_rng(seed)

    f_resp = (rr_bpm / 60.0) * float(rng.uniform(0.97, 1.03))
    a_resp = float(rng.uniform(2.0, 6.0))     # 现实静息呼吸位移幅度
    a_card = float(rng.uniform(0.10, 0.5))    # 现实心搏位移幅度（亚毫米级）
    sigma_c = float(rng.uniform(0.06, 0.12))
    drift_amp = float(rng.uniform(0.5, 3.0))

    d_resp = make_respiration(t, f_resp, a_resp, rng,
                              n_harm=cfg["n_resp_harmonics"],
                              decay=cfg["resp_harm_decay"])
    d_card, beats, hr_mean = make_cardiac(
        t, fs, hr_bpm, a_card, sigma_c, f_resp, rng,
        rsa=cfg["rsa"], riav=cfg["riav"],
    )
    d_mot, n_burst = make_motion(t, fs, rng, scale=cfg["motion_scale"])
    d_drift = make_drift(t, fs, rng, drift_amp, scale=cfg["drift_scale"])

    sigma_n, p_card = noise_sigma_for_cardiac_snr(d_card, fs, snr_db)
    n_ref = rng.normal(0.0, sigma_n, n)

    d_mix = d_resp + d_card + d_mot + d_drift + n_ref

    # 诊断：运动在心搏带内的功率相对心搏功率（dB）。
    # 运动强度不随 SNR 变化，若该值过大，SNR 扫描会被运动污染 ——
    # 做"只变一件事"的消融时应用 --motion-scale 0 关闭运动。
    mot_inband_db = 10.0 * math.log10(
        max(bandpower(d_mot, fs, SNR_REF_BAND), 1e-18) / max(p_card, 1e-18)
    )

    # 多通道：共享真实位移 + 独立噪声 + 轻微增益差
    c = int(cfg["channels"])
    gains = 1.0 + rng.normal(0.0, 0.05, c)
    sigma_ch = cfg["channel_noise_ratio"] * sigma_n
    mix = np.empty((c, n), dtype=np.float32)
    for ci in range(c):
        e = rng.normal(0.0, sigma_ch, n)
        mix[ci] = (gains[ci] * d_mix + e).astype(np.float32)

    snr_achieved = 10.0 * math.log10(p_card / max(bandpower(n_ref, fs, SNR_REF_BAND), 1e-18))

    return {
        "t": t,
        "mix": mix,
        "resp": d_resp.astype(np.float32),
        "card": d_card.astype(np.float32),
        "mot": d_mot.astype(np.float32),
        "drift": d_drift.astype(np.float32),
        "noise_ref": n_ref.astype(np.float32),
        "gains": gains.astype(np.float32),
        "meta": {
            "seed": int(seed),
            "snr_target_db": float(snr_db),
            "snr_achieved_db": float(snr_achieved),
            "hr_nominal_bpm": float(hr_bpm),
            "hr_mean_bpm": float(hr_mean),
            "rr_bpm": float(rr_bpm),
            "f_resp_hz": float(f_resp),
            "a_resp_mm": float(a_resp),
            "a_card_mm": float(a_card),
            "sigma_c_s": float(sigma_c),
            "rsa": float(cfg["rsa"]),
            "riav": float(cfg["riav"]),
            "n_beats": int(beats.size),
            "n_motion_bursts": int(n_burst),
            "drift_amp_mm": float(drift_amp),
            "mot_inband_rel_db": float(mot_inband_db),
            "n_channels": c,
            "repeat_index": int(k),
            "bin_index": -1,
        },
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
README_TXT = """合成雷达位移域生命体征数据集 — 字段说明

文件：<split>.npz
  t           (T,)            时间轴（秒），fs 由 config.json 指定
  mix         (M, C, T)       多通道观测位移（mm），这是"输入信号"
  resp        (M, T)          干净呼吸分量（金标，干扰源）
  card        (M, T)          干净心跳分量（金标，目标源）
  mot         (M, T)          干净运动分量（金标）
  drift       (M, T)          干净基线漂移分量（金标）
  noise_ref   (M, T)          参考通道上的加性噪声（金标）
  gains       (M, C)          各通道增益（真实位移的缩放系数）
  sample_ids  (M,)            样本编号，与 manifest.csv 的 sample_id 对应

文件：manifest.csv  每行一条样本，字段含义见列名（snr_target_db 为设计值，
  snr_achieved_db 为实测值，二者应接近 —— 这是生成器的自检指标）。

说明：
  1) 本数据集刻意绕过空间前端，只合成位移域时间序列，使"小波环节"成为
     唯一变量；因此 bin_index 恒为 -1。
  2) 心搏带内 SNR 的定义：干净心跳分量在 1.0-2.5 Hz 内的功率 / 噪声在
     同带内的功率。噪声 sigma 用同一 Butterworth 滤波器实测标定，故
     snr_achieved_db 与 snr_target_db 的【均值偏差】应接近 0（自检判据）。
  3) hr_mean_bpm 是由实际搏动时刻反算的平均心率，作为真实心率金标
     （因含 RSA，它可能略偏离 hr_nominal_bpm）。
  4) 【重要】运动与漂移的强度【不随 SNR 变化】。manifest 的
     mot_inband_rel_db 记录了运动在心搏带内相对心搏的功率（dB）；
     若要做"只变 SNR"的干净消融，请用 --motion-scale 0 --drift-scale 0
     重新生成一份，否则运动/漂移会成为混杂因素。
  5) 多通道：各通道共享同一真实位移，叠加独立噪声（--channel-noise-ratio）
     与轻微增益差（sigma=0.05），供 RC 类准则（跨通道相干）使用。
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="合成雷达位移域生命体征数据集生成器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", type=str, default="synthetic_data", help="输出目录")
    p.add_argument("--split", type=str, default="grid", help="数据集分片名（决定 npz 文件名）")
    p.add_argument("--fs", type=float, default=100.0, help="采样率 Hz")
    p.add_argument("--duration", type=float, default=30.0, help="每条样本时长 s")
    p.add_argument("--snr", type=float, nargs="+", default=[-10.0, -5.0, 0.0, 5.0, 10.0],
                   help="心搏带内 SNR 网格 dB")
    p.add_argument("--hr", type=float, nargs="+", default=[50.0, 65.0, 80.0, 100.0, 120.0],
                   help="心率网格 BPM")
    p.add_argument("--rr", type=float, nargs="+", default=[12.0, 16.0, 20.0],
                   help="呼吸率网格 次/分")
    p.add_argument("--n-per-combo", type=int, default=3, help="每个网格组合的重复次数")
    p.add_argument("--channels", type=int, default=8, help="通道数（RC 准则需要 >1）")
    p.add_argument("--channel-noise-ratio", type=float, default=0.7, help="通道独立噪声相对于主噪声的幅度比")
    p.add_argument("--rsa", type=float, default=0.03, help="RSA 调制深度（心率随呼吸起伏）")
    p.add_argument("--riav", type=float, default=0.15, help="RIAV 调制深度（心搏幅度随呼吸起伏）")
    p.add_argument("--motion-scale", type=float, default=1.0,
                   help="运动强度缩放；0 = 关闭运动（做纯 SNR 扫描时用）")
    p.add_argument("--drift-scale", type=float, default=1.0,
                   help="基线漂移强度缩放；0 = 关闭漂移")
    p.add_argument("--resp-harmonics", type=int, default=6,
                   help="呼吸谐波阶数（模拟非正弦呼吸；其谐波落进心搏带是本项目头号失败模式）")
    p.add_argument("--resp-harm-decay", type=float, default=3.0,
                   help="呼吸谐波幅度衰减指数（a_resp/k^decay）。3.0=现实中等不对称；"
                        "2.0=谐波主导的极端对照条件")
    p.add_argument("--master-seed", type=int, default=20261018, help="主随机种子")
    p.add_argument("--demo-plot", action="store_true", help="额外输出单样本可视化自检图 demo.png")
    return p.parse_args(argv)


def build_dataset(args) -> dict:
    cfg = {
        "fs": float(args.fs),
        "duration_s": float(args.duration),
        "channels": int(args.channels),
        "channel_noise_ratio": float(args.channel_noise_ratio),
        "rsa": float(args.rsa),
        "riav": float(args.riav),
        "motion_scale": float(args.motion_scale),
        "drift_scale": float(args.drift_scale),
        "n_resp_harmonics": int(args.resp_harmonics),
        "resp_harm_decay": float(args.resp_harm_decay),
        "snr_grid_db": [float(v) for v in args.snr],
        "hr_grid_bpm": [float(v) for v in args.hr],
        "rr_grid_bpm": [float(v) for v in args.rr],
        "n_per_combo": int(args.n_per_combo),
        "master_seed": int(args.master_seed),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = [(s, h, r) for s in cfg["snr_grid_db"] for h in cfg["hr_grid_bpm"] for r in cfg["rr_grid_bpm"]]
    total = len(combos) * cfg["n_per_combo"]
    print(f"[gen] grid: {len(cfg['snr_grid_db'])} SNR x {len(cfg['hr_grid_bpm'])} HR "
          f"x {len(cfg['rr_grid_bpm'])} RR x {cfg['n_per_combo']} = {total} 条样本")

    n = int(round(cfg["duration_s"] * cfg["fs"]))
    mix = np.empty((total, cfg["channels"], n), dtype=np.float32)
    resp = np.empty((total, n), dtype=np.float32)
    card = np.empty((total, n), dtype=np.float32)
    mot = np.empty((total, n), dtype=np.float32)
    drift = np.empty((total, n), dtype=np.float32)
    noise_ref = np.empty((total, n), dtype=np.float32)
    gains = np.empty((total, cfg["channels"]), dtype=np.float32)
    rows: list[dict] = []
    first_sample = None

    idx = 0
    for ci, combo in enumerate(combos):
        for k in range(cfg["n_per_combo"]):
            seed = int(cfg["master_seed"]) * 1_000_019 + ci * 1_003 + k
            s = make_one_sample(cfg, combo, k, seed)
            mix[idx] = s["mix"]
            resp[idx] = s["resp"]
            card[idx] = s["card"]
            mot[idx] = s["mot"]
            drift[idx] = s["drift"]
            noise_ref[idx] = s["noise_ref"]
            gains[idx] = s["gains"]
            row = {"sample_id": idx, **s["meta"]}
            rows.append(row)
            if first_sample is None:
                first_sample = s
            idx += 1

    t = np.arange(n, dtype=float) / cfg["fs"]

    np.savez_compressed(
        out_dir / f"{args.split}.npz",
        t=t.astype(np.float32),
        mix=mix, resp=resp, card=card, mot=mot, drift=drift,
        noise_ref=noise_ref, gains=gains,
        sample_ids=np.arange(total, dtype=np.int32),
    )

    # manifest.csv
    cols = ["sample_id", "seed", "snr_target_db", "snr_achieved_db", "hr_nominal_bpm", "hr_mean_bpm",
            "rr_bpm", "f_resp_hz", "a_resp_mm", "a_card_mm", "sigma_c_s", "rsa", "riav",
            "n_beats", "n_motion_bursts", "drift_amp_mm", "mot_inband_rel_db",
            "n_channels", "repeat_index", "bin_index"]
    import csv
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(README_TXT)

    # ---- 自检：目标 SNR 与实测 SNR 的一致性 ----
    # 说明：实测 SNR 由"滤波后的带内噪声功率"得到，属统计估计量，
    # 其相对标准差约 1/sqrt(T·B)（T=窗长秒数，B=带宽 Hz）。以默认
    # 30 s / 1.5 Hz 计约 0.67 dB，故判据只看【均值偏差】，最大值仅作信息。
    tgt = np.array([r["snr_target_db"] for r in rows])
    ach = np.array([r["snr_achieved_db"] for r in rows])
    err = ach - tgt
    bias, sd = float(err.mean()), float(err.std())
    tb = cfg["duration_s"] * (SNR_REF_BAND[1] - SNR_REF_BAND[0])
    print("[selfcheck] SNR 标定: "
          f"bias={bias:+.3f} dB ({'OK' if abs(bias) < 0.3 else 'WARN'}, 判据 |bias|<0.3), "
          f"sd={sd:.3f} dB, max|err|={np.abs(err).max():.3f} dB "
          f"(理论涨落 sd~{10.0 / math.log(10.0) / math.sqrt(max(tb, 1e-9)):.2f} dB)")
    hrm = np.array([r["hr_mean_bpm"] for r in rows])
    print(f"[selfcheck] 实际平均心率范围: {np.nanmin(hrm):.1f} - {np.nanmax(hrm):.1f} BPM")
    amc = np.array([r["a_card_mm"] for r in rows])
    print(f"[selfcheck] 心搏幅度范围: {amc.min():.3f} - {amc.max():.3f} mm (亚毫米量级)")
    mir = np.array([r["mot_inband_rel_db"] for r in rows])
    print(f"[selfcheck] 运动的心搏带内相对功率: {np.nanmin(mir):+.1f} ~ {np.nanmax(mir):+.1f} dB "
          f"({'运动未开启' if cfg['motion_scale'] <= 0 else '注意: 运动强度不随 SNR 变化，'
             '如需纯 SNR 扫描请用 --motion-scale 0'})")

    if args.demo_plot and first_sample is not None:
        _demo_plot(first_sample, cfg, out_dir / "demo.png")

    size_mb = (out_dir / f"{args.split}.npz").stat().st_size / 1e6
    print(f"[gen] 完成: {total} 条 -> {out_dir / (args.split + '.npz')} ({size_mb:.1f} MB)")
    print(f"[gen] 附带: manifest.csv / config.json / README.txt")
    return {"cfg": cfg, "rows": rows, "total": total, "first_sample": first_sample}


def _setup_cjk_font() -> None:
    """让 matplotlib 能显示中文（Windows 常见的微软雅黑/黑体）。"""
    import matplotlib
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Source Han Sans SC"):
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _demo_plot(s: dict, cfg: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_cjk_font()

    fs = cfg["fs"]
    t = s["t"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(t, s["resp"], color="tab:blue", lw=1.0)
    axes[0].set_ylabel("呼吸 (mm)"); axes[0].set_title("干净分量（金标）")

    axes[1].plot(t, s["card"], color="tab:red", lw=1.0)
    axes[1].set_ylabel("心跳 (mm)")

    axes[2].plot(t, s["mot"], color="tab:orange", lw=0.8, label="运动")
    axes[2].plot(t, s["drift"], color="tab:green", lw=1.0, label="漂移")
    axes[2].set_ylabel("干扰 (mm)"); axes[2].legend(loc="upper right", fontsize=8)

    axes[3].plot(t, s["mix"][0], color="0.25", lw=0.8)
    axes[3].set_ylabel("混合 (mm)"); axes[3].set_xlabel("时间 (s)")
    axes[3].set_title(
        f"观测（通道 0）  SNR_target={s['meta']['snr_target_db']:.0f} dB, "
        f"HR={s['meta']['hr_mean_bpm']:.1f} BPM, RR={s['meta']['rr_bpm']:.0f}/min"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

    # 频谱自检
    from scipy.signal import welch
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    for name, key, col in [("呼吸", "resp", "tab:blue"), ("心跳", "card", "tab:red"),
                           ("混合", None, "0.25")]:
        x = s["mix"][0] if key is None else s[key]
        f, P = welch(np.asarray(x, float) - np.mean(x), fs=fs, nperseg=min(2048, len(x)))
        ax.semilogy(f, P, lw=1.2, label=name, color=col)
    ax.axvspan(*CARD_HR_BAND, color="orange", alpha=0.12, label="心率带 1.0-2.5 Hz")
    ax.set_xlim(0, 5); ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("PSD")
    ax.set_title("频谱自检：心搏带内 心跳 vs 混合")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_name("demo_psd.png"), dpi=130)
    plt.close(fig)
    print(f"[gen] 可视化自检图: {path.name} / demo_psd.png")


def main(argv=None) -> int:
    args = parse_args(argv)
    build_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
