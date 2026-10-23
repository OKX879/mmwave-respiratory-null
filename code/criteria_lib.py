#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""无参考评价准则库：14 个 bin 级打分函数，供 matrix_eval 与 calibrate_criteria 调用。
Reference-free criteria library: 14 bin-level scoring functions used by matrix_eval and calibrate_criteria."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import signal
from scipy.signal import find_peaks, hilbert, welch

# ---------------------------------------------------------------------------
# 频段常量（与生产链一致）
# ---------------------------------------------------------------------------
RESP_BAND = (0.08, 0.6)
CARD_BAND = (1.0, 2.5)          # 心率带 / SNR 参考带
DISPLAY_BAND = (0.7, 3.0)       # 心搏波形展示带


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _bandpass(x: np.ndarray, fs: float, band: tuple[float, float], order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [band[0], band[1]], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=float))


def _psd(x: np.ndarray, fs: float, nperseg: Optional[int] = None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if nperseg is None:
        nperseg = min(256, n) if fs < 30.0 else min(2048, n)
    return welch(x - x.mean(), fs=fs, nperseg=max(16, nperseg))


def _psd_at(f: np.ndarray, P: np.ndarray, fq) -> np.ndarray:
    """在任意频率处取 PSD（线性插值，端点截断）。
PSD at an arbitrary frequency, with linear interpolation and clamped endpoints."""
    return np.interp(np.asarray(fq, dtype=float), f, np.maximum(P, 1e-12))


def _pairwise_coherence(rows: Sequence[np.ndarray]) -> float:
    """一组（复数或实数）分量的两两相干均值。
Mean pairwise coherence over a set of complex or real components."""
    rows = [np.asarray(r) for r in rows]
    C = len(rows)
    if C < 2:
        return float("nan")
    vals = []
    for i in range(C):
        for j in range(i + 1, C):
            a = rows[i] - rows[i].mean()
            b = rows[j] - rows[j].mean()
            num = abs(np.sum(a * np.conj(b)))
            den = math.sqrt(float(np.sum(np.abs(a) ** 2)) * float(np.sum(np.abs(b) ** 2))) + 1e-12
            vals.append(num / den)
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# PB —— 生理先验类
# ---------------------------------------------------------------------------
def pb1_spectral_concentration(x: np.ndarray, fs: float, band: tuple[float, float] = CARD_BAND) -> float:
    """心搏带能量集中度：带内功率 / 全带功率。方向 up。
Cardiac-band energy concentration: in-band power over total power. Direction up."""
    f, P = _psd(x, fs)
    m = (f >= band[0]) & (f <= band[1])
    tot = float(np.sum(P))
    if not m.any() or tot <= 0:
        return float("nan")
    return float(np.sum(P[m]) / tot)


def _refine_peak_freq(
    x: np.ndarray, fs: float, band: tuple[float, float], f_coarse: float,
    half_width_hz: float, df_target: float = 0.004, prefilter: bool = True, order: int = 4,
) -> float:
    """在 f_coarse ± half_width 内精化峰频（零填充加窗 FFT + 抛物线插值）。
Refine the peak within f_coarse +- half_width using a zero-padded windowed FFT and parabolic interpolation."""
    if not np.isfinite(f_coarse):
        return float(f_coarse)
    y = np.asarray(x, dtype=float)
    if prefilter:
        lo = max(band[0] * 0.85, 1e-9)
        hi = min(band[1] * 1.15, 0.45 * fs)
        if hi <= lo:
            return float(f_coarse)
        y = _bandpass(y, fs, (lo, hi), order=order)
    n = len(y)
    if n < 64 or not np.isfinite(y).all():
        return float(f_coarse)
    n_fft = int(2 ** math.ceil(math.log2(max(n, fs / df_target))))
    n_fft = max(n_fft, 2 * n)
    S = np.abs(np.fft.rfft(y * np.hanning(n), n=n_fft))
    f = np.fft.rfftfreq(n_fft, 1.0 / fs)
    m = (f >= f_coarse - half_width_hz) & (f <= f_coarse + half_width_hz)
    if int(m.sum()) < 3:
        return float(f_coarse)
    idx = np.flatnonzero(m)
    k = int(idx[int(np.argmax(S[idx]))])
    if 0 < k < len(S) - 1:
        y_m1, y_0, y_p1 = float(S[k - 1]), float(S[k]), float(S[k + 1])
        den = y_m1 - 2.0 * y_0 + y_p1
        if abs(den) > 1e-18:
            d = float(np.clip(0.5 * (y_m1 - y_p1) / den, -0.5, 0.5))
            return float(f[k] + d * (f[1] - f[0]))
    return float(f[k])


def estimate_resp_freq(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float] = (0.14, 0.5),
    valid_bpm: tuple[float, float] = (7.0, 28.0),
    snr_min: float = 1.8,
    refine: bool = True,
) -> Optional[float]:
    """估计呼吸基频 f_resp（Hz），口径对齐生产链 RHC 的估计器。
Estimate the respiratory fundamental f_resp in Hz, matching the production RHC estimator."""
    n = len(x)
    if n < 256:
        return None
    y = _bandpass(x, fs, band)
    f, P = _psd(y, fs)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any() or float(P[m].max()) <= 0:
        return None
    pk = float(P[m].max())
    fp = float(f[m][int(np.argmax(P[m]))])
    if not (valid_bpm[0] / 60.0 <= fp <= valid_bpm[1] / 60.0):
        return None
    bg = float(np.median(P[m]))
    if bg <= 0 or pk / bg < snr_min:
        return None
    if refine and len(f) > 1:
        fp_r = _refine_peak_freq(y, fs, band, fp, 1.5 * float(f[1] - f[0]),
                                 prefilter=False)
        # 边界保护：精化后的值仍须落在生理带与频带内，否则回退栅格值
        if (valid_bpm[0] / 60.0 <= fp_r <= valid_bpm[1] / 60.0
                and band[0] <= fp_r <= band[1]):
            fp = float(fp_r)
    return fp


def pb2_resp_harmonic_residue(
    x_card: np.ndarray,
    fs: float,
    f_resp: Optional[float] = None,
    band: tuple[float, float] = CARD_BAND,
    tol_bpm: float = 3.0,
    k_max: int = 6,
    x_for_fresp: Optional[np.ndarray] = None,
) -> float:
    """呼吸谐波残余比 RHR：心率带内【落在 k·f_resp 附近】的能量 / 非谐波区能量。方向 down。
Residual harmonic ratio: energy near k*f_resp inside the heart band over non-harmonic energy. Direction down."""
    f, P = _psd(x_card, fs)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        return float("nan")
    if f_resp is None or not np.isfinite(f_resp) or f_resp <= 0:
        src = x_for_fresp if x_for_fresp is not None else x_card
        f_resp = estimate_resp_freq(src, fs)
    if f_resp is None or not np.isfinite(f_resp) or f_resp <= 0:
        return float("nan")
    harm = np.zeros_like(f, dtype=bool)
    tol = tol_bpm / 60.0
    for k in range(2, k_max + 1):
        fk = k * f_resp
        if band[0] <= fk <= band[1]:
            harm |= np.abs(f - fk) <= tol
    e_h = float(np.sum(P[m & harm]))
    e_r = float(np.sum(P[m & ~harm]))
    return float(e_h / max(e_r, 1e-18))


def pb3_ibi_physio_yield(
    x_card: np.ndarray,
    fs: float,
    min_ms: float = 333.0,
    max_ms: float = 1500.0,
    height_factor: float = 0.5,
) -> dict:
    """IBI 生理合格率（用生理先验当"软参考"）。主方向 up（合格率）。
IBI physiological validity rate, using priors as a soft reference. Direction up."""
    x = np.asarray(x_card, dtype=float)
    std = float(np.std(x))
    if std <= 1e-12:
        return {"yield": 0.0, "cv": float("nan"), "n": 0}
    min_dist = max(1, int(round(min_ms / 1000.0 * fs)))
    pk, _ = find_peaks(x, distance=min_dist, height=height_factor * std)
    if pk.size < 2:
        return {"yield": 0.0, "cv": float("nan"), "n": int(pk.size)}
    ibi = np.diff(pk) / fs * 1000.0
    ok = (ibi >= min_ms) & (ibi <= max_ms)
    if ok.sum() < 2:
        return {"yield": float(ok.mean()), "cv": float("nan"), "n": int(pk.size)}
    ibi_ok = ibi[ok]
    return {
        "yield": float(ok.mean()),
        "cv": float(np.std(ibi_ok) / max(np.mean(ibi_ok), 1e-9)),
        "n": int(pk.size),
    }


def pb4_cardiac_amp_plausibility(x_card: np.ndarray, lo_mm: float = 0.02, hi_mm: float = 1.0) -> float:
    """心搏位移幅度合理性（亚毫米）→ 返回 [0,1] 梯形隶属度。方向 up。
Plausibility of the beat displacement (sub-millimetre) as a trapezoidal membership in [0,1]. Direction up."""
    rms = float(np.sqrt(np.mean(np.asarray(x_card, dtype=float) ** 2)))
    if rms <= 0 or hi_mm <= lo_mm:
        return 0.0
    if lo_mm <= rms <= hi_mm:
        return 1.0
    return float(np.clip(rms / lo_mm if rms < lo_mm else hi_mm / rms, 0.0, 1.0))


def pb5_resp_cardiac_amp_ratio(
    x_resp: np.ndarray, x_card: np.ndarray, lo: float = 5.0, hi: float = 50.0
) -> float:
    """呼吸/心跳幅度比合理性 → [0,1]。方向 up。
Plausibility of the respiration-to-beat amplitude ratio, mapped to [0,1]. Direction up."""
    rr = float(np.sqrt(np.mean(np.asarray(x_resp, dtype=float) ** 2)))
    rc = float(np.sqrt(np.mean(np.asarray(x_card, dtype=float) ** 2)))
    if rc <= 0 or hi <= lo:
        return 0.0
    ratio = rr / rc
    if lo <= ratio <= hi:
        return 1.0
    return float(np.clip(ratio / lo if ratio < lo else hi / ratio, 0.0, 1.0))


# ---------------------------------------------------------------------------
# RC —— 冗余一致性类（最可靠：不依赖生理假设）
# ---------------------------------------------------------------------------
def rc1_cross_channel_coherence(card_ch: Sequence[np.ndarray]) -> float:
    """跨通道心搏相干（8 通道）。方向 up。真心跳在所有通道同步。
Cross-channel beat coherence over 8 channels. Direction up; a real beat is synchronous everywhere."""
    return _pairwise_coherence(card_ch)


def rc2_cross_bin_coherence(card_bins: Sequence[np.ndarray]) -> float:
    """跨距离格心搏相干（中心格 ±1）。方向 up。
Cross-range-bin beat coherence over the centre bin +/-1. Direction up."""
    return _pairwise_coherence(card_bins)


def estimate_hr_psd(x: np.ndarray, fs: float, band: tuple[float, float] = CARD_BAND,
                    refine: bool = True) -> float:
    """简易心率估计（Welch 谱主峰 + 峰位精化）。口径同生产链的单路 PSD 主峰。
Simple heart-rate estimate from the Welch peak with refinement, matching the production single-channel PSD peak."""
    f, P = _psd(x, fs)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any() or float(P[m].max()) <= 0:
        return float("nan")
    f_coarse = float(f[m][int(np.argmax(P[m]))])
    if refine and len(f) > 1:
        f_coarse = _refine_peak_freq(x, fs, band, f_coarse, 1.5 * float(f[1] - f[0]))
    return float(f_coarse * 60.0)


def rc3_cross_window_stability(
    x_card: np.ndarray,
    fs: float,
    n_sub: int = 4,
    hr_fn=estimate_hr_psd,
) -> float:
    """跨窗稳定性：把窗切成 n_sub 个子窗各自估 HR，取 std（BPM）。方向 down。
Cross-window stability: split the window into n_sub sub-windows, take the sd of HR in BPM. Direction down."""
    x = np.asarray(x_card, dtype=float)
    L = len(x) // max(1, n_sub)
    if L < 64:
        return float("nan")
    hrs = [hr_fn(x[k * L:(k + 1) * L], fs) for k in range(n_sub)]
    hrs = [h for h in hrs if np.isfinite(h)]
    return float(np.std(hrs)) if len(hrs) >= 2 else float("nan")


def rc4_subband_consistency(card_per_res: Sequence[np.ndarray]) -> float:
    """子带一致性：不同分辨率层重构出的心搏分量之间的一致度。方向 up。
Subband consistency: agreement between beat components reconstructed at different resolution levels. Direction up."""
    return _pairwise_coherence(card_per_res)


# ---------------------------------------------------------------------------
# SM —— 结构 / 数学类
# ---------------------------------------------------------------------------
def sm1_energy_conservation(x: np.ndarray, coeffs: Sequence[np.ndarray]) -> float:
    """能量守恒偏差 |Σc² - Σx²| / Σx²。方向 down（仅对正交基有效）。
Energy conservation error |sum c^2 - sum x^2| / sum x^2. Direction down, orthogonal bases only."""
    e_t = float(np.sum(np.asarray(x, dtype=float) ** 2))
    e_w = sum(float(np.sum(np.asarray(c, dtype=float) ** 2)) for c in coeffs)
    return float(abs(e_w - e_t) / max(e_t, 1e-18))


def sm2_component_leakage(
    x_resp: np.ndarray, x_card: np.ndarray, fs: float, band: tuple[float, float] = DISPLAY_BAND
) -> float:
    """分量间泄漏：呼吸分量与心搏分量在频域的带内重叠度。方向 down。
Inter-component leakage: in-band spectral overlap between the respiration and beat components. Direction down."""
    f, Pr = _psd(x_resp, fs)
    _, Pc = _psd(x_card, fs)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        return float("nan")
    num = float(np.sum(np.minimum(Pr[m], Pc[m])))
    den = float(np.sum(np.maximum(Pr[m], Pc[m]))) + 1e-18
    return float(num / den)


def sm3_retained_energy_ratio(x: np.ndarray, recon: np.ndarray) -> float:
    """重构保留能量比。**非单调** —— 只能做趋势监控，禁止进评分。
Reconstruction energy ratio. Non-monotonic, so monitoring only and never used for scoring."""
    e_x = float(np.sum(np.asarray(x, dtype=float) ** 2))
    e_r = float(np.sum(np.asarray(recon, dtype=float) ** 2))
    return float(e_r / max(e_x, 1e-18))


def sm4_hps_prominence(
    x: np.ndarray, fs: float, band: tuple[float, float] = CARD_BAND, order: int = 3
) -> float:
    """HPS 谐波结构突出度：HPS(f)=P(f)·P(2f)·P(3f)… 主峰 / 带内中位背景。方向 up。
HPS harmonic prominence: product of the first harmonics, peak over median in-band background. Direction up."""
    f, P = _psd(x, fs)
    Ps = np.maximum(P, 1e-12)
    hps = Ps.copy()
    for h in range(2, order + 1):
        hps = hps * _psd_at(f, P, h * f)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        return float("nan")
    ratio = float(np.max(hps[m]) / max(float(np.median(hps[m])), 1e-18))
    return float(math.log10(max(ratio, 1e-18)))   # 取 log10：原始比值跨 12 个数量级，不便聚合


def sm5_ridge_smoothness(x: np.ndarray, fs: float, band: tuple[float, float] = CARD_BAND) -> float:
    """时频脊线连续性：心搏带瞬时频率（Hilbert 解析信号相位差分）的 std（Hz）。方向 down。
Time-frequency ridge continuity: sd of the instantaneous beat-band frequency from the Hilbert phase. Direction down."""
    y = _bandpass(x, fs, band)
    z = hilbert(y)
    ph = np.unwrap(np.angle(z))
    if len(ph) < 3:
        return float("nan")
    inst_f = np.diff(ph) * fs / (2.0 * np.pi)
    m = (inst_f >= band[0]) & (inst_f <= band[1])
    if int(m.sum()) < 10:
        return float("nan")
    return float(np.std(inst_f[m]))


# ---------------------------------------------------------------------------
# 注册表与统一入口
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Criterion:
    key: str
    name: str
    group: str            # PB / RC / SM
    direction: str        # "up" | "down"
    report_only: bool     # True = 只作趋势监控，不参与评分
    production_ready: bool  # True = 可直接复用生产链既有实现
    desc: str


CRITERIA: tuple[Criterion, ...] = (
    Criterion("pb1", "心搏带能量集中度", "PB", "up", False, False, "带内心搏能量占比；单用易被窄带假峰骗"),
    Criterion("pb2", "呼吸谐波残余比", "PB", "down", False, True, "心率带内 k·f_resp 附近能量占比；f_resp 取自宽带信号"),
    Criterion("pb3", "IBI 生理合格率", "PB", "up", False, True, "IBI 落在 333-1500ms 的比例；复用 EventDetector 思路"),
    Criterion("pb4", "心搏幅度合理性", "PB", "up", False, False, "心搏位移 RMS 落在亚毫米区间的隶属度；需标定量纲"),
    Criterion("pb5", "呼吸/心跳幅度比", "PB", "up", False, False, "比值落区间的隶属度；区间场景相关"),
    Criterion("rc1", "跨通道心搏相干", "RC", "up", False, True, "8 通道两两相干；真心跳同步出现"),
    Criterion("rc2", "跨距离格心搏相干", "RC", "up", False, True, "中心格±1 的相干"),
    Criterion("rc3", "跨窗稳定性", "RC", "down", False, False, "子窗 HR 的 std；区分稳定/不稳定而非心跳/呼吸"),
    Criterion("rc4", "子带一致性", "RC", "up", False, False, "不同分辨率层重构分量的一致度"),
    Criterion("sm1", "能量守恒偏差", "SM", "down", False, False, "|Σc²-Σx²|/Σx²；仅正交基有效"),
    Criterion("sm2", "分量间泄漏", "SM", "down", False, False, "呼吸与心搏分量的带内重叠度"),
    Criterion("sm3", "保留能量比", "SM", "n/a", True, False, "非单调，只作趋势监控，禁止进评分"),
    Criterion("sm4", "HPS 谐波突出度", "SM", "up", False, True, "log10(P(f)P(2f)P(3f) 主峰/背景)；复用既有 HPS"),
    Criterion("sm5", "脊线平滑度", "SM", "down", False, False, "瞬时频率 std；可被平滑假输出骗，须与 sm4 合用"),
)

_BY_KEY = {c.key: c for c in CRITERIA}


def evaluate_all(res: dict, fs: float, *, resp_freq: Optional[float] = None,
                 pb5_range: tuple[float, float] = (5.0, 50.0)) -> dict:
    """对一份分解产物计算全部可用准则。
Compute every applicable criterion for one decomposition product."""
    out: dict[str, float] = {}
    card = res.get("card")
    resp = res.get("resp")
    mix = res.get("mix")
    recon = res.get("recon")

    nan = float("nan")
    if card is not None:
        card = np.asarray(card, dtype=float)
        out["pb1"] = pb1_spectral_concentration(card, fs)
        out["pb2"] = pb2_resp_harmonic_residue(
            card, fs, resp_freq,
            x_for_fresp=(res.get("x_fresp") if res.get("x_fresp") is not None
                         else res.get("mix") if res.get("mix") is not None
                         else res.get("resp")),
        )
        pb3 = pb3_ibi_physio_yield(card, fs)
        out["pb3"] = pb3["yield"]
        out["pb4"] = pb4_cardiac_amp_plausibility(card)
        out["rc3"] = rc3_cross_window_stability(card, fs)
        out["sm4"] = sm4_hps_prominence(card, fs)
        out["sm5"] = sm5_ridge_smoothness(card, fs)
    else:
        for k in ("pb1", "pb2", "pb3", "pb4", "rc3", "sm4", "sm5"):
            out[k] = nan

    out["pb5"] = (
        pb5_resp_cardiac_amp_ratio(resp, card, *pb5_range)
        if (card is not None and resp is not None) else nan
    )
    out["sm2"] = (
        sm2_component_leakage(resp, card, fs)
        if (card is not None and resp is not None) else nan
    )
    out["rc1"] = rc1_cross_channel_coherence(res["card_ch"]) if res.get("card_ch") is not None else nan
    out["rc2"] = rc2_cross_bin_coherence(res["card_bins"]) if res.get("card_bins") is not None else nan
    out["rc4"] = rc4_subband_consistency(res["card_per_res"]) if res.get("card_per_res") else nan
    out["sm1"] = (
        sm1_energy_conservation(mix, res["coeffs"])
        if (mix is not None and res.get("coeffs")) else nan
    )
    out["sm3"] = (
        sm3_retained_energy_ratio(mix, recon)
        if (mix is not None and recon is not None) else nan
    )
    return out


def criteria_table() -> list[dict]:
    return [asdict(c) for c in CRITERIA]


# ---------------------------------------------------------------------------
# 自检：验证每个准则在"理想分解"与"未分解"两个极端上方向是否正确
# ---------------------------------------------------------------------------
def _selftest(npz_path: Path, n_samples: int = 60, channels: int = 8) -> int:
    data = np.load(npz_path)
    card_gt = data["card"]          # (M, T)
    resp_gt = data["resp"]
    mix = data["mix"]               # (M, C, T)
    M, T = card_gt.shape
    n = min(n_samples, M)
    # 采样率优先读同目录 config.json（由生成器写出），否则退回 100 Hz
    fs = 100.0
    cfg_path = npz_path.parent / "config.json"
    if cfg_path.exists():
        try:
            fs = float(json.loads(cfg_path.read_text(encoding="utf-8"))["fs"])
        except Exception:
            pass
    rng = np.random.default_rng(0)

    ideal_rows: list[dict] = []
    raw_rows: list[dict] = []
    for i in range(n):
        tile_card = np.tile(card_gt[i][None, :], (channels, 1))
        res_ideal = {
            "card": card_gt[i],
            "resp": resp_gt[i],
            "mix": mix[i, 0],
            "recon": resp_gt[i] + card_gt[i],
            "card_ch": tile_card,
        }
        res_raw = {
            "card": mix[i, 0],
            "resp": mix[i, 0],
            "mix": mix[i, 0],
            "recon": mix[i, 0],
            "card_ch": rng.normal(0.0, 1.0, (channels, T)),   # 完全不相干的极端反例
        }
        ideal_rows.append(evaluate_all(res_ideal, fs))
        raw_rows.append(evaluate_all(res_raw, fs))

    def _stat(rows, key):
        """取【中位数】：比值型准则尾部重，均值会被离群样本主导。
Take the median: ratio-type criteria have heavy tails and the mean is dominated by outliers."""
        v = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    print(f"[selftest] npz={npz_path.name}  样本数={n}  通道={channels}  fs={fs:.0f} Hz")
    print(f"{'准则':28s} {'方向':4s} {'理想分解':>12s} {'未分解(混合)':>14s}   判定")
    print("-" * 76)
    n_pass = n_fail = n_info = 0
    for c in CRITERIA:
        a = _stat(ideal_rows, c.key)
        b = _stat(raw_rows, c.key)
        if not (np.isfinite(a) and np.isfinite(b)):
            verdict, tag = "数据不足/需额外输入", "INFO"
            n_info += 1
        elif c.direction == "n/a" or c.report_only:
            verdict, tag = "非单调，仅趋势监控", "INFO"
            n_info += 1
        else:
            ok = (a > b) if c.direction == "up" else (a < b)
            verdict = "方向正确" if ok else "方向不符(需检查)"
            tag = "PASS" if ok else "FAIL"
            n_pass += int(ok)
            n_fail += int(not ok)
        print(f"{c.key+' '+c.name:28s} {c.direction:4s} {a:12.4g} {b:14.4g}   [{tag}] {verdict}")
    print("-" * 76)
    print(f"[selftest] PASS={n_pass}  FAIL={n_fail}  INFO={n_info}（INFO 属预期：需额外输入或非单调）")
    print("[selftest] 说明：'未分解'列把【混合信号本身】当作心搏分量，是最坏反例；")
    print("           rc1 的未分解列用【完全不相干的通道噪声】，为极端反例。")
    return 0 if n_fail == 0 else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="无参考准则库自检")
    p.add_argument("--selftest", action="store_true", help="在合成数据上验证准则方向")
    p.add_argument("--npz", type=str, default="synthetic_data/grid.npz", help="合成数据 npz 路径")
    p.add_argument("--samples", type=int, default=60, help="自检取样条数")
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--list", action="store_true", help="列出全部准则定义")
    args = p.parse_args(argv)

    if args.list:
        for c in CRITERIA:
            print(f"{c.key:5s} [{c.group}] {c.name:16s} dir={c.direction:4s} "
                  f"report_only={int(c.report_only)} prod_ready={int(c.production_ready)}  {c.desc}")
        return 0
    if args.selftest:
        return _selftest(Path(args.npz), n_samples=args.samples, channels=args.channels)
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
