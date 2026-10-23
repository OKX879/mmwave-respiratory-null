#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选矩阵批量评估：对每个（方法, 带, 层）组合算指标，写 manifest.csv 与 matrix_config.json。
Batch evaluation of the candidate matrix: metrics for every (method, band, level) combination, written to manifest.csv and matrix_config.json."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pywt
from scipy.signal import butter, find_peaks, sosfiltfilt

from criteria_lib import (
    CARD_BAND,
    DISPLAY_BAND,
    RESP_BAND,
    evaluate_all,
    estimate_hr_psd,
    estimate_resp_freq,
)

# ---------------------------------------------------------------------------
# DWT 频带归属
# ---------------------------------------------------------------------------
def level_bands(level: int, fs: float) -> dict[str, tuple[float, float]]:
    """给出各层频带：d1..dL 为细节带（由高到低），aL 为近似带。
Band edges per level: d1 to dL are detail bands from high to low, aL is the approximation band."""
    nyq = fs / 2.0
    out: dict[str, tuple[float, float]] = {}
    for j in range(1, level + 1):
        out[f"d{j}"] = (nyq / 2 ** j, nyq / 2 ** (j - 1))
    out[f"a{level}"] = (0.0, nyq / 2 ** level)
    return out


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def assign_component(band: tuple[float, float]) -> str:
    """把某一层的频带划给 呼吸 / 心搏 / 丢弃（"层归属"模式用）。
Assign one level's band to respiration, beat or discard, used by the level-assignment mode."""
    lo, hi = band
    if hi <= DISPLAY_BAND[0]:
        return "resp"
    if lo >= DISPLAY_BAND[0]:
        return "card"
    return "resp"


def _bp(x: np.ndarray, fs: float, band: tuple[float, float], order: int = 4) -> np.ndarray:
    sos = butter(order, [band[0], band[1]], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=float))


# ---------------------------------------------------------------------------
# 阈值规则
# ---------------------------------------------------------------------------
def _sigma_mad(c: np.ndarray) -> float:
    c = np.asarray(c, dtype=float)
    if c.size == 0:
        return 0.0
    return float(np.median(np.abs(c)) / 0.6745)


def _thr_universal(c: np.ndarray) -> float:
    n = max(int(np.size(c)), 2)
    return _sigma_mad(c) * math.sqrt(2.0 * math.log(n))


def _thr_sure(c: np.ndarray) -> float:
    """Stein 无偏风险估计阈值（标准实现）。
Stein's unbiased risk estimate threshold, standard implementation."""
    c = np.asarray(c, dtype=float)
    n = c.size
    if n < 4:
        return 0.0
    a = np.sort(np.abs(c)) ** 2
    cum = np.cumsum(a)
    kk = np.arange(1, n + 1, dtype=float)
    risk = np.empty(n + 1)
    risk[0] = float(n)
    risk[1:] = (n - 2.0 * kk + cum + (n - kk) * a) / n
    kbest = int(np.argmin(risk))
    return 0.0 if kbest == 0 else math.sqrt(float(a[kbest - 1]))


def _thr_minimax(c: np.ndarray) -> float:
    n = max(int(np.size(c)), 2)
    if n > 32:
        return _sigma_mad(c) * (0.3936 + 0.1829 * math.log2(n))
    return 0.0


_THR_FUNCS = {
    "universal-soft": ( _thr_universal, "soft"),
    "universal-hard": ( _thr_universal, "hard"),
    "sure":           ( _thr_sure,      "soft"),
    "minimax":        ( _thr_minimax,   "soft"),
}


def apply_threshold(coeffs: list[np.ndarray], rule: str) -> list[np.ndarray]:
    """对细节系数逐层做阈值（近似系数 aL 不处理）。rule='none' 时原样返回。
Threshold detail coefficients level by level, leaving aL untouched; rule='none' returns them unchanged."""
    if rule == "none":
        return [np.array(c, dtype=float) for c in coeffs]
    fn, mode = _THR_FUNCS[rule]           # coeffs[0]=aL, 其余为 dL..d1
    out = [np.array(coeffs[0], dtype=float)]
    for c in coeffs[1:]:
        cc = np.asarray(c, dtype=float)
        thr = float(fn(cc))
        if mode == "soft":
            out.append(np.sign(cc) * np.maximum(np.abs(cc) - thr, 0.0))
        else:
            out.append(np.where(np.abs(cc) > thr, cc, 0.0))
    return out


# ---------------------------------------------------------------------------
# DWT 分解 → （呼吸, 心搏, 全重构, 系数, 各层心搏重构）
# ---------------------------------------------------------------------------
def decompose_dwt(
    x: np.ndarray,
    wavelet: str,
    level: int,
    threshold_rule: str,
    boundary: str,
    fs: float,
    separation: str = "bandpass",
    need_full: bool = True,
) -> Optional[dict]:
    """DWT 分解 →（呼吸, 心搏, 全重构, 系数, 各层心搏重构）。
DWT decomposition returning respiration, beat, full reconstruction, coefficients and per-level beat reconstructions."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    try:
        coeffs = pywt.wavedec(x, wavelet, mode=boundary, level=level)
    except Exception:
        return None
    if len(coeffs) != level + 1:
        return None
    th = apply_threshold(coeffs, threshold_rule)

    keys = [f"a{level}"] + [f"d{j}" for j in range(level, 0, -1)]
    try:
        recon = np.real(pywt.waverec(th, wavelet, mode=boundary))[:n]
    except Exception:
        return None

    if separation == "bandpass":
        card = _bp(recon, fs, DISPLAY_BAND)
        owner = {"mode": "bandpass"}
        card_levels: list[np.ndarray] = []
        resp = None
        if need_full:
            resp = _bp(recon, fs, RESP_BAND)
            try:
                c2 = pywt.wavedec(card, wavelet, mode=boundary, level=level)
                for kk, cc in zip(keys, c2):
                    only = [cc if k2 == kk else np.zeros_like(c2i) for k2, c2i in zip(keys, c2)]
                    card_levels.append(np.real(pywt.waverec(only, wavelet, mode=boundary))[:n])
            except Exception:
                card_levels = []
    else:
        bands = level_bands(level, fs)
        owner = {k: assign_component(bands[k]) for k in keys}
        resp_c, card_c = [], []
        card_levels = []
        for k, c in zip(keys, th):
            zeros = np.zeros_like(c)
            if owner[k] == "resp":
                resp_c.append(c); card_c.append(zeros)
            elif owner[k] == "card":
                resp_c.append(zeros); card_c.append(c)
            else:
                resp_c.append(zeros); card_c.append(zeros)
        for k in keys:
            if owner[k] != "card":
                continue
            only = [c if kk == k else np.zeros_like(cc) for kk, cc in zip(keys, th)]
            try:
                card_levels.append(np.real(pywt.waverec(only, wavelet, mode=boundary))[:n])
            except Exception:
                pass
        try:
            resp = np.real(pywt.waverec(resp_c, wavelet, mode=boundary))[:n]
            card = np.real(pywt.waverec(card_c, wavelet, mode=boundary))[:n]
        except Exception:
            return None

    return {"resp": resp, "card": card, "recon": recon, "coeffs": th,
            "card_per_res": card_levels, "owner": owner}


# ---------------------------------------------------------------------------
# CWT 分解（pywt.cwt 可用；pywt.icwt 在 1.8.0 已移除 → 用自实现 _icwt_like）
# ---------------------------------------------------------------------------
def cwt_scales(wavelet: str, fs: float, band: tuple[float, float], n_scales: int) -> np.ndarray:
    fc = float(pywt.central_frequency(wavelet))
    s_lo = fc * fs / band[1]     # 对应带内最高频 → 最小尺度
    s_hi = fc * fs / band[0]     # 对应带内最低频 → 最大尺度
    return np.geomspace(max(s_lo, 1e-6), max(s_hi, 1e-6), n_scales)


def _icwt_like(coeffs: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Torrence & Compo 式尺度求和重构（自实现，替代已移除的 pywt.icwt）。
Torrence and Compo scale-sum reconstruction, reimplemented because pywt.icwt was removed."""
    s = np.asarray(scales, dtype=float)
    dj = float(np.log2(s[-1] / s[0]) / (len(s) - 1)) if len(s) > 1 else 1.0
    return np.real(np.sum(np.asarray(coeffs) / np.sqrt(s)[:, None], axis=0)) * dj


def decompose_cwt(
    x: np.ndarray, wavelet: str, scales: np.ndarray, fs: float, band: tuple[float, float]
) -> Optional[dict]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    try:
        coeffs, freqs = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / fs)
    except Exception:
        return None
    if not np.isfinite(coeffs).all():
        return None
    m_resp = (freqs >= RESP_BAND[0]) & (freqs <= RESP_BAND[1])
    m_card = (freqs >= DISPLAY_BAND[0]) & (freqs <= DISPLAY_BAND[1])
    resp = _icwt_like(coeffs * m_resp[:, None], scales)[:n]
    card = _icwt_like(coeffs * m_card[:, None], scales)[:n]
    recon = _icwt_like(coeffs, scales)[:n]
    if len(resp) != n or len(card) != n:
        return None
    # 各尺度分带重构（取心搏带内的几个子段）用于 rc4
    card_levels = []
    idx = np.where(m_card)[0]
    if idx.size >= 2:
        chunks = np.array_split(idx, min(3, idx.size))
        for ch in chunks:
            mk = np.zeros_like(freqs, dtype=bool)
            mk[ch] = True
            card_levels.append(_icwt_like(coeffs * mk[:, None], scales)[:n])
    return {"resp": resp, "card": card, "recon": recon, "coeffs": None,
            "card_per_res": card_levels, "owner": {"cwt": "by-band"}}


# ---------------------------------------------------------------------------
# 金标指标（仅合成数据可算）
# ---------------------------------------------------------------------------
def gt_metrics(card_est: np.ndarray, resp_est: np.ndarray, *,
               card_gt: np.ndarray, resp_gt: np.ndarray,
               hr_gt_bpm: float, rr_gt_bpm: float, fs: float) -> dict:
    """合成数据上的"真实指标"（有干净源 → 可算）。
Ground-truth metrics on synthetic data, available because the clean source is known."""
    # 心率误差：在展示带上估心率（见文件头口径说明）
    hr_est = estimate_hr_psd(card_est, fs, band=DISPLAY_BAND)
    hr_err = abs(hr_est - hr_gt_bpm) if np.isfinite(hr_est) else float("nan")

    # 呼吸率误差
    f_resp_est = estimate_resp_freq(resp_est, fs)
    rr_err = abs(f_resp_est * 60.0 - rr_gt_bpm) if f_resp_est else float("nan")

    # 尺度不变 SDR：card_est 解释 card_gt 的能力
    a = np.asarray(card_gt, dtype=float) - np.mean(card_gt)
    b = np.asarray(card_est, dtype=float) - np.mean(card_est)
    denom = float(np.dot(b, b)) + 1e-18
    alpha = float(np.dot(a, b)) / denom
    err = a - alpha * b
    sdr = 10.0 * math.log10(max(float(np.dot(a, a)), 1e-18) / max(float(np.dot(err, err)), 1e-18))
    # 相关（未标准化前的形状一致性）
    corr = float(np.dot(a, b) / (math.sqrt(float(np.dot(a, a)) * denom) + 1e-18))

    return {"hr_mae_bpm": hr_err, "rr_mae_bpm": rr_err,
            "sdr_db": sdr, "corr_card": corr}


# ---------------------------------------------------------------------------
# 候选集合构建
# ---------------------------------------------------------------------------
def build_candidates(args) -> list[dict]:
    cands: list[dict] = []
    if args.kernels in ("dwt", "both"):
        for w in args.dwt_wavelets:
            for lv in args.dwt_levels:
                for th in args.dwt_thresholds:
                    for bd in args.dwt_boundaries:
                        cands.append({
                            "kind": "dwt", "wavelet": w, "level": int(lv),
                            "threshold": th, "boundary": bd,
                            "separation": args.separation,
                            "id": f"dwt|{w}|L{lv}|{th}|{bd}|{args.separation}",
                        })
    if args.kernels in ("cwt", "both"):
        for w in args.cwt_wavelets:
            cands.append({
                "kind": "cwt", "wavelet": w, "level": -1,
                "threshold": "none", "boundary": "n/a",
                "id": f"cwt|{w}|scales{args.cwt_scales}",
            })
    return cands


# ---------------------------------------------------------------------------
# 单候选评估（供串行与多进程共用）
# ---------------------------------------------------------------------------
_WORKER_CTX: dict = {}


def _init_worker(ctx: dict) -> None:
    """multiprocessing 初始化：把上下文（已切片的信号）注入每个工作进程。
multiprocessing initializer that injects the sliced signals into each worker."""
    _WORKER_CTX.clear()
    _WORKER_CTX.update(ctx)


def _worker(cand: dict):
    return eval_one_candidate(cand, _WORKER_CTX)


def eval_one_candidate(cand: dict, ctx: dict):
    """算一个候选在全部（已选）样本上的结果 → (聚合行, 逐样本行, 失败数)。
Evaluate one candidate on all selected samples, returning the aggregate row, per-sample rows and failure count."""
    mix = ctx["mix"]                # (n_samples, C, T) 已按 sample_idx 切片
    mix_raw = ctx["mix_raw"]        # 同上；未开 RHC 时与 mix 相同
    card_gt = ctx["card_gt"]
    resp_gt = ctx["resp_gt"]
    metas = ctx["metas"]            # {sample_id: meta}
    rhc_trust = ctx["rhc_trust"]
    fs = ctx["fs"]
    C = ctx["C"]
    scales = ctx["scales"]
    sid = ctx["sid"]

    per_sample: list[dict] = []
    fail = 0
    for pos, i in enumerate(sid):
        x = mix[pos]
        xr = mix_raw[pos] if mix_raw is not None else x
        cards_ch: list[np.ndarray] = []
        res_ref: Optional[dict] = None
        for ch in range(C):
            if cand["kind"] == "dwt":
                d = decompose_dwt(x[ch], cand["wavelet"], cand["level"],
                                  cand["threshold"], cand["boundary"], fs,
                                  separation=cand.get("separation", "bandpass"),
                                  need_full=(ch == 0))
            else:
                d = decompose_cwt(x[ch], cand["wavelet"], scales, fs,
                                  (RESP_BAND[0], DISPLAY_BAND[1]))
            if d is None:
                break
            cards_ch.append(d["card"])
            if ch == 0:
                res_ref = d
        if res_ref is None or len(cards_ch) != C:
            fail += 1
            continue

        meta = metas.get(i, {})
        res = {
            "card": res_ref["card"],
            "resp": res_ref["resp"],
            "mix": x[0],
            "x_fresp": xr[0],
            "recon": res_ref["recon"],
            "card_ch": cards_ch,
            "card_per_res": res_ref.get("card_per_res") or [],
            "coeffs": res_ref.get("coeffs") if cand["kind"] == "dwt" else None,
        }
        crit = evaluate_all(res, fs)
        gt = gt_metrics(
            res["card"], res["resp"],
            card_gt=card_gt[pos], resp_gt=resp_gt[pos],
            hr_gt_bpm=float(meta.get("hr_mean_bpm", "nan") or "nan"),
            rr_gt_bpm=float(meta.get("rr_bpm", "nan") or "nan"),
            fs=fs,
        )
        per_sample.append({"candidate_id": cand["id"], "sample_id": int(i),
                           **crit, **gt, "rhc": ctx["rhc"],
                           "rhc_trust": float(rhc_trust.get(int(i), float("nan")))})

    if not per_sample:
        return None, [], fail

    agg = {"candidate_id": cand["id"], "kind": cand["kind"], "wavelet": cand["wavelet"],
           "level": cand["level"], "threshold": cand["threshold"], "boundary": cand["boundary"],
           "separation": cand.get("separation", "n/a"),
           "max_level": cand.get("max_level", -1),
           "level_over_max": cand.get("level_over_max", 0),
           "n_ok": len(per_sample), "n_fail": fail}
    keys = [k for k in per_sample[0].keys()
            if k not in ("candidate_id", "sample_id", "rhc")]
    for k in keys:
        v = np.array([r[k] for r in per_sample], dtype=float)
        v = v[np.isfinite(v)]
        agg[k] = float(np.median(v)) if v.size else float("nan")
    return agg, per_sample, fail


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_matrix(args) -> int:
    npz_path = Path(args.npz)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    t = data["t"]
    mix = data["mix"]                 # (M, C, T)
    card_gt = data["card"]            # (M, T)
    resp_gt = data["resp"]
    M, C, T = mix.shape

    cfg_path = npz_path.parent / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    fs = float(cfg.get("fs", 100.0))

    # 金标元数据
    man_path = npz_path.parent / "manifest.csv"
    metas: list[dict] = []
    if man_path.exists():
        with open(man_path, "r", encoding="utf-8") as f:
            metas = list(csv.DictReader(f))
    if len(metas) < M:
        metas = [{}] * M

    n_samples = M if args.samples <= 0 else min(args.samples, M)
    # 分层抽样：数据集按 (SNR, HR, RR) 分块排列，直接取前 N 条会全部落在
    # 最低 SNR 块上（实测：取前 40 条 = 全为 -10 dB），导致统计被最低 SNR
    # 霸占。改为沿索引均匀取点，使抽样覆盖整个网格。
    if args.samples <= 0 or n_samples >= M:
        sample_idx = list(range(M))
    else:
        sample_idx = np.unique(np.linspace(0, M - 1, n_samples).astype(int)).tolist()
    n_samples = len(sample_idx)
    cands = build_candidates(args)
    # 超层数标记（T4）：pywt.wavedec 在 level > dwt_max_level 时**不报错**，
    # 只发 UserWarning 并继续分解 —— 但此时顶层近似带只剩 T/2^L 个样点
    # （如 T=3000、L=10 → 约 3 点），是**退化配置**，必须单独标记，
    # 否则会把"基底 × 层数"的比较污染成"谁在退化区更差"。
    n_over = 0
    for c in cands:
        if c["kind"] != "dwt":
            continue
        try:
            c["max_level"] = int(pywt.dwt_max_level(T, pywt.Wavelet(c["wavelet"]).dec_len))
        except Exception:
            c["max_level"] = -1
        c["level_over_max"] = int(c["max_level"] >= 0 and c["level"] > c["max_level"])
        n_over += c["level_over_max"]
    print(f"[matrix] 样本={n_samples}/{M}  通道={C}  fs={fs:.0f} Hz  候选数={len(cands)}"
          f"（其中超最大层数 {n_over} 个，已标记 level_over_max）")
    print(f"[matrix] 频带归属：呼吸{RESP_BAND} 心搏{DISPLAY_BAND}（心搏用展示带，见文件头口径）")
    print(f"[matrix] 前端 RHC：{'开启（与生产链同源）' if args.rhc == 'on' else '关闭（谐波未对消的对照条件）'}")

    # ---------------------------------------------------------------------
    # RHC 前端（可选）：与生产链共享同一份实现（rhc_frontend.py 按路径加载
    # analysis/respiratory_harmonic_cancellation.py）。
    # 关键：RHC 与"小波基"无关，因此每个 (样本, 通道) 只算一次并就地替换，
    # 全部候选共用 —— 既保证"只变小波基这一件事"，又避免重复计算。
    # ---------------------------------------------------------------------
    rhc_trust: dict[int, float] = {}
    rhc_mod_path = None
    mix_raw = None
    if args.rhc == "on":
        from rhc_frontend import make_rhc_frontend
        front = make_rhc_frontend(fs)
        rhc_mod_path = front.prod_path
        print(f"[matrix] RHC 模块：{rhc_mod_path}")
        mix = np.array(mix, dtype=float)           # 可写副本（原地替换）
        # 保留【对消前】的原始信号副本：pb2 的 f_resp 必须从原始信号估计，
        # 因为 RHC 会把呼吸基频(k=1)与 2 次谐波(k=2)作为"安全谐波"减除，
        # 对消后信号里已无呼吸基频（生产链的 tracker 也是在输入信号上估计的）。
        mix_raw = mix.copy()
        t_rhc = time.time()
        for i in sample_idx:
            for ch in range(C):
                out = front(mix[i, ch])
                mix[i, ch] = out["x_heart"]
                if ch == 0:
                    rhc_trust[int(i)] = float(out["rhc_trust"])
        tr = np.array(list(rhc_trust.values()), dtype=float)
        print(f"[matrix] RHC 前端完成：{len(sample_idx)}×{C} 通道，耗时 {time.time()-t_rhc:.1f}s；"
              f"呼吸跟踪可信度 trust 中位={np.median(tr):.3f} 最小={tr.min():.3f}")


    scales = None
    if any(c["kind"] == "cwt" for c in cands):
        scales = cwt_scales(args.cwt_wavelets[0], fs, (RESP_BAND[0], DISPLAY_BAND[1]), args.cwt_scales)

    raw_rows: list[dict] = []
    agg_rows: list[dict] = []
    t0 = time.time()

    # ---- 构建并行上下文（已按 sample_idx 切片，避免逐任务 pickle 大数组）----
    ctx = {
        "mix": np.array([mix[i] for i in sample_idx], dtype=float),
        "mix_raw": (np.array([mix_raw[i] for i in sample_idx], dtype=float)
                    if mix_raw is not None else None),
        "card_gt": np.array([card_gt[i] for i in sample_idx], dtype=float),
        "resp_gt": np.array([resp_gt[i] for i in sample_idx], dtype=float),
        "metas": {int(i): metas[i] for i in sample_idx},
        "rhc_trust": rhc_trust, "fs": fs, "C": C, "scales": scales,
        "sid": [int(i) for i in sample_idx], "rhc": args.rhc,
    }

    jobs = int(getattr(args, "jobs", -1))
    if jobs == -1:
        jobs = min(os.cpu_count() or 1, 16)
    jobs = max(1, jobs)

    def _absorb(cid: str, agg, raw, fail: int) -> None:
        if agg is None:
            print(f"[matrix] {cid}: 全部样本分解失败（fail={fail}），跳过")
            return
        agg_rows.append(agg)
        raw_rows.extend(raw)

    if jobs > 1:
        import multiprocessing as mp
        print(f"[matrix] 多进程并行：{jobs} 进程（候选间完全独立）")
        with mp.Pool(processes=jobs, initializer=_init_worker, initargs=(ctx,)) as pool:
            for ci, (cand, (agg, raw, fail)) in enumerate(
                    zip(cands, pool.imap(_worker, cands, chunksize=1))):
                _absorb(cand["id"], agg, raw, fail)
                if agg is not None and (args.verbose or (ci + 1) % 50 == 0
                                        or ci == len(cands) - 1):
                    print(f"[matrix] {ci+1}/{len(cands)} {cand['id']:<40s} "
                          f"hr_mae={agg.get('hr_mae_bpm', float('nan')):6.2f}  "
                          f"sdr={agg.get('sdr_db', float('nan')):6.2f}  "
                          f"elapsed={time.time()-t0:5.1f}s")
    else:
        for ci, cand in enumerate(cands):
            agg, raw, fail = eval_one_candidate(cand, ctx)
            _absorb(cand["id"], agg, raw, fail)
            if agg is not None and (args.verbose or (ci + 1) % 10 == 0
                                    or ci == len(cands) - 1):
                print(f"[matrix] {ci+1}/{len(cands)} {cand['id']:<40s} "
                      f"hr_mae={agg.get('hr_mae_bpm', float('nan')):6.2f}  "
                      f"sdr={agg.get('sdr_db', float('nan')):6.2f}  "
                      f"elapsed={time.time()-t0:5.1f}s")

    if not agg_rows:
        print("[matrix] 没有任何候选成功完成，退出")
        return 1

    # 写文件
    import csv as _csv
    def _write(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        cols = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    _write(out_dir / "matrix_results.csv", agg_rows)
    _write(out_dir / "matrix_raw.csv", raw_rows)
    with open(out_dir / "matrix_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "npz": str(npz_path), "n_samples": n_samples, "channels": C, "fs": fs,
            "rhc": args.rhc, "rhc_module": rhc_mod_path,
            "n_candidates_over_max_level": n_over,
            "resp_band": list(RESP_BAND), "card_display_band": list(DISPLAY_BAND),
            "hr_estimate_band": list(DISPLAY_BAND),
            "candidates": cands, "n_candidates": len(cands),
        }, f, ensure_ascii=False, indent=2)

    # 汇总：按金标指标排序（合成数据有真值 → 这是"真值排名"，
    # 后续 §6 要用无参考准则去预测它）
    print("\n[matrix] 按金标心率误差排序（前 12；越小越好）")
    print(f"{'candidate_id':36s} {'HR_MAE':>8s} {'RR_MAE':>8s} {'SDR':>8s} {'corr':>7s}")
    print("-" * 72)
    for r in sorted(agg_rows, key=lambda d: (np.nan_to_num(d.get("hr_mae_bpm", np.inf))))[:12]:
        print(f"{r['candidate_id']:36s} {r.get('hr_mae_bpm', float('nan')):8.2f} "
              f"{r.get('rr_mae_bpm', float('nan')):8.2f} {r.get('sdr_db', float('nan')):8.2f} "
              f"{r.get('corr_card', float('nan')):7.3f}")

    print(f"\n[matrix] 完成：{len(agg_rows)} 个候选 × {n_samples} 样本，"
          f"耗时 {time.time()-t0:.1f}s")
    print(f"[matrix] 输出：{out_dir/'matrix_results.csv'} / matrix_raw.csv / matrix_config.json")
    print("[matrix] 下一步：用 matrix_raw.csv 做 §6 校准（无参考准则 vs 金标指标的 Spearman ρ）")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="候选小波基矩阵批量评估",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--npz", type=str, default="synthetic_data_clean/grid.npz", help="合成数据 npz")
    p.add_argument("--out", type=str, default="matrix_out", help="输出目录")
    p.add_argument("--kernels", choices=["dwt", "cwt", "both"], default="dwt")
    p.add_argument("--separation", choices=["bandpass", "levels"], default="bandpass",
                   help="bandpass=基只负责去噪/分带用固定滤波器（主路径）；"
                        "levels=用基自身的二进频带划分来分离（结构对照）")
    p.add_argument("--dwt-wavelets", type=str, nargs="+",
                   default=["haar", "db4", "db8", "sym8", "coif4", "bior3.5"])
    p.add_argument("--dwt-levels", type=int, nargs="+", default=[4, 5, 6])
    p.add_argument("--dwt-thresholds", type=str, nargs="+",
                   default=["none", "universal-soft"])
    p.add_argument("--dwt-boundaries", type=str, nargs="+", default=["symmetric"])
    p.add_argument("--cwt-wavelets", type=str, nargs="+",
                   default=["morl", "mexh", "cmor1.5-1.0", "gaus4"])
    p.add_argument("--cwt-scales", type=int, default=64)
    p.add_argument("--samples", type=int, default=0, help="取样条数；0=全部")
    p.add_argument("--rhc", choices=["off", "on"], default="off",
                   help="前端 RHC（呼吸谐波对消）开关。on=与生产链对齐；"
                        "off=谐波未对消的对照条件（见 T3）")
    p.add_argument("--jobs", type=int, default=-1,
                   help="并行进程数；-1=自动（cpu_count，上限 16），1=串行（用于逐位核对）")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_matrix(parse_args()))
