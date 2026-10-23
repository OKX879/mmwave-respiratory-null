#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生产链 RHC 前端的薄封装（T3）。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# 生产链 RHC 模块（不在本目录，按路径加载，避免复制实现）
PROD_RHC_PATH = (Path(__file__).resolve().parents[1]
                 / "analysis" / "respiratory_harmonic_cancellation.py")

_MOD = None


def _load_prod_module():
    """按路径加载生产链 RHC 模块（只加载一次）。"""
    if not PROD_RHC_PATH.exists():
        raise FileNotFoundError(f"找不到生产链 RHC 模块：{PROD_RHC_PATH}")
    spec = importlib.util.spec_from_file_location("prod_rhc", PROD_RHC_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prod_rhc"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_rhc_frontend(fs: float):
    """返回一个 RHC 前端函数 f(x) -> process 结果 dict。"""
    global _MOD
    if _MOD is None:
        _MOD = _load_prod_module()
    proc = _MOD.RespiratoryHarmonicCancellation(fs=float(fs))

    def front(x: np.ndarray) -> dict:
        return proc.process(np.asarray(x, dtype=float))

    front.prod_path = str(PROD_RHC_PATH)     # 便于在报告里标注来源
    return front


# ---------------------------------------------------------------------------
def _selftest() -> int:
    """自检：撞心率场景（RR=18BPM 的 4 次谐波 = HR=72BPM 的基频）。"""
    fs, dur = 100.0, 30.0
    n = int(dur * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    hr, rr = 1.2, 0.3          # Hz
    x = (0.3 * np.sin(2 * np.pi * hr * t)
         + 6.0 * np.sin(2 * np.pi * rr * t)
         + 6.0 * 0.2 * np.sin(2 * np.pi * 2 * rr * t)
         + rng.normal(0, 0.05, n))
    front = make_rhc_frontend(fs)
    out = front(x)
    f_resp = out["resp_track"].f_resp_mean
    trust = out["rhc_trust"]
    protected = [(k, round(f, 3)) for k, f, _, _ in out["reg_result"].protected_harmonics]

    F = np.abs(np.fft.rfft(out["x_heart"]))
    fr = np.fft.rfftfreq(n, 1 / fs)

    def mag(fq):
        return float(F[np.argmin(np.abs(fr - fq))])

    print(f"[selftest] 生产模块：{PROD_RHC_PATH}")
    print(f"[selftest] f_resp 估计 = {f_resp:.4f} Hz（真值 0.300）  trust = {trust:.3f}")
    print(f"[selftest] 受保护谐波 = {protected}")
    print(f"[selftest] 残差 FFT 幅度： 0.3Hz={mag(0.3):.3f}  0.6Hz={mag(0.6):.3f}"
          f"  0.9Hz={mag(0.9):.3f}  1.2Hz(真心率)={mag(1.2):.1f}")
    ok = (abs(f_resp - 0.3) < 0.02 and trust > 0.5 and mag(1.2) > 10 * mag(0.9))
    print(f"[selftest] {'PASS' if ok else 'FAIL'}：呼吸基频估准 + 心搏频点被保护 + 谐波被压掉")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
