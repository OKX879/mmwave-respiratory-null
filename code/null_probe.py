#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多径零点检验：在 Zhang 等 2024 的 Range-FFT 立方体上直接量复平面轨迹。
Null test: measure the complex trajectory directly on the Range-FFT cube of Zhang et al. 2024."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATA = Path(r"C:\Users\yxh\Desktop\capsule-4604022-data\Minimum_dataset")
FS = 100.0


def load_cube(stem: str) -> np.ndarray:
    with h5py.File(DATA / f"{stem}.mat", "r") as f:
        raw = f[f["data_all"][0, 0]][:]
    return (raw["real"] + 1j * raw["imag"]).astype(np.complex128)


def kasa_fit(z: np.ndarray) -> tuple[float, float, float, float]:
    """代数圆拟合（Kasa）。返回 (cx, cy, R, 残差比例)。
Algebraic circle fit (Kasa), returning centre, radius and residual ratio."""
    x, y = z.real, z.imag
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    R = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0)))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    resid = float(np.std(r) / max(R, 1e-12))
    return float(cx), float(cy), R, resid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="多径零点检验（复平面轨迹）")
    ap.add_argument("--patient", type=int, default=1)
    ap.add_argument("--angle", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5, help="按运动能量选前 k 个 bin")
    args = ap.parse_args(argv)

    z = load_cube(f"patient_{args.patient}")
    seg_all = z[:, args.angle, :, :]                    # (ch, bin, T)
    # 去静态后按 (ch,bin) 的运动能量排序，取 topk
    ac = seg_all - seg_all.mean(axis=2, keepdims=True)
    energy = ac.std(axis=2).mean(axis=0)                # (bin,)
    order = np.argsort(energy)[::-1][:args.topk]

    print(f"=== patient_{args.patient}  angle={args.angle}  (T={seg_all.shape[2]}, {FS:.0f} Hz) ===")
    print(f"  运动能量最强的 bin: {order.tolist()}")
    print()
    print(f"{'bin':>4s}{'|As|/R':>9s}{'R/|As|':>9s}{'圆残差':>8s}"
          f"{'|z|凹陷比':>10s}{'凹陷频':>9s}{'相位跨越':>9s}")
    rows = []
    for b in order:
        vals = []
        for ch in range(seg_all.shape[0]):
            zz = seg_all[ch, b, :]
            As = float(np.abs(zz.mean()))
            zc = zz - zz.mean()
            cx, cy, R, resid = kasa_fit(zc)
            cen = float(np.hypot(cx, cy))
            ratio = cen / max(R, 1e-12)                 # |中心|/R：>1 表示不绕原点
            # |z| 包络的凹陷：用 1 s 滑窗的 min/max
            w = int(1.0 * FS)
            a = np.abs(zz)
            mins = np.array([a[i:i + w].min() for i in range(0, len(a) - w, w // 2)])
            maxs = np.array([a[i:i + w].max() for i in range(0, len(a) - w, w // 2)])
            dip = float(np.median(mins) / max(np.median(maxs), 1e-12))
            # 凹陷频率：对 |z| 做 FFT，取 0.05–2 Hz 的峰（用户关注"周期性"）
            A = np.abs(np.fft.rfft(a - a.mean()))
            f = np.fft.rfftfreq(len(a), 1 / FS)
            m = (f >= 0.05) & (f <= 2.0)
            fpk = float(f[m][int(np.argmax(A[m]))]) if m.any() else float("nan")
            # 是否绕原点：中心到原点距离 < R 则轨迹包含原点
            crosses = bool(ratio < 1.0)
            vals.append((As, R, resid, dip, fpk, crosses))
        As, R, resid, dip, fpk, crosses = [np.median([v[i] for v in vals]) for i in range(6)]
        n_cross = int(np.sum([v[5] for v in vals]))
        print(f"{b:>4d}{As/max(R,1e-12):>9.2f}{R/max(As,1e-12):>9.2f}{resid:>8.3f}"
              f"{dip:>10.2f}{fpk:>9.2f}{f'{n_cross}/{len(vals)}':>9s}")
        rows.append({"bin": int(b), "As_over_R": As / max(R, 1e-12), "R_over_As": R / max(As, 1e-12),
                     "circle_resid": resid, "dip_ratio": dip, "dip_freq_hz": fpk,
                     "n_ch_crossing_origin": n_cross, "n_ch": len(vals)})

    print("\n解读（三条判据）")
    print("  ① |As|/R ≫ 1  → 轨迹远离原点 → **零点在这种条件下被抑制**（不是被增强）")
    print("  ② |z|凹陷比（中位 min/max）接近 1 → 没有幅度塌陷；显著 <0.5 才算有")
    print("  ③ '相位跨越' 列 = 有多少通道的中心距原点 < 半径（即轨迹能绕/穿原点）")
    print("     —— 只有这一步成立，'零点'才有几何可能")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
