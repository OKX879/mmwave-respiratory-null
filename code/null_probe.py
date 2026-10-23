#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多径零点检验：在 Zhang 等 2024 的 Range-FFT 立方体上直接量复平面轨迹。

要回答的问题（按证据强度从弱到强）
--------------------------------
Q1 复平面轨迹的**圆心偏移**有多大？（即静态分量 A_s 相对运动分量的大小）
Q2 轨迹上是否存在**周期性幅度塌陷**（|z| 的极小值）？深度多少？频率是多少？
Q3 是否存在**相位跳变/绕原点**？—— 这是"零点穿越"的硬证据。

必须先说清的物理（否则会得出相反结论）
------------------------------------
双相量模型 z(t) = A_s + A_1 e^{jφ1} + A_2 e^{jφ2}：
- **零点（null）来自两个"运动"相量的相对相位扫过 π**；
- **静态杂波 A_s 只把整个轨迹平移**；当 |A_s| ≫ |A_1|,|A_2| 时，
  轨迹变成一个**远离原点的小圆**，**永远不会穿越原点 → 零点被抑制，而不是被产生**。
所以"因为没去杂波、圆心偏离原点，所以更容易看到零点塌陷"这个推理**方向是反的**。
本脚本用 |中心|/R 直接量化这一条。

另一条必须排除的平凡解释
------------------------
距离维只有 30 格且未加窗：**目标跨越 bin 边界**也会造成幅度周期性起伏
（与多径无关）。判别办法：看幅度凹陷的**频率**是否与呼吸/心跳一致，
以及凹陷时**相位是否连续**（bin 边界跨越会让相位连续、只是幅度变；真零点会让相位快速翻转）。

用法：python null_probe.py --patient 1
"""
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
    """代数圆拟合（Kasa）。返回 (cx, cy, R, 残差比例)。"""
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
