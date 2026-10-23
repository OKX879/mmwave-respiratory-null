#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨数据集零点分析：capsule（20 例，100 Hz）+ Parralejo 110 人（440 段，10 Hz）。"""
from __future__ import annotations

import json
import pickle
import warnings
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

warnings.filterwarnings("ignore", category=DeprecationWarning)

HERE = Path(__file__).resolve().parent
OUT = HERE / "null_crossdataset"
CAP_ROOT = Path(r"C:\Users\yxh\Desktop\capsule-4604022-data\Minimum_dataset")
PAR_ROOT = Path(r"C:\Users\yxh\Desktop\db_records")
R_BIN_PAR = 0.31228381041666664
MIN_SEP_S = 0.25
PROM_FRAC = 0.15
DEEP_FRAC = 0.35


# ---------------------------------------------------------------- 基础工具
def bp_complex(x, fs, lo=0.1, hi=0.6, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x.real) + 1j * sosfiltfilt(sos, x.imag)


def phase_B(z, fs):
    zb = bp_complex(z, fs)
    return (np.unwrap(np.angle(zb)) / (2 * np.pi)) % 1.0


def phase_A(z, fs):
    zb = bp_complex(z, fs)
    return (np.unwrap(np.angle(hilbert(zb.real))) / (2 * np.pi)) % 1.0


def rayleigh(ph):
    n = len(ph)
    if n < 5:
        return float("nan"), float("nan")
    r = float(abs(np.mean(np.exp(1j * 2 * np.pi * ph))))
    return r, float(np.exp(-n * r * r))


def deep_nulls(env, fs):
    pk, _ = find_peaks(-env, distance=int(MIN_SEP_S * fs),
                      prominence=PROM_FRAC * np.std(env))
    thr = DEEP_FRAC * float(np.median(env))
    return pk[env[pk] < thr]


def kasa_R(z):
    x, y = z.real, z.imag
    if len(x) < 10:
        return float("nan")
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0], sol[1]
    return float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0)))


def peak_freq(x, fs, lo, hi):
    v = x - np.mean(x)
    F = np.abs(np.fft.rfft(v))
    f = np.fft.rfftfreq(len(v), 1 / fs)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(F[m])]) if m.any() else float("nan")


def metrics(z, fs, bin_lo=0, bin_hi=None):
    """z: (bin, T) 复数。返回该段的全部指标。"""
    zm = z[bin_lo:bin_hi]
    b = bin_lo + int(np.argmax(np.std(zm, axis=1)))
    zc = z[b]
    env = np.abs(zc)
    pk = deep_nulls(env, fs)
    phB = phase_B(zc, fs)
    phA = phase_A(zc, fs)
    rB, pB = rayleigh(phB[pk])
    rA, _ = rayleigh(phA[pk])
    zb = bp_complex(zc, fs)
    f_resp = peak_freq(zb.real, fs, 0.1, 0.6)
    f_dip = peak_freq(env, fs, 0.1, 2.0)
    rel = abs(f_dip - f_resp) / f_resp if np.isfinite(f_resp) and f_resp > 0 else np.nan
    As = abs(complex(np.mean(zc)))
    R = kasa_R(zc - np.mean(zc))
    return {
        "bin": int(b), "n_deep": int(len(pk)), "rate_per_min": len(pk) / (len(zc) / fs) * 60,
        "Rbar_B": rB, "p_B": pB, "Rbar_A": rA,
        "f_resp": f_resp, "f_dip": f_dip, "rel_err": rel,
        "As_over_R": As / R if R and np.isfinite(R) else np.nan,
        "peaks": pk, "z": zc, "env": env,
    }


# ---------------------------------------------------------------- 数据加载
def load_capsule(stem):
    import h5py
    with h5py.File(CAP_ROOT / f"{stem}.mat", "r") as f:
        raw = f[f["data_all"][0, 0]][:]
        lbl = np.asarray(f[f["data_all"][1, 0]][:]).ravel()
    z = (raw["real"] + 1j * raw["imag"]).astype(np.complex128)   # (ch, ang, bin, T)
    return z[:, 0, :, :].mean(axis=0), 100.0, lbl   # (bin, T)


def load_parralejo(sid, cond):
    d = PAR_ROOT / sid / cond
    obj = pickle.loads(zlib.decompress((d / "radar_rFFTs.zlib").read_bytes()))
    cube = obj[0]                                  # (T, ch, bin)
    return cube.mean(axis=1).T, 10.0               # (bin, T)


def parse_bh(path):
    d = {}
    for ln in path.read_text(encoding="utf-8").strip().splitlines():
        k, v = ln.split(",", 1)
        d[k.strip()] = datetime.fromisoformat(v.strip())
    return d["begin"], d["end"]


def first_frame_time(path):
    with path.open(encoding="utf-8") as f:
        return datetime.fromisoformat(f.readline().strip())


# ---------------------------------------------------------------- 主流程
def run_capsule():
    rows = []
    for i in range(1, 21):
        stem = f"patient_{i}"
        try:
            z, fs, lbl = load_capsule(stem)
            m = metrics(z, fs)
            gold = 60000.0 / np.median(np.diff(lbl) / fs * 1000.0) if len(lbl) > 1 else np.nan
            rows.append({k: v for k, v in m.items() if k not in ("peaks", "z", "env")}
                        | {"record": stem, "gold_hr": gold})
        except Exception as e:
            print("capsule fail", stem, e)
    return rows


def run_parralejo():
    rows = []
    conds = ["Sitting/Rest", "Sitting/Post-exercise",
             "Lying/Rest", "Lying/Post-exercise"]
    for i in range(1, 111):
        sid = f"P{i:03d}"
        for cond in conds:
            try:
                z, fs = load_parralejo(sid, cond)
                m = metrics(z, fs, bin_lo=1, bin_hi=7)
                # 屏气窗（仅 Rest）
                d = PAR_ROOT / sid / cond
                bh = {"bh_dur": np.nan, "n_in": np.nan, "n_out": np.nan,
                      "rate_in": np.nan, "rate_out": np.nan}
                bhf = d / "non_breathing_ts.csv"
                if bhf.exists():
                    b, e = parse_bh(bhf)
                    t0 = first_frame_time(d / "radar_timestamps.csv")
                    i0 = int(round((b - t0).total_seconds() * fs))
                    i1 = int(round((e - t0).total_seconds() * fs))
                    i0, i1 = max(0, i0), min(m["env"].size, i1)
                    mask = np.zeros(len(m["env"]), dtype=bool)
                    mask[i0:i1] = True
                    pk = m["peaks"]
                    nin = int(mask[pk].sum())
                    nout = int(len(pk) - nin)
                    dur_in = (i1 - i0) / fs
                    dur_out = (len(m["env"]) - (i1 - i0)) / fs
                    bh = {"bh_dur": dur_in, "n_in": nin, "n_out": nout,
                          "rate_in": nin / dur_in * 60 if dur_in > 0 else np.nan,
                          "rate_out": nout / dur_out * 60 if dur_out > 0 else np.nan}
                rows.append({k: v for k, v in m.items() if k not in ("peaks", "z", "env")}
                            | {"subject": sid, "condition": cond,
                               "range_m": m["bin"] * R_BIN_PAR} | bh)
            except Exception as e:
                print("par fail", sid, cond, e)
    return rows


def summarize_capsule(rows):
    v = [r for r in rows if np.isfinite(r["Rbar_B"])]
    el = [r for r in v if r["n_deep"] >= 5]
    return {
        "n": len(rows),
        "median_n_deep": float(np.median([r["n_deep"] for r in rows])),
        "eligible": len(el),
        "locked": int(sum(1 for r in el if r["Rbar_B"] > 0.4)),
        "median_Rbar": float(np.median([r["Rbar_B"] for r in el])) if el else np.nan,
        "median_AsR": float(np.nanmedian([r["As_over_R"] for r in rows])),
        "frac_dip_at_resp": float(np.mean([r["rel_err"] < 0.2 for r in rows
                                           if np.isfinite(r["rel_err"])])),
    }


def summarize_par(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    el = df[df["n_deep"] >= 5]
    out = {
        "n_records": int(len(df)),
        "median_n_deep": float(df["n_deep"].median()),
        "median_rate_per_min": float(df["rate_per_min"].median()),
        "eligible": int(len(el)),
        "locked": int((el["Rbar_B"] > 0.4).sum()),
        "median_Rbar": float(el["Rbar_B"].median()) if len(el) else np.nan,
        "median_AsR": float(df["As_over_R"].median()),
        "frac_dip_at_resp": float((df["rel_err"] < 0.2).mean()),
        "by_condition": {},
    }
    for c, g in df.groupby("condition"):
        ge = g[g["n_deep"] >= 5]
        out["by_condition"][c] = {
            "n": int(len(g)),
            "median_rate": float(g["rate_per_min"].median()),
            "eligible": int(len(ge)),
            "locked": int((ge["Rbar_B"] > 0.4).sum()),
            "median_Rbar": float(ge["Rbar_B"].median()) if len(ge) else np.nan,
            "frac_dip_at_resp": float((g["rel_err"] < 0.2).mean()),
        }
    bh = df[df["bh_dur"].notna()]
    if len(bh):
        out["breath_hold"] = {
            "n": int(len(bh)),
            "median_rate_in": float(bh["rate_in"].median()),
            "median_rate_out": float(bh["rate_out"].median()),
            "median_ratio": float((bh["rate_in"] / bh["rate_out"]).median()),
            "frac_lower_inside": float((bh["rate_in"] < bh["rate_out"]).mean()),
        }
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(bh["rate_in"], bh["rate_out"])
            out["breath_hold"]["wilcoxon_p"] = float(p)
        except Exception:
            pass
    return out, df


def main():
    OUT.mkdir(exist_ok=True)
    print("[capsule] 20 records ...")
    cap = run_capsule()
    dicts = [{k: v for k, v in r.items()} for r in cap]
    import pandas as pd
    pd.DataFrame(dicts).to_csv(OUT / "capsule_B.csv", index=False, encoding="utf-8")
    print("[parralejo] 440 records ...")
    par = run_parralejo()
    summary_par, df = summarize_par(par)
    df.to_csv(OUT / "parralejo_summary.csv", index=False, encoding="utf-8")
    summary = {"capsule": summarize_capsule(cap), "parralejo": summary_par}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
