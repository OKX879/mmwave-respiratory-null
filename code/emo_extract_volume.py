#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从单个 zip 分卷中提取完整包含的文件（分卷 zip 无法用标准库直接解压）。

用途：emo 数据集 raw 是 15 个 3 GiB 分卷 + 末尾 .zip；首卷已完整包含 P01 的
00-17.mat（18.mat 跨卷）。本脚本扫描本地文件头，把完整落在本卷内的条目解出，
供试点分析使用；正式全量解压仍建议装 7-Zip。

用法：
  python emo_extract_volume.py --volume <z01> --out <dir> --prefix 01_raw_data/mmwave/P01/
"""
from __future__ import annotations

import argparse
import mmap
import struct
import zlib
from pathlib import Path


def scan(mm):
    pos, out = 0, []
    while True:
        i = mm.find(b"PK\x03\x04", pos)
        if i < 0:
            break
        (sig, ver, flag, method, mtime, mdate, crc, csize, usize,
         nlen, elen) = struct.unpack("<IHHHHHIIIHH", mm[i:i + 30])
        name = mm[i + 30:i + 30 + nlen].decode("utf-8", "replace")
        out.append({"name": name, "method": method, "flag": flag,
                    "csize": csize, "usize": usize, "off": i,
                    "nlen": nlen, "elen": elen})
        if csize in (0, 0xFFFFFFFF):
            pos = i + 4
        else:
            pos = i + 30 + nlen + elen + csize
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    vol = Path(args.volume)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = 0

    with vol.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        size = len(mm)
        for e in scan(mm):
            if not e["name"].startswith(args.prefix) or e["name"].endswith("/"):
                continue
            start = e["off"] + 30 + e["nlen"] + e["elen"]
            end = start + e["csize"]
            if end > size:
                print(f"[skip] {e['name']} spans volume (end {end} > {size})")
                n_skip += 1
                continue
            buf = mm[start:end]
            if e["method"] == 0:
                raw = buf
            elif e["method"] == 8:
                raw = zlib.decompress(buf, -15)
            else:
                print(f"[skip] {e['name']} method {e['method']}")
                n_skip += 1
                continue
            if len(raw) != e["usize"]:
                print(f"[warn] {e['name']} size {len(raw)} != {e['usize']}")
            dst = out / f"{Path(e['name']).parent.name}_{Path(e['name']).name}"
            dst.write_bytes(raw)
            print(f"[ok] {dst.name} {len(raw)/2**20:.1f} MiB")
            n_ok += 1
        mm.close()
    print(f"extracted {n_ok}, skipped {n_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
