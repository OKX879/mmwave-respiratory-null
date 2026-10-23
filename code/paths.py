"""统一的路径解析。

所有外部数据集与输出根目录都可被环境变量覆盖，默认按仓库根/桌面推导，
因此仓库克隆到任何机器上都不需要改代码。

环境变量：
  MMWAVE_DATA_ROOT     外部数据集根目录（默认: ~/Desktop）
  MMWAVE_OUT_ROOT      交叉数据集输出目录（默认: <repo>/null_crossdataset）
  MMWAVE_PRODUCT_ROOT  生产管线目录（默认: <repo>）
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("MMWAVE_DATA_ROOT", Path.home() / "Desktop"))
OUT_ROOT = Path(os.environ.get("MMWAVE_OUT_ROOT", REPO_ROOT / "null_crossdataset"))
PRODUCT_ROOT = Path(os.environ.get("MMWAVE_PRODUCT_ROOT", REPO_ROOT))
FIGS = REPO_ROOT / "figs"

# 常用数据集
CAPSULE = DATA_ROOT / "capsule-4604022-data" / "Minimum_dataset"
DB_RECORDS = DATA_ROOT / "db_records"
EMO_PROCESSED = DATA_ROOT / "emo_dataset" / "processed" / "02_processed_data"
EMO_MULTI = DATA_ROOT / "emo_dataset" / "raw" / "_multi"
EMO_PILOT = DATA_ROOT / "emo_dataset" / "raw" / "_pilot"
FMCW_DUAL = (DATA_ROOT / "FMCW radar-based multi-person vital sign monitoring data"
             / "FMCW radar-based multi-person vital sign monitoring data")


def ensure_out():
    """确保输出目录存在。"""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUT_ROOT
