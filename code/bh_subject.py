#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稳健性验证 2：屏气对照按“被试”聚合，避免 440 段的伪重复。
Robustness check 2: aggregate breath-hold by subject to avoid pseudo-replication over 440 segments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
OUT = HERE / "null_crossdataset"
DUR = 50.0


def main():
    df = pd.read_csv(OUT / "parralejo_summary.csv")
    df = df[df["bh_dur"].notna()].copy()
    assert len(df) == 220, len(df)

    # 每个被试合并两个 Rest 段
    g = df.groupby("subject").agg(
        n_in=("n_in", "sum"), n_out=("n_out", "sum"),
        dur_in=("bh_dur", "sum"),
        n_rec=("condition", "count"),
    )
    g["dur_out"] = DUR * g["n_rec"] - g["dur_in"]
    g["rate_in"] = g["n_in"] / g["dur_in"] * 60
    g["rate_out"] = g["n_out"] / g["dur_out"] * 60
    g["ratio"] = g["rate_in"] / g["rate_out"].replace(0, np.nan)

    stat, p = wilcoxon(g["rate_in"], g["rate_out"])
    summary = {
        "subjects": int(len(g)),
        "median_rate_in": float(g["rate_in"].median()),
        "median_rate_out": float(g["rate_out"].median()),
        "median_ratio": float(g["ratio"].median()),
        "frac_lower_inside": float((g["rate_in"] < g["rate_out"]).mean()),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p),
    }

    # Poisson GLM，按被试聚类
    long = pd.concat([
        pd.DataFrame({"subject": g.index, "count": g["n_in"].values,
                      "dur": g["dur_in"].values, "inside": 1}),
        pd.DataFrame({"subject": g.index, "count": g["n_out"].values,
                      "dur": g["dur_out"].values, "inside": 0}),
    ], ignore_index=True)
    X = sm.add_constant(long[["inside"]].astype(float))
    model = sm.GLM(long["count"], X, family=sm.families.Poisson(),
                   offset=np.log(long["dur"]))
    res = model.fit(cov_type="cluster", cov_kwds={"groups": long["subject"]})
    coef = float(res.params["inside"])
    ci = res.conf_int().loc["inside"].tolist()
    summary["poisson"] = {
        "rate_ratio": float(np.exp(coef)),
        "ci95": [float(np.exp(ci[0])), float(np.exp(ci[1]))],
        "p": float(res.pvalues["inside"]),
        "n_subjects": int(long["subject"].nunique()),
    }

    (OUT / "bh_subject.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    g.to_csv(OUT / "bh_subject.csv", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
