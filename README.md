# Respiratory-Phase Multipath Nulls in mmWave Cardiac Monitoring

Code and results for the manuscript *"Respiratory-Phase Multipath Nulls in mmWave Cardiac
Monitoring: Depth Bound, Saturation, and a Deployable Criterion"*, submitted to
**IEEE Transactions on Microwave Theory and Techniques**.

---

## The problem in one paragraph

A range bin of an FMCW radar watching a seated person is not a point scatterer. It sums a
**static phasor** $A_s$ (chair, torso, wall, desk) and a **small phasor** $A_c\,e^{j\theta(t)}$
that rotates with respiration. Both live in one complex number:

$$z(t) = A_s + A_c\,e^{j\theta(t)}, \qquad \rho = A_c/A_s .$$

Draw that trajectory and it is a circle of radius $A_c$ centred at $A_s$. When $\rho = 1$ the
centre sits exactly one radius from the origin, so **the circle passes through the origin**: the
magnitude $|z|$ collapses to zero, the phase is undefined, and every phase-based vital-sign
estimate fails. This is not noise and not a calibration error; it is geometry, and no amount of
DC-offset removal changes where the circle's centre sits *relative to its own radius*.

We give the depth of that collapse a closed form,

$$r = \frac{|\rho-1|}{1+\rho}, \qquad 0 \le r \le 1,$$

show that the interval within which any usable threshold is exceeded is **narrow**
($r > 0.15 \iff \rho \in [0.7391,\ 1.3529]$), and identify **chirp bandwidth** — which sets the
bin's range resolution and therefore how much static clutter shares the bin with the chest — as
the one design knob that moves a bin into or out of that band.

---

## What is in this repository

| Path | Contents |
|---|---|
| `code/` | **80 Python scripts**, flat by necessity (see note below) |
| `figs/` | The 9 figures of the paper and the supplement, as vector PDF and PNG |
| `paper/` | Submitted manuscript, supplementary material, graphical abstract |

### Why `code/` is flat

The scripts import each other as top-level modules — `from lei_pilot import load, ROOT`,
`from criteria_lib import CRITERIA`, `from null_crossdataset import ...` — and several of them
resolve that with `sys.path.insert(0, HERE)`, where `HERE` is the script's own directory.
Splitting them into subpackages silently breaks those imports, so they stay in one directory.
Grouping is by filename prefix instead:

> **Language note.** Module docstrings and inline comments in `code/` are written in Chinese,
> because that is how the analysis was actually done and annotated. Everything that faces
> outward — this README, figure labels, table headers, the manuscript — is in English.
> The docstrings are being translated gradually; a file with an English header has been done.

| Prefix / name | Purpose |
|---|---|
| `exp_*.py` | The experiments behind the paper's five evidence claims |
| `lei_*.py` | Everything touching the Lei dual-subject dataset (layouts, bins, coverage) |
| `null_*.py` | Null probability, cross-dataset transfer, alignment gate, robustness |
| `parralejo_*.py`, `emo_*.py`, `real_*.py`, `matrix_*.py`, `cardiac_*.py` | The other three datasets and their evaluation |
| `fig_*.py`, `paper_figures_*.py`, `null_phase_figure.py`, `make_graphical_abstract.py` | Figure generation |
| `criteria_lib.py`, `calibrate_criteria.py`, `bridge_null_criteria.py` | The decision criteria library and its calibration |
| `xref_audit.py`, `md2tex.py` | LaTeX cross-reference audit, Markdown→TeX conversion |

### Figures and the scripts that produce them

| Figure | File | Script |
|---|---|---|
| Fig. 1 | `fig_model_depth` | `paper_figures_v3.py` |
| Fig. 2 | `fig_failure_map` | `fig_failure_map_v6.py` |
| Fig. 3 | `fig_v5_resolution` | `paper_figures_v5.py` |
| Fig. 4 | `fig_c2_shape` | `exp_C2_shape.py` |
| Fig. S1 | `fig_bh_conditions` | `paper_figures_v3.py` |
| Fig. S2 | `fig_window` | `paper_figures_v3.py` |
| Fig. S3 | `fig_null_phase` | `null_phase_figure.py` |
| Fig. S4 | `fig_v5_example` | `paper_figures_v5.py` |
| Fig. S5 | `fig_alignment_phase` | `paper_figures_v3.py` |

---

## What is **not** in this repository

**The data.** All four datasets are public but large, and two of them carry licences that do not
permit redistribution. `DATA.md` lists each one with its DOI and the directory name the scripts
expect.

Because of that, the scripts will not run out of the box: they read bare relative paths such as
`matrix_out_t4/`, `real_data/`, `synthetic_data_clean/grid.npz`. Put the data in those directories
at the repository root and they work unchanged.

---

## Quick start

```bash
python -m pip install -r requirements.txt

# once the data directories exist at the repo root:
python code/paper_figures_v3.py        # Fig. 1, S1, S2, S5
python code/fig_failure_map_v6.py      # Fig. 2
python code/paper_figures_v5.py        # Fig. 3, S4
python code/exp_C2_shape.py            # Fig. 4
python code/null_phase_figure.py       # Fig. S3
```

Run scripts **from the repository root**, not from inside `code/`. The relative data paths assume
the root is the working directory.

---

## Three things that will bite you

**1. Never seed with `hash()`.** Python's `hash()` depends on `PYTHONHASHSEED` and changes between
processes. An early version used `abs(hash((layout, pos, bw, k)))` and produced 15.3% on one run
and 14.4% on the next from identical data. Every script here uses

```python
import zlib
seed = zlib.crc32(f"{layout}|{pos}|{bw}|{k}".encode()) % (2**31)
```

If you rewrite a bootstrap, do the same.

**2. Two intervals that look alike and are not.**

| Expression | Meaning | Value |
|---|---|---|
| $\lvert\rho-1\rvert/(1+\rho) = 0.15$ | the paper's 15% criterion, a *relative* difference | $\rho \in [0.7391,\ 1.3529]$ |
| $\lvert\rho-1\rvert = 0.15$ | an *absolute* 15% difference | $\rho \in [0.85,\ 1.15]$ |

The first is the criterion. The second appears only as the sweep range of the synthetic
experiment, where it is legitimate. Mixing them produces a paper that quotes two bands for one
threshold.

**3. `D` is scale-invariant, so removing the centre does not help.** Subtracting the fitted static
phasor changes the *denominator* of the depth ratio, not the ratio's existence. On these records
it raises the record-level null rate from 39.8% to 65.2%. That result is in the paper (and
supplementary Table S5) precisely because it contradicts the intuitive fix.

---

## Data

See **`DATA.md`**. Four datasets, all public: a dual-subject mmWave dataset with nine layouts, a
Code Ocean capsule, a 110-subject Zenodo record, and an emotion-classification Zenodo record.

## Citation

`CITATION.cff` is included, so GitHub will render a "Cite this repository" button. If you use this
code, please cite the manuscript.

## Licence

Code: **MIT** (`LICENSE`). Figures: **CC BY 4.0**. The manuscript text is the author's and, on
acceptance, will be transferred to IEEE — see `paper/NOTICE.md`.

## Status

Manuscript submitted to IEEE TMTT, October 2026. The PDFs under `paper/` are the **submitted
manuscript**, not the version of record.

---

## 中文说明

这是论文《室内毫米波心脏监测中呼吸相位驱动的多径零点：深度界、饱和与可部署判据》的代码与结果仓库。

**一句话结论**：一个距离单元里同时有静止回波和随呼吸旋转的回波，当两者幅度可比（$\rho\to1$），
复平面轨迹会穿过原点，相位塌陷——不是噪声，是几何。论文给出塌陷深度的闭式界
$r=|\rho-1|/(1+\rho)$，证明超过任何可用阈值的区间很窄（$\rho\in[0.7391,1.3529]$），
并指出 **chirp 带宽** 是唯一能把某个格推进或推出该区间的设计旋钮。

**仓库里有什么**：`code/` 80 个脚本（必须平铺，脚本之间互相 import）、`figs/` 9 张图（矢量 PDF + PNG）、
`paper/` 投稿版正文与补充材料。

**仓库里没有什么**：数据。四个数据集都是公开的，但体积大且部分不允许再分发，见 `DATA.md`。
脚本读的是 `matrix_out_t4/`、`real_data/` 这类相对路径，把数据放到仓库根目录下同名目录即可直接运行。

**三个坑**：① 随机种子不能用 `hash()`，必须用 `zlib.crc32`；② 两个 ρ 区间别混
（`[0.7391,1.3529]` 是判据，`[0.85,1.15]` 只是合成实验的扫描范围）；③ 减圆心不会降低零点率，
反而从 39.8% 升到 65.2%，因为深度是比值。
