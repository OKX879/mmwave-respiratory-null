# Respiratory-Phase Multipath Nulls in mmWave Cardiac Monitoring

Code and results for the manuscript *"Respiratory-Phase Multipath Nulls in mmWave Cardiac
Monitoring: Depth Bound, Saturation, and a Deployable Criterion"*, submitted to IEEE TMTT.

A range bin sums a static phasor $A_s$ and a breathing phasor $A_c e^{j\theta}$. When
$\rho = A_c/A_s \approx 1$ the circle passes through the origin, $|z| \to 0$, and phase-based
vital-sign estimates fail. The paper bounds that depth, $r = |\rho-1|/(1+\rho)$, and shows chirp
bandwidth is the design knob that sets how often it happens.

## Contents

| Path | Contents |
|---|---|
| `code/` | 80 Python scripts, flat |
| `figs/` | 9 figures, vector PDF and PNG |
| `paper/` | Manuscript, supplementary material, graphical abstract |

## Run

```bash
pip install -r requirements.txt
```

Run from the repository root; data directories must be named as in `DATA.md`.

```bash
python code/paper_figures_v3.py      # Fig. 1, S1, S2, S5
python code/fig_failure_map_v6.py    # Fig. 2
python code/paper_figures_v5.py      # Fig. 3, S4
python code/exp_C2_shape.py          # Fig. 4
python code/null_phase_figure.py     # Fig. S3
```

`reproduce.sh` runs all five.

## Notes

- Scripts import each other by bare module name (`from lei_pilot import load`), so `code/` is
  flat. Splitting it into subpackages breaks the imports.
- Docstrings and comments are in Chinese. README, figure labels and the manuscript are English.
- Seeds use `zlib.crc32`, not `hash()`: `hash()` changes between processes and produced 15.3% on
  one run and 14.4% on the next from identical data.
- Two intervals that look alike: $\lvert\rho-1\rvert/(1+\rho)=0.15 \Rightarrow \rho \in
  [0.7391, 1.3529]$ is the 15% criterion; $\lvert\rho-1\rvert=0.15 \Rightarrow \rho \in
  [0.85, 1.15]$ is only the sweep range of the synthetic experiment.

## Figures

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

## Data

Not included. Four public datasets, listed in `DATA.md` with DOIs and the directory names the
scripts expect.

## Citation

`CITATION.cff` is included, so GitHub shows a "Cite this repository" button.

## Licence

Code MIT (`LICENSE`), figures CC BY 4.0. The manuscript is not covered by either licence; see
`paper/NOTICE.md`.

## Status

Submitted to IEEE TMTT, October 2026. The PDFs in `paper/` are the submitted version, not the
version of record.
