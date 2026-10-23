#!/usr/bin/env bash
# Regenerate every figure of the paper and the supplement.
# Run from the repository root:  bash reproduce.sh
# Requires the data directories described in DATA.md to be present.

set -euo pipefail

PY=${PYTHON:-python}

echo "== Fig. 1, S1, S2, S5  (geometry, breath-hold, drift window, alignment) =="
$PY code/paper_figures_v3.py

echo "== Fig. 2  (failure map over rho and theta_max) =="
$PY code/fig_failure_map_v6.py

echo "== Fig. 3, S4  (range resolution, three-bandwidth example) =="
$PY code/paper_figures_v5.py

echo "== Fig. 4  (phase-step shape test) =="
$PY code/exp_C2_shape.py

echo "== Fig. S3  (capsule phase portrait) =="
$PY code/null_phase_figure.py

echo "== Graphical abstract =="
$PY code/make_graphical_abstract.py

echo
echo "Done. Figures are written to figs/ ; compare against the committed copies."
