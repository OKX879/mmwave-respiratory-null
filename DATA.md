# Data

No data is redistributed here. All four datasets are public and must be obtained from their
publishers. Each row gives the DOI, what it is used for in the paper, and the **directory name the
scripts expect at the repository root**.

| Dataset | DOI | Used for | Expected directory |
|---|---|---|---|
| Dual-subject mmWave, 9 layouts (Lei) | `10.17632/684v4r8wfr.1` | Same-bin vs different-bin suppression; the four-group analysis in §VIII; breath-hold pairs | `real_data/`, `matrix_out*/` |
| Code Ocean capsule 4604022 | `10.24433/CO.9082638.v4` | Cross-dataset replication | `null_crossdataset/` |
| Zenodo 110-subject record | `10.5281/zenodo.18599983` | Large-sample null rate | `real_pilot/`, `real_rc2/`, `real_matrix/` |
| Zenodo emotion-classification record | `10.5281/zenodo.16900785` | Control dataset | `emo_matrix/`, `emo_matrix_full/` |

## Directory convention

The scripts read **bare relative paths**, for example:

```python
DATA = "matrix_out_t4"          # a derived cache of the Lei dataset
GRID = "synthetic_data_clean/grid.npz"
```

So the working directory must be the repository root, and the data must sit in directories with
exactly these names. Renaming them will break the scripts — twenty of the directories are
referenced by hard-coded relative paths, which is why they are not reorganised.

## Derived caches

Most `matrix_out*` directories are not raw data but caches produced by earlier passes
(`matrix_eval.py`, `exp_A_group.py`, `lei_scan_all.py` and friends). If you start from the raw
dataset you can regenerate them, but the regeneration order matters: the bandwidth sweep has to run
before anything that reads `matrix_out_*`.

## Licensing of the data

Each dataset keeps its own licence. Nothing here overrides it. In particular, do not re-host the
raw records; link to the DOI instead.
