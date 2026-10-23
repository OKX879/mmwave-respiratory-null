# Data

No data is redistributed here. All four datasets are public; obtain them from the publisher.

| Dataset | DOI | Used for | Directory the scripts expect |
|---|---|---|---|
| Dual-subject mmWave, 9 layouts (Lei) | `10.17632/684v4r8wfr.1` | Same-bin vs different-bin suppression; breath-hold pairs | `real_data/`, `matrix_out*/` |
| Code Ocean capsule 4604022 | `10.24433/CO.9082638.v4` | Cross-dataset replication | `null_crossdataset/` |
| Zenodo 110-subject record | `10.5281/zenodo.18599983` | Large-sample null rate | `real_pilot/`, `real_rc2/`, `real_matrix/` |
| Zenodo emotion record | `10.5281/zenodo.16900785` | Control dataset | `emo_matrix/`, `emo_matrix_full/` |

Scripts resolve external roots through `code/paths.py` (`MMWAVE_DATA_ROOT`,
`MMWAVE_OUT_ROOT`, `MMWAVE_PRODUCT_ROOT`), defaulting to `~/Desktop` and the repository root.
No absolute machine-specific path is hard-coded, so a fresh clone runs without edits.

Most `matrix_out*` directories are caches, not raw data. Regenerate them with `matrix_eval.py`,
`exp_A_group.py` and `lei_scan_all.py`; the bandwidth sweep has to run before anything that reads
`matrix_out_*`. A snapshot of the derived tables (without the `*_raw*.csv` intermediates) is
committed under `results/`.

Each dataset keeps its own licence. Link to the DOI rather than re-hosting the records.
