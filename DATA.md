# Data

No data is redistributed here. All four datasets are public; obtain them from the publisher.

| Dataset | DOI | Used for | Directory the scripts expect |
|---|---|---|---|
| Dual-subject mmWave, 9 layouts (Lei) | `10.17632/684v4r8wfr.1` | Same-bin vs different-bin suppression; breath-hold pairs | `real_data/`, `matrix_out*/` |
| Code Ocean capsule 4604022 | `10.24433/CO.9082638.v4` | Cross-dataset replication | `null_crossdataset/` |
| Zenodo 110-subject record | `10.5281/zenodo.18599983` | Large-sample null rate | `real_pilot/`, `real_rc2/`, `real_matrix/` |
| Zenodo emotion record | `10.5281/zenodo.16900785` | Control dataset | `emo_matrix/`, `emo_matrix_full/` |

Scripts read bare relative paths, for example `"matrix_out_t4"` or
`"synthetic_data_clean/grid.npz"`, with the repository root as working directory. Renaming a
directory breaks them; twenty of these paths are hard-coded.

Most `matrix_out*` directories are caches, not raw data. Regenerate them with `matrix_eval.py`,
`exp_A_group.py` and `lei_scan_all.py`; the bandwidth sweep has to run before anything that reads
`matrix_out_*`.

Each dataset keeps its own licence. Link to the DOI rather than re-hosting the records.
