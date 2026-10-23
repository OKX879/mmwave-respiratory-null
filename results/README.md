# Results snapshot

Derived tables produced by the scripts in `code/`, committed so the reported numbers can be
checked without re-running the pipeline. **No raw data is included.**

## What is here

| Subdirectory | Origin | Content |
|---|---|---|
| `matrix_out*/` | `matrix_eval.py`, `exp_A_group.py`, bandwidth sweep | Per-configuration criteria tables, closed-loop folds, Table A/B/C |
| `exp_out/` | `exp_*.py` | Experiment tables: dips, straddle, baselines, synthetic shapes |
| `null_crossdataset/` | `null_crossdataset.py`, `lei_*.py`, `emo_*.py` | Cross-dataset null statistics (capsule, dual-subject, Zenodo, emotion) |
| `emo_matrix*/`, `real_matrix/`, `real_rc2/`, `real_pilot/` | corresponding `*_eval.py` | Per-record result and calibration tables |
| `data/` | `bridge_null_criteria.py`, `parralejo_*.py` | Bridged null criteria and record-level QC summaries |
| `null_probe_out/` | `null_probe.py`, `cardiac_null_probe.py` | Null summaries |

## What is deliberately excluded

- **Raw datasets** — obtain from the DOIs in `../DATA.md`.
- **Large per-record matrices** — every `*_raw*.csv` (145 MB total) and
  `data/parralejo_coverage.csv` (2.9 MB). These are regenerable intermediates, not results.

## Note on paths

These files are a snapshot of the directory layout the scripts use. The scripts themselves
resolve roots through `code/paths.py`; set `MMWAVE_OUT_ROOT` if you want them to write
somewhere other than `<repo>/null_crossdataset`.
