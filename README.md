# Hongloumeng translation-edition attribution

Reproducibility materials for **Multi-feature entropy stylometry and ensemble machine learning for translation-edition attribution of Hongloumeng** (PLOS ONE manuscript PONE-D-26-25036R1).

## Current revision: 11 November 2026

Use the versioned [computational companion](revisions/2026-11-11/README.md) for the corrected manuscript. It contains the corrected feature matrices, complete analysis code and configurations, final numerical results, model predictions, SHAP values, and editable figure sources. These materials are also supplied with the manuscript as S2 File.

The revised analysis includes 120 source-chapter pairs, 240 chapter-level feature rows, and 722 fixed-length segments (361 per edition). `hawkes` identifies the **Hawkes–Minford edition** and `yang` the **Yang–Yang edition**. Segments are matched by source chapter; their translated content spans are not manually aligned.

The 12 November publication-format update uses letter labels for the tables and figures inside S1 Appendix and 8–12 pt Arial labels in all main figures. Numerical analysis outputs are unchanged.

### Start here

```text
cd revisions/2026-11-11
python -m pip install -r requirements.txt
python statistics/analysis.py --feature-dir data --original-punctuation-csv data/original_table3_punctuation_values.csv --out statistics/reproduced_results
```

The [revision README](revisions/2026-11-11/README.md) contains the complete commands for the reference robustness analyses, grouped validation, sampling sensitivity, SHAP, figures, and source-feature reconstruction. The executed reference environment was Python 3.13.0; exact package and language-model versions are recorded in the companion.

### Final data and results

| Material | Location |
|---|---|
| Corrected chapter and segment features | [data](revisions/2026-11-11/data) |
| Paired tests, multiplicity, effect intervals and length adjustment | [statistical results](revisions/2026-11-11/statistics/results) |
| Recomputed baseline and four preceding robustness analyses | [robustness](revisions/2026-11-11/robustness) |
| Grouped hold-out, position sensitivity, predictions and SHAP | [machine-learning results](revisions/2026-11-11/ml/results) |
| Plotting code, numeric source tables and editable SVGs | [figures](revisions/2026-11-11/figures) |
| Previous submitted values versus corrected values | [accuracy comparison](revisions/2026-11-11/comparison/accuracy_changes.csv) |
| File checksums | [SHA-256 manifest](revisions/2026-11-11/MANIFEST_SHA256.json) |

The current companion distributes derived data and instructions for using legally obtained source editions; it does not include the books' full texts, model caches or temporary rendering files. It records the chapter-boundary corrections, sampling coverage, software versions and validation checks.

## Historical root files

Root-level scripts and previously uploaded assets are retained as historical material. The corrected manuscript uses the versioned companion linked above. In particular, the corrected fixed-length dataset contains 361 segments per edition, replacing the earlier 359-segment dataset. Do not combine historical features or outputs with the revised tables and figures.

## Interpretation

The task is attribution between two translation editions. Paired feature comparisons account for source-chapter matching; grouped predictive analyses keep chapter groups together. Held-out bootstrap intervals condition on the fitted models. Repeated-split and fold SDs are descriptive. SHAP explains Random Forest predictions and does not establish causal effects or isolate an individual translator's style. See the companion for full methodological details.
