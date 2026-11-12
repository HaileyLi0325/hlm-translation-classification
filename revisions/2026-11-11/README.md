# Computational companion for the 11 November 2026 revision

Manuscript PONE-D-26-25036R1. This archive supports the corrected R2 manuscript.
It contains final code, configurations, derived data and numerical outputs, not
copyrighted book text, rendering intermediates or model caches. It is supplied
as S2 File and is also available at https://github.com/HaileyLi0325/hlm-translation-classification/tree/main/revisions/2026-11-11.

## Corpus and labels

`hawkes` means the Hawkes–Minford edition; `yang` means the Yang–Yang edition.
These codes identify editions, not two individual translators. Data contain
240 complete-chapter rows and 722 fixed-length-segment rows: 120 source chapter
pairs and 361 segments per edition. The segments are chapter matched, not
aligned to identical source spans. `word_count` is alphabetic spaCy tokens;
the 1,500-word segmentation rule uses a separate regex.

## Environment

The executed scientific environment was Python 3.13.0. Install requirements.txt
in an isolated environment. The exact English model is en_core_web_sm 3.8.0.
For limited machines, restrict BLAS/OpenMP threads; the reference code can be
computationally expensive. Reproduction was run with OMP_NUM_THREADS=1,
MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and LOKY_MAX_CPU_COUNT=2 for the four
reference robustness analyses. Scientific results do not depend on local paths.

## Data-first reproduction

Run from this archive root:

```text
python statistics/analysis.py --feature-dir data --original-punctuation-csv data/original_table3_punctuation_values.csv --out statistics/reproduced_results
python statistics/validate_statistics.py --feature-dir data --result-dir statistics/reproduced_results
python robustness/02_shared_inputs/run_original_analysis_replication.py
python "robustness/03_experiments/review 2-comment 1/run_experiment.py"
python "robustness/03_experiments/review 2-comment 2/run_experiment.py"
python "robustness/03_experiments/review 2-comment 3/run_experiment.py"
python "robustness/03_experiments/review 2-comment 4/run_experiment.py"
python ml/run_ml_revision.py --source-root source --feature-root data --output ml/results --stage heldout
python ml/run_ml_revision.py --source-root source --feature-root data --output ml/results --stage sampling
python ml/audit_ml_outputs.py --results ml/results --feature-root data
python figures/generate_figures.py --features data --baseline robustness/02_shared_inputs/original_analysis_replication --ml-results ml/results --output regenerated_figures
```

Sampling evaluation uses the distributed final sampling_features.csv. Every
row is linked to its scheme, source chapter, edition and window positions.
The reference wrapper also computes a diagnostic against old manuscript
headlines; those historical checks are not expected to pass after corrections.
The authoritative old-to-new comparison is comparison/accuracy_changes.csv.

## Reconstructing features from legally obtained editions

Obtain the editions identified by manuscript references 23 and 24. Compare
your PDFs with source/source_file_manifest.json. A different scan or PDF may
require a new extraction audit even when it represents the same book.
Place matching local inputs at source/data/raw_pdfs/Hawkes.pdf and source/data/raw_pdfs/Yangs.pdf.
Run `python source/run_hlm_pipeline.py`. This creates local chapter/window
text and matrices under source/results. Copy the two feature CSVs into data/
only when intentionally replacing them; compare values and hashes first.
No copyrighted source PDFs or extracted chapter text are distributed here.

Alternative-position features can be reconstructed without persisting book
text using the command in ml/README_ML.txt. The corrected heading parser
accepts CHAPTER102, selects first occurrences and removes running headings.
The audit records HM101/102 and YY38/61/102/109 corrections; all other 234
chapter files remained identical. The incremental correction script is an
optional audit route requiring the old matrices; full reconstruction does not.

## Result map and interpretation

- data/: corrected input matrices and feature-change ledger.
- statistics/results/: paired tests, all 36 BH-adjusted comparisons, effects,
  intervals, original punctuation discrepancy, correlations and length controls.
- robustness/: recomputed reference baseline and four R1 robustness analyses.
- ml/results/: new grouped hold-out metrics and intervals, five-position
  grouped CV, exact predictions, SHAP arrays, correlations and split manifests.
- figures/: editable SVGs and numeric source tables for all ten main figures.
- comparison/: actual R1 submitted accuracy values versus corrected values.

Paired effects use HM minus YY. Main BH correction pools 36 paired tests;
chapter-length and segment-mean-length contrasts each have separate BH18
families. Chapter bootstrap intervals are pointwise and assume exchangeable
chapter pairs. Held-out intervals condition on the fixed fitted model.
Repeated-split and grouped-fold SDs are descriptive; they are not independent
replicate standard errors. Sample-level reference CV may share source chapters;
the grouped analyses keep them together. Hard-voting AUC is unavailable.
SHAP explains held-out Random Forest predictions and is neither causal nor
an individual-translator attribution. Correlated predictors can share credit.

OpenAI Codex assisted with revision, code auditing, analysis-code preparation
and figure scripts. Numerical results were computed from the supplied corpus.
Validation evidence includes input hashes, source-boundary checks, feature
agreement, partition checks, exact voting reconstruction and SHAP additivity.
The authors remain responsible for final scientific review and interpretation.
