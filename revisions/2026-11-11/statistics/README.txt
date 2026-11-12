Statistical code and final numerical evidence

This folder uses relative paths and contains no machine-specific configuration.
The accepted edition storage identifiers are hawkes (Hawkes-Minford edition)
and yang (Yang-Yang edition). They are edition labels, not individual authors.

Requirements for statistical analysis only
Python 3.13.0; numpy 2.1.3; pandas 2.2.3; scipy 1.15.1.
For the optional independent numerical validation, install statsmodels 0.14.4.
The original extraction stack is required only for feature reconstruction:
spaCy 3.8.14, en_core_web_sm 3.8.0, NLTK 3.9.4, wordfreq 3.1.1,
pypdf 6.10.2, plus the source pipeline's other recorded dependencies.

Reproduce statistical outputs from the released derived data
Run from this folder:

python analysis.py --feature-dir ../data --original-punctuation-csv ../data/original_table3_punctuation_values.csv --out reproduced_statistics

Or supply arbitrary input paths explicitly:

python analysis.py --chapter-csv PATH/chapter_level_features.csv --segment-csv PATH/window_1500_features.csv --out reproduced_statistics

The exact input SHA-256 digests and statistical configuration are saved in
analysis_metadata.json. The corrected feature files contain 240 chapter rows
and 722 segment rows, covering 120 matched source chapters. Effect directions
are Hawkes-Minford minus Yang-Yang. C2G, C3G, C4G, WB and WT follow manuscript
Table 2; these are labels for the character 2/3/4-gram and word bi/trigram
features, respectively.

The default run does not require or redistribute the copyrighted source text.
To verify punctuation entropy directly after reconstructing the source texts
from legally obtained PDFs, add:

--text-root PATH_TO_CORRECTED_HLM/data

This writes punctuation_raw_text_audit.csv. Omitting --text-root leaves that
source-text-only check unperformed; it does not alter the feature-based tests.

Main outputs
paired_tests_36.csv: paired t, raw p, BH36 q, Holm36, paired mean differences
  and t confidence intervals, Cohen dz and 10,000-pair bootstrap intervals.
segment_chapter_means.csv: 120 observations per edition for segment inference.
descriptive_statistics.csv: full chapter, individual segment, and segment
  chapter-mean summaries, kept explicitly separate.
table3_punctuation_verification.json: distinguishes prior rounded summaries,
  prior unrounded observations, and corrected paired inference.
token_count_entropy_correlations.csv: 36 within-edition chapter correlations.
length_adjusted_paired_regression.csv: 18 chapter fixed-effect-equivalent
  equal-token-count contrasts with HC3 standard errors and BH18 adjustment.
length_adjusted_segment_means_regression.csv: corresponding secondary
  adjustment of segment chapter means, with a separate BH18 family.
exploratory_paired_tests_by_historical_block.csv: edition comparisons within
  chapters 1-80 and 81-120, with BH72 across the exploratory family.
analysis_metadata.json: versions, input hashes, seeds, assumptions and counts.
statistical_numerical_validation.json: independent numerical verification.

The 95% intervals are pointwise, not simultaneous confidence intervals.
Chapter bootstrap intervals require treating source-chapter pairs as exchangeable
sampling units. The analyses are conditional on this work and these editions.
Length adjustment uses a common slope and has limited overlap at equal token
counts; it must not be read as a causal isolation of individual translator style.

Feature reconstruction and original submission discrepancy
The corrected source pipeline is the primary route for a complete rebuild from
legally obtained copies of the editions. The included incremental helper is an
audit convenience for readers who also have the preceding submission's feature
matrices and extracted windows:

python rebuild_corrected_features.py --source-root PATH_TO_CORRECTED_HLM --previous-feature-dir PATH_TO_PREVIOUS_FEATURES --previous-window-dir PATH_TO_PREVIOUS_HLM/data/windows_1500 --out corrected_features

Use a separate corrected source directory: the helper writes reconstructed
windows to that directory. It never edits the supplied previous matrices or
previous-window directory. It recomputes six corrected chapter records:
Hawkes-Minford 101 and 102; Yang-Yang 38, 61, 102 and 109. It regenerates the
matched windows for chapters 101 and 102 in both editions. All other retained
prefix windows are unaffected. A recomputed Hawkes-Minford chapter 100 checks
feature-extraction consistency. The change ledger is feature_correction_audit.csv.
The helper is optional: the archived final matrices suffice for the statistical
analysis, and a complete source-pipeline rebuild does not require old matrices.

To run the optional independent comparison with statsmodels from the package
root, use:

python statistics/validate_statistics.py --feature-dir data --result-dir statistics/results

The defaults resolve relative to the script location, so running
python validate_statistics.py from this statistics folder also uses the packaged
data and results. To check a new analysis output, pass its directory with
--result-dir. The helper verifies BH/Holm adjustments and all 36 HC3 regressions
and writes statistical_numerical_validation.json to the selected result directory.
It does not modify the input CSVs. The recorded raw-text punctuation check is
reported only when present in analysis_metadata.json; this helper does not need
or re-read the source books.

Archive layout note: this release stores corrected_features as ../data and final_results as results. Use explicit CLI paths shown in the root README.
