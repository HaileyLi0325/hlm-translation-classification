# hlm

This folder contains a self-contained Hongloumeng translation-classification pipeline.

## Included

- `run_hlm_pipeline.py`: extracts chapters, builds aligned datasets, and computes 18 entropy-based features.
- `data/raw_pdfs/`: copies of the two source PDFs.
- `data/chapters/`: per-chapter plain-text exports.
- `data/windows_1500/`: chapter-aligned non-overlapping 1500-word windows.
- `results/`: CSV/XLSX feature tables and alignment metadata.

## Dataset split used

The pipeline generates two datasets:

1. `chapter-level`
2. `window_1500`

For the fixed-length dataset, each chapter is split independently into non-overlapping 1500-word windows. For a given chapter, both editions keep the same number of windows, equal to the minimum available window count across the two editions.
