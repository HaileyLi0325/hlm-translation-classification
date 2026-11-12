from __future__ import annotations

import json
import math
import re
import shutil
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import spacy
from nltk.util import ngrams
from pypdf import PdfReader
from spacy.language import Language
from wordfreq import zipf_frequency


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
CHAPTER_DIR = PROJECT_ROOT / "data" / "chapters"
WINDOW_DIR = PROJECT_ROOT / "data" / "windows_1500"
RESULTS_DIR = PROJECT_ROOT / "results"

# Supply legally obtained PDFs under data/raw_pdfs with these names.
SOURCE_PDFS = (RAW_DIR / "Hawkes.pdf", RAW_DIR / "Yangs.pdf")

WINDOW_WORDS = 1500
CHAPTER_PATTERN = re.compile(r"(?mi)^[ \t]*ch(?:apter|atper)[ \t.]*(\d{1,3})[ \t]*$")
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
PUNCT_PATTERN = re.compile(r"\.\.\.|--|[.,;:?!()\"']")
CLAUSE_MARKERS = {
    "after",
    "although",
    "as",
    "because",
    "before",
    "if",
    "once",
    "since",
    "that",
    "though",
    "unless",
    "until",
    "when",
    "where",
    "whether",
    "which",
    "while",
    "who",
    "whom",
    "whose",
}
COORD_MARKERS = {"and", "but", "or", "nor", "yet", "so"}
SUBORD_LABELS = {"mark", "advcl", "ccomp", "xcomp", "acl", "relcl", "csubj", "csubjpass"}
COORD_LABELS = {"cc", "conj"}
FREQ_BANDS = (
    ("very_common", 6.0, math.inf),
    ("common", 5.0, 6.0),
    ("mid", 4.0, 5.0),
    ("low", 3.0, 4.0),
    ("rare", -math.inf, 3.0),
)

FEATURE_COLUMNS = [
    "wordform_unigram_entropy",
    "lemma_unigram_entropy",
    "function_word_entropy",
    "word_length_entropy",
    "lexical_frequency_band_entropy",
    "punctuation_entropy",
    "pos_unigram_entropy",
    "pos_bigram_entropy",
    "pos_trigram_entropy",
    "clause_marker_entropy",
    "dependency_relation_entropy",
    "coordination_subordination_entropy",
    "character_entropy",
    "character_2gram_entropy",
    "character_3gram_entropy",
    "character_4gram_entropy",
    "word_bigram_entropy",
    "word_trigram_entropy",
]


@dataclass
class ChapterRecord:
    edition: str
    chapter: int
    title: str
    text: str


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = re.sub(r"(?m)^\s*\*\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def chapter_title(chapter_text: str) -> str:
    lines = [line.strip() for line in chapter_text.splitlines() if line.strip()]
    if len(lines) >= 3:
        return f"{lines[1]} / {lines[2]}"
    if len(lines) >= 2:
        return lines[1]
    return ""


def edition_label(pdf_path: Path) -> str:
    if "霍克斯" in pdf_path.stem or pdf_path.stem.lower() == "hawkes":
        return "hawkes"
    if "杨宪益" in pdf_path.stem or pdf_path.stem.lower() in {"yang", "yangs"}:
        return "yang"
    return pdf_path.stem.lower().replace(" ", "_")


def extract_chapters(full_text: str, edition: str) -> list[ChapterRecord]:
    # Accept headings without whitespace (e.g. CHAPTER102). The first
    # occurrence starts a chapter; repeated same-number running headings
    # inside that chapter are removed before feature extraction.
    matches = [m for m in CHAPTER_PATTERN.finditer(full_text) if 1 <= int(m.group(1)) <= 120]
    first_matches = []
    seen = set()
    for match in matches:
        chapter_no = int(match.group(1))
        if chapter_no not in seen:
            seen.add(chapter_no)
            first_matches.append(match)
    chapter_numbers = [int(m.group(1)) for m in first_matches]
    if chapter_numbers != list(range(1, 121)):
        raise ValueError(f"{edition}: expected first headings 1-120, got {chapter_numbers}")
    chapters = []
    for index, match in enumerate(first_matches):
        chapter_no = int(match.group(1))
        end = first_matches[index + 1].start() if index + 1 < len(first_matches) else len(full_text)
        fragment = full_text[match.start():end]
        local_matches = [m for m in CHAPTER_PATTERN.finditer(fragment) if 1 <= int(m.group(1)) <= 120]
        for duplicate in reversed(local_matches[1:]):
            if int(duplicate.group(1)) != chapter_no:
                raise ValueError(f"{edition}: inconsistent running heading in chapter {chapter_no}")
            fragment = fragment[:duplicate.start()] + "\n\n" + fragment[duplicate.end():]
        text = normalize_text(fragment.strip())
        chapters.append(ChapterRecord(edition, chapter_no, chapter_title(text), text))
    return chapters


def entropy(items: Iterable[str]) -> float:
    sequence = list(items)
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def token_spans(text: str) -> list[re.Match[str]]:
    return list(WORD_PATTERN.finditer(text))


def words_from_text(text: str) -> list[str]:
    return [match.group(0).lower() for match in token_spans(text)]


def save_chapters(chapters: Sequence[ChapterRecord]) -> None:
    for record in chapters:
        target = CHAPTER_DIR / record.edition
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"chapter_{record.chapter:03d}.txt"
        path.write_text(record.text, encoding="utf-8")


def aligned_windows(text: str, window_words: int, keep: int) -> list[str]:
    spans = token_spans(text)
    windows: list[str] = []
    for index in range(keep):
        start_idx = index * window_words
        end_idx = start_idx + window_words - 1
        if end_idx >= len(spans):
            break
        char_start = spans[start_idx].start()
        char_end = spans[end_idx].end()
        windows.append(text[char_start:char_end].strip())
    return windows


def save_windows(edition: str, chapter: int, windows: Sequence[str]) -> None:
    target = WINDOW_DIR / edition / f"chapter_{chapter:03d}"
    target.mkdir(parents=True, exist_ok=True)
    for index, window_text in enumerate(windows, start=1):
        path = target / f"window_{index:02d}.txt"
        path.write_text(window_text, encoding="utf-8")


def copy_raw_pdfs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for pdf in SOURCE_PDFS:
        if not pdf.exists():
            raise FileNotFoundError(f"Place a legally obtained source PDF at {pdf}")


def build_nlp() -> Language:
    return spacy.load("en_core_web_sm", disable=["ner"])


def char_sequence(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text.lower())
    return [char for char in cleaned if not char.isspace()]


def ngram_strings(tokens: Sequence[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    return [" ".join(gram) for gram in ngrams(tokens, n)]


def classify_frequency_band(word: str) -> str:
    score = zipf_frequency(word, "en")
    for label, lower, upper in FREQ_BANDS:
        if lower <= score < upper:
            return label
    return "rare"


def punctuation_tokens(text: str) -> list[str]:
    raw = PUNCT_PATTERN.findall(text)
    normalized = []
    for token in raw:
        if token == "...":
            normalized.append("ellipsis")
        elif token == "--":
            normalized.append("dash")
        else:
            normalized.append(token)
    return normalized


def clause_marker_tokens(doc) -> list[str]:
    tokens: list[str] = []
    for token in doc:
        lower = token.text.lower()
        if lower in CLAUSE_MARKERS and token.pos_ in {"SCONJ", "PRON", "ADV"}:
            tokens.append(lower)
    return tokens


def coord_subord_tokens(doc) -> list[str]:
    tokens: list[str] = []
    for token in doc:
        lower = token.text.lower()
        if token.dep_ in COORD_LABELS or lower in COORD_MARKERS:
            tokens.append("coordination")
        elif token.dep_ in SUBORD_LABELS or lower in CLAUSE_MARKERS:
            tokens.append("subordination")
    return tokens


def dependency_distance_tokens(doc) -> list[str]:
    tokens: list[str] = []
    for token in doc:
        if token.i == token.head.i:
            continue
        distance = abs(token.i - token.head.i)
        if distance <= 2:
            bucket = "1_2"
        elif distance <= 5:
            bucket = "3_5"
        elif distance <= 10:
            bucket = "6_10"
        else:
            bucket = "11_plus"
        tokens.append(bucket)
    return tokens


def compute_feature_row(doc, edition: str, sample_id: str, sample_type: str, chapter: int, title: str, window_id: int | None) -> dict[str, float | int | str]:
    words = [token.text.lower() for token in doc if token.is_alpha]
    lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
    function_words = [token.text.lower() for token in doc if token.is_alpha and token.is_stop]
    word_lengths = [str(len(word)) for word in words]
    freq_bands = [classify_frequency_band(word) for word in words]
    puncts = punctuation_tokens(doc.text)
    pos_tokens = [token.pos_ for token in doc if token.is_alpha]
    clause_tokens = clause_marker_tokens(doc)
    dep_labels = [token.dep_ for token in doc if token.dep_]
    coord_subord = coord_subord_tokens(doc)
    chars = char_sequence(doc.text)

    row: dict[str, float | int | str] = {
        "edition": edition,
        "sample_id": sample_id,
        "sample_type": sample_type,
        "chapter": chapter,
        "title": title,
        "window_id": window_id or 0,
        "word_count": len(words),
        "wordform_unigram_entropy": entropy(words),
        "lemma_unigram_entropy": entropy(lemmas),
        "function_word_entropy": entropy(function_words),
        "word_length_entropy": entropy(word_lengths),
        "lexical_frequency_band_entropy": entropy(freq_bands),
        "punctuation_entropy": entropy(puncts),
        "pos_unigram_entropy": entropy(pos_tokens),
        "pos_bigram_entropy": entropy(ngram_strings(pos_tokens, 2)),
        "pos_trigram_entropy": entropy(ngram_strings(pos_tokens, 3)),
        "clause_marker_entropy": entropy(clause_tokens),
        "dependency_relation_entropy": entropy(dep_labels),
        "coordination_subordination_entropy": entropy(coord_subord),
        "character_entropy": entropy(chars),
        "character_2gram_entropy": entropy(ngram_strings(chars, 2)),
        "character_3gram_entropy": entropy(ngram_strings(chars, 3)),
        "character_4gram_entropy": entropy(ngram_strings(chars, 4)),
        "word_bigram_entropy": entropy(ngram_strings(words, 2)),
        "word_trigram_entropy": entropy(ngram_strings(words, 3)),
    }
    return row


def build_feature_frame(samples: Sequence[dict[str, str | int]], nlp: Language) -> pd.DataFrame:
    docs = list(nlp.pipe((str(sample["text"]) for sample in samples), batch_size=8))
    rows = []
    for sample, doc in zip(samples, docs):
        rows.append(
            compute_feature_row(
                doc=doc,
                edition=str(sample["edition"]),
                sample_id=str(sample["sample_id"]),
                sample_type=str(sample["sample_type"]),
                chapter=int(sample["chapter"]),
                title=str(sample["title"]),
                window_id=int(sample["window_id"]) if sample["window_id"] is not None else None,
            )
        )
    return pd.DataFrame(rows)


def chapter_samples(chapter_map: dict[str, list[ChapterRecord]]) -> list[dict[str, str | int]]:
    samples: list[dict[str, str | int]] = []
    for edition, chapters in chapter_map.items():
        for record in chapters:
            samples.append(
                {
                    "edition": edition,
                    "sample_id": f"{edition}_chapter_{record.chapter:03d}",
                    "sample_type": "chapter",
                    "chapter": record.chapter,
                    "title": record.title,
                    "window_id": None,
                    "text": record.text,
                }
            )
    return samples


def window_samples(chapter_map: dict[str, list[ChapterRecord]]) -> tuple[list[dict[str, str | int]], list[dict[str, int]]]:
    samples: list[dict[str, str | int]] = []
    counts: list[dict[str, int]] = []
    hawkes = {record.chapter: record for record in chapter_map["hawkes"]}
    yang = {record.chapter: record for record in chapter_map["yang"]}
    for chapter_no in range(1, 121):
        hawkes_words = len(words_from_text(hawkes[chapter_no].text))
        yang_words = len(words_from_text(yang[chapter_no].text))
        keep = min(hawkes_words // WINDOW_WORDS, yang_words // WINDOW_WORDS)
        counts.append(
            {
                "chapter": chapter_no,
                "hawkes_windows_available": hawkes_words // WINDOW_WORDS,
                "yang_windows_available": yang_words // WINDOW_WORDS,
                "windows_kept_per_edition": keep,
            }
        )
        if keep == 0:
            continue
        hawkes_windows = aligned_windows(hawkes[chapter_no].text, WINDOW_WORDS, keep)
        yang_windows = aligned_windows(yang[chapter_no].text, WINDOW_WORDS, keep)
        save_windows("hawkes", chapter_no, hawkes_windows)
        save_windows("yang", chapter_no, yang_windows)
        for edition, record, edition_windows in (
            ("hawkes", hawkes[chapter_no], hawkes_windows),
            ("yang", yang[chapter_no], yang_windows),
        ):
            for index, window_text in enumerate(edition_windows, start=1):
                samples.append(
                    {
                        "edition": edition,
                        "sample_id": f"{edition}_chapter_{chapter_no:03d}_window_{index:02d}",
                        "sample_type": "window_1500",
                        "chapter": chapter_no,
                        "title": record.title,
                        "window_id": index,
                        "text": window_text,
                    }
                )
    return samples, counts


def write_results(chapter_df: pd.DataFrame, window_df: pd.DataFrame, window_counts: list[dict[str, int]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    chapter_csv = RESULTS_DIR / "chapter_level_features.csv"
    chapter_xlsx = RESULTS_DIR / "chapter_level_features.xlsx"
    window_csv = RESULTS_DIR / "window_1500_features.csv"
    window_xlsx = RESULTS_DIR / "window_1500_features.xlsx"
    counts_csv = RESULTS_DIR / "window_alignment_counts.csv"
    summary_json = RESULTS_DIR / "summary.json"

    chapter_df.to_csv(chapter_csv, index=False, encoding="utf-8-sig")
    chapter_df.to_excel(chapter_xlsx, index=False)
    window_df.to_csv(window_csv, index=False, encoding="utf-8-sig")
    window_df.to_excel(window_xlsx, index=False)
    pd.DataFrame(window_counts).to_csv(counts_csv, index=False, encoding="utf-8-sig")

    summary = {
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "chapter_samples": int(len(chapter_df)),
        "chapter_samples_per_edition": chapter_df["edition"].value_counts().to_dict(),
        "window_samples": int(len(window_df)),
        "window_samples_per_edition": window_df["edition"].value_counts().to_dict(),
        "window_words": WINDOW_WORDS,
        "split_strategy": "chapter-level plus chapter-aligned non-overlapping 1500-word windows, keeping min(window_count) across editions for each chapter",
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme() -> None:
    readme = PROJECT_ROOT / "README.md"
    readme.write_text(
        """# hlm

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
""",
        encoding="utf-8",
    )


def write_requirements() -> None:
    requirements = PROJECT_ROOT / "requirements.txt"
    requirements.write_text(
        "\n".join(
            [
                "pandas",
                "pypdf",
                "spacy",
                "wordfreq",
                "openpyxl",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    copy_raw_pdfs()
    write_readme()
    write_requirements()
    nlp = build_nlp()

    chapter_map: dict[str, list[ChapterRecord]] = {}
    for pdf_path in SOURCE_PDFS:
        edition = edition_label(pdf_path)
        chapters = extract_chapters(read_pdf_text(pdf_path), edition)
        chapter_map[edition] = chapters
        save_chapters(chapters)

    chapter_df = build_feature_frame(chapter_samples(chapter_map), nlp)
    window_data, window_counts = window_samples(chapter_map)
    window_df = build_feature_frame(window_data, nlp)
    write_results(chapter_df, window_df, window_counts)

    print("hlm pipeline completed")
    print(f"Chapter samples: {len(chapter_df)}")
    print(f"Window samples: {len(window_df)}")
    print(f"Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
