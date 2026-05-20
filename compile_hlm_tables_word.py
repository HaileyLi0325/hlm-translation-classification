from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from scipy.stats import mannwhitneyu

from run_hlm_classification import ALL_FEATURES, FEATURE_GROUPS


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
CLASSIFICATION_DIR = RESULTS_DIR / "classification"
OUTPUT_DOCX = CLASSIFICATION_DIR / "hlm_compiled_tables.docx"
FALLBACK_OUTPUT_DOCX = CLASSIFICATION_DIR / "hlm_compiled_tables_updated.docx"

CAPTION_FILL = "A9ACD5"
HEADER_FILL = "EFEFF7"
GROUP_LABELS = {
    "lexical": "Lexical",
    "syntactic": "Syntactic",
    "ngram": "N-gram",
    "all_features": "Overall",
}
DATASET_LABELS = {
    "chapter_level": "Chapter-level",
    "window_1500": "1,500-word segments",
}
EDITION_LABELS = {
    "hawkes": "Hawkes",
    "yang": "Yang",
}

FEATURE_META = {
    "wordform_unigram_entropy": ("lexical", "Wordform unigram entropy", "WFU"),
    "lemma_unigram_entropy": ("lexical", "Lemma unigram entropy", "LU"),
    "function_word_entropy": ("lexical", "Function-word entropy", "FW"),
    "word_length_entropy": ("lexical", "Word-length entropy", "WL"),
    "lexical_frequency_band_entropy": ("lexical", "Frequency-band entropy", "FB"),
    "punctuation_entropy": ("lexical", "Punctuation entropy", "PUNC"),
    "pos_unigram_entropy": ("syntactic", "POS unigram entropy", "POSU"),
    "pos_bigram_entropy": ("syntactic", "POS bigram entropy", "POSB"),
    "pos_trigram_entropy": ("syntactic", "POS trigram entropy", "POST"),
    "clause_marker_entropy": ("syntactic", "Clause-marker entropy", "CM"),
    "dependency_relation_entropy": ("syntactic", "Dependency-relation entropy", "DR"),
    "coordination_subordination_entropy": ("syntactic", "Coord/Subord entropy", "CS"),
    "character_entropy": ("ngram", "Character entropy", "CHAR"),
    "character_2gram_entropy": ("ngram", "Character 2-gram entropy", "C2G"),
    "character_3gram_entropy": ("ngram", "Character 3-gram entropy", "C3G"),
    "character_4gram_entropy": ("ngram", "Character 4-gram entropy", "C4G"),
    "word_bigram_entropy": ("ngram", "Word bigram entropy", "WB"),
    "word_trigram_entropy": ("ngram", "Word trigram entropy", "WT"),
}


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def fmt_int(value: float) -> str:
    return f"{int(round(float(value))):,}"


def fmt_accuracy(mean_value: float, std_value: float) -> str:
    return f"{float(mean_value):.2f} ({float(std_value):.2f})"


def fmt_percent(value: float) -> str:
    return f"{float(value):.2f}%"


def sig_marker(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_borders(cell, color: str = "000000", size: str = "8") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def write_cell(cell, text: object, *, bold: bool = False, font_size: float = 7.0, align: str = "left") -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    if align == "center":
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "right":
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    else:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = paragraph.add_run("" if text is None else str(text))
    run.bold = bold
    run.font.size = Pt(font_size)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(
    doc: Document,
    caption: str,
    columns: list[str],
    rows: Iterable[Iterable[object]],
    *,
    note: str | None = None,
    font_size: float = 7.0,
) -> None:
    rows = list(rows)
    table = doc.add_table(rows=len(rows) + 2, cols=len(columns))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    caption_cell = table.rows[0].cells[0]
    for cell in table.rows[0].cells[1:]:
        caption_cell = caption_cell.merge(cell)
    write_cell(caption_cell, caption, bold=True, font_size=8.5)
    set_cell_shading(caption_cell, CAPTION_FILL)

    for cell, column in zip(table.rows[1].cells, columns):
        write_cell(cell, column, bold=True, font_size=font_size, align="center")
        set_cell_shading(cell, HEADER_FILL)

    for row_idx, row_values in enumerate(rows, start=2):
        for cell, value in zip(table.rows[row_idx].cells, row_values):
            write_cell(cell, value, font_size=font_size)

    for row in table.rows:
        for cell in row.cells:
            set_cell_borders(cell)

    if note:
        paragraph = doc.add_paragraph(note)
        paragraph.paragraph_format.space_after = Pt(2)
        for run in paragraph.runs:
            run.font.size = Pt(7)
    doc.add_paragraph()


def set_document_layout(doc: Document) -> None:
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.top_margin = Inches(0.45)
    section.bottom_margin = Inches(0.45)
    section.left_margin = Inches(0.45)
    section.right_margin = Inches(0.45)

    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(9)
    styles["Title"].font.name = "Arial"
    styles["Title"].font.size = Pt(15)
    styles["Heading 1"].font.name = "Arial"
    styles["Heading 1"].font.size = Pt(11)


def source_map() -> dict[str, str]:
    sources = {"hawkes": "", "yang": ""}
    raw_dir = ROOT / "data" / "raw_pdfs"
    for path in raw_dir.glob("*.pdf"):
        name = path.name
        if "霍克斯" in name:
            sources["hawkes"] = name
        elif "杨宪益" in name:
            sources["yang"] = name
    return sources


def build_corpus_rows(chapter_df: pd.DataFrame, window_df: pd.DataFrame) -> list[list[object]]:
    sources = source_map()
    rows = []
    for edition in ("hawkes", "yang"):
        chapter_part = chapter_df[chapter_df["edition"] == edition]
        window_part = window_df[window_df["edition"] == edition]
        rows.append(
            [
                EDITION_LABELS[edition],
                "English",
                sources.get(edition, ""),
                len(chapter_part),
                fmt_int(chapter_part["word_count"].sum()),
                fmt_float(chapter_part["word_count"].mean(), 0),
                len(window_part),
                fmt_int(window_part["word_count"].sum()),
                fmt_float(window_part["word_count"].mean(), 0),
            ]
        )
    return rows


def build_feature_index_rows() -> list[list[object]]:
    rows = []
    for feature in ALL_FEATURES:
        group, label, code = FEATURE_META[feature]
        rows.append([GROUP_LABELS[group], label, code])
    return rows


def build_descriptive_rows(
    data_by_dataset: dict[str, pd.DataFrame],
    features: list[str],
) -> list[list[object]]:
    rows = []
    for dataset_name, df in data_by_dataset.items():
        for feature in features:
            _, label, code = FEATURE_META[feature]
            hawkes = df.loc[df["edition"] == "hawkes", feature]
            yang = df.loc[df["edition"] == "yang", feature]
            p_value = mannwhitneyu(hawkes, yang, alternative="two-sided").pvalue
            rows.append(
                [
                    DATASET_LABELS[dataset_name],
                    code,
                    fmt_float(hawkes.mean()),
                    fmt_float(hawkes.std(ddof=1)),
                    fmt_float(hawkes.median()),
                    fmt_float(yang.mean()),
                    fmt_float(yang.std(ddof=1)),
                    fmt_float(yang.median()),
                    sig_marker(p_value),
                    label,
                ]
            )
    return rows


def build_individual_classification_rows(dataset_name: str) -> tuple[list[str], list[list[object]]]:
    mean_df = pd.read_csv(CLASSIFICATION_DIR / dataset_name / "individual_feature_accuracy_mean.csv").set_index("feature")
    models = [column for column in mean_df.columns]
    rows = []
    for feature in ALL_FEATURES:
        group, _, code = FEATURE_META[feature]
        row = [GROUP_LABELS[group], code]
        for model in models:
            row.append(fmt_percent(mean_df.loc[feature, model]))
        rows.append(row)
    return ["Level", "Code", *models], rows


def build_group_classification_rows() -> tuple[list[str], list[list[object]]]:
    rows = []
    models: list[str] | None = None
    for dataset_name in ("chapter_level", "window_1500"):
        mean_df = pd.read_csv(CLASSIFICATION_DIR / dataset_name / "feature_group_accuracy_mean.csv").set_index("feature_group")
        if models is None:
            models = [column for column in mean_df.columns]
        for feature_set in ("lexical", "syntactic", "ngram", "all_features"):
            row = [DATASET_LABELS[dataset_name], GROUP_LABELS[feature_set], len(FEATURE_GROUPS[feature_set])]
            for model in models:
                row.append(fmt_percent(mean_df.loc[feature_set, model]))
            rows.append(row)
    return ["Dataset", "Feature set", "N", *(models or [])], rows


def add_intro(doc: Document) -> None:
    with (RESULTS_DIR / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    paragraph = doc.add_paragraph()
    paragraph.add_run("Data source: ").bold = True
    paragraph.add_run(
        "existing HLM feature tables and classification outputs under results/. "
        f"The pipeline contains {summary['feature_count']} entropy-based features, "
        f"{summary['chapter_samples']} chapter-level samples, and "
        f"{summary['window_samples']} aligned 1,500-word segment samples."
    )


def main() -> None:
    chapter_df = pd.read_csv(RESULTS_DIR / "chapter_level_features.csv")
    window_df = pd.read_csv(RESULTS_DIR / "window_1500_features.csv")
    data_by_dataset = {
        "chapter_level": chapter_df,
        "window_1500": window_df,
    }

    doc = Document()
    set_document_layout(doc)
    title = doc.add_paragraph("HLM Translation Classification Tables")
    title.style = doc.styles["Title"]
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_intro(doc)

    add_table(
        doc,
        "Table 1 Summary of corpus.",
        [
            "Sub-corpus",
            "Language",
            "Source",
            "Chapter Text Count",
            "Chapter Overall Size",
            "Mean Chapter Length",
            "Segment Count",
            "Segment Overall Size",
            "Mean Segment Length",
        ],
        build_corpus_rows(chapter_df, window_df),
        note="Word counts and text counts are computed from the existing chapter-level and aligned 1,500-word feature tables.",
        font_size=7.2,
    )

    add_table(
        doc,
        "Table 2 Summary of proposed simplified indices.",
        ["Level", "Proposed simplified index", "Code"],
        build_feature_index_rows(),
        font_size=7.4,
    )

    descriptive_columns = [
        "Dataset",
        "Code",
        "Hawkes Mean",
        "Hawkes SD",
        "Hawkes Median",
        "Yang Mean",
        "Yang SD",
        "Yang Median",
        "Sig",
        "Index",
    ]
    for table_no, group_name in enumerate(("lexical", "syntactic", "ngram"), start=3):
        add_table(
            doc,
            f"Table {table_no} Descriptive results of the {GROUP_LABELS[group_name].lower()} indices by translation.",
            descriptive_columns,
            build_descriptive_rows(data_by_dataset, FEATURE_GROUPS[group_name]),
            note="* p < .05, ** p < .01, *** p < .001; two-sided Mann-Whitney U test between Hawkes and Yang within each dataset.",
            font_size=6.8,
        )

    columns, rows = build_individual_classification_rows("chapter_level")
    add_table(
        doc,
        "Table 6 Classification accuracy by individual index and algorithm: chapter-level samples.",
        columns,
        rows,
        note="Cells show mean accuracy (%) from 10-fold stratified cross-validation.",
        font_size=6.1,
    )

    columns, rows = build_individual_classification_rows("window_1500")
    add_table(
        doc,
        "Table 7 Classification accuracy by individual index and algorithm: 1,500-word aligned segments.",
        columns,
        rows,
        note="Cells show mean accuracy (%) from 10-fold stratified cross-validation.",
        font_size=6.1,
    )

    columns, rows = build_group_classification_rows()
    add_table(
        doc,
        "Table 8 Classification accuracy of combined feature sets and all features.",
        columns,
        rows,
        note="Cells show mean accuracy (%) from 10-fold stratified cross-validation. Voting Ensemble is hard voting over the eight base classifiers.",
        font_size=6.0,
    )

    OUTPUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    try:
        doc.save(OUTPUT_DOCX)
        try:
            doc.save(FALLBACK_OUTPUT_DOCX)
        except PermissionError:
            pass
        print(OUTPUT_DOCX)
    except PermissionError:
        doc.save(FALLBACK_OUTPUT_DOCX)
        print(FALLBACK_OUTPUT_DOCX)


if __name__ == "__main__":
    main()
