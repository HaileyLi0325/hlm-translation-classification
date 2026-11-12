from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[1]
SOURCE_ROOT = PACKAGE_ROOT.parent / "source"
FEATURE_ROOT = PACKAGE_ROOT.parent / "data"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
LABELS = ["hawkes", "yang"]


def load_source_module():
    path = SOURCE_ROOT / "run_hlm_classification.py"
    spec = importlib.util.spec_from_file_location("original_classification_r2c3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )


def plot_results(metrics: pd.DataFrame, per_chapter: pd.DataFrame):
    style()
    order = [
        "Logistic Regression", "Naive Bayes", "SVM", "K-NN", "Random Forest",
        "Ridge Classifier", "Gradient Boosting", "Neural Network", "Voting Ensemble",
    ]
    short = ["LR", "NB", "SVM", "K-NN", "RF", "Ridge", "GB", "NN", "Vote"]
    palette = {"chapter_level": "#484878", "window_1500": "#E4CCD8"}
    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ax_top = fig.add_subplot(gs[0, :])
    axes_bottom = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    x = np.arange(len(order))
    for i, dataset in enumerate(("chapter_level", "window_1500")):
        part = metrics[metrics["dataset"] == dataset].set_index("algorithm").loc[order]
        ax_top.bar(
            x + (i - 0.5) * 0.36, part["test_accuracy"], width=0.36,
            color=palette[dataset], edgecolor="#272727", linewidth=0.5,
            label="Chapter-level" if dataset == "chapter_level" else "1,500-word segments",
        )
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(short, rotation=25, ha="right")
    ax_top.set_ylabel("Late-chapter test accuracy (%)")
    ax_top.set_ylim(55, 101)
    ax_top.legend(loc="lower left", ncol=2)
    ax_top.text(-0.04, 1.02, "a", transform=ax_top.transAxes, fontweight="bold", fontsize=8)

    for ax, dataset, label in zip(axes_bottom, ("chapter_level", "window_1500"), "bc"):
        part = per_chapter[
            (per_chapter["dataset"] == dataset) & (per_chapter["algorithm"] == "Voting Ensemble")
        ].copy()
        part["chapter_bin"] = pd.cut(
            part["chapter"], bins=[80, 90, 100, 110, 120], labels=["81-90", "91-100", "101-110", "111-120"]
        )
        summary = part.groupby("chapter_bin", observed=True)["accuracy"].agg(["mean", "std", "count"]).reset_index()
        half_width = 1.96 * summary["std"].fillna(0) / np.sqrt(summary["count"])
        lower = np.maximum(0, summary["mean"] - half_width)
        upper = np.minimum(1, summary["mean"] + half_width)
        ax.errorbar(
            summary["chapter_bin"].astype(str), summary["mean"] * 100,
            yerr=np.vstack(((summary["mean"] - lower) * 100, (upper - summary["mean"]) * 100)),
            color=palette[dataset], marker="o",
            linewidth=1.5, capsize=2,
        )
        ax.set_title("Chapter-level" if dataset == "chapter_level" else "1,500-word segments")
        ax.set_ylabel("Voting accuracy by chapter block (%)")
        ax.set_xlabel("Later chapter block")
        ax.set_ylim(60, 101)
        ax.text(-0.12, 1.02, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {"svg": {}, "pdf": {}, "png": {"dpi": 300}, "tiff": {"dpi": 600}}.items():
        fig.savefig(FIGURES / f"early_to_late_chronological_split.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main():
    source = load_source_module()
    builders = dict(source.MODEL_BUILDERS)
    builders["Voting Ensemble"] = source.make_voting_ensemble
    RESULTS.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    chapter_rows = []
    predictions = []
    for dataset, filename in {"chapter_level": "chapter_level_features.csv", "window_1500": "window_1500_features.csv"}.items():
        df = pd.read_csv(FEATURE_ROOT / filename)
        train = df["chapter"] <= 80
        test = df["chapter"] >= 81
        X_train, X_test = df.loc[train, source.ALL_FEATURES], df.loc[test, source.ALL_FEATURES]
        y_train, y_test = df.loc[train, "edition"], df.loc[test, "edition"]
        for algorithm, builder in builders.items():
            model = clone(builder())
            model.fit(X_train, y_train)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, pred_test, labels=LABELS, average="macro", zero_division=0
            )
            cm = confusion_matrix(y_test, pred_test, labels=LABELS)
            metric_rows.append(
                {
                    "dataset": dataset, "algorithm": algorithm,
                    "train_chapters": "1-80", "test_chapters": "81-120",
                    "n_train": len(y_train), "n_test": len(y_test),
                    "train_accuracy": accuracy_score(y_train, pred_train) * 100,
                    "test_accuracy": accuracy_score(y_test, pred_test) * 100,
                    "generalization_gap": (accuracy_score(y_train, pred_train) - accuracy_score(y_test, pred_test)) * 100,
                    "balanced_accuracy": balanced_accuracy_score(y_test, pred_test) * 100,
                    "macro_precision": precision * 100, "macro_recall": recall * 100, "macro_f1": f1 * 100,
                    "hawkes_correct": int(cm[0, 0]), "hawkes_as_yang": int(cm[0, 1]),
                    "yang_as_hawkes": int(cm[1, 0]), "yang_correct": int(cm[1, 1]),
                }
            )
            pred_frame = df.loc[test, ["sample_id", "edition", "chapter", "window_id"]].copy()
            pred_frame["dataset"] = dataset
            pred_frame["algorithm"] = algorithm
            pred_frame["prediction"] = pred_test
            pred_frame["correct"] = pred_frame["edition"] == pred_frame["prediction"]
            predictions.append(pred_frame)
            per = pred_frame.groupby("chapter", as_index=False)["correct"].mean().rename(columns={"correct": "accuracy"})
            per["dataset"] = dataset
            per["algorithm"] = algorithm
            chapter_rows.append(per)

    metrics = pd.DataFrame(metric_rows)
    per_chapter = pd.concat(chapter_rows, ignore_index=True)
    pred_df = pd.concat(predictions, ignore_index=True)
    metrics.to_csv(RESULTS / "chronological_split_metrics.csv", index=False, encoding="utf-8-sig")
    per_chapter.to_csv(RESULTS / "per_chapter_accuracy.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(RESULTS / "test_predictions.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "split_rationale": (
            "Chapters 1-80 were used for training and chapters 81-120 for testing to align with the conventional "
            "textual boundary discussed by the reviewer. This is a robustness test, not evidence about authorial intention."
        ),
        "translation_context": (
            "Both analyzed editions contain all 120 chapters. In the Penguin edition, Hawkes translated chapters 1-80 "
            "and Minford translated chapters 81-120, so the chronological shift also includes a translator-composition shift."
        ),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "features": source.ALL_FEATURES,
    }
    (RESULTS / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(metrics, per_chapter)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
