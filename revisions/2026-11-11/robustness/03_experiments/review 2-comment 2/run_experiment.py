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
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[1]
SOURCE_ROOT = PACKAGE_ROOT.parent / "source"
FEATURE_ROOT = PACKAGE_ROOT.parent / "data"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
LABELS = ["hawkes", "yang"]


def load_source_module():
    module_path = SOURCE_ROOT / "run_hlm_classification.py"
    spec = importlib.util.spec_from_file_location("original_classification_r2c2", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixed_splits(df: pd.DataFrame):
    indices = np.arange(len(df))
    sample_train, sample_test = train_test_split(
        indices, test_size=0.20, stratify=df["edition"], random_state=42
    )

    chapters = np.array(sorted(df["chapter"].unique()))
    rng = np.random.default_rng(42)
    test_chapters = np.sort(rng.choice(chapters, size=24, replace=False))
    grouped_test = indices[df["chapter"].isin(test_chapters)]
    grouped_train = indices[~df["chapter"].isin(test_chapters)]
    return {
        "sample_stratified_80_20": (sample_train, sample_test, None),
        "chapter_grouped_80_20": (grouped_train, grouped_test, test_chapters.tolist()),
    }


def metric_row(dataset: str, scheme: str, algorithm: str, y_train, pred_train, y_test, pred_test, chapters):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred_test, labels=LABELS, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, pred_test, labels=LABELS)
    return {
        "dataset": dataset,
        "split_scheme": scheme,
        "algorithm": algorithm,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "train_accuracy": accuracy_score(y_train, pred_train) * 100,
        "test_accuracy": accuracy_score(y_test, pred_test) * 100,
        "generalization_gap": (accuracy_score(y_train, pred_train) - accuracy_score(y_test, pred_test)) * 100,
        "balanced_accuracy": balanced_accuracy_score(y_test, pred_test) * 100,
        "macro_precision": precision * 100,
        "macro_recall": recall * 100,
        "macro_f1": f1 * 100,
        "hawkes_correct": int(cm[0, 0]),
        "hawkes_as_yang": int(cm[0, 1]),
        "yang_as_hawkes": int(cm[1, 0]),
        "yang_correct": int(cm[1, 1]),
        "test_chapters": "" if chapters is None else ",".join(map(str, chapters)),
    }


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


def plot_results(df: pd.DataFrame):
    style()
    order = [
        "Logistic Regression", "Naive Bayes", "SVM", "K-NN", "Random Forest",
        "Ridge Classifier", "Gradient Boosting", "Neural Network", "Voting Ensemble",
    ]
    short = ["LR", "NB", "SVM", "K-NN", "RF", "Ridge", "GB", "NN", "Vote"]
    colors = {"sample_stratified_80_20": "#7884B4", "chapter_grouped_80_20": "#E4CCD8"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for col, dataset in enumerate(("chapter_level", "window_1500")):
        part = df[df["dataset"] == dataset]
        x = np.arange(len(order))
        for i, scheme in enumerate(colors):
            vals = part[part["split_scheme"] == scheme].set_index("algorithm").loc[order]
            axes[0, col].bar(
                x + (i - 0.5) * 0.36, vals["test_accuracy"], width=0.36,
                color=colors[scheme], edgecolor="#272727", linewidth=0.5,
                label=scheme.replace("_", " "),
            )
            axes[1, col].bar(
                x + (i - 0.5) * 0.36, vals["generalization_gap"], width=0.36,
                color=colors[scheme], edgecolor="#272727", linewidth=0.5,
            )
        axes[0, col].set_title("Chapter-level" if dataset == "chapter_level" else "1,500-word segments")
        axes[0, col].set_ylabel("Test accuracy (%)")
        axes[0, col].set_ylim(max(0, float(part["test_accuracy"].min()) - 7), 101)
        axes[1, col].set_ylabel("Train-test gap (percentage points)")
        axes[1, col].axhline(0, color="#767676", linewidth=0.7)
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xticks(x)
            ax.set_xticklabels(short, rotation=35, ha="right")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, fontsize=6)
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.12, 1.02, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {
        "svg": {}, "pdf": {}, "png": {"dpi": 300}, "tiff": {"dpi": 600}
    }.items():
        fig.savefig(FIGURES / f"fixed_80_20_train_test_analysis.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main():
    source = load_source_module()
    builders = dict(source.MODEL_BUILDERS)
    builders["Voting Ensemble"] = source.make_voting_ensemble
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    split_manifest = {}
    for dataset, filename in {
        "chapter_level": "chapter_level_features.csv",
        "window_1500": "window_1500_features.csv",
    }.items():
        df = pd.read_csv(FEATURE_ROOT / filename)
        X = df[source.ALL_FEATURES]
        y = df["edition"]
        split_manifest[dataset] = {}
        for scheme, (train_idx, test_idx, chapters) in fixed_splits(df).items():
            split_manifest[dataset][scheme] = {
                "train_samples": len(train_idx),
                "test_samples": len(test_idx),
                "test_chapters": chapters,
                "train_class_counts": y.iloc[train_idx].value_counts().to_dict(),
                "test_class_counts": y.iloc[test_idx].value_counts().to_dict(),
            }
            for algorithm, builder in builders.items():
                model = clone(builder())
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                rows.append(
                    metric_row(
                        dataset, scheme, algorithm,
                        y.iloc[train_idx], model.predict(X.iloc[train_idx]),
                        y.iloc[test_idx], model.predict(X.iloc[test_idx]), chapters,
                    )
                )
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "fixed_80_20_metrics.csv", index=False, encoding="utf-8-sig")
    (RESULTS / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "random_seed": 42, "features": source.ALL_FEATURES,
    }
    (RESULTS / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    plot_results(results)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
