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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[1]
SOURCE_ROOT = PACKAGE_ROOT.parent / "source"
FEATURE_ROOT = PACKAGE_ROOT.parent / "data"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
TRAIN_FRACTIONS = [0.50, 0.70, 0.80, 0.90]
SEEDS = list(range(10))


def load_source_module():
    path = SOURCE_ROOT / "run_hlm_classification.py"
    spec = importlib.util.spec_from_file_location("original_classification_r2c4", path)
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


def heatmap(ax, matrix, xlabels, ylabels, title, cmap, vmin=None, vmax=None, fmt=".1f"):
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(xlabels)), labels=xlabels)
    ax.set_yticks(np.arange(len(ylabels)), labels=ylabels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            rgba = image.cmap(image.norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, format(value, fmt), ha="center", va="center", color="white" if luminance < 0.48 else "black", fontsize=6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_results(summary: pd.DataFrame):
    style()
    order = [
        "Logistic Regression", "Naive Bayes", "SVM", "K-NN", "Random Forest",
        "Ridge Classifier", "Gradient Boosting", "Neural Network", "Voting Ensemble",
    ]
    short = ["LR", "NB", "SVM", "K-NN", "RF", "Ridge", "GB", "NN", "Vote"]
    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8])
    top_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_delta = fig.add_subplot(gs[1, :])
    all_deltas = []
    for ax, dataset, label in zip(top_axes, ("chapter_level", "window_1500"), "ab"):
        part = summary[summary["dataset"] == dataset]
        pivot = part.pivot(index="algorithm", columns="train_fraction", values="mean_test_accuracy").loc[order, TRAIN_FRACTIONS]
        im = heatmap(
            ax, pivot.to_numpy(), ["50%", "70%", "80%", "90%"], short,
            "Chapter-level" if dataset == "chapter_level" else "1,500-word segments",
            cmap="Blues", vmin=max(50, float(pivot.min().min()) - 2), vmax=min(100, float(pivot.max().max()) + 1),
        )
        ax.set_xlabel("Training fraction")
        ax.set_ylabel("Classifier")
        ax.text(-0.14, 1.02, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Mean test accuracy (%)")
        delta = pivot[0.50] - pivot[0.90]
        all_deltas.append((dataset, delta))

    x = np.arange(len(order))
    colors = ["#484878", "#E4CCD8"]
    for i, ((dataset, delta), color) in enumerate(zip(all_deltas, colors)):
        ax_delta.bar(
            x + (i - 0.5) * 0.36, delta.loc[order], width=0.36, color=color,
            edgecolor="#272727", linewidth=0.5,
            label="Chapter-level" if dataset == "chapter_level" else "1,500-word segments",
        )
    ax_delta.axhline(0, color="#767676", linewidth=0.8)
    ax_delta.set_xticks(x, short)
    ax_delta.set_ylabel("Accuracy change: 50% minus 90% training (points)")
    ax_delta.legend(loc="lower left")
    ax_delta.text(-0.04, 1.02, "c", transform=ax_delta.transAxes, fontweight="bold", fontsize=8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {"svg": {}, "pdf": {}, "png": {"dpi": 300}, "tiff": {"dpi": 600}}.items():
        fig.savefig(FIGURES / f"training_size_sensitivity.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main():
    source = load_source_module()
    builders = dict(source.MODEL_BUILDERS)
    builders["Voting Ensemble"] = source.make_voting_ensemble
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, filename in {"chapter_level": "chapter_level_features.csv", "window_1500": "window_1500_features.csv"}.items():
        df = pd.read_csv(FEATURE_ROOT / filename)
        X, y = df[source.ALL_FEATURES], df["edition"]
        indices = np.arange(len(df))
        for fraction in TRAIN_FRACTIONS:
            for seed in SEEDS:
                train_idx, test_idx = train_test_split(
                    indices, train_size=fraction, stratify=y, random_state=seed
                )
                for algorithm, builder in builders.items():
                    model = clone(builder())
                    model.fit(X.iloc[train_idx], y.iloc[train_idx])
                    train_accuracy = accuracy_score(y.iloc[train_idx], model.predict(X.iloc[train_idx])) * 100
                    test_accuracy = accuracy_score(y.iloc[test_idx], model.predict(X.iloc[test_idx])) * 100
                    rows.append(
                        {
                            "dataset": dataset, "algorithm": algorithm,
                            "train_fraction": fraction, "seed": seed,
                            "n_train": len(train_idx), "n_test": len(test_idx),
                            "train_accuracy": train_accuracy, "test_accuracy": test_accuracy,
                            "generalization_gap": train_accuracy - test_accuracy,
                        }
                    )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["dataset", "algorithm", "train_fraction"], as_index=False)
        .agg(
            mean_test_accuracy=("test_accuracy", "mean"),
            sd_test_accuracy=("test_accuracy", "std"),
            mean_train_accuracy=("train_accuracy", "mean"),
            mean_generalization_gap=("generalization_gap", "mean"),
            n_repeats=("seed", "count"),
        )
    )
    raw.to_csv(RESULTS / "training_size_repeated_splits_raw.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS / "training_size_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "design": "Repeated stratified hold-out evaluation at 50%, 70%, 80%, and 90% training fractions.",
        "repeats": len(SEEDS), "seeds": SEEDS,
        "interpretation": "Variability is descriptive sample SD across overlapping predefined splits, not an independent-experiment confidence interval.",
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "features": source.ALL_FEATURES,
    }
    (RESULTS / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_results(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
