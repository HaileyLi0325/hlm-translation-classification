from __future__ import annotations

import importlib.util
import json
import platform
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[1]
SOURCE_ROOT = PACKAGE_ROOT.parent / "source"
FEATURE_ROOT = PACKAGE_ROOT.parent / "data"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"


def load_source_module():
    path = SOURCE_ROOT / "run_hlm_classification.py"
    spec = importlib.util.spec_from_file_location("original_classification_r2c1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SEARCHES = {
    "Logistic Regression": ("grid", {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}),
    "Naive Bayes": ("grid", {"var_smoothing": np.logspace(-12, -6, 7).tolist()}),
    "SVM": ("grid", {"clf__C": [0.1, 1.0, 10.0, 100.0], "clf__gamma": ["scale", 0.01, 0.1, 1.0]}),
    "K-NN": (
        "grid",
        {"clf__n_neighbors": [3, 5, 7, 9, 11], "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]},
    ),
    "Random Forest": (
        "random",
        {
            "n_estimators": [200, 500, 800], "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.5, None],
        },
    ),
    "Ridge Classifier": ("grid", {"clf__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
    "Gradient Boosting": (
        "random",
        {
            "n_estimators": [50, 100, 200], "learning_rate": [0.03, 0.1, 0.2],
            "max_depth": [1, 2, 3], "min_samples_leaf": [1, 2, 4],
        },
    ),
    "Neural Network": (
        "random",
        {
            "clf__hidden_layer_sizes": [(25,), (50,), (100,), (50, 25)],
            "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "clf__learning_rate_init": [0.0005, 0.001, 0.01],
            "clf__activation": ["relu", "tanh"],
        },
    ),
}


def build_search(estimator, algorithm: str, inner_cv, seed: int):
    strategy, space = SEARCHES[algorithm]
    if algorithm == "Random Forest":
        estimator.set_params(n_jobs=1)
    common = dict(estimator=estimator, scoring="accuracy", cv=inner_cv, refit=True, n_jobs=-1, error_score="raise")
    if strategy == "grid":
        return GridSearchCV(param_grid=space, **common)
    return RandomizedSearchCV(param_distributions=space, n_iter=8, random_state=seed, **common)


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


def plot_results(summary: pd.DataFrame):
    style()
    order = [
        "Logistic Regression", "Naive Bayes", "SVM", "K-NN", "Random Forest",
        "Ridge Classifier", "Gradient Boosting", "Neural Network",
    ]
    short = ["LR", "NB", "SVM", "K-NN", "RF", "Ridge", "GB", "NN"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    x = np.arange(len(order))
    colors = {"default": "#7884B4", "tuned": "#E4CCD8"}
    for col, dataset in enumerate(("chapter_level", "window_1500")):
        part = summary[summary["dataset"] == dataset].set_index("algorithm").loc[order]
        for i, (kind, mean_col, sd_col) in enumerate(
            (("default", "default_test_mean", "default_test_sd"), ("tuned", "tuned_test_mean", "tuned_test_sd"))
        ):
            axes[0, col].bar(
                x + (i - 0.5) * 0.36, part[mean_col], yerr=part[sd_col], width=0.36,
                color=colors[kind], edgecolor="#272727", linewidth=0.5,
                error_kw={"elinewidth": 0.7, "capsize": 1.5}, label=kind.capitalize(),
            )
        axes[0, col].set_title("Chapter-level" if dataset == "chapter_level" else "1,500-word segments")
        axes[0, col].set_ylabel("Outer-fold test accuracy (%)")
        lower = max(0, float(part[["default_test_mean", "tuned_test_mean"]].min().min()) - 7)
        upper = min(102, float((part[["default_test_mean", "tuned_test_mean"]].max(axis=1) + part[["default_test_sd", "tuned_test_sd"]].max(axis=1)).max()) + 2)
        axes[0, col].set_ylim(lower, upper)
        axes[1, col].bar(x, part["tuned_gap_mean"], color="#E4CCD8", edgecolor="#272727", linewidth=0.5)
        axes[1, col].axhline(0, color="#767676", linewidth=0.8)
        axes[1, col].set_ylabel("Tuned train-test gap (points)")
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xticks(x, short, rotation=35, ha="right")
    axes[0, 0].legend(loc="lower left")
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.12, 1.02, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {"svg": {}, "pdf": {}, "png": {"dpi": 300}, "tiff": {"dpi": 600}}.items():
        fig.savefig(FIGURES / f"hyperparameter_nested_cv_robustness.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    source = load_source_module()
    RESULTS.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    selected_rows = []
    for dataset, filename in {"chapter_level": "chapter_level_features.csv", "window_1500": "window_1500_features.csv"}.items():
        df = pd.read_csv(FEATURE_ROOT / filename)
        X, y = df[source.ALL_FEATURES], df["edition"]
        outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for algorithm, builder in source.MODEL_BUILDERS.items():
            print(f"{dataset}: {algorithm}", flush=True)
            for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
                default_model = clone(builder())
                default_model.fit(X.iloc[train_idx], y.iloc[train_idx])
                default_train = accuracy_score(y.iloc[train_idx], default_model.predict(X.iloc[train_idx])) * 100
                default_test = accuracy_score(y.iloc[test_idx], default_model.predict(X.iloc[test_idx])) * 100

                inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000 + fold)
                search = build_search(clone(builder()), algorithm, inner, seed=2000 + fold)
                search.fit(X.iloc[train_idx], y.iloc[train_idx])
                tuned_train = accuracy_score(y.iloc[train_idx], search.best_estimator_.predict(X.iloc[train_idx])) * 100
                tuned_test = accuracy_score(y.iloc[test_idx], search.best_estimator_.predict(X.iloc[test_idx])) * 100
                fold_rows.append(
                    {
                        "dataset": dataset, "algorithm": algorithm, "outer_fold": fold,
                        "n_train": len(train_idx), "n_test": len(test_idx),
                        "default_train_accuracy": default_train, "default_test_accuracy": default_test,
                        "tuned_train_accuracy": tuned_train, "tuned_test_accuracy": tuned_test,
                        "default_gap": default_train - default_test, "tuned_gap": tuned_train - tuned_test,
                        "inner_best_accuracy": search.best_score_ * 100,
                        "search_strategy": SEARCHES[algorithm][0],
                    }
                )
                selected_rows.append(
                    {
                        "dataset": dataset, "algorithm": algorithm, "outer_fold": fold,
                        "best_parameters": json.dumps(search.best_params_, sort_keys=True, default=str),
                        "inner_best_accuracy": search.best_score_ * 100,
                    }
                )
    folds = pd.DataFrame(fold_rows)
    selected = pd.DataFrame(selected_rows)
    summary = (
        folds.groupby(["dataset", "algorithm"], as_index=False)
        .agg(
            default_test_mean=("default_test_accuracy", "mean"),
            default_test_sd=("default_test_accuracy", "std"),
            tuned_test_mean=("tuned_test_accuracy", "mean"),
            tuned_test_sd=("tuned_test_accuracy", "std"),
            tuned_train_mean=("tuned_train_accuracy", "mean"),
            tuned_gap_mean=("tuned_gap", "mean"),
            tuned_gap_sd=("tuned_gap", "std"),
        )
    )
    correlations = []
    for dataset in ("chapter_level", "window_1500"):
        part = summary[summary["dataset"] == dataset]
        rho, p = spearmanr(part["default_test_mean"], part["tuned_test_mean"])
        correlations.append({"dataset": dataset, "spearman_rho": rho, "p_value": p})
    folds.to_csv(RESULTS / "nested_cv_outer_fold_metrics.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(RESULTS / "selected_hyperparameters_by_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS / "nested_cv_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(correlations).to_csv(RESULTS / "default_vs_tuned_rank_correlation.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "outer_evaluation": "10-fold stratified cross-validation, shuffle=True, random_state=42",
        "inner_optimization": "5-fold stratified cross-validation within each outer training fold",
        "optimization_metric": "accuracy",
        "search_spaces": {k: {"strategy": v[0], "parameters": v[1]} for k, v in SEARCHES.items()},
        "random_search_trials": 8,
        "leakage_control": "Scaling pipelines and hyperparameter search were fitted only within each outer training fold.",
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "features": source.ALL_FEATURES,
    }
    (RESULTS / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    plot_results(summary)
    print(summary.to_string(index=False))
    print(pd.DataFrame(correlations).to_string(index=False))


if __name__ == "__main__":
    main()
