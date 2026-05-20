from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
CLASSIFICATION_DIR = RESULTS_DIR / "classification"

FEATURE_FILES = {
    "chapter_level": RESULTS_DIR / "chapter_level_features.csv",
    "window_1500": RESULTS_DIR / "window_1500_features.csv",
}

LEXICAL_FEATURES = [
    "wordform_unigram_entropy",
    "lemma_unigram_entropy",
    "function_word_entropy",
    "word_length_entropy",
    "lexical_frequency_band_entropy",
    "punctuation_entropy",
]
SYNTACTIC_FEATURES = [
    "pos_unigram_entropy",
    "pos_bigram_entropy",
    "pos_trigram_entropy",
    "clause_marker_entropy",
    "dependency_relation_entropy",
    "coordination_subordination_entropy",
]
NGRAM_FEATURES = [
    "character_entropy",
    "character_2gram_entropy",
    "character_3gram_entropy",
    "character_4gram_entropy",
    "word_bigram_entropy",
    "word_trigram_entropy",
]
ALL_FEATURES = LEXICAL_FEATURES + SYNTACTIC_FEATURES + NGRAM_FEATURES

FEATURE_GROUPS = {
    "lexical": LEXICAL_FEATURES,
    "syntactic": SYNTACTIC_FEATURES,
    "ngram": NGRAM_FEATURES,
    "all_features": ALL_FEATURES,
}


def make_logistic_regression() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def make_naive_bayes() -> GaussianNB:
    return GaussianNB()


def make_svm() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", gamma="scale", probability=False, random_state=42)),
        ]
    )


def make_knn() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]
    )


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
    )


def make_ridge_classifier() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(random_state=42)),
        ]
    )


def make_gradient_boosting() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=42)


def make_neural_network() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, random_state=42)),
        ]
    )


MODEL_BUILDERS: dict[str, Callable[[], object]] = {
    "Logistic Regression": make_logistic_regression,
    "Naive Bayes": make_naive_bayes,
    "SVM": make_svm,
    "K-NN": make_knn,
    "Random Forest": make_random_forest,
    "Ridge Classifier": make_ridge_classifier,
    "Gradient Boosting": make_gradient_boosting,
    "Neural Network": make_neural_network,
}


def make_voting_ensemble() -> VotingClassifier:
    estimators = [
        ("lr", make_logistic_regression()),
        ("nb", make_naive_bayes()),
        ("svm", make_svm()),
        ("knn", make_knn()),
        ("rf", make_random_forest()),
        ("ridge", make_ridge_classifier()),
        ("gb", make_gradient_boosting()),
        ("nn", make_neural_network()),
    ]
    return VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1)


def evaluate_model(X: pd.DataFrame, y: pd.Series, model) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_validate(
        clone(model),
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=False,
    )
    mean_accuracy = float(scores["test_score"].mean() * 100)
    std_accuracy = float(scores["test_score"].std(ddof=0) * 100)
    return mean_accuracy, std_accuracy


def evaluate_individual_features(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["edition"]
    mean_rows = []
    std_rows = []
    long_rows = []

    for feature in ALL_FEATURES:
        row_mean = {"feature": feature}
        row_std = {"feature": feature}
        X = df[[feature]]
        for model_name, builder in MODEL_BUILDERS.items():
            mean_acc, std_acc = evaluate_model(X, y, builder())
            row_mean[model_name] = round(mean_acc, 2)
            row_std[model_name] = round(std_acc, 2)
            long_rows.append(
                {
                    "dataset": dataset_name,
                    "analysis_level": "individual_feature",
                    "feature_set": feature,
                    "feature_count": 1,
                    "algorithm": model_name,
                    "mean_accuracy": round(mean_acc, 4),
                    "std_accuracy": round(std_acc, 4),
                }
            )
        mean_rows.append(row_mean)
        std_rows.append(row_std)

    return pd.DataFrame(mean_rows), pd.DataFrame(std_rows), pd.DataFrame(long_rows)


def evaluate_feature_groups(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["edition"]
    mean_rows = []
    std_rows = []
    long_rows = []

    for group_name, features in FEATURE_GROUPS.items():
        row_mean = {"feature_group": group_name}
        row_std = {"feature_group": group_name}
        X = df[features]
        for model_name, builder in MODEL_BUILDERS.items():
            mean_acc, std_acc = evaluate_model(X, y, builder())
            row_mean[model_name] = round(mean_acc, 2)
            row_std[model_name] = round(std_acc, 2)
            long_rows.append(
                {
                    "dataset": dataset_name,
                    "analysis_level": "feature_group",
                    "feature_set": group_name,
                    "feature_count": len(features),
                    "algorithm": model_name,
                    "mean_accuracy": round(mean_acc, 4),
                    "std_accuracy": round(std_acc, 4),
                }
            )

        ensemble_mean, ensemble_std = evaluate_model(X, y, make_voting_ensemble())
        row_mean["Voting Ensemble"] = round(ensemble_mean, 2)
        row_std["Voting Ensemble"] = round(ensemble_std, 2)
        long_rows.append(
            {
                "dataset": dataset_name,
                "analysis_level": "feature_group",
                "feature_set": group_name,
                "feature_count": len(features),
                "algorithm": "Voting Ensemble",
                "mean_accuracy": round(ensemble_mean, 4),
                "std_accuracy": round(ensemble_std, 4),
            }
        )
        mean_rows.append(row_mean)
        std_rows.append(row_std)

    return pd.DataFrame(mean_rows), pd.DataFrame(std_rows), pd.DataFrame(long_rows)


def build_best_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    grouped = long_df.groupby(["dataset", "analysis_level", "feature_set"], as_index=False)
    for _, group in grouped:
        best = group.sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True]).iloc[0]
        best_rows.append(
            {
                "dataset": best["dataset"],
                "analysis_level": best["analysis_level"],
                "feature_set": best["feature_set"],
                "feature_count": best["feature_count"],
                "best_algorithm": best["algorithm"],
                "best_mean_accuracy": round(float(best["mean_accuracy"]), 4),
                "best_std_accuracy": round(float(best["std_accuracy"]), 4),
            }
        )
    return pd.DataFrame(best_rows)


def save_tables(dataset_name: str, individual_mean: pd.DataFrame, individual_std: pd.DataFrame, group_mean: pd.DataFrame, group_std: pd.DataFrame) -> None:
    dataset_dir = CLASSIFICATION_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    individual_mean.to_csv(dataset_dir / "individual_feature_accuracy_mean.csv", index=False, encoding="utf-8-sig")
    individual_std.to_csv(dataset_dir / "individual_feature_accuracy_std.csv", index=False, encoding="utf-8-sig")
    group_mean.to_csv(dataset_dir / "feature_group_accuracy_mean.csv", index=False, encoding="utf-8-sig")
    group_std.to_csv(dataset_dir / "feature_group_accuracy_std.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(dataset_dir / "classification_tables.xlsx") as writer:
        individual_mean.to_excel(writer, sheet_name="individual_mean", index=False)
        individual_std.to_excel(writer, sheet_name="individual_std", index=False)
        group_mean.to_excel(writer, sheet_name="group_mean", index=False)
        group_std.to_excel(writer, sheet_name="group_std", index=False)


def build_markdown_summary(all_results: pd.DataFrame, best_summary: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# hlm 分类分析结果")
    lines.append("")
    lines.append("## 1. 方法设置")
    lines.append("")
    lines.append("本分析参照《Simplification in interpreting: the text classification of spoken and interpreted Chinese through ensemble learning techniques》的分类思路，采用 10-fold stratified cross-validation，对单特征、三大类特征和全部特征进行二分类实验。")
    lines.append("")
    lines.append("本次保留 8 个基础分类器：")
    lines.append("")
    for model_name in MODEL_BUILDERS:
        lines.append(f"- `{model_name}`")
    lines.append("")
    lines.append("同时增加一个 `Voting Ensemble`，采用硬投票方式综合 8 个基础分类器。之所以使用硬投票，是为了在保持 8 模型面板完整的同时，避免部分分类器缺少概率输出接口。")
    lines.append("")
    lines.append("## 2. 数据集")
    lines.append("")
    lines.append("- `chapter_level`：120+120 个样本")
    lines.append("- `window_1500`：359+359 个样本")
    lines.append("")
    lines.append("## 3. 最优结果概览")
    lines.append("")
    for dataset in ["chapter_level", "window_1500"]:
        lines.append(f"### {dataset}")
        subset = best_summary[best_summary["dataset"] == dataset].copy()
        for analysis_level in ["individual_feature", "feature_group"]:
            level_subset = subset[subset["analysis_level"] == analysis_level]
            if level_subset.empty:
                continue
            lines.append(f"- `{analysis_level}`")
            for _, row in level_subset.sort_values(["best_mean_accuracy", "feature_set"], ascending=[False, True]).iterrows():
                lines.append(
                    f"  - `{row['feature_set']}`: best = `{row['best_algorithm']}` ({row['best_mean_accuracy']:.2f}% ± {row['best_std_accuracy']:.2f})"
                )
        lines.append("")
    lines.append("## 4. 输出文件")
    lines.append("")
    lines.append("- `classification/<dataset>/individual_feature_accuracy_mean.csv`")
    lines.append("- `classification/<dataset>/individual_feature_accuracy_std.csv`")
    lines.append("- `classification/<dataset>/feature_group_accuracy_mean.csv`")
    lines.append("- `classification/<dataset>/feature_group_accuracy_std.csv`")
    lines.append("- `classification/<dataset>/classification_tables.xlsx`")
    lines.append("- `classification/all_results_long.csv`")
    lines.append("- `classification/best_results_summary.csv`")
    return "\n".join(lines) + "\n"


def main() -> None:
    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    all_long_frames = []
    for dataset_name, path in FEATURE_FILES.items():
        df = pd.read_csv(path)
        individual_mean, individual_std, individual_long = evaluate_individual_features(df, dataset_name)
        group_mean, group_std, group_long = evaluate_feature_groups(df, dataset_name)
        save_tables(dataset_name, individual_mean, individual_std, group_mean, group_std)
        all_long_frames.extend([individual_long, group_long])

    all_results = pd.concat(all_long_frames, ignore_index=True)
    best_summary = build_best_summary(all_results)

    all_results.to_csv(CLASSIFICATION_DIR / "all_results_long.csv", index=False, encoding="utf-8-sig")
    best_summary.to_csv(CLASSIFICATION_DIR / "best_results_summary.csv", index=False, encoding="utf-8-sig")
    summary_md = build_markdown_summary(all_results, best_summary)
    (CLASSIFICATION_DIR / "classification_summary.md").write_text(summary_md, encoding="utf-8")

    metadata = {
        "cross_validation": "10-fold stratified cross-validation",
        "base_classifiers": list(MODEL_BUILDERS.keys()),
        "ensemble": "Voting Ensemble (hard voting over 8 base classifiers)",
        "feature_groups": {key: value for key, value in FEATURE_GROUPS.items()},
    }
    (CLASSIFICATION_DIR / "classification_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Classification outputs saved to: {CLASSIFICATION_DIR}")


if __name__ == "__main__":
    main()
