from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT.parent / "source"
FEATURE_ROOT = PACKAGE_ROOT.parent / "data"
OUTPUT_ROOT = Path(__file__).resolve().parent / "original_analysis_replication"


def load_source_module():
    module_path = SOURCE_ROOT / "run_hlm_classification.py"
    spec = importlib.util.spec_from_file_location("original_run_hlm_classification", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    source = load_source_module()
    source.CLASSIFICATION_DIR = OUTPUT_ROOT
    source.FEATURE_FILES = {
        "chapter_level": FEATURE_ROOT / "chapter_level_features.csv",
        "window_1500": FEATURE_ROOT / "window_1500_features.csv",
    }
    source.main()

    all_results = pd.read_csv(OUTPUT_ROOT / "all_results_long.csv")
    key = all_results[
        (all_results["analysis_level"] == "feature_group")
        & (all_results["feature_set"] == "all_features")
    ].copy()
    expected = {
        ("chapter_level", "Ridge Classifier"): 97.50,
        ("chapter_level", "Voting Ensemble"): 95.00,
        ("window_1500", "Neural Network"): 94.44,
        ("window_1500", "Voting Ensemble"): 93.60,
    }
    checks = []
    for (dataset, algorithm), manuscript_value in expected.items():
        observed = float(
            key.loc[
                (key["dataset"] == dataset) & (key["algorithm"] == algorithm),
                "mean_accuracy",
            ].iloc[0]
        )
        checks.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "manuscript_accuracy": manuscript_value,
                "replicated_accuracy": observed,
                "absolute_difference": round(abs(observed - manuscript_value), 4),
                "pass_at_0.01_percentage_point": abs(observed - manuscript_value) <= 0.01,
            }
        )
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(OUTPUT_ROOT / "manuscript_key_accuracy_check.csv", index=False, encoding="utf-8-sig")
    summary = {
        "all_key_checks_pass": bool(checks_df["pass_at_0.01_percentage_point"].all()),
        "checks": checks,
        "note": "The checks compare regenerated 10-fold results with the four headline values stated in the submitted manuscript.",
    }
    (OUTPUT_ROOT / "replication_check.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
