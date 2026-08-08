from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "features" / "training_dataset_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "hf_watch_top20_model.joblib"
OUTPUT_PATH = PROJECT_ROOT / "results" / "record_aggregation_benchmark.csv"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                    C=1.0,
                ),
            ),
        ]
    )


def metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "accuracy": accuracy_score(y_true, pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, score),
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    bundle = joblib.load(MODEL_PATH)
    features = bundle["feature_columns"]

    required = set(features) | {"label", "patient_id"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = data[features]
    y = data["label"].astype(int)
    groups = data["patient_id"].astype(str)

    gkf = GroupKFold(n_splits=5)
    patient_rows: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        pipeline = build_pipeline()
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])

        probs = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
        fold_df = pd.DataFrame(
            {
                "patient_id": groups.iloc[test_idx].to_numpy(),
                "label": y.iloc[test_idx].to_numpy(),
                "probability": probs,
            }
        )
        fold_df["window_prediction"] = (fold_df["probability"] >= 0.5).astype(int)

        grouped = (
            fold_df.groupby("patient_id", as_index=False)
            .agg(
                label=("label", "first"),
                windows=("probability", "size"),
                mean_probability=("probability", "mean"),
                median_probability=("probability", "median"),
                majority_fraction=("window_prediction", "mean"),
            )
        )
        grouped["fold"] = fold
        patient_rows.extend(grouped.to_dict("records"))

    patients = pd.DataFrame(patient_rows)
    y_patient = patients["label"].astype(int).to_numpy()

    methods = {
        "mean_probability": patients["mean_probability"].to_numpy(float),
        "median_probability": patients["median_probability"].to_numpy(float),
        "majority_fraction": patients["majority_fraction"].to_numpy(float),
    }

    rows = []
    for name, score in methods.items():
        row = {"method": name, **metrics(y_patient, score)}
        rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        ["roc_auc", "accuracy"], ascending=[False, False]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("HF-Watch-AI record aggregation benchmark")
    print("Development GroupKFold only; patients are never split across train and test.")
    print("All aggregation methods use the fixed 0.50 decision threshold.\n")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSaved:", OUTPUT_PATH)
    print("This benchmark does not use the external 83-record evaluation set for selection.")


if __name__ == "__main__":
    main()
