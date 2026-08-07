from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "features" / "training_dataset_clean.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "results" / "shap_feature_importance.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "regularization_benchmark.csv"
C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    importance = pd.read_csv(IMPORTANCE_PATH)
    features = importance["feature"].tolist()[:20]
    X = data[features]
    y = data["label"].astype(int)
    groups = data["patient_id"]

    rows = []
    splitter = GroupKFold(n_splits=5)

    for c in C_VALUES:
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=c,
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            )
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = pipe.predict(X.iloc[test_idx])
            prob = pipe.predict_proba(X.iloc[test_idx])[:, 1]
            fold_rows.append(
                {
                    "accuracy": accuracy_score(y.iloc[test_idx], pred),
                    "f1": f1_score(y.iloc[test_idx], pred),
                    "roc_auc": roc_auc_score(y.iloc[test_idx], prob),
                }
            )

        folds = pd.DataFrame(fold_rows)
        rows.append(
            {
                "C": c,
                "mean_accuracy": folds["accuracy"].mean(),
                "std_accuracy": folds["accuracy"].std(),
                "mean_f1": folds["f1"].mean(),
                "mean_roc_auc": folds["roc_auc"].mean(),
            }
        )

    results = pd.DataFrame(rows).sort_values("mean_roc_auc", ascending=False)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(results.to_string(index=False))
    print("\nSaved:", OUTPUT_PATH)
    print("This benchmark uses development GroupKFold only and does not touch the external evaluation set.")


if __name__ == "__main__":
    main()
