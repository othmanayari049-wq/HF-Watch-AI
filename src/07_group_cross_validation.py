from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "features" / "training_dataset_clean.csv"
RESULTS_PATH = PROJECT_ROOT / "results" / "group_cv_results.csv"

METADATA_COLUMNS = [
    "patient_id",
    "window_index",
    "start_seconds",
    "number_of_beats",
    "label",
]


def main() -> None:
    data = pd.read_csv(DATA_PATH)

    feature_columns = [
        column
        for column in data.columns
        if column not in METADATA_COLUMNS
    ]

    X = data[feature_columns]
    y = data["label"]
    groups = data["patient_id"]

    group_kfold = GroupKFold(n_splits=5)

    fold_results = []

    for fold, (train_index, test_index) in enumerate(
        group_kfold.split(X, y, groups),
        start=1,
    ):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        train_groups = groups.iloc[train_index]
        test_groups = groups.iloc[test_index]

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities)

        result = {
            "fold": fold,
            "train_patients": train_groups.nunique(),
            "test_patients": test_groups.nunique(),
            "train_windows": len(train_index),
            "test_windows": len(test_index),
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc": roc_auc,
        }

        fold_results.append(result)

        print(f"\nFold {fold}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("Test patients:", sorted(test_groups.unique()))

    results = pd.DataFrame(fold_results)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    print("\nCross-validation summary")
    print(results[["accuracy", "f1_score", "roc_auc"]])

    print("\nMean results")
    print(f"Accuracy: {results['accuracy'].mean():.4f}")
    print(f"F1-score: {results['f1_score'].mean():.4f}")
    print(f"ROC-AUC: {results['roc_auc'].mean():.4f}")

    print("\nStandard deviation")
    print(f"Accuracy: {results['accuracy'].std():.4f}")
    print(f"F1-score: {results['f1_score'].std():.4f}")
    print(f"ROC-AUC: {results['roc_auc'].std():.4f}")

    print("\nSaved to:", RESULTS_PATH)


if __name__ == "__main__":
    main()