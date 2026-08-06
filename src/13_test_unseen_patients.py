from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "features" / "training_dataset_clean.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "results" / "shap_feature_importance.csv"

TEST_PATIENTS = [
    "19090", "19093", "19140", "19830",
    "chf12", "chf13", "chf14", "chf15",
]


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    importance = pd.read_csv(IMPORTANCE_PATH)

    features = importance["feature"].head(20).tolist()

    train = data[~data["patient_id"].isin(TEST_PATIENTS)].copy()
    test = data[data["patient_id"].isin(TEST_PATIENTS)].copy()

    X_train = train[features]
    y_train = train["label"]

    X_test = test[features]
    y_test = test["label"]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Training patients:", train["patient_id"].nunique())
    print("Unseen test patients:", test["patient_id"].nunique())
    print("Test windows:", len(test))

    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()