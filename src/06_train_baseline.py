from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_PATH = FEATURES_DIR / "training_dataset_clean.csv"
MODEL_PATH = MODELS_DIR / "logistic_regression_baseline.joblib"
RESULTS_PATH = RESULTS_DIR / "baseline_results.txt"

METADATA_COLUMNS = [
    "patient_id",
    "window_index",
    "start_seconds",
    "number_of_beats",
    "label",
]

# Fixed patient-level split
TEST_HEALTHY = ["19090", "19093", "19140", "19830"]
TEST_CHF = ["chf12", "chf13", "chf14", "chf15"]


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    data = pd.read_csv(INPUT_PATH)

    test_patients = TEST_HEALTHY + TEST_CHF

    train_data = data[~data["patient_id"].isin(test_patients)].copy()
    test_data = data[data["patient_id"].isin(test_patients)].copy()

    feature_columns = [
        column
        for column in data.columns
        if column not in METADATA_COLUMNS
    ]

    X_train = train_data[feature_columns]
    y_train = train_data["label"]

    X_test = test_data[feature_columns]
    y_test = test_data["label"]

    print("Training patients:", train_data["patient_id"].nunique())
    print("Test patients:", test_data["patient_id"].nunique())
    print("Training windows:", len(train_data))
    print("Test windows:", len(test_data))

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
    roc_auc = roc_auc_score(y_test, probabilities)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)

    results_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"ROC-AUC: {roc_auc:.4f}\n\n"
        f"Confusion Matrix:\n{matrix}\n\n"
        f"Classification Report:\n{report}\n"
    )

    print("\n" + results_text)

    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": feature_columns,
            "test_patients": test_patients,
        },
        MODEL_PATH,
    )

    RESULTS_PATH.write_text(results_text, encoding="utf-8")

    print("Model saved to:", MODEL_PATH)
    print("Results saved to:", RESULTS_PATH)


if __name__ == "__main__":
    main()