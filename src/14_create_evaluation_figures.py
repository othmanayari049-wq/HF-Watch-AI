from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "features" / "training_dataset_clean.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "results" / "shap_feature_importance.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

TEST_PATIENTS = [
    "19090", "19093", "19140", "19830",
    "chf12", "chf13", "chf14", "chf15",
]


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

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

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Healthy", "CHF"],
    )
    plt.title("Confusion Matrix — Unseen Patients")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "confusion_matrix_unseen.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        name="Top-20 Logistic Regression",
    )
    plt.title("ROC Curve — Unseen Patients")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "roc_curve_unseen.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    PrecisionRecallDisplay.from_predictions(
        y_test,
        probabilities,
        name="Top-20 Logistic Regression",
    )
    plt.title("Precision–Recall Curve — Unseen Patients")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "precision_recall_unseen.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Saved:")
    print("-", RESULTS_DIR / "confusion_matrix_unseen.png")
    print("-", RESULTS_DIR / "roc_curve_unseen.png")
    print("-", RESULTS_DIR / "precision_recall_unseen.png")


if __name__ == "__main__":
    main()