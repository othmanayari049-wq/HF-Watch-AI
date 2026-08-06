from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "features" / "training_dataset_clean.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "results" / "shap_feature_importance.csv"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    importance = pd.read_csv(IMPORTANCE_PATH)

    features = importance["feature"].head(20).tolist()

    X = data[features]
    y = data["label"]
    groups = data["patient_id"]

    group_kfold = GroupKFold(n_splits=5)

    fold_rows = []

    for fold, (train_index, test_index) in enumerate(
        group_kfold.split(X, y, groups),
        start=1,
    ):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

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

        fold_rows.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(y_test, predictions),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "test_patients": groups.iloc[test_index].nunique(),
            }
        )

    results = pd.DataFrame(fold_rows)

    print(results)
    print("\nMean accuracy:", round(results["accuracy"].mean(), 4))
    print("Mean ROC-AUC:", round(results["roc_auc"].mean(), 4))

    print(
        "\nImportant limitation:\n"
        "CHF and healthy recordings come from different databases. "
        "High performance may partly reflect acquisition or database differences."
    )


if __name__ == "__main__":
    main()