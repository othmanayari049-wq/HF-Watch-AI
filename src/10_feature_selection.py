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
OUTPUT_PATH = PROJECT_ROOT / "results" / "feature_selection_results.csv"

FEATURE_COUNTS = [5, 10, 20, 40, 90]


def evaluate_features(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:
    X = data[feature_columns]
    y = data["label"]
    groups = data["patient_id"]

    group_kfold = GroupKFold(n_splits=5)

    accuracies = []
    f1_scores = []
    roc_aucs = []

    for train_index, test_index in group_kfold.split(X, y, groups):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

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

        accuracies.append(accuracy_score(y_test, predictions))
        f1_scores.append(f1_score(y_test, predictions))
        roc_aucs.append(roc_auc_score(y_test, probabilities))

    return {
        "accuracy": sum(accuracies) / len(accuracies),
        "f1_score": sum(f1_scores) / len(f1_scores),
        "roc_auc": sum(roc_aucs) / len(roc_aucs),
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    importance = pd.read_csv(IMPORTANCE_PATH)

    ranked_features = importance["feature"].tolist()

    results = []

    for feature_count in FEATURE_COUNTS:
        selected_features = ranked_features[:feature_count]

        metrics = evaluate_features(
            data=data,
            feature_columns=selected_features,
        )

        result = {
            "number_of_features": feature_count,
            **metrics,
        }

        results.append(result)

        print(f"\nTop {feature_count} features")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSummary:")
    print(results_df)

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()