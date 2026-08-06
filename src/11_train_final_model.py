from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "features" / "training_dataset_clean.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "results" / "shap_feature_importance.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "hf_watch_top20_model.joblib"

TOP_N = 20


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    importance = pd.read_csv(IMPORTANCE_PATH)

    selected_features = importance["feature"].head(TOP_N).tolist()

    X = data[selected_features]
    y = data["label"]

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

    pipeline.fit(X, y)

    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": selected_features,
            "window_seconds": 300,
            "model_purpose": "Experimental CHF vs healthy ECG classification",
        },
        MODEL_PATH,
    )

    print("Final model trained.")
    print("Features used:", TOP_N)

    for feature in selected_features:
        print("-", feature)

    print("Saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()