from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "features" / "training_dataset_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_baseline.joblib"
RESULTS_DIR = PROJECT_ROOT / "results"

SHAP_PLOT_PATH = RESULTS_DIR / "shap_summary.png"
FEATURE_PATH = RESULTS_DIR / "shap_feature_importance.csv"

METADATA_COLUMNS = [
    "patient_id",
    "window_index",
    "start_seconds",
    "number_of_beats",
    "label",
]

RANDOM_STATE = 42
MAX_SAMPLES = 1000


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    data = pd.read_csv(DATA_PATH)

    saved_model = joblib.load(MODEL_PATH)
    pipeline = saved_model["pipeline"]
    feature_columns = saved_model["feature_columns"]

    X = data[feature_columns]

    if len(X) > MAX_SAMPLES:
        X_sample = X.sample(
            n=MAX_SAMPLES,
            random_state=RANDOM_STATE,
        )
    else:
        X_sample = X.copy()

    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    X_scaled = scaler.transform(X_sample)

    explainer = shap.LinearExplainer(
        model,
        X_scaled,
    )

    shap_values = explainer(X_scaled)

    mean_absolute_shap = np.abs(shap_values.values).mean(axis=0)

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_absolute_shap": mean_absolute_shap,
        }
    ).sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    importance.to_csv(
        FEATURE_PATH,
        index=False,
    )

    print("\nTop 20 features:")
    print(importance.head(20).to_string(index=False))

    shap.summary_plot(
        shap_values.values,
        X_scaled,
        feature_names=feature_columns,
        max_display=20,
        show=False,
    )

    plt.tight_layout()
    plt.savefig(
        SHAP_PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("\nSaved SHAP plot to:", SHAP_PLOT_PATH)
    print("Saved feature importance to:", FEATURE_PATH)


if __name__ == "__main__":
    main()