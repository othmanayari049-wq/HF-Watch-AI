from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "features" / "training_dataset_clean.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "results" / "shap_feature_importance.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "calibration_benchmark.csv"

N_FEATURES = 20
OUTER_SPLITS = 5
CALIBRATION_SIZE = 0.25
RANDOM_STATE = 42
EPS = 1e-6


def make_base_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def safe_logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def evaluate(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0 - EPS)
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "brier": brier_score_loss(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    importance = pd.read_csv(IMPORTANCE_PATH)

    feature_columns = importance["feature"].tolist()[:N_FEATURES]

    X = data[feature_columns]
    y = data["label"].astype(int).to_numpy()
    groups = data["patient_id"].astype(str).to_numpy()

    outer_cv = GroupKFold(n_splits=OUTER_SPLITS)

    fold_rows: list[dict[str, float | int | str]] = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups),
        start=1,
    ):
        X_outer_train = X.iloc[train_idx]
        y_outer_train = y[train_idx]
        g_outer_train = groups[train_idx]

        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        inner_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=CALIBRATION_SIZE,
            random_state=RANDOM_STATE + fold,
        )
        fit_rel_idx, cal_rel_idx = next(
            inner_splitter.split(X_outer_train, y_outer_train, g_outer_train)
        )

        X_fit = X_outer_train.iloc[fit_rel_idx]
        y_fit = y_outer_train[fit_rel_idx]
        X_cal = X_outer_train.iloc[cal_rel_idx]
        y_cal = y_outer_train[cal_rel_idx]

        base = make_base_model()
        base.fit(X_fit, y_fit)

        p_cal = base.predict_proba(X_cal)[:, 1]
        p_test = base.predict_proba(X_test)[:, 1]

        # Uncalibrated baseline.
        methods: dict[str, np.ndarray] = {
            "uncalibrated": p_test,
        }

        # Sigmoid / Platt scaling: fit a one-dimensional logistic model
        # on held-out patient groups using the base model logit as input.
        sigmoid = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        sigmoid.fit(safe_logit(p_cal).reshape(-1, 1), y_cal)
        methods["sigmoid"] = sigmoid.predict_proba(
            safe_logit(p_test).reshape(-1, 1)
        )[:, 1]

        # Isotonic regression: non-parametric monotonic calibration,
        # also fit only on held-out patient groups from the training fold.
        isotonic = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )
        isotonic.fit(p_cal, y_cal)
        methods["isotonic"] = isotonic.predict(p_test)

        for method, probabilities in methods.items():
            metrics = evaluate(y_test, probabilities)
            fold_rows.append(
                {
                    "fold": fold,
                    "method": method,
                    **metrics,
                }
            )

    fold_df = pd.DataFrame(fold_rows)

    summary = (
        fold_df.groupby("method", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_f1=("f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_brier=("brier", "mean"),
            mean_log_loss=("log_loss", "mean"),
        )
        .sort_values(["mean_brier", "mean_log_loss"], ascending=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print("HF-Watch-AI calibration benchmark")
    print("Patient groups are separated in both the outer test split and the inner calibration split.")
    print("Lower Brier score and lower log loss indicate better probability calibration.\n")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nSaved:", OUTPUT_PATH)
    print("This benchmark uses development data only and does not use the external evaluation set for model selection.")


if __name__ == "__main__":
    main()
