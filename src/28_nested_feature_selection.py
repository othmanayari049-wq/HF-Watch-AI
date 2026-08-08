from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "features" / "training_dataset_clean.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

OUTER_RESULTS = RESULTS_DIR / "nested_feature_selection_outer_folds.csv"
INNER_RESULTS = RESULTS_DIR / "nested_feature_selection_inner_scores.csv"
SUMMARY_RESULTS = RESULTS_DIR / "nested_feature_selection_summary.csv"
FEATURE_FREQUENCY = RESULTS_DIR / "nested_feature_selection_feature_frequency.csv"

METADATA_COLUMNS = [
    "patient_id",
    "window_index",
    "start_seconds",
    "number_of_beats",
    "label",
]

FEATURE_COUNTS = [5, 10, 20, 40, 90]
OUTER_SPLITS = 5
INNER_SPLITS = 4
RANDOM_STATE = 42
MAX_SHAP_SAMPLES = 1000


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def rank_features_with_training_data_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
) -> list[str]:
    """Rank features using only the supplied training partition.

    This reproduces the project's linear-model SHAP logic while ensuring that
    no validation/test rows are used to construct the ranking.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    model = LogisticRegression(
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    X_imputed = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imputed)
    model.fit(X_scaled, y_train)

    rng = np.random.default_rng(random_state)
    n_rows = len(X_scaled)
    if n_rows > MAX_SHAP_SAMPLES:
        sample_idx = np.sort(
            rng.choice(n_rows, size=MAX_SHAP_SAMPLES, replace=False)
        )
        X_explain = X_scaled[sample_idx]
    else:
        X_explain = X_scaled

    explainer = shap.LinearExplainer(model, X_explain)
    shap_values = explainer(X_explain)
    importance = np.abs(shap_values.values).mean(axis=0)

    ranked_idx = np.argsort(-importance)
    return [X_train.columns[i] for i in ranked_idx]


def evaluate_selected_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    features: list[str],
) -> dict[str, float]:
    pipe = make_pipeline()
    pipe.fit(X_train[features], y_train)

    pred = pipe.predict(X_valid[features])
    prob = pipe.predict_proba(X_valid[features])[:, 1]

    return {
        "accuracy": accuracy_score(y_valid, pred),
        "f1": f1_score(y_valid, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_valid, prob),
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH)

    feature_columns = [
        c for c in data.columns if c not in METADATA_COLUMNS
    ]
    candidate_counts = sorted(
        {min(n, len(feature_columns)) for n in FEATURE_COUNTS}
    )

    X = data[feature_columns]
    y = data["label"].astype(int)
    groups = data["patient_id"].astype(str)

    print("HF-Watch-AI fully nested subject-grouped feature-selection benchmark")
    print(f"Rows: {len(data)}")
    print(f"Patients: {groups.nunique()}")
    print(f"Candidate feature counts: {candidate_counts}")
    print("Outer test folds are untouched until inner feature-count selection is complete.\n")

    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    outer_rows: list[dict[str, object]] = []
    inner_rows: list[dict[str, object]] = []
    selected_feature_lists: list[list[str]] = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(X, y, groups),
        start=1,
    ):
        X_outer_train = X.iloc[outer_train_idx]
        y_outer_train = y.iloc[outer_train_idx]
        g_outer_train = groups.iloc[outer_train_idx]

        X_outer_test = X.iloc[outer_test_idx]
        y_outer_test = y.iloc[outer_test_idx]
        g_outer_test = groups.iloc[outer_test_idx]

        inner_cv = StratifiedGroupKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + outer_fold,
        )

        per_count_metrics: dict[int, list[dict[str, float]]] = {
            n: [] for n in candidate_counts
        }

        for inner_fold, (inner_train_rel, inner_valid_rel) in enumerate(
            inner_cv.split(X_outer_train, y_outer_train, g_outer_train),
            start=1,
        ):
            X_inner_train = X_outer_train.iloc[inner_train_rel]
            y_inner_train = y_outer_train.iloc[inner_train_rel]
            X_inner_valid = X_outer_train.iloc[inner_valid_rel]
            y_inner_valid = y_outer_train.iloc[inner_valid_rel]

            ranked = rank_features_with_training_data_only(
                X_inner_train,
                y_inner_train,
                random_state=(RANDOM_STATE + outer_fold * 100 + inner_fold),
            )

            for n_features in candidate_counts:
                selected = ranked[:n_features]
                metrics = evaluate_selected_features(
                    X_inner_train,
                    y_inner_train,
                    X_inner_valid,
                    y_inner_valid,
                    selected,
                )
                per_count_metrics[n_features].append(metrics)
                inner_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "n_features": n_features,
                        **metrics,
                    }
                )

        count_summary = []
        for n_features, metric_list in per_count_metrics.items():
            frame = pd.DataFrame(metric_list)
            count_summary.append(
                {
                    "n_features": n_features,
                    "mean_roc_auc": frame["roc_auc"].mean(),
                    "mean_f1": frame["f1"].mean(),
                    "mean_accuracy": frame["accuracy"].mean(),
                }
            )

        count_summary_df = pd.DataFrame(count_summary).sort_values(
            ["mean_roc_auc", "mean_f1", "mean_accuracy", "n_features"],
            ascending=[False, False, False, True],
        )
        chosen_n = int(count_summary_df.iloc[0]["n_features"])

        # After the feature count is selected using only inner folds, rank again
        # on the complete outer-training partition. The outer test fold remains
        # untouched until this point.
        outer_ranking = rank_features_with_training_data_only(
            X_outer_train,
            y_outer_train,
            random_state=RANDOM_STATE + outer_fold * 1000,
        )
        selected_features = outer_ranking[:chosen_n]
        selected_feature_lists.append(selected_features)

        outer_metrics = evaluate_selected_features(
            X_outer_train,
            y_outer_train,
            X_outer_test,
            y_outer_test,
            selected_features,
        )

        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "train_patients": g_outer_train.nunique(),
                "test_patients": g_outer_test.nunique(),
                "train_windows": len(outer_train_idx),
                "test_windows": len(outer_test_idx),
                "selected_n_features": chosen_n,
                "accuracy": outer_metrics["accuracy"],
                "f1": outer_metrics["f1"],
                "roc_auc": outer_metrics["roc_auc"],
                "selected_features": ";".join(selected_features),
            }
        )

        print(
            f"Outer fold {outer_fold}: selected {chosen_n} features | "
            f"accuracy={outer_metrics['accuracy']:.4f} | "
            f"F1={outer_metrics['f1']:.4f} | "
            f"ROC-AUC={outer_metrics['roc_auc']:.4f}"
        )

    outer_df = pd.DataFrame(outer_rows)
    inner_df = pd.DataFrame(inner_rows)

    summary = pd.DataFrame(
        [
            {
                "mean_accuracy": outer_df["accuracy"].mean(),
                "std_accuracy": outer_df["accuracy"].std(),
                "mean_f1": outer_df["f1"].mean(),
                "std_f1": outer_df["f1"].std(),
                "mean_roc_auc": outer_df["roc_auc"].mean(),
                "std_roc_auc": outer_df["roc_auc"].std(),
                "outer_folds": OUTER_SPLITS,
                "inner_folds": INNER_SPLITS,
            }
        ]
    )

    all_selected = [feature for fold in selected_feature_lists for feature in fold]
    frequency = (
        pd.Series(all_selected, name="feature")
        .value_counts()
        .rename_axis("feature")
        .reset_index(name="selection_count")
    )
    frequency["selection_fraction_of_outer_folds"] = (
        frequency["selection_count"] / OUTER_SPLITS
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outer_df.to_csv(OUTER_RESULTS, index=False)
    inner_df.to_csv(INNER_RESULTS, index=False)
    summary.to_csv(SUMMARY_RESULTS, index=False)
    frequency.to_csv(FEATURE_FREQUENCY, index=False)

    print("\nNested outer-fold summary:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nFeature count selected in each outer fold:")
    print(
        outer_df[["outer_fold", "selected_n_features"]]
        .to_string(index=False)
    )

    print("\nMost frequently selected features across outer folds:")
    print(frequency.head(20).to_string(index=False))

    print("\nSaved:")
    print(" -", OUTER_RESULTS)
    print(" -", INNER_RESULTS)
    print(" -", SUMMARY_RESULTS)
    print(" -", FEATURE_FREQUENCY)
    print(
        "\nIMPORTANT: This analysis is development-only. The external 83-record "
        "evaluation set is not accessed or used for feature ranking, feature-count "
        "selection, model fitting, or threshold selection."
    )


if __name__ == "__main__":
    main()
