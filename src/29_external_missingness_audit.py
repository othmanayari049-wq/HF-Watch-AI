from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_FILE = PROJECT_ROOT / "features" / "external_rr_features.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "hf_watch_top20_model.joblib"
SUMMARY_FILE = PROJECT_ROOT / "results" / "external_missingness_feature_summary.csv"
RECORD_FILE = PROJECT_ROOT / "results" / "external_missingness_record_summary.csv"


def main() -> None:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {FEATURE_FILE}. Run external feature extraction first."
        )
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_FILE}. The frozen model bundle is required to identify the 20 deployed features."
        )

    df = pd.read_csv(FEATURE_FILE)
    bundle = joblib.load(MODEL_FILE)
    features = list(bundle["feature_columns"])

    required_meta = {"record", "dataset", "label"}
    missing_meta = sorted(required_meta.difference(df.columns))
    if missing_meta:
        raise ValueError(f"Missing metadata columns: {missing_meta}")

    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ValueError(
            "External feature table is missing deployed model features: "
            + ", ".join(missing_features)
        )

    X = df[features].replace([np.inf, -np.inf], np.nan)
    n_rows = len(X)

    feature_rows: list[dict[str, object]] = []
    for feature in features:
        mask = X[feature].isna()
        feature_rows.append(
            {
                "feature": feature,
                "missing_values": int(mask.sum()),
                "missing_fraction": float(mask.mean()),
                "missing_percent": float(mask.mean() * 100.0),
                "non_missing_values": int((~mask).sum()),
            }
        )

    feature_summary = pd.DataFrame(feature_rows).sort_values(
        ["missing_values", "feature"], ascending=[False, True]
    )

    per_row_missing = X.isna().sum(axis=1)
    row_frame = df[["record", "dataset", "label"]].copy()
    row_frame["missing_selected_features"] = per_row_missing
    row_frame["any_missing_selected_feature"] = per_row_missing > 0

    record_summary = (
        row_frame.groupby(["record", "dataset", "label"], as_index=False)
        .agg(
            windows=("missing_selected_features", "size"),
            windows_with_any_missing=("any_missing_selected_feature", "sum"),
            mean_missing_features_per_window=("missing_selected_features", "mean"),
            max_missing_features_in_window=("missing_selected_features", "max"),
        )
    )
    record_summary["fraction_windows_with_any_missing"] = (
        record_summary["windows_with_any_missing"] / record_summary["windows"]
    )

    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    feature_summary.to_csv(SUMMARY_FILE, index=False)
    record_summary.to_csv(RECORD_FILE, index=False)

    total_cells = n_rows * len(features)
    total_missing = int(X.isna().sum().sum())
    rows_any_missing = int((per_row_missing > 0).sum())
    records_any_missing = int(
        (record_summary["windows_with_any_missing"] > 0).sum()
    )

    print("HF-Watch-AI external selected-feature missingness audit")
    print("Descriptive audit only; no model, threshold, or imputation strategy is changed.\n")
    print(f"External windows: {n_rows}")
    print(f"External records: {df['record'].nunique()}")
    print(f"Selected model features: {len(features)}")
    print(f"Total selected-feature cells: {total_cells}")
    print(f"Missing/non-finite selected-feature cells: {total_missing}")
    print(f"Overall missing selected-feature fraction: {total_missing / total_cells:.6f}")
    print(f"Windows with >=1 missing selected feature: {rows_any_missing}/{n_rows} ({rows_any_missing / n_rows:.2%})")
    print(
        f"Records with >=1 affected window: {records_any_missing}/{df['record'].nunique()} "
        f"({records_any_missing / df['record'].nunique():.2%})"
    )

    print("\nPer-feature missingness among the 20 deployed features:")
    print(
        feature_summary.to_string(
            index=False,
            formatters={
                "missing_fraction": lambda x: f"{x:.6f}",
                "missing_percent": lambda x: f"{x:.3f}",
            },
        )
    )

    print("\nDataset-level summary:")
    dataset_summary = (
        row_frame.groupby("dataset")
        .agg(
            windows=("missing_selected_features", "size"),
            windows_with_any_missing=("any_missing_selected_feature", "sum"),
            mean_missing_features_per_window=("missing_selected_features", "mean"),
            max_missing_features_in_window=("missing_selected_features", "max"),
        )
    )
    dataset_summary["fraction_windows_with_any_missing"] = (
        dataset_summary["windows_with_any_missing"] / dataset_summary["windows"]
    )
    print(dataset_summary.to_string())

    worst_records = record_summary.sort_values(
        ["fraction_windows_with_any_missing", "mean_missing_features_per_window"],
        ascending=False,
    ).head(15)
    print("\nRecords with the most selected-feature missingness:")
    print(worst_records.to_string(index=False))

    print(f"\nSaved feature summary: {SUMMARY_FILE}")
    print(f"Saved record summary:  {RECORD_FILE}")
    print(
        "\nIMPORTANT: The frozen pipeline already uses median imputation. This audit quantifies how often "
        "that imputation is invoked on the external data; it does not justify changing the imputation method "
        "using the external evaluation set."
    )


if __name__ == "__main__":
    main()
