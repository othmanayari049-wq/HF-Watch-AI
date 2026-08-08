from __future__ import annotations

from pathlib import Path

import pandas as pd


SUMMARY_FILE = Path("results/external_validation_record_summary.csv")
WINDOW_FILE = Path("results/external_validation_predictions.csv")
OUTPUT_FILE = Path("results/external_error_analysis.csv")


def main() -> None:
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_FILE}. Run `python src/19_external_validation.py` first."
        )

    summary = pd.read_csv(SUMMARY_FILE)

    required = {
        "record",
        "dataset",
        "label",
        "windows",
        "mean_chf_probability",
        "predicted_chf_fraction",
        "record_prediction",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Missing required columns in record summary: {missing}")

    summary = summary.copy()
    summary["label"] = summary["label"].astype(int)
    summary["record_prediction"] = summary["record_prediction"].astype(int)

    def error_type(row: pd.Series) -> str:
        y = int(row["label"])
        pred = int(row["record_prediction"])
        if y == 0 and pred == 0:
            return "TN"
        if y == 0 and pred == 1:
            return "FP"
        if y == 1 and pred == 0:
            return "FN"
        return "TP"

    summary["error_type"] = summary.apply(error_type, axis=1)
    summary["correct"] = summary["label"] == summary["record_prediction"]
    summary["distance_from_threshold"] = (summary["predicted_chf_fraction"] - 0.5).abs()

    # Optional descriptive window-level statistics. These do not change the model or threshold.
    if WINDOW_FILE.exists():
        windows = pd.read_csv(WINDOW_FILE)
        if {"record", "chf_probability", "predicted_label"}.issubset(windows.columns):
            wstats = (
                windows.groupby("record")
                .agg(
                    min_window_probability=("chf_probability", "min"),
                    max_window_probability=("chf_probability", "max"),
                    median_window_probability=("chf_probability", "median"),
                    window_probability_std=("chf_probability", "std"),
                )
                .reset_index()
            )
            summary = summary.merge(wstats, on="record", how="left")

    errors = summary[~summary["correct"]].copy()
    errors = errors.sort_values(
        ["error_type", "distance_from_threshold", "record"],
        ascending=[True, True, True],
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(OUTPUT_FILE, index=False)

    print("HF-Watch-AI external record error analysis")
    print("Descriptive analysis only: no model, feature, calibration, or threshold tuning is performed.\n")

    counts = summary["error_type"].value_counts().reindex(["TN", "FP", "FN", "TP"], fill_value=0)
    print("Record outcomes:")
    for name, value in counts.items():
        print(f"  {name}: {int(value)}")

    print(f"\nCorrect records: {int(summary['correct'].sum())}/{len(summary)}")
    print(f"Misclassified records: {len(errors)}/{len(summary)}")

    if errors.empty:
        print("\nNo misclassified external records.")
    else:
        print("\nMisclassified external records:")
        columns = [
            "record",
            "dataset",
            "error_type",
            "label",
            "record_prediction",
            "windows",
            "mean_chf_probability",
            "predicted_chf_fraction",
            "distance_from_threshold",
        ]
        optional = [
            "min_window_probability",
            "median_window_probability",
            "max_window_probability",
            "window_probability_std",
        ]
        columns.extend([c for c in optional if c in errors.columns])
        print(errors[columns].to_string(index=False))

        print("\nSummary by error type:")
        grouped = (
            errors.groupby("error_type")
            .agg(
                records=("record", "size"),
                mean_record_probability=("mean_chf_probability", "mean"),
                mean_chf_window_fraction=("predicted_chf_fraction", "mean"),
                mean_distance_from_threshold=("distance_from_threshold", "mean"),
            )
            .reset_index()
        )
        print(grouped.to_string(index=False))

    print(f"\nSaved: {OUTPUT_FILE.resolve()}")
    print(
        "Important: these external errors may be inspected to understand failure modes, "
        "but the external set must not be used to choose a new threshold or model variant."
    )


if __name__ == "__main__":
    main()
