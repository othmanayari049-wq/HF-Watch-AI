from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "features"

INPUT_PATH = FEATURES_DIR / "training_dataset.csv"
OUTPUT_PATH = FEATURES_DIR / "training_dataset_clean.csv"


def main() -> None:
    data = pd.read_csv(INPUT_PATH)

    print("Original shape:", data.shape)
    print("Original missing values:", data.isna().sum().sum())

    data = data.replace([np.inf, -np.inf], np.nan)

    completely_missing = data.columns[data.isna().all()].tolist()

    print("\nCompletely missing columns:")
    for column in completely_missing:
        print("-", column)

    data = data.drop(columns=completely_missing)

    metadata_columns = [
        "patient_id",
        "window_index",
        "start_seconds",
        "number_of_beats",
        "label",
    ]

    feature_columns = [
        column
        for column in data.columns
        if column not in metadata_columns
    ]

    missing_percentage = (
        data[feature_columns].isna().mean() * 100
    ).sort_values(ascending=False)

    high_missing_columns = missing_percentage[
        missing_percentage > 30
    ].index.tolist()

    print("\nColumns with more than 30% missing:")
    for column in high_missing_columns:
        print(
            f"- {column}: "
            f"{missing_percentage[column]:.2f}%"
        )

    data = data.drop(columns=high_missing_columns)

    data.to_csv(OUTPUT_PATH, index=False)

    print("\nCleaned shape:", data.shape)
    print("Remaining missing values:", data.isna().sum().sum())
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()