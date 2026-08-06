from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "features"

CHF_PATH = FEATURES_DIR / "chf_5min_windows.csv"
HEALTHY_PATH = FEATURES_DIR / "healthy_5min_windows.csv"
OUTPUT_PATH = FEATURES_DIR / "training_dataset.csv"

RANDOM_STATE = 42


def main() -> None:
    chf = pd.read_csv(CHF_PATH)
    healthy = pd.read_csv(HEALTHY_PATH)

    print("CHF shape:", chf.shape)
    print("Healthy shape:", healthy.shape)

    # Balance the classes by keeping all CHF windows
    # and randomly sampling the same number of healthy windows.
    healthy_balanced = healthy.sample(
        n=len(chf),
        random_state=RANDOM_STATE,
        replace=False,
    )

    combined = pd.concat(
        [chf, healthy_balanced],
        ignore_index=True,
    )

    combined = combined.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    # Replace infinite values with missing values.
    combined = combined.replace([np.inf, -np.inf], np.nan)

    combined.to_csv(OUTPUT_PATH, index=False)

    print("\nFinal shape:", combined.shape)
    print("\nClass counts:")
    print(combined["label"].value_counts().sort_index())
    print("\nPatients per class:")
    print(combined.groupby("label")["patient_id"].nunique())
    print("\nMissing values:", combined.isna().sum().sum())
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()