from __future__ import annotations

from pathlib import Path
import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nsrdb"
FEATURES_DIR = PROJECT_ROOT / "features"

WINDOW_SECONDS = 300
MINIMUM_BEATS = 150

RECORDS = [
    "16265", "16272", "16273", "16420", "16483", "16539",
    "16773", "16786", "16795", "17052", "17453", "18177",
    "18184", "19088", "19090", "19093", "19140", "19830",
]


def process_record(record_id: str) -> pd.DataFrame:
    record = wfdb.rdrecord(str(DATA_DIR / record_id))

    sampling_rate = int(record.fs)
    ecg = record.p_signal[:, 0]

    window_samples = sampling_rate * WINDOW_SECONDS
    number_of_windows = len(ecg) // window_samples

    rows: list[pd.DataFrame] = []

    for window_index in tqdm(
        range(number_of_windows),
        desc=record_id,
        unit="window",
    ):
        start = window_index * window_samples
        end = start + window_samples

        segment = ecg[start:end]

        try:
            if np.isnan(segment).mean() > 0.05:
                continue

            segment = (
                pd.Series(segment)
                .interpolate(limit_direction="both")
                .to_numpy()
            )

            cleaned = nk.ecg_clean(
                segment,
                sampling_rate=sampling_rate,
            )

            _, info = nk.ecg_process(
                cleaned,
                sampling_rate=sampling_rate,
            )

            number_of_beats = len(info["ECG_R_Peaks"])

            if number_of_beats < MINIMUM_BEATS:
                continue

            features = nk.hrv(
                info,
                sampling_rate=sampling_rate,
                show=False,
            )

            features.insert(0, "patient_id", record_id)
            features.insert(1, "window_index", window_index)
            features.insert(2, "start_seconds", start / sampling_rate)
            features.insert(3, "number_of_beats", number_of_beats)
            features.insert(4, "label", 0)

            rows.append(features)

        except Exception as error:
            print(
                f"\nSkipped {record_id}, window {window_index}: {error}"
            )

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Healthy data folder:", DATA_DIR)
    print("Features folder:", FEATURES_DIR)

    all_files: list[Path] = []

    for record_id in RECORDS:
        output_path = FEATURES_DIR / f"healthy_{record_id}_5min_windows.csv"

        if output_path.exists():
            print(f"Skipping {record_id}: already processed")
            all_files.append(output_path)
            continue

        print(f"\nProcessing healthy record {record_id}")

        try:
            features = process_record(record_id)

            if features.empty:
                print(f"No valid windows for {record_id}")
                continue

            features.to_csv(output_path, index=False)
            all_files.append(output_path)

            print(
                f"Saved {len(features)} valid windows "
                f"to {output_path.name}"
            )

        except Exception as error:
            print(f"Failed to process {record_id}: {error}")

    combined_frames = [
        pd.read_csv(path)
        for path in all_files
        if path.exists()
    ]

    if not combined_frames:
        print("No healthy feature files were created.")
        return

    combined = pd.concat(combined_frames, ignore_index=True)

    combined_path = FEATURES_DIR / "healthy_5min_windows.csv"
    combined.to_csv(combined_path, index=False)

    print("\nFinished.")
    print("Combined healthy shape:", combined.shape)
    print("Saved to:", combined_path)


if __name__ == "__main__":
    main()