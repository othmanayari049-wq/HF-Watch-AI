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
DATA_DIR = PROJECT_ROOT / "data" / "chfdb" / "files"
FEATURES_DIR = PROJECT_ROOT / "features"

WINDOW_SECONDS = 300
MINIMUM_BEATS = 150


def process_patient(patient_id: str) -> pd.DataFrame:
    record_path = DATA_DIR / patient_id
    record = wfdb.rdrecord(str(record_path))

    sampling_rate = int(record.fs)
    ecg = record.p_signal[:, 0]

    window_samples = sampling_rate * WINDOW_SECONDS
    number_of_windows = len(ecg) // window_samples

    patient_rows: list[pd.DataFrame] = []

    for window_index in tqdm(
        range(number_of_windows),
        desc=patient_id,
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

            features.insert(0, "patient_id", patient_id)
            features.insert(1, "window_index", window_index)
            features.insert(2, "start_seconds", start / sampling_rate)
            features.insert(3, "number_of_beats", number_of_beats)
            features.insert(4, "label", 1)

            patient_rows.append(features)

        except Exception as error:
            print(
                f"\nSkipped {patient_id}, window {window_index}: {error}"
            )

    if not patient_rows:
        return pd.DataFrame()

    return pd.concat(patient_rows, ignore_index=True)


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Data folder:", DATA_DIR)
    print("Features folder:", FEATURES_DIR)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}"
        )

    all_patient_files: list[Path] = []

    for patient_number in range(1, 16):
        patient_id = f"chf{patient_number:02d}"
        output_path = FEATURES_DIR / f"{patient_id}_5min_windows.csv"

        if output_path.exists():
            print(f"Skipping {patient_id}: already processed")
            all_patient_files.append(output_path)
            continue

        print(f"\nProcessing {patient_id}")

        try:
            patient_features = process_patient(patient_id)

            if patient_features.empty:
                print(f"No valid windows found for {patient_id}")
                continue

            patient_features.to_csv(output_path, index=False)
            all_patient_files.append(output_path)

            print(
                f"Saved {len(patient_features)} valid windows "
                f"to {output_path.name}"
            )

        except Exception as error:
            print(f"Failed to process {patient_id}: {error}")

    if not all_patient_files:
        print("No patient feature files were created.")
        return

    combined_frames = [
        pd.read_csv(file_path)
        for file_path in all_patient_files
        if file_path.exists()
    ]

    combined_dataset = pd.concat(
        combined_frames,
        ignore_index=True,
    )

    combined_path = FEATURES_DIR / "chf_5min_windows.csv"
    combined_dataset.to_csv(combined_path, index=False)

    print("\nFinished.")
    print("Combined shape:", combined_dataset.shape)
    print("Saved combined dataset to:", combined_path)


if __name__ == "__main__":
    main()