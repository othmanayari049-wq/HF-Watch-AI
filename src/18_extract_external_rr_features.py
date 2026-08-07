from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk


WINDOW_SECONDS = 300
MIN_BEATS = 120

DATASETS = [
    ("data/chf2db", "chf", 1),
    ("data/nsr2db", "nsr", 0),
]


def extract_windows(record_path: Path, label_name: str, label: int):
    header = wfdb.rdheader(str(record_path))
    ann = wfdb.rdann(str(record_path), "ecg")

    fs = header.fs
    samples = np.asarray(ann.sample)
    symbols = np.asarray(ann.symbol)

    # Keep normal beats only
    normal_samples = samples[symbols == "N"]

    if len(normal_samples) < MIN_BEATS:
        return []

    start_sample = normal_samples[0]
    end_sample = normal_samples[-1]

    window_samples = int(WINDOW_SECONDS * fs)

    rows = []

    for window_start in range(start_sample, end_sample - window_samples, window_samples):
        window_end = window_start + window_samples

        beats = normal_samples[
            (normal_samples >= window_start)
            & (normal_samples < window_end)
        ]

        if len(beats) < MIN_BEATS:
            continue

        relative_beats = beats - window_start

        try:
            hrv = nk.hrv(relative_beats, sampling_rate=fs, show=False)

            row = hrv.iloc[0].to_dict()

            row["record"] = record_path.name
            row["dataset"] = label_name
            row["label"] = label
            row["window_start_sec"] = window_start / fs
            row["n_beats"] = len(beats)

            rows.append(row)

        except Exception as exc:
            print(
                f"Skipped {record_path.name} "
                f"window {window_start / fs:.0f}s: {exc}"
            )

    return rows


def process_dataset(folder: str, prefix: str, label: int):
    folder_path = Path(folder)

    records = sorted(
        p.with_suffix("")
        for p in folder_path.glob("*.hea")
        if p.stem.startswith(prefix)
    )

    all_rows = []

    print(f"\nProcessing {folder}")
    print(f"Records found: {len(records)}")

    for i, record in enumerate(records, start=1):
        print(f"[{i}/{len(records)}] {record.name}")

        rows = extract_windows(record, prefix, label)
        all_rows.extend(rows)

        print(f"    Valid windows: {len(rows)}")

    return all_rows


def main():
    output_rows = []

    for folder, prefix, label in DATASETS:
        rows = process_dataset(folder, prefix, label)
        output_rows.extend(rows)

    df = pd.DataFrame(output_rows)

    output_path = Path("features/external_rr_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\n======================================")
    print("EXTERNAL RR FEATURE EXTRACTION FINISHED")
    print("======================================")
    print("Shape:", df.shape)

    if not df.empty:
        print("\nClass counts:")
        print(df["label"].value_counts())

        print("\nDataset counts:")
        print(df["dataset"].value_counts())

        print("\nUnique records:")
        print(df.groupby("dataset")["record"].nunique())

    print("\nSaved to:")
    print(output_path)


if __name__ == "__main__":
    main()