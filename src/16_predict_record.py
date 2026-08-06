from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hf_watch_top20_model.joblib"
WINDOW_SECONDS = 300


def extract_hrv_window(
    record_path: Path,
    channel: int,
    start_seconds: int,
) -> tuple[pd.DataFrame, int, int]:
    record = wfdb.rdrecord(str(record_path))
    sampling_rate = int(record.fs)

    if channel < 0 or channel >= record.p_signal.shape[1]:
        raise ValueError(
            f"Channel {channel} is invalid. Record has "
            f"{record.p_signal.shape[1]} channel(s)."
        )

    start_sample = int(start_seconds * sampling_rate)
    end_sample = start_sample + int(WINDOW_SECONDS * sampling_rate)

    if end_sample > len(record.p_signal):
        raise ValueError(
            "The selected record does not contain a complete 5-minute "
            "window from the requested start time."
        )

    segment = record.p_signal[start_sample:end_sample, channel]

    if np.isnan(segment).mean() > 0.05:
        raise ValueError("More than 5% of the ECG window is missing.")

    segment = (
        pd.Series(segment)
        .interpolate(limit_direction="both")
        .to_numpy()
    )

    cleaned = nk.ecg_clean(segment, sampling_rate=sampling_rate)
    _, info = nk.ecg_process(cleaned, sampling_rate=sampling_rate)

    beat_count = len(info["ECG_R_Peaks"])
    if beat_count < 150:
        raise ValueError(
            f"Only {beat_count} beats were detected; at least 150 are required."
        )

    hrv = nk.hrv(info, sampling_rate=sampling_rate, show=False)
    return hrv, sampling_rate, beat_count


def predict_record(
    record_path: Path,
    model_path: Path,
    channel: int,
    start_seconds: int,
) -> dict[str, object]:
    saved = joblib.load(model_path)
    pipeline = saved["pipeline"]
    feature_columns = saved["feature_columns"]

    hrv, sampling_rate, beat_count = extract_hrv_window(
        record_path=record_path,
        channel=channel,
        start_seconds=start_seconds,
    )

    missing_features = [
        feature for feature in feature_columns if feature not in hrv.columns
    ]
    if missing_features:
        raise ValueError(f"Missing required HRV features: {missing_features}")

    X = hrv[feature_columns]
    prediction = int(pipeline.predict(X)[0])
    probability = float(pipeline.predict_proba(X)[0, 1])

    return {
        "record_path": str(record_path),
        "sampling_rate": sampling_rate,
        "channel": channel,
        "start_seconds": start_seconds,
        "window_seconds": WINDOW_SECONDS,
        "detected_beats": beat_count,
        "prediction": prediction,
        "chf_probability": probability,
        "interpretation": (
            "CHF-like HRV pattern" if prediction == 1 else "Healthy-like HRV pattern"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the experimental HF-Watch-AI model on one 5-minute "
            "window from a WFDB ECG record."
        )
    )
    parser.add_argument(
        "record_path",
        type=Path,
        help="WFDB record path without .dat or .hea, for example data/chfdb/files/chf01",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained joblib model.",
    )
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--start-seconds", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_record(
        record_path=args.record_path,
        model_path=args.model_path,
        channel=args.channel,
        start_seconds=args.start_seconds,
    )

    print("\nHF-Watch-AI experimental result")
    print("Record:", result["record_path"])
    print("Sampling rate:", result["sampling_rate"], "Hz")
    print("Detected beats:", result["detected_beats"])
    print("Prediction:", result["prediction"])
    print(f"CHF probability: {result['chf_probability']:.4f}")
    print("Interpretation:", result["interpretation"])
    print("\nResearch prototype only — not a clinical diagnosis.")


if __name__ == "__main__":
    main()
