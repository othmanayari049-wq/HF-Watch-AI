from __future__ import annotations

from pathlib import Path

import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb


WINDOW_SECONDS = 300
MINIMUM_BEATS = 150


def extract_hrv_features(
    record_path: Path,
    channel: int = 0,
    start_seconds: int = 0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    record = wfdb.rdrecord(str(record_path))
    sampling_rate = int(record.fs)

    if channel < 0 or channel >= record.p_signal.shape[1]:
        raise ValueError(
            f"Channel {channel} is invalid. The record contains "
            f"{record.p_signal.shape[1]} channel(s)."
        )

    start_sample = int(start_seconds * sampling_rate)
    end_sample = start_sample + int(WINDOW_SECONDS * sampling_rate)

    if start_sample < 0:
        raise ValueError("start_seconds cannot be negative.")

    if end_sample > len(record.p_signal):
        raise ValueError(
            "A complete 5-minute window is not available from the selected start time."
        )

    segment = record.p_signal[start_sample:end_sample, channel]

    missing_fraction = float(np.isnan(segment).mean())
    if missing_fraction > 0.05:
        raise ValueError(
            f"The ECG window contains {missing_fraction:.1%} missing values."
        )

    segment = (
        pd.Series(segment)
        .interpolate(limit_direction="both")
        .to_numpy()
    )

    cleaned = nk.ecg_clean(segment, sampling_rate=sampling_rate)
    _, info = nk.ecg_process(cleaned, sampling_rate=sampling_rate)

    beat_count = len(info["ECG_R_Peaks"])
    if beat_count < MINIMUM_BEATS:
        raise ValueError(
            f"Only {beat_count} beats were detected; at least {MINIMUM_BEATS} are required."
        )

    hrv = nk.hrv(info, sampling_rate=sampling_rate, show=False)
    metadata = {
        "sampling_rate": sampling_rate,
        "detected_beats": beat_count,
        "window_seconds": WINDOW_SECONDS,
        "channel": channel,
        "start_seconds": start_seconds,
    }
    return hrv, metadata


def predict_wfdb_record(
    record_path: Path,
    model_path: Path,
    channel: int = 0,
    start_seconds: int = 0,
) -> dict[str, object]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    saved = joblib.load(model_path)
    pipeline = saved["pipeline"]
    feature_columns = saved["feature_columns"]

    hrv, metadata = extract_hrv_features(
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
        **metadata,
        "record_path": str(record_path),
        "prediction": prediction,
        "chf_probability": probability,
        "interpretation": (
            "CHF-like HRV pattern" if prediction == 1 else "Healthy-like HRV pattern"
        ),
        "selected_features": feature_columns,
        "feature_values": X.iloc[0].to_dict(),
    }
