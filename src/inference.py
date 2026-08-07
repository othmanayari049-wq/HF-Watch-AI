from __future__ import annotations

from pathlib import Path

import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb

from quality import assess_window_quality


WINDOW_SECONDS = 300
MINIMUM_BEATS = 150


def extract_hrv_features(
    record_path: Path,
    channel: int = 0,
    start_seconds: int = 0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Extract HRV features and visualization metadata from one ECG window."""
    record = wfdb.rdrecord(str(record_path))
    sampling_rate = int(record.fs)

    if record.p_signal is None or record.p_signal.ndim != 2:
        raise ValueError("The selected WFDB record does not contain a readable ECG signal.")

    if channel < 0 or channel >= record.p_signal.shape[1]:
        raise ValueError(
            f"Channel {channel} is invalid. The record contains "
            f"{record.p_signal.shape[1]} channel(s)."
        )

    if start_seconds < 0:
        raise ValueError("start_seconds cannot be negative.")

    start_sample = int(start_seconds * sampling_rate)
    end_sample = start_sample + int(WINDOW_SECONDS * sampling_rate)

    if start_sample >= len(record.p_signal):
        raise ValueError("The selected start time is beyond the end of the recording.")

    if end_sample > len(record.p_signal):
        available_seconds = max(0.0, (len(record.p_signal) - start_sample) / sampling_rate)
        raise ValueError(
            "A complete 5-minute window is not available from the selected start time. "
            f"Only {available_seconds:.1f} seconds remain."
        )

    raw_segment = record.p_signal[start_sample:end_sample, channel].astype(float)
    missing_fraction = float(np.isnan(raw_segment).mean())
    if missing_fraction > 0.05:
        raise ValueError(
            f"The ECG window contains {missing_fraction:.1%} missing values; "
            "the allowed maximum is 5%."
        )

    segment = pd.Series(raw_segment).interpolate(limit_direction="both").to_numpy(dtype=float)
    cleaned = nk.ecg_clean(segment, sampling_rate=sampling_rate)
    _, info = nk.ecg_process(cleaned, sampling_rate=sampling_rate)

    r_peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    beat_count = len(r_peaks)
    if beat_count < MINIMUM_BEATS:
        raise ValueError(
            f"Only {beat_count} beats were detected; at least {MINIMUM_BEATS} are required."
        )

    quality = assess_window_quality(r_peaks, sampling_rate, missing_fraction)
    if quality["status"] == "fail":
        raise ValueError(str(quality["message"]))

    hrv = nk.hrv(info, sampling_rate=sampling_rate, show=False)
    rr_ms = np.diff(r_peaks) / sampling_rate * 1000.0
    mean_hr_bpm = float(60_000.0 / np.mean(rr_ms)) if len(rr_ms) else float("nan")

    signal_name = (
        record.sig_name[channel]
        if record.sig_name and channel < len(record.sig_name)
        else f"Channel {channel}"
    )

    metadata: dict[str, object] = {
        "sampling_rate": sampling_rate,
        "detected_beats": beat_count,
        "window_seconds": WINDOW_SECONDS,
        "channel": channel,
        "signal_name": signal_name,
        "start_seconds": start_seconds,
        "missing_fraction": missing_fraction,
        "mean_hr_bpm": mean_hr_bpm,
        "raw_signal": segment,
        "cleaned_signal": np.asarray(cleaned, dtype=float),
        "r_peaks": r_peaks,
        "quality": quality,
    }
    return hrv, metadata


def _logistic_contributions(pipeline, X: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    """Compute exact per-feature contributions to the logistic-regression logit."""
    try:
        imputed = pipeline.named_steps["imputer"].transform(X)
        scaled = pipeline.named_steps["scaler"].transform(imputed)
        model = pipeline.named_steps["model"]
        coefficients = np.asarray(model.coef_[0], dtype=float)
        contributions = np.asarray(scaled[0], dtype=float) * coefficients
        return {name: float(value) for name, value in zip(feature_columns, contributions)}
    except Exception:
        return {}


def predict_wfdb_record(
    record_path: Path,
    model_path: Path,
    channel: int = 0,
    start_seconds: int = 0,
) -> dict[str, object]:
    """Run the experimental classifier on one 5-minute WFDB ECG window."""
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

    missing_features = [feature for feature in feature_columns if feature not in hrv.columns]
    if missing_features:
        raise ValueError(f"Missing required HRV features: {missing_features}")

    X = hrv[feature_columns]
    prediction = int(pipeline.predict(X)[0])
    probability = float(pipeline.predict_proba(X)[0, 1])
    contributions = _logistic_contributions(pipeline, X, feature_columns)

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
        "feature_contributions": contributions,
    }
