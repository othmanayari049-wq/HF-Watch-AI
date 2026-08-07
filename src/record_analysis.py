from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from inference import WINDOW_SECONDS, predict_wfdb_record


def get_record_duration_seconds(record_path: Path) -> float:
    record = wfdb.rdrecord(str(record_path), sampto=1)
    header = wfdb.rdheader(str(record_path))
    return float(header.sig_len / header.fs)


def analyze_full_record(
    record_path: Path,
    model_path: Path,
    channel: int = 0,
    step_seconds: int = WINDOW_SECONDS,
    max_windows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Analyze consecutive complete 5-minute windows across a WFDB record."""
    header = wfdb.rdheader(str(record_path))
    duration_seconds = float(header.sig_len / header.fs)

    starts = list(
        range(
            0,
            max(0, int(duration_seconds) - WINDOW_SECONDS + 1),
            int(step_seconds),
        )
    )
    if max_windows is not None:
        starts = starts[:max_windows]

    rows: list[dict[str, object]] = []
    for start in starts:
        try:
            result = predict_wfdb_record(
                record_path=record_path,
                model_path=model_path,
                channel=channel,
                start_seconds=start,
            )
            rows.append(
                {
                    "window_start_sec": start,
                    "window_end_sec": start + WINDOW_SECONDS,
                    "chf_probability": float(result["chf_probability"]),
                    "prediction": int(result["prediction"]),
                    "detected_beats": int(result["detected_beats"]),
                    "mean_hr_bpm": float(result["mean_hr_bpm"]),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "window_start_sec": start,
                    "window_end_sec": start + WINDOW_SECONDS,
                    "chf_probability": np.nan,
                    "prediction": np.nan,
                    "detected_beats": np.nan,
                    "mean_hr_bpm": np.nan,
                    "status": "skipped",
                    "error": str(exc),
                }
            )

    df = pd.DataFrame(rows)
    valid = df[df["status"] == "ok"].copy() if not df.empty else pd.DataFrame()

    if valid.empty:
        summary = {
            "valid_windows": 0,
            "skipped_windows": len(df),
            "mean_chf_probability": float("nan"),
            "median_chf_probability": float("nan"),
            "chf_like_window_fraction": float("nan"),
            "record_prediction": None,
            "record_interpretation": "No valid windows",
        }
        return df, summary

    fraction = float(valid["prediction"].mean())
    record_prediction = int(fraction >= 0.5)
    summary = {
        "valid_windows": int(len(valid)),
        "skipped_windows": int(len(df) - len(valid)),
        "mean_chf_probability": float(valid["chf_probability"].mean()),
        "median_chf_probability": float(valid["chf_probability"].median()),
        "chf_like_window_fraction": fraction,
        "record_prediction": record_prediction,
        "record_interpretation": (
            "CHF-like record pattern" if record_prediction == 1 else "Healthy-like record pattern"
        ),
    }
    return df, summary
