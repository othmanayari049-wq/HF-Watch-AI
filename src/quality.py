from __future__ import annotations

import numpy as np


RR_MIN_MS = 300.0
RR_MAX_MS = 2000.0
WARN_OUTLIER_FRACTION = 0.10
FAIL_OUTLIER_FRACTION = 0.20


def assess_window_quality(
    r_peaks: np.ndarray,
    sampling_rate: int,
    missing_fraction: float,
) -> dict[str, object]:
    """Return transparent ECG/RR quality indicators without altering model inputs."""
    r_peaks = np.asarray(r_peaks, dtype=int)
    rr_ms = np.diff(r_peaks) / float(sampling_rate) * 1000.0

    if rr_ms.size == 0:
        return {
            "status": "fail",
            "message": "No usable R-R intervals were detected.",
            "rr_outlier_fraction": 1.0,
            "rr_count": 0,
            "rr_min_ms": float("nan"),
            "rr_max_ms": float("nan"),
        }

    outliers = (rr_ms < RR_MIN_MS) | (rr_ms > RR_MAX_MS)
    outlier_fraction = float(np.mean(outliers))

    if missing_fraction > 0.05 or outlier_fraction > FAIL_OUTLIER_FRACTION:
        status = "fail"
        message = (
            "Signal quality is insufficient for a reliable research prediction. "
            "Review the ECG and R-peak detection."
        )
    elif outlier_fraction > WARN_OUTLIER_FRACTION:
        status = "warning"
        message = (
            "The window contains an elevated fraction of unusual R-R intervals. "
            "Interpret the research output cautiously."
        )
    else:
        status = "pass"
        message = "Basic ECG/R-R quality checks passed."

    return {
        "status": status,
        "message": message,
        "rr_outlier_fraction": outlier_fraction,
        "rr_count": int(rr_ms.size),
        "rr_min_ms": float(np.min(rr_ms)),
        "rr_max_ms": float(np.max(rr_ms)),
    }
