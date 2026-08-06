from pathlib import Path

import joblib
import neurokit2 as nk
import pandas as pd
import wfdb


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "hf_watch_top20_model.joblib"
CHF_DATA_DIR = PROJECT_ROOT / "data" / "chfdb" / "files"

RECORD_ID = "chf01"
WINDOW_SECONDS = 300


def main() -> None:
    saved = joblib.load(MODEL_PATH)

    pipeline = saved["pipeline"]
    feature_columns = saved["feature_columns"]

    record = wfdb.rdrecord(str(CHF_DATA_DIR / RECORD_ID))

    sampling_rate = int(record.fs)
    window_samples = sampling_rate * WINDOW_SECONDS

    ecg = record.p_signal[:window_samples, 0]

    cleaned = nk.ecg_clean(
        ecg,
        sampling_rate=sampling_rate,
    )

    _, info = nk.ecg_process(
        cleaned,
        sampling_rate=sampling_rate,
    )

    hrv = nk.hrv(
        info,
        sampling_rate=sampling_rate,
        show=False,
    )

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in hrv.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    X = hrv[feature_columns]

    prediction = int(pipeline.predict(X)[0])
    probability = float(pipeline.predict_proba(X)[0, 1])

    print("Record:", RECORD_ID)
    print("Prediction:", prediction)
    print(f"CHF probability: {probability:.4f}")

    if prediction == 1:
        print("Experimental result: CHF-like pattern detected.")
    else:
        print("Experimental result: Healthy-like pattern detected.")


if __name__ == "__main__":
    main()