from __future__ import annotations

import argparse
from pathlib import Path

from inference import predict_wfdb_record


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hf_watch_top20_model.joblib"


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
    result = predict_wfdb_record(
        record_path=args.record_path,
        model_path=args.model_path,
        channel=args.channel,
        start_seconds=args.start_seconds,
    )

    print("\nHF-Watch-AI experimental result")
    print("Record:", result["record_path"])
    print("Signal:", result["signal_name"])
    print("Sampling rate:", result["sampling_rate"], "Hz")
    print("Detected beats:", result["detected_beats"])
    print(f"Mean heart rate: {result['mean_hr_bpm']:.1f} bpm")
    print("Prediction:", result["prediction"])
    print(f"CHF-like probability: {result['chf_probability']:.4f}")
    print("Interpretation:", result["interpretation"])
    print("\nResearch prototype only — not a clinical diagnosis.")


if __name__ == "__main__":
    main()
