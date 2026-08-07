from __future__ import annotations

import argparse
from pathlib import Path

from record_analysis import analyze_full_record


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hf_watch_top20_model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze consecutive 5-minute windows from one WFDB ECG record.")
    parser.add_argument("record_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--step-seconds", type=int, default=300)
    parser.add_argument("--max-windows", type=int, default=0, help="0 means all complete windows")
    args = parser.parse_args()

    windows, summary = analyze_full_record(
        record_path=args.record_path,
        model_path=args.model_path,
        channel=args.channel,
        step_seconds=args.step_seconds,
        max_windows=None if args.max_windows == 0 else args.max_windows,
    )

    print("\nHF-Watch-AI full-record research analysis")
    print("Valid windows:", summary["valid_windows"])
    print("Skipped windows:", summary["skipped_windows"])
    print(f"Mean CHF-like probability: {summary['mean_chf_probability']:.4f}")
    print(f"Median CHF-like probability: {summary['median_chf_probability']:.4f}")
    print(f"CHF-like window fraction: {summary['chf_like_window_fraction']:.4f}")
    print("Record interpretation:", summary["record_interpretation"])

    out = PROJECT_ROOT / "results" / "full_record_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(out, index=False)
    print("Saved:", out)
    print("\nResearch prototype only — not a clinical diagnosis.")


if __name__ == "__main__":
    main()
