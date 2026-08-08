from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("results/external_validation_predictions.csv")
OUTPUT = Path("results/external_consistency_audit.csv")


def consistency_label(agreement: float) -> str:
    """Descriptive presentation heuristic only; not a clinical threshold."""
    if agreement >= 0.90:
        return "High"
    if agreement >= 0.75:
        return "Moderate"
    return "Low / mixed"


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run `python src/19_external_validation.py` first."
        )

    df = pd.read_csv(INPUT)

    required = {
        "record",
        "dataset",
        "label",
        "predicted_label",
        "chf_probability",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    rows = []
    for (record, dataset, label), g in df.groupby(["record", "dataset", "label"]):
        probs = g["chf_probability"].astype(float).to_numpy()
        preds = g["predicted_label"].astype(int).to_numpy()
        chf_fraction = float(preds.mean())
        record_prediction = int(chf_fraction >= 0.5)
        agreement = chf_fraction if record_prediction == 1 else 1.0 - chf_fraction

        if label == 0 and record_prediction == 0:
            outcome = "TN"
        elif label == 0 and record_prediction == 1:
            outcome = "FP"
        elif label == 1 and record_prediction == 0:
            outcome = "FN"
        else:
            outcome = "TP"

        rows.append(
            {
                "record": record,
                "dataset": dataset,
                "label": int(label),
                "record_prediction": record_prediction,
                "outcome": outcome,
                "windows": int(len(g)),
                "mean_chf_probability": float(np.mean(probs)),
                "median_chf_probability": float(np.median(probs)),
                "predicted_chf_fraction": chf_fraction,
                "window_agreement": agreement,
                "pattern_consistency": consistency_label(agreement),
                "score_std": float(np.std(probs, ddof=0)),
                "score_min": float(np.min(probs)),
                "score_max": float(np.max(probs)),
                "score_range": float(np.max(probs) - np.min(probs)),
            }
        )

    out = pd.DataFrame(rows).sort_values(["outcome", "window_agreement", "record"])

    print("HF-Watch-AI external consistency audit")
    print("Descriptive audit only. The external set is not used to choose model, threshold, or consistency cutoffs.\n")

    summary = (
        out.groupby(["outcome", "pattern_consistency"])
        .size()
        .rename("records")
        .reset_index()
    )
    print("Consistency counts by record outcome:")
    print(summary.to_string(index=False))

    print("\nMisclassified records with consistency information:")
    wrong = out[out["outcome"].isin(["FP", "FN"])].copy()
    cols = [
        "record",
        "dataset",
        "outcome",
        "windows",
        "mean_chf_probability",
        "predicted_chf_fraction",
        "window_agreement",
        "pattern_consistency",
        "score_std",
        "score_min",
        "score_max",
    ]
    print(wrong[cols].to_string(index=False))

    print("\nMean window agreement by outcome:")
    means = out.groupby("outcome")["window_agreement"].mean().sort_index()
    for name, value in means.items():
        print(f"  {name}: {value:.3f}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT.resolve()}")
    print(
        "Important: High/Moderate/Low are presentation heuristics only. "
        "This audit does not validate them as uncertainty thresholds."
    )


if __name__ == "__main__":
    main()
