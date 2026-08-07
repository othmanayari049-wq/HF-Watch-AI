from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


INPUT = Path("results/external_validation_record_summary.csv")


def bootstrap_ci(y, pred, prob, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)

    values = {
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1": [],
        "auc": [],
    }

    n = len(y)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)

        y_b = y[idx]
        pred_b = pred[idx]
        prob_b = prob[idx]

        # AUC requires both classes
        if len(np.unique(y_b)) < 2:
            continue

        tn, fp, fn, tp = confusion_matrix(
            y_b,
            pred_b,
            labels=[0, 1]
        ).ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

        values["accuracy"].append(
            accuracy_score(y_b, pred_b)
        )

        values["sensitivity"].append(
            recall_score(
                y_b,
                pred_b,
                zero_division=0
            )
        )

        values["specificity"].append(
            specificity
        )

        values["precision"].append(
            precision_score(
                y_b,
                pred_b,
                zero_division=0
            )
        )

        values["f1"].append(
            f1_score(
                y_b,
                pred_b,
                zero_division=0
            )
        )

        values["auc"].append(
            roc_auc_score(y_b, prob_b)
        )

    cis = {}

    for name, vals in values.items():
        vals = np.asarray(vals)

        cis[name] = (
            np.percentile(vals, 2.5),
            np.percentile(vals, 97.5),
        )

    return cis


def main():
    df = pd.read_csv(INPUT)

    y = df["label"].astype(int).to_numpy()
    pred = df["record_prediction"].astype(int).to_numpy()
    prob = df["mean_chf_probability"].astype(float).to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y, pred)
    sensitivity = recall_score(
        y,
        pred,
        zero_division=0
    )

    specificity = tn / (tn + fp)

    precision = precision_score(
        y,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        pred,
        zero_division=0
    )

    auc = roc_auc_score(y, prob)

    print("======================================")
    print("EXTERNAL RECORD-LEVEL METRICS")
    print("======================================")

    print(f"Total records:   {len(df)}")
    print(f"Healthy records: {(y == 0).sum()}")
    print(f"CHF records:     {(y == 1).sum()}")

    print("\nConfusion matrix:")
    print([[tn, fp], [fn, tp]])

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    print("\nMetrics:")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"F1 score:     {f1:.4f}")
    print(f"ROC-AUC:      {auc:.4f}")

    print("\nCalculating bootstrap 95% confidence intervals...")

    cis = bootstrap_ci(
        y,
        pred,
        prob
    )

    print("\n95% bootstrap confidence intervals:")

    metrics = {
        "Accuracy": (accuracy, "accuracy"),
        "Sensitivity": (sensitivity, "sensitivity"),
        "Specificity": (specificity, "specificity"),
        "Precision": (precision, "precision"),
        "F1": (f1, "f1"),
        "ROC-AUC": (auc, "auc"),
    }

    for display, (value, key) in metrics.items():
        low, high = cis[key]

        print(
            f"{display:12} "
            f"{value:.4f} "
            f"(95% CI {low:.4f}-{high:.4f})"
        )


if __name__ == "__main__":
    main()