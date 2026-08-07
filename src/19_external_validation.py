from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


FEATURE_FILE = Path("features/external_rr_features.csv")
MODEL_FILE = Path("models/hf_watch_top20_model.joblib")
RESULT_FILE = Path("results/external_validation_predictions.csv")


def main():
    print("Loading external features...")
    df = pd.read_csv(FEATURE_FILE)

    print("Loading trained model...")
    bundle = joblib.load(MODEL_FILE)

    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    print("Expected features:", len(feature_columns))

    missing = [col for col in feature_columns if col not in df.columns]

    if missing:
        print("\nERROR: Missing model features:")
        for col in missing:
            print(" -", col)
        return

    X = df[feature_columns].copy()
    y = df["label"].astype(int)

    print("External samples:", len(df))
    print("CHF windows:", int((y == 1).sum()))
    print("Healthy windows:", int((y == 0).sum()))

    print("\nRunning external predictions...")

    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    sensitivity = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    auc = roc_auc_score(y, probabilities)

    cm = confusion_matrix(y, predictions)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    print("\n======================================")
    print("EXTERNAL VALIDATION RESULTS")
    print("======================================")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"F1 score:     {f1:.4f}")
    print(f"ROC-AUC:      {auc:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    result_df = df[
        ["record", "dataset", "label", "window_start_sec", "n_beats"]
    ].copy()

    result_df["predicted_label"] = predictions
    result_df["chf_probability"] = probabilities

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(RESULT_FILE, index=False)

    print("\n======================================")
    print("PER-RECORD SUMMARY")
    print("======================================")

    summary = (
        result_df
        .groupby(["record", "dataset", "label"])
        .agg(
            windows=("predicted_label", "size"),
            mean_chf_probability=("chf_probability", "mean"),
            predicted_chf_fraction=("predicted_label", "mean"),
        )
        .reset_index()
    )

    summary["record_prediction"] = (
        summary["predicted_chf_fraction"] >= 0.5
    ).astype(int)

    print(summary.to_string(index=False))

    record_accuracy = accuracy_score(
        summary["label"],
        summary["record_prediction"]
    )

    record_auc = roc_auc_score(
        summary["label"],
        summary["mean_chf_probability"]
    )

    print("\n======================================")
    print("RECORD-LEVEL RESULTS")
    print("======================================")
    print(f"Record accuracy: {record_accuracy:.4f}")
    print(f"Record ROC-AUC:  {record_auc:.4f}")

    record_cm = confusion_matrix(
        summary["label"],
        summary["record_prediction"]
    )

    print("\nRecord confusion matrix:")
    print(record_cm)

    summary_path = Path("results/external_validation_record_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nSaved window predictions to:")
    print(RESULT_FILE)

    print("\nSaved record summary to:")
    print(summary_path)

    print("\nIMPORTANT:")
    print(
        "This is external database validation of an experimental research "
        "classifier. It is not clinical validation or a diagnostic test."
    )


if __name__ == "__main__":
    main()