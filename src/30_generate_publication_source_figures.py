from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import predict_wfdb_record  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "publication_figures"
MODEL_PATH = PROJECT_ROOT / "models" / "hf_watch_top20_model.joblib"
CHF_RECORD = PROJECT_ROOT / "data" / "chfdb" / "files" / "chf01"
NSR_RECORD = PROJECT_ROOT / "data" / "nsrdb" / "16265"
WINDOW_PRED = PROJECT_ROOT / "results" / "external_validation_predictions.csv"
RECORD_PRED = PROJECT_ROOT / "results" / "external_validation_record_summary.csv"
NESTED_FREQ = PROJECT_ROOT / "results" / "nested_feature_selection_feature_frequency.csv"


def save_figure(fig: plt.Figure, stem: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    png = RESULTS_DIR / f"{stem}.png"
    pdf = RESULTS_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def ecg_and_rr_examples() -> None:
    chf = predict_wfdb_record(CHF_RECORD, MODEL_PATH, channel=0, start_seconds=0)
    nsr = predict_wfdb_record(NSR_RECORD, MODEL_PATH, channel=0, start_seconds=0)

    examples = [("CHF development example", chf), ("NSR development example", nsr)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    for row, (title, result) in enumerate(examples):
        fs = int(result["sampling_rate"])
        cleaned = np.asarray(result["cleaned_signal"], dtype=float)
        peaks = np.asarray(result["r_peaks"], dtype=int)

        n15 = min(len(cleaned), 15 * fs)
        t = np.arange(n15) / fs
        visible_peaks = peaks[peaks < n15]

        ax = axes[row, 0]
        ax.plot(t, cleaned[:n15], linewidth=0.9)
        ax.scatter(visible_peaks / fs, cleaned[visible_peaks], s=16, zorder=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ECG amplitude")
        ax.set_title(
            f"{title}: ECG and detected R peaks\n"
            f"score={float(result['chf_probability']):.3f}, beats={int(result['detected_beats'])}"
        )

        rr_ms = np.diff(peaks) / fs * 1000.0
        rr_t = peaks[1:] / fs
        ax = axes[row, 1]
        ax.plot(rr_t, rr_ms, linewidth=0.9)
        ax.set_xlabel("Time within 5-min window (s)")
        ax.set_ylabel("R–R interval (ms)")
        ax.set_title(f"{title}: R–R interval series")

    fig.suptitle("Representative raw-ECG pipeline outputs used by HF-Watch-AI", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "figure_ecg_rr_examples")


def external_discrimination_curves() -> None:
    windows = pd.read_csv(WINDOW_PRED)
    records = pd.read_csv(RECORD_PRED)

    y_w = windows["label"].astype(int).to_numpy()
    p_w = windows["chf_probability"].astype(float).to_numpy()
    y_r = records["label"].astype(int).to_numpy()
    p_r = records["mean_chf_probability"].astype(float).to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    fpr, tpr, _ = roc_curve(y_w, p_w)
    axes[0, 0].plot(fpr, tpr, linewidth=1.5)
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[0, 0].set_xlabel("False-positive rate")
    axes[0, 0].set_ylabel("True-positive rate")
    axes[0, 0].set_title(f"Window-level ROC (AUC={roc_auc_score(y_w, p_w):.3f})")

    precision, recall, _ = precision_recall_curve(y_w, p_w)
    axes[0, 1].plot(recall, precision, linewidth=1.5)
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title(
        f"Window-level precision–recall (AP={average_precision_score(y_w, p_w):.3f})"
    )

    fpr, tpr, _ = roc_curve(y_r, p_r)
    axes[1, 0].plot(fpr, tpr, linewidth=1.5)
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("False-positive rate")
    axes[1, 0].set_ylabel("True-positive rate")
    axes[1, 0].set_title(f"Record-level ROC (AUC={roc_auc_score(y_r, p_r):.3f})")

    precision, recall, _ = precision_recall_curve(y_r, p_r)
    axes[1, 1].plot(recall, precision, linewidth=1.5)
    axes[1, 1].set_xlabel("Recall")
    axes[1, 1].set_ylabel("Precision")
    axes[1, 1].set_title(
        f"Record-level precision–recall (AP={average_precision_score(y_r, p_r):.3f})"
    )

    fig.suptitle("Frozen external cross-database discrimination", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "figure_external_roc_pr")


def nested_feature_stability() -> None:
    freq = pd.read_csv(NESTED_FREQ)
    required = {"feature", "selection_count", "selection_fraction_of_outer_folds"}
    missing = required.difference(freq.columns)
    if missing:
        raise ValueError(f"Missing columns in nested feature-frequency file: {sorted(missing)}")

    top = freq.sort_values(
        ["selection_count", "feature"], ascending=[False, True]
    ).head(20).copy()
    top = top.sort_values("selection_fraction_of_outer_folds", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["selection_fraction_of_outer_folds"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Fraction of outer folds in which feature was selected")
    ax.set_ylabel("HRV feature")
    ax.set_title("Feature-selection stability in fully nested grouped evaluation")
    fig.tight_layout()
    save_figure(fig, "figure_nested_feature_stability")


def local_contributions() -> None:
    chf = predict_wfdb_record(CHF_RECORD, MODEL_PATH, channel=0, start_seconds=0)
    nsr = predict_wfdb_record(NSR_RECORD, MODEL_PATH, channel=0, start_seconds=0)
    examples = [("CHF development example", chf), ("NSR development example", nsr)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (title, result) in zip(axes, examples):
        contrib = pd.Series(result["feature_contributions"], dtype=float)
        top_names = contrib.abs().sort_values(ascending=False).head(10).index
        plot_values = contrib.loc[top_names].sort_values()
        ax.barh(plot_values.index, plot_values.values)
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Contribution to logistic-regression logit")
        ax.set_title(
            f"{title}\nCHF-like model score={float(result['chf_probability']):.3f}"
        )

    fig.suptitle("Exact local feature contributions for representative 5-minute windows", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "figure_local_feature_contributions")


def main() -> None:
    required_paths = [MODEL_PATH, WINDOW_PRED, RECORD_PRED, NESTED_FREQ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Required files are missing:\n- " + "\n- ".join(missing))

    ecg_and_rr_examples()
    external_discrimination_curves()
    nested_feature_stability()
    local_contributions()

    print("\nPublication source-figure generation finished.")
    print("No model fitting, threshold tuning, or external model selection is performed by this script.")


if __name__ == "__main__":
    main()
