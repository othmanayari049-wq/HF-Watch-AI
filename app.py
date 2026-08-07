from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import predict_wfdb_record  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "models" / "hf_watch_top20_model.joblib"
DEFAULT_RECORD = PROJECT_ROOT / "data" / "chfdb" / "files" / "chf01"
DECISION_THRESHOLD = 0.50


st.set_page_config(
    page_title="HF-Watch-AI",
    page_icon="❤️",
    layout="wide",
)

st.title("❤️ HF-Watch-AI")
st.caption(
    "Experimental 5-minute ECG → HRV → machine-learning research prototype. "
    "Not a medical device and not intended for diagnosis or treatment decisions."
)

with st.sidebar:
    st.header("ECG input")
    record_text = st.text_input(
        "WFDB record path",
        value=str(DEFAULT_RECORD),
        help="Enter the shared path without .dat or .hea.",
    )
    channel = st.number_input(
        "ECG channel",
        min_value=0,
        value=0,
        step=1,
        help="0 selects the first ECG channel in the WFDB record.",
    )
    start_seconds = st.number_input(
        "Window start time (seconds)",
        min_value=0,
        value=0,
        step=300,
        help="0 analyzes the first five minutes; 300 analyzes minutes 5–10.",
    )
    run_prediction = st.button(
        "Analyze 5-minute ECG window",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.markdown("**Model**")
    st.caption("Top-20 HRV logistic-regression pipeline")
    st.caption("Decision threshold: 0.50")

st.info(
    "The app expects local WFDB files such as `record.dat` + `record.hea`. "
    "Enter the record path without either extension."
)

with st.expander("How to interpret the output"):
    st.markdown(
        "- **CHF-like probability** is the classifier's output score, not the clinical "
        "probability that a person has heart failure.\n"
        "- **Detected beats** are the R-peaks found in the selected 5-minute ECG window.\n"
        "- **Mean heart rate** is derived from the detected R-R intervals.\n"
        "- A score at or above **0.50** is labeled *CHF-like*; below 0.50 is labeled "
        "*Healthy-like*.\n"
        "- All outputs are for research/education only."
    )

if not MODEL_PATH.exists():
    st.error(
        f"Trained model not found at `{MODEL_PATH}`. "
        "Run `python src/11_train_final_model.py` first."
    )

if run_prediction:
    try:
        record_path = Path(record_text.strip())

        with st.spinner("Cleaning ECG, detecting R-peaks, extracting HRV, and running the model..."):
            result = predict_wfdb_record(
                record_path=record_path,
                model_path=MODEL_PATH,
                channel=int(channel),
                start_seconds=int(start_seconds),
            )

        probability = float(result["chf_probability"])
        prediction = int(result["prediction"])

        st.subheader("Experimental result")

        if prediction == 1:
            st.warning(
                "**CHF-like HRV pattern detected.** This is a research classification, "
                "not a heart-failure diagnosis."
            )
        else:
            st.success(
                "**Healthy-like HRV pattern detected.** This does not rule out heart "
                "failure or establish that a person is clinically healthy."
            )

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("CHF-like probability", f"{probability:.1%}")
        metric_col2.metric("Detected beats", int(result["detected_beats"]))
        metric_col3.metric("Mean heart rate", f"{float(result['mean_hr_bpm']):.1f} bpm")
        metric_col4.metric("Sampling rate", f"{result['sampling_rate']} Hz")

        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            f"Selected signal: {result['signal_name']} · Channel {result['channel']} · "
            f"Window: {result['start_seconds']}–"
            f"{int(result['start_seconds']) + int(result['window_seconds'])} s · "
            f"Missing/interpolated samples: {float(result['missing_fraction']):.2%}"
        )

        signal_tab, hrv_tab, validation_tab = st.tabs(
            ["ECG & R-peaks", "HRV features", "Validation context"]
        )

        with signal_tab:
            sampling_rate = int(result["sampling_rate"])
            cleaned_signal = np.asarray(result["cleaned_signal"], dtype=float)
            r_peaks = np.asarray(result["r_peaks"], dtype=int)

            preview_seconds = min(15, int(result["window_seconds"]))
            preview_samples = min(len(cleaned_signal), preview_seconds * sampling_rate)
            preview_signal = cleaned_signal[:preview_samples]
            preview_time = (
                np.arange(preview_samples) / sampling_rate
                + float(result["start_seconds"])
            )
            preview_peaks = r_peaks[r_peaks < preview_samples]

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(preview_time, preview_signal, linewidth=0.9, label="Cleaned ECG")
            if len(preview_peaks):
                ax.scatter(
                    preview_peaks / sampling_rate + float(result["start_seconds"]),
                    cleaned_signal[preview_peaks],
                    marker="o",
                    s=28,
                    label="Detected R-peaks",
                )
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("ECG amplitude")
            ax.set_title(f"First {preview_seconds} seconds of the analyzed window")
            ax.legend(loc="upper right")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            if len(r_peaks) > 1:
                rr_ms = np.diff(r_peaks) / sampling_rate * 1000.0
                rr_time = (
                    r_peaks[1:] / sampling_rate + float(result["start_seconds"])
                )
                rr_df = pd.DataFrame(
                    {
                        "Time (s)": rr_time,
                        "R-R interval (ms)": rr_ms,
                    }
                ).set_index("Time (s)")
                st.markdown("**R-R interval series across the 5-minute window**")
                st.line_chart(rr_df, use_container_width=True)

        with hrv_tab:
            st.markdown(
                "These are the **20 HRV features actually supplied to the trained model**."
            )
            feature_table = pd.DataFrame(
                {
                    "Feature": list(result["feature_values"].keys()),
                    "Value": list(result["feature_values"].values()),
                }
            )
            feature_table["Value"] = pd.to_numeric(
                feature_table["Value"], errors="coerce"
            )
            st.dataframe(
                feature_table,
                use_container_width=True,
                hide_index=True,
            )

        with validation_tab:
            st.markdown("**External record-level validation (independent RR databases)**")
            val1, val2, val3 = st.columns(3)
            val1.metric("Record accuracy", "84.34%")
            val2.metric("Sensitivity", "72.41%")
            val3.metric("Specificity", "90.74%")
            val4, val5, val6 = st.columns(3)
            val4.metric("Precision", "80.77%")
            val5.metric("F1 score", "76.36%")
            val6.metric("ROC-AUC", "0.8902")
            st.caption(
                "External evaluation used 83 long-term records (29 CHF, 54 healthy). "
                "These results are substantially lower than internal development results "
                "and are a more realistic estimate of cross-database generalization."
            )
            st.warning(
                "External CHF and healthy records still came from separate source databases, "
                "so database-source bias is reduced but not eliminated."
            )

        st.error(
            "Research use only: this output must not be used to diagnose, rule out, "
            "or manage heart failure or any other medical condition."
        )

    except FileNotFoundError as error:
        st.error(str(error))
        st.caption(
            "Check that the path exists and that you entered the WFDB record name "
            "without `.dat` or `.hea`."
        )
    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.exception(error)

st.divider()
st.markdown("### Research status")
st.markdown(
    "HF-Watch-AI is an experimental prototype. Internal performance was very high, "
    "but external record-level validation produced **84.34% accuracy** and "
    "**0.8902 ROC-AUC**, showing both useful HRV signal and meaningful generalization "
    "limitations."
)
st.caption(
    "No clinical claims are made. Prospective validation on independently collected, "
    "clinically characterized cohorts would be required before clinical interpretation."
)
