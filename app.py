from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import predict_wfdb_record  # noqa: E402
from record_analysis import analyze_full_record  # noqa: E402
from reporting import MODEL_VERSION, result_to_csv_bytes, result_to_html_bytes, result_to_json_bytes  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "models" / "hf_watch_top20_model.joblib"
DEFAULT_RECORD = PROJECT_ROOT / "data" / "chfdb" / "files" / "chf01"

st.set_page_config(page_title="HF-Watch-AI", page_icon="❤️", layout="wide")


def _uploaded_record_path(files) -> tuple[Path | None, tempfile.TemporaryDirectory | None]:
    if not files:
        return None, None
    names = {Path(f.name).suffix.lower(): f for f in files}
    if ".hea" not in names or ".dat" not in names:
        st.error("Upload the matching WFDB `.hea` and `.dat` files together.")
        return None, None
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    for f in files:
        (tmp_path / Path(f.name).name).write_bytes(f.getvalue())
    hea_name = Path(names[".hea"].name).name
    return tmp_path / Path(hea_name).stem, tmp


def _ecg_plot(result: dict[str, object]):
    fs = int(result["sampling_rate"])
    cleaned = np.asarray(result["cleaned_signal"], dtype=float)
    r_peaks = np.asarray(result["r_peaks"], dtype=int)
    n = min(len(cleaned), fs * 15)
    x = np.arange(n) / fs
    peaks = r_peaks[r_peaks < n]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, cleaned[:n], linewidth=1.0, label="Cleaned ECG")
    ax.scatter(peaks / fs, cleaned[peaks], s=24, label="Detected R-peaks")
    ax.set_title("First 15 seconds of the analyzed window")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("ECG amplitude")
    ax.legend()
    fig.tight_layout()
    return fig


def _rr_dataframe(result: dict[str, object]) -> pd.DataFrame:
    fs = float(result["sampling_rate"])
    peaks = np.asarray(result["r_peaks"], dtype=float)
    rr = np.diff(peaks) / fs * 1000.0
    t = peaks[1:] / fs + float(result["start_seconds"])
    return pd.DataFrame({"Time (s)": t, "R-R interval (ms)": rr})


def _render_window_result(result: dict[str, object]):
    probability = float(result["chf_probability"])
    prediction = int(result["prediction"])
    quality = result.get("quality", {})

    st.header("Experimental result")
    if prediction == 1:
        st.warning("**CHF-like HRV pattern detected.** This is a research classification, not a heart-failure diagnosis.")
    else:
        st.success("**Healthy-like HRV pattern detected.** This does not rule out heart failure or establish that a person is clinically healthy.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CHF-like probability", f"{probability:.1%}")
    c2.metric("Detected beats", int(result["detected_beats"]))
    c3.metric("Mean heart rate", f"{float(result['mean_hr_bpm']):.1f} bpm")
    c4.metric("Sampling rate", f"{result['sampling_rate']} Hz")
    st.progress(min(max(probability, 0.0), 1.0))

    st.caption(
        f"Selected signal: {result['signal_name']} · Channel {result['channel']} · "
        f"Window: {result['start_seconds']}–{int(result['start_seconds']) + int(result['window_seconds'])} s · "
        f"Missing/interpolated samples: {float(result['missing_fraction']):.2%}"
    )

    if quality:
        if quality.get("status") == "warning":
            st.warning(str(quality.get("message")))
        else:
            st.success(
                f"Signal quality check: {quality.get('message')} "
                f"R-R outlier fraction: {float(quality.get('rr_outlier_fraction', 0)):.1%}."
            )

    tabs = st.tabs(["ECG & R-peaks", "HRV features", "Why this score?", "Validation context", "Export"])

    with tabs[0]:
        fig = _ecg_plot(result)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.subheader("R-R interval series across the 5-minute window")
        st.line_chart(_rr_dataframe(result), x="Time (s)", y="R-R interval (ms)", use_container_width=True)

    with tabs[1]:
        feature_table = pd.DataFrame(
            {"feature": list(result["feature_values"].keys()), "value": list(result["feature_values"].values())}
        )
        st.dataframe(feature_table, use_container_width=True, hide_index=True)
        st.caption("These are the 20 HRV measurements passed to the final logistic-regression model.")

    with tabs[2]:
        contrib = result.get("feature_contributions", {})
        if contrib:
            df = pd.DataFrame({"feature": list(contrib.keys()), "logit_contribution": list(contrib.values())})
            df["absolute"] = df["logit_contribution"].abs()
            df = df.sort_values("absolute", ascending=False).head(10).drop(columns="absolute")
            st.bar_chart(df.set_index("feature")["logit_contribution"], use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(
                "Positive contributions push the logistic-regression score toward CHF-like; negative contributions push it toward healthy-like. "
                "This explains the mathematical model score, not a biological cause."
            )
        else:
            st.info("Local feature contributions are unavailable for this model pipeline.")

    with tabs[3]:
        st.subheader("External record-level validation")
        a, b, c = st.columns(3)
        a.metric("Record accuracy", "84.34%")
        b.metric("Sensitivity", "72.41%")
        c.metric("Specificity", "90.74%")
        d, e, f = st.columns(3)
        d.metric("Precision", "80.77%")
        e.metric("F1 score", "76.36%")
        f.metric("ROC-AUC", "0.8902")
        st.caption("External evaluation used 83 long-term records: 29 CHF and 54 healthy. Record-level metrics are emphasized because windows from one record are not independent.")
        st.warning("External CHF and healthy records still came from separate source databases, so database-source bias is reduced but not eliminated.")

    with tabs[4]:
        st.download_button("Download CSV", result_to_csv_bytes(result), "hf_watch_result.csv", "text/csv")
        st.download_button("Download JSON", result_to_json_bytes(result), "hf_watch_result.json", "application/json")
        st.download_button("Download HTML research report", result_to_html_bytes(result), "hf_watch_report.html", "text/html")
        st.caption(f"Model version: {MODEL_VERSION}")


st.title("❤️ HF-Watch-AI")
st.caption("Experimental 5-minute ECG → HRV → machine-learning research prototype. Not a medical device and not intended for diagnosis or treatment decisions.")

if not MODEL_PATH.exists():
    st.error(
        f"Trained model not found at `{MODEL_PATH}`. This repository intentionally does not publish the generated model binary by default. "
        "Train it locally with `python src/11_train_final_model.py` or add the validated model artifact before deployment."
    )

with st.sidebar:
    st.header("ECG input")
    input_mode = st.radio("Input source", ["Local WFDB path", "Upload WFDB files"])
    uploaded_files = None
    record_text = ""
    if input_mode == "Local WFDB path":
        record_text = st.text_input("WFDB record path", value=str(DEFAULT_RECORD), help="Path without .dat or .hea")
    else:
        uploaded_files = st.file_uploader("Upload matching .hea and .dat files", type=["hea", "dat"], accept_multiple_files=True)

    channel = st.number_input("ECG channel", min_value=0, value=0, step=1)
    start_seconds = st.number_input("Window start time (seconds)", min_value=0, value=0, step=300)
    run_prediction = st.button("Analyze 5-minute ECG window", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Full-record research analysis")
    step_seconds = st.selectbox("Window step", [300, 150], index=0, format_func=lambda x: f"{x} s")
    max_windows_input = st.number_input("Maximum windows (0 = all)", min_value=0, value=12, step=1)
    run_full = st.button("Analyze recording", use_container_width=True)
    st.divider()
    st.caption(f"Model: {MODEL_VERSION}")
    st.caption("Decision threshold: 0.50")

st.info("Use a local WFDB record path without an extension, or upload a matching `.hea` + `.dat` pair. The model analyzes complete 5-minute ECG windows.")

with st.expander("How to interpret the output"):
    st.markdown(
        "The model first detects heartbeats (R-peaks), converts their timing into R-R intervals, calculates HRV features, and then outputs a CHF-like model score. "
        "A score above 0.50 is classified as CHF-like. This number is **not** the patient's true medical probability of heart failure."
    )

record_path: Path | None = None
tmp_holder = None
if input_mode == "Local WFDB path":
    record_path = Path(record_text) if record_text else None
else:
    record_path, tmp_holder = _uploaded_record_path(uploaded_files)

if run_prediction and record_path is not None:
    try:
        with st.spinner("Cleaning ECG, detecting R-peaks, checking signal quality, extracting HRV, and running the model..."):
            result = predict_wfdb_record(record_path, MODEL_PATH, int(channel), int(start_seconds))
        _render_window_result(result)
    except Exception as error:
        st.error(str(error))

if run_full and record_path is not None:
    try:
        max_windows = None if int(max_windows_input) == 0 else int(max_windows_input)
        with st.spinner("Analyzing consecutive 5-minute windows. This can take several minutes for long recordings..."):
            windows, summary = analyze_full_record(
                record_path=record_path,
                model_path=MODEL_PATH,
                channel=int(channel),
                step_seconds=int(step_seconds),
                max_windows=max_windows,
            )
        st.header("Full-record research summary")
        if summary["valid_windows"] == 0:
            st.error("No valid 5-minute windows were available.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Valid windows", summary["valid_windows"])
            c2.metric("Mean CHF-like score", f"{summary['mean_chf_probability']:.1%}")
            c3.metric("CHF-like window fraction", f"{summary['chf_like_window_fraction']:.1%}")
            c4.metric("Skipped windows", summary["skipped_windows"])
            st.info(f"Experimental record aggregation: **{summary['record_interpretation']}**. This majority-window rule is a research summary, not a clinical diagnosis.")
            valid = windows[windows["status"] == "ok"]
            st.line_chart(valid, x="window_start_sec", y="chf_probability", use_container_width=True)
            st.dataframe(windows, use_container_width=True, hide_index=True)
            st.download_button("Download full-record window results", windows.to_csv(index=False).encode("utf-8"), "hf_watch_full_record.csv", "text/csv")
    except Exception as error:
        st.error(str(error))

st.divider()
st.subheader("Research status")
st.markdown(
    "HF-Watch-AI is an experimental prototype. Internal performance was very high, while external record-level validation produced **84.34% accuracy** and **0.8902 ROC-AUC**, showing both useful HRV signal and meaningful generalization limitations."
)
st.caption("No clinical claims are made. Prospective validation on independently collected, clinically characterized cohorts is required before clinical interpretation.")
st.error("Research use only: this output must not be used to diagnose, rule out, or manage heart failure or any other medical condition.")
