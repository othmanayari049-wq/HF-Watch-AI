from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import predict_wfdb_record  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "models" / "hf_watch_top20_model.joblib"
DEFAULT_RECORD = PROJECT_ROOT / "data" / "chfdb" / "files" / "chf01"


st.set_page_config(
    page_title="HF-Watch-AI",
    page_icon="❤️",
    layout="wide",
)

st.title("HF-Watch-AI")
st.caption(
    "Experimental ECG-derived HRV research prototype. "
    "Not a medical device and not intended for diagnosis."
)

with st.sidebar:
    st.header("Input")
    record_text = st.text_input(
        "WFDB record path (without .dat or .hea)",
        value=str(DEFAULT_RECORD),
    )
    channel = st.number_input(
        "ECG channel",
        min_value=0,
        value=0,
        step=1,
    )
    start_seconds = st.number_input(
        "Window start time (seconds)",
        min_value=0,
        value=0,
        step=300,
    )
    run_prediction = st.button("Analyze 5-minute ECG window", type="primary")

st.info(
    "The application expects local WFDB files such as `record.dat` and "
    "`record.hea`. Enter the shared path without the extension."
)

if not MODEL_PATH.exists():
    st.error(
        f"Trained model not found at `{MODEL_PATH}`. "
        "Run `python src/11_train_final_model.py` first."
    )

if run_prediction:
    try:
        with st.spinner("Cleaning ECG, detecting R-peaks, and extracting HRV..."):
            result = predict_wfdb_record(
                record_path=Path(record_text),
                model_path=MODEL_PATH,
                channel=int(channel),
                start_seconds=int(start_seconds),
            )

        probability = float(result["chf_probability"])

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("CHF-like probability", f"{probability:.1%}")
        metric_col2.metric("Detected beats", int(result["detected_beats"]))
        metric_col3.metric("Sampling rate", f"{result['sampling_rate']} Hz")

        if int(result["prediction"]) == 1:
            st.warning("Experimental output: CHF-like HRV pattern detected.")
        else:
            st.success("Experimental output: Healthy-like HRV pattern detected.")

        st.progress(min(max(probability, 0.0), 1.0))

        st.subheader("Model input features")
        feature_table = pd.DataFrame(
            {
                "feature": list(result["feature_values"].keys()),
                "value": list(result["feature_values"].values()),
            }
        )
        st.dataframe(feature_table, use_container_width=True, hide_index=True)

        st.warning(
            "Research limitation: the CHF and healthy classes were derived from "
            "different PhysioNet databases, so performance may partly reflect "
            "dataset or acquisition differences."
        )

    except Exception as error:
        st.exception(error)

st.divider()
st.markdown(
    "**Research use only.** Results require external validation on comparable, "
    "independently collected cohorts before any clinical interpretation."
)
