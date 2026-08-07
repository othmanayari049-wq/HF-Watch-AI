from __future__ import annotations

from pathlib import Path
import sys

import joblib
import neurokit2
import numpy
import pandas
import sklearn
import wfdb

import quality
import record_analysis
import reporting


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hf_watch_top20_model.joblib"
CHF_DATA_DIR = PROJECT_ROOT / "data" / "chfdb" / "files"
HEALTHY_DATA_DIR = PROJECT_ROOT / "data" / "nsrdb"


def status(label: str, ok: bool, detail: str = "") -> None:
    marker = "PASS" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{marker}] {label}{suffix}")


def main() -> None:
    print("HF-Watch-AI environment smoke test\n")

    status("Python version", sys.version_info >= (3, 10), sys.version.split()[0])
    status("NumPy import", True, numpy.__version__)
    status("pandas import", True, pandas.__version__)
    status("scikit-learn import", True, sklearn.__version__)
    status("WFDB import", True, wfdb.__version__)
    status("NeuroKit2 import", True, neurokit2.__version__)
    status("Quality module", hasattr(quality, "assess_window_quality"))
    status("Full-record module", hasattr(record_analysis, "analyze_full_record"))
    status("Reporting module", hasattr(reporting, "result_to_html_bytes"))

    status("CHF dataset folder", CHF_DATA_DIR.exists(), str(CHF_DATA_DIR))
    status("Healthy dataset folder", HEALTHY_DATA_DIR.exists(), str(HEALTHY_DATA_DIR))
    status("Final model file", MODEL_PATH.exists(), str(MODEL_PATH))

    if MODEL_PATH.exists():
        saved = joblib.load(MODEL_PATH)
        required_keys = {"pipeline", "feature_columns"}
        missing = required_keys.difference(saved)
        status(
            "Model structure",
            not missing,
            "valid" if not missing else f"missing keys: {sorted(missing)}",
        )
        if not missing:
            status("Model feature count", len(saved["feature_columns"]) == 20, str(len(saved["feature_columns"])))

    print("\nA FAIL for local datasets or model means that file is not present locally.")
    print("The repository intentionally excludes large datasets and trained models by default.")


if __name__ == "__main__":
    main()
