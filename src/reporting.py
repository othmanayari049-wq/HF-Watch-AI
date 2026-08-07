from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json

import pandas as pd


MODEL_VERSION = "HF-Watch-AI v1.1 / LR-top20"


def result_to_csv_bytes(result: dict[str, object]) -> bytes:
    row = {
        "model_version": MODEL_VERSION,
        "record_path": result.get("record_path"),
        "signal_name": result.get("signal_name"),
        "channel": result.get("channel"),
        "window_start_sec": result.get("start_seconds"),
        "window_seconds": result.get("window_seconds"),
        "sampling_rate_hz": result.get("sampling_rate"),
        "detected_beats": result.get("detected_beats"),
        "mean_hr_bpm": result.get("mean_hr_bpm"),
        "chf_like_probability": result.get("chf_probability"),
        "prediction": result.get("prediction"),
        "interpretation": result.get("interpretation"),
    }
    for name, value in result.get("feature_values", {}).items():
        row[name] = value
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8")


def result_to_json_bytes(result: dict[str, object]) -> bytes:
    payload = {
        "model_version": MODEL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_path": result.get("record_path"),
        "signal_name": result.get("signal_name"),
        "channel": result.get("channel"),
        "window_start_sec": result.get("start_seconds"),
        "window_seconds": result.get("window_seconds"),
        "sampling_rate_hz": result.get("sampling_rate"),
        "detected_beats": result.get("detected_beats"),
        "mean_hr_bpm": result.get("mean_hr_bpm"),
        "chf_like_probability": result.get("chf_probability"),
        "prediction": result.get("prediction"),
        "interpretation": result.get("interpretation"),
        "quality": result.get("quality"),
        "feature_values": result.get("feature_values", {}),
        "disclaimer": "Research use only; not a medical diagnosis or medical device.",
    }
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def result_to_html_bytes(result: dict[str, object]) -> bytes:
    probability = float(result.get("chf_probability", float("nan")))
    features = result.get("feature_values", {})
    feature_rows = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{escape(f'{float(v):.6g}')}</td></tr>"
        for k, v in features.items()
    )
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>HF-Watch-AI research report</title>
<style>body{{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;line-height:1.5}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px;text-align:left}}.warn{{padding:12px;background:#fff3cd}}</style></head>
<body><h1>HF-Watch-AI Research Report</h1>
<p><strong>Model:</strong> {escape(MODEL_VERSION)}</p>
<p><strong>Record:</strong> {escape(str(result.get('record_path', '')))}</p>
<p><strong>Window:</strong> {result.get('start_seconds', 0)}–{int(result.get('start_seconds', 0)) + int(result.get('window_seconds', 300))} s</p>
<h2>Experimental result</h2>
<p><strong>{escape(str(result.get('interpretation', '')))}</strong></p>
<p>CHF-like model score: <strong>{probability:.1%}</strong></p>
<p>Detected beats: {result.get('detected_beats')} | Mean heart rate: {float(result.get('mean_hr_bpm', float('nan'))):.1f} bpm | Sampling rate: {result.get('sampling_rate')} Hz</p>
<div class='warn'><strong>Research use only.</strong> This output is not a diagnosis, does not rule heart failure in or out, and must not be used for treatment decisions.</div>
<h2>Model input HRV features</h2><table><tr><th>Feature</th><th>Value</th></tr>{feature_rows}</table>
</body></html>"""
    return html.encode("utf-8")
