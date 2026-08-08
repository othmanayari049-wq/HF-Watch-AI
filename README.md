# HF-Watch-AI

HF-Watch-AI is an experimental research pipeline for heart-failure-related ECG analysis using heart-rate variability (HRV) features and explainable machine learning.

> **Research use only.** HF-Watch-AI is not a medical device and must not be used to diagnose, rule out, triage, or manage heart failure or any other medical condition.

## What the system does

For one 5-minute ECG window, the pipeline performs:

```text
ECG
  ↓
signal cleaning
  ↓
R-peak detection
  ↓
R-R intervals
  ↓
HRV feature extraction
  ↓
20 selected HRV features
  ↓
logistic-regression model
  ↓
CHF-like model score + research classification
```

The model output is a **CHF-like model score**, not a patient's true medical probability of heart failure.

## Current version

**HF-Watch-AI v1.1 / LR-top20**

- Window duration: 300 seconds
- Model: logistic regression
- Selected features: 20 HRV features
- Decision threshold: 0.50
- Preprocessing: median imputation + standardization

## Main capabilities

### Single-window ECG analysis

The Streamlit application can:

- read a local WFDB ECG record;
- accept an uploaded matching `.hea` + `.dat` pair;
- choose ECG channel and window start time;
- clean a complete 5-minute ECG window;
- detect R-peaks;
- calculate R-R intervals and HRV features;
- display CHF-like versus healthy-like research output;
- display detected beats, mean heart rate, sampling rate, and missing-data percentage;
- plot cleaned ECG with detected R-peaks;
- plot the R-R interval series;
- display all 20 model input features;
- display local logistic-regression feature contributions;
- export CSV, JSON, and HTML research reports.

### Signal-quality checks

Before prediction, the inference pipeline performs basic engineering checks for:

- excessive missing ECG samples;
- too few detected beats;
- a high fraction of physiologically unusual R-R intervals.

A severely poor-quality window is rejected instead of forcing a prediction. These quality checks are engineering safeguards and are **not clinically validated signal-quality criteria**.

### Full-record research analysis

`src/22_full_record_analysis.py` and the Streamlit app can analyze consecutive 5-minute windows across a longer ECG recording.

The output includes:

- one CHF-like score per valid window;
- a probability timeline;
- mean and median CHF-like score;
- fraction of windows classified CHF-like;
- a majority-window record summary;
- skipped-window information;
- **window agreement** with the majority class;
- descriptive pattern consistency (`High`, `Moderate`, `Low / mixed`);
- score spread and score range;
- downloadable window-level CSV results.

The majority-window rule is a research aggregation method, not a clinically validated patient-level decision rule.

**Important:** pattern consistency measures whether the analyzed windows agree with each other. It does **not** measure whether the classification is correct. A record can be highly consistent and still be misclassified.

### Explainability

The application reports exact per-feature contributions to the final logistic-regression logit after imputation and scaling.

- positive contribution → pushes the mathematical score toward CHF-like;
- negative contribution → pushes the mathematical score toward healthy-like.

These contributions explain the model calculation and should not be interpreted as biological causality.

## Datasets

### Development / internal evaluation

- BIDMC Congestive Heart Failure Database
- MIT-BIH Normal Sinus Rhythm Database

### External validation

- Congestive Heart Failure RR Interval Database (`chf2db`) — 29 CHF records
- Normal Sinus Rhythm RR Interval Database (`nsr2db`) — 54 healthy records

The raw datasets are not included in this repository.

The PTB Diagnostic ECG Database was investigated, but its records were much shorter than the 5-minute window required by the trained HRV model, so it was **not** used as direct validation of this model.

## Internal results

The top-20-feature logistic-regression model achieved:

- Mean GroupKFold accuracy: **99.42%**
- Mean F1-score: **99.45%**
- Mean ROC-AUC: **0.9996**
- Unseen-patient holdout accuracy: **99.01%**
- Unseen-patient ROC-AUC: **0.9993**

These very high internal results must be interpreted cautiously because the CHF and healthy development cohorts originate from different source databases.

## External validation

The trained model was evaluated **without retraining** on independent long-term RR-interval databases using the same 5-minute duration.

### Window level

22,874 external windows:

- 7,626 CHF windows
- 15,248 healthy windows

Results:

- Accuracy: **74.84%**
- Precision: **61.08%**
- Sensitivity: **67.58%**
- Specificity: **78.46%**
- F1-score: **64.17%**
- ROC-AUC: **0.7985**

### Record level

Because repeated windows from the same long-term record are correlated, record-level metrics are emphasized.

83 records:

- 29 CHF records
- 54 healthy records

Results:

- Accuracy: **84.34%** (95% bootstrap CI: 75.90%–91.57%)
- Sensitivity: **72.41%** (95% CI: 55.56%–88.00%)
- Specificity: **90.74%** (95% CI: 82.14%–98.04%)
- Precision: **80.77%** (95% CI: 64.70%–95.45%)
- F1-score: **76.36%** (95% CI: 62.22%–87.50%)
- ROC-AUC: **0.8902** (95% CI: 0.7918–0.9661)

Record-level confusion matrix:

| | Predicted Healthy | Predicted CHF |
|---|---:|---:|
| Actual Healthy | 49 | 5 |
| Actual CHF | 8 | 21 |

The external results are substantially lower than the internal results. This supports the presence of useful HRV signal while also showing meaningful cross-database generalization limitations.

## Robustness and follow-up experiments

All model-selection experiments below used **development data only**. The external 83-record evaluation set was not used to choose a new model, threshold, calibration method, or aggregation method.

### Regularization benchmark

`src/23_regularization_benchmark.py` tested logistic-regression `C` values from 0.01 to 10 using patient-grouped cross-validation.

- Current model (`C=1`): accuracy **99.42%**, F1 **99.45%**, ROC-AUC **0.999589**
- Best numeric result (`C=10`): accuracy **99.50%**, F1 **99.52%**, ROC-AUC **0.999695**

The gain was only about **0.08 percentage points in accuracy**, so the existing `C=1` model was retained.

### Calibration benchmark

`src/24_calibration_benchmark.py` compared the current score with sigmoid and isotonic calibration using grouped development validation.

| Method | Mean Brier | Mean log loss |
|---|---:|---:|
| Uncalibrated | **0.006144** | **0.028100** |
| Sigmoid | 0.008103 | 0.033361 |
| Isotonic | 0.008215 | 0.063078 |

Lower is better. The current uncalibrated model performed best, so no calibration layer was added. This development result does **not** establish that the score is a clinically calibrated disease probability.

### Record aggregation benchmark

`src/25_record_aggregation_benchmark.py` compared mean probability, median probability, and majority-window aggregation on grouped development predictions. All three achieved 100% record-level performance on that benchmark, so the simple majority-window research rule was retained rather than selecting a more complex rule without evidence of benefit.

## External error analysis

`src/26_external_error_analysis.py` performs a **descriptive** audit of external record errors without changing the model or threshold.

Out of 83 records:

- Correct: **70**
- Misclassified: **13**
- False positives: **5**
- False negatives: **8**

The failures are not explained by a simple threshold issue. Some incorrect records lie near 0.50, while others are strongly on the wrong side. Individual misclassified records can also contain 5-minute windows spanning almost the full 0–1 score range.

## External consistency audit

`src/27_external_consistency_audit.py` descriptively examined the new full-record consistency display. It did **not** choose or tune the consistency labels using the external set.

Among the 13 misclassified records:

- `Low / mixed`: **10**
- `Moderate`: **1**
- `High`: **2**

Among the 70 correctly classified records, **18** were `Low / mixed`.

Mean window agreement was lower for errors than for correct records:

- False negatives: **0.739**
- False positives: **0.633**
- True negatives: **0.828**
- True positives: **0.846**

This suggests that low agreement is often a useful warning of a difficult or heterogeneous record. However, it is **not a correctness detector**. Two false-negative CHF records (`chf201` and `chf219`) had `High` consistency despite being incorrectly classified.

Therefore:

> **High consistency means the analyzed windows agree with each other. It does not mean the model is correct.**

The `High`, `Moderate`, and `Low / mixed` labels are descriptive presentation heuristics only. They are not validated uncertainty thresholds, clinical confidence scores, or probabilities of correctness.

## Important limitations

- Development CHF and healthy cohorts originate from different PhysioNet databases, creating database-source-bias risk.
- External CHF and healthy cohorts also come from separate source databases, so external validation reduces but does not eliminate source bias.
- Beat annotations are used for the external RR datasets, while raw-ECG inference detects R-peaks algorithmically.
- Repeated 5-minute windows from one record are correlated.
- The 0.50 decision threshold was not tuned on the external evaluation set.
- The quality rules are engineering safeguards, not validated clinical ECG-quality standards.
- Full-record majority-window aggregation is exploratory.
- Pattern consistency reflects within-record agreement, not correctness or clinical confidence.
- Development-only calibration testing does not establish clinical probability calibration.
- No prospective clinical validation has been performed.
- No validated subgroup fairness, rhythm-specific, acquisition-device, or site-robustness study has been completed.
- A healthy-like output does not establish clinical health and does not rule out heart failure.
- A CHF-like output does not diagnose heart failure.

See [`MODEL_CARD.md`](MODEL_CARD.md) for the model card.

## Installation

```bash
python -m venv hf_env
hf_env\Scripts\activate
python -m pip install -r requirements.txt
```

## Local model and datasets

Large datasets, generated features, generated results, and `models/*.joblib` are ignored by Git by default.

To reproduce the current local workflow, the expected final model path is:

```text
models/hf_watch_top20_model.joblib
```

Run the environment check with:

```bash
python src/17_smoke_test.py
```

## Run the Streamlit application

```bash
streamlit run app.py
```

Example CHF development record:

```text
C:\Users\othma\HF-Watch-AI\data\chfdb\files\chf01
```

Example healthy development record:

```text
C:\Users\othma\HF-Watch-AI\data\nsrdb\16265
```

For the examples above:

- `ECG channel = 0` means use the first signal channel;
- `Window start time = 0` means analyze seconds 0–300;
- start time `300` analyzes seconds 300–600.

## Uploaded WFDB files

The application can also accept a matching WFDB pair through the browser:

```text
record.hea
record.dat
```

The `.hea` file must correspond to the uploaded `.dat` file.

## Full-record CLI

Analyze consecutive complete windows:

```bash
python src/22_full_record_analysis.py data/chfdb/files/chf01 --channel 0 --max-windows 12
```

Set `--max-windows 0` to process all complete windows.

## Research scripts

### Development and internal evaluation

- `src/02_extract_windows.py`
- `src/03_extract_healthy_windows.py`
- `src/04_build_training_dataset.py`
- `src/05_clean_dataset.py`
- `src/06_train_baseline.py`
- `src/07_group_cross_validation.py`
- `src/08_compare_models.py`
- `src/09_shap_analysis.py`
- `src/10_feature_selection.py`
- `src/11_train_final_model.py`
- `src/12_test_single_record.py`
- `src/13_test_unseen_patients.py`
- `src/14_create_evaluation_figures.py`
- `src/15_dataset_bias_check.py`

### Inference and application

- `src/16_predict_record.py`
- `src/17_smoke_test.py`
- `src/inference.py`
- `src/quality.py`
- `src/record_analysis.py`
- `src/reporting.py`
- `app.py`

### External validation

- `src/18_extract_external_rr_features.py`
- `src/19_external_validation.py`
- `src/20_external_validation_plots.py`
- `src/21_record_level_metrics.py`

### Extended research tools

- `src/22_full_record_analysis.py`
- `src/23_regularization_benchmark.py`
- `src/24_calibration_benchmark.py`
- `src/25_record_aggregation_benchmark.py`
- `src/26_external_error_analysis.py`
- `src/27_external_consistency_audit.py`

## Deployment note

The Streamlit UI is upload-ready, but a public deployment still needs the trained model artifact to be available to the deployed application. The repository intentionally ignores `models/*.joblib` by default. Do not claim a public deployment is reproducible until the exact validated model artifact and its provenance/versioning are intentionally packaged.

## Disclaimer

For research and educational use only. HF-Watch-AI must not be used to diagnose, rule out, screen, triage, or manage heart failure or any other medical condition.
