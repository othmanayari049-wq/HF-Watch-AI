# HF-Watch-AI

HF-Watch-AI is an experimental research pipeline for heart-failure screening using ECG-derived heart-rate variability (HRV) features and explainable machine learning.

> **Research use only.** HF-Watch-AI is not a medical device and must not be used to diagnose, rule out, or manage heart failure or any other medical condition.

## Project overview

The pipeline:

1. Loads long-term ECG recordings.
2. Splits them into 5-minute windows.
3. Cleans ECG signals and detects R-peaks, or uses validated beat annotations when available.
4. Extracts HRV features.
5. Trains patient-level machine-learning models.
6. Uses SHAP for explainability.
7. Evaluates performance on unseen patients.
8. Performs external database validation on independent long-term RR-interval datasets.
9. Provides reusable command-line inference and an interactive Streamlit research demo.

## Datasets

### Development / internal evaluation

- BIDMC Congestive Heart Failure Database
- MIT-BIH Normal Sinus Rhythm Database

### External validation

- Congestive Heart Failure RR Interval Database (`chf2db`) — 29 CHF records
- Normal Sinus Rhythm RR Interval Database (`nsr2db`) — 54 healthy records

The raw datasets are not included in this repository.

A PTB Diagnostic ECG Database copy may also be used for exploratory work, but it is **not used as external validation for the current 5-minute HRV model** because many PTB records are much shorter than the model's required 300-second analysis window.

## Final research model

The final classifier is a logistic-regression pipeline using 20 selected HRV features extracted from 5-minute windows.

The saved pipeline includes:

- median imputation
- standardization
- class-balanced logistic regression
- fixed 0.50 decision threshold for the current research demo

The output is described as a **CHF-like HRV pattern** or **Healthy-like HRV pattern**. The reported CHF-like probability is the model's output score; it is **not** an estimate of a patient's true clinical probability of heart failure.

## Internal results

The top-20-feature logistic-regression model achieved:

- Mean GroupKFold accuracy: 99.42%
- Mean F1-score: 99.45%
- Mean ROC-AUC: 0.9996
- Unseen-patient holdout accuracy: 99.01%
- Unseen-patient ROC-AUC: 0.9993

These very high internal results should be interpreted cautiously because the CHF and healthy development cohorts originate from different source databases.

## External validation

The trained model was evaluated **without retraining or threshold tuning** on independent PhysioNet long-term RR-interval databases using 5-minute HRV windows.

### Window level

22,874 external 5-minute windows were evaluated:

- 7,626 CHF windows
- 15,248 healthy windows

Results:

- Accuracy: 74.84%
- Precision: 61.08%
- Sensitivity: 67.58%
- Specificity: 78.46%
- F1-score: 64.17%
- ROC-AUC: 0.7985

### Record level

Because many 5-minute windows come from the same long-term record, the record-level analysis is the more important external result. It uses 83 independent records:

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

The external results are substantially lower than the internal results. This indicates that the classifier contains useful HRV signal while also confirming meaningful cross-database generalization limits and likely source effects in the development results.

## Streamlit research demo

The Streamlit interface now provides:

- local WFDB record-path input
- ECG channel selection
- selectable 5-minute start time
- CHF-like model score
- detected beat count
- mean heart rate
- sampling rate and selected signal name
- cleaned ECG preview
- detected R-peak overlay
- R-R interval series across the analyzed window
- the 20 HRV features supplied to the model
- external-validation context and limitations
- explicit research-only safety messaging

Run it with:

```bash
streamlit run app.py
```

Example development records:

```text
C:\Users\<USER>\HF-Watch-AI\data\chfdb\files\chf01
C:\Users\<USER>\HF-Watch-AI\data\nsrdb\16265
```

Enter the WFDB record path **without** `.dat` or `.hea`.

For the default first 5-minute window:

- ECG channel: `0`
- Window start time: `0`

A start time of `300` analyzes minutes 5–10, `600` analyzes minutes 10–15, and so on, provided a complete 5-minute window remains in the recording.

## Command-line inference

Run the same shared inference pipeline from the terminal:

```bash
python src/16_predict_record.py data/chfdb/files/chf01
python src/16_predict_record.py data/nsrdb/16265
```

Optional parameters:

```bash
python src/16_predict_record.py RECORD_PATH --channel 0 --start-seconds 300
```

## Installation

```bash
python -m venv hf_env
hf_env\Scripts\activate
python -m pip install -r requirements.txt
```

## Main scripts

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
- `app.py`

### External validation

- `src/18_extract_external_rr_features.py`
- `src/19_external_validation.py`
- `src/20_external_validation_plots.py`
- `src/21_record_level_metrics.py`

## Generated outputs

The pipeline can generate artifacts such as:

- internal confusion matrix, ROC and precision-recall figures
- SHAP summary figure
- external confusion matrix
- external ROC curve
- external precision-recall curve
- external probability distribution
- external window-level prediction table
- external record-level summary table

Large datasets and generated intermediate feature tables should remain outside version control.

## Important limitations

- The development CHF and healthy cohorts originate from different PhysioNet databases, creating a risk of database-source bias.
- The external CHF and healthy cohorts also come from separate databases, so external validation reduces but does not eliminate source bias.
- Multiple 5-minute windows are derived from each long-term record; record-level metrics are therefore emphasized over window-level metrics.
- Beat annotations are used for the external RR datasets, whereas the raw-ECG inference workflow detects R-peaks from ECG signals. These pipelines are related but not identical.
- No threshold tuning was performed on the external evaluation set.
- The model has not been prospectively tested in a clinical population.
- The current app accepts local WFDB ECG records; it is not designed for arbitrary consumer wearable files or clinical ECG formats without preprocessing.
- A Healthy-like result does not rule out disease, and a CHF-like result does not establish a diagnosis.
- Prospective validation on independently collected, clinically characterized cohorts is required before any clinical interpretation.

## Disclaimer

For research and educational use only. HF-Watch-AI is an experimental machine-learning prototype and is not a clinically validated diagnostic system.
