# HF-Watch-AI

HF-Watch-AI is an experimental research pipeline for heart-failure screening using ECG-derived heart-rate variability (HRV) features and explainable machine learning.

## Project overview

The pipeline:

1. Loads long-term ECG recordings.
2. Splits them into 5-minute windows.
3. Cleans ECG signals and detects R-peaks.
4. Extracts HRV features.
5. Trains patient-level machine-learning models.
6. Uses SHAP for explainability.
7. Evaluates performance on unseen patients.

## Datasets

- BIDMC Congestive Heart Failure Database
- MIT-BIH Normal Sinus Rhythm Database

The raw datasets are not included in this repository.

## Results

The top-20-feature logistic-regression model achieved:

- Mean GroupKFold accuracy: 99.42%
- Mean F1-score: 99.45%
- Mean ROC-AUC: 0.9996
- Unseen-patient holdout accuracy: 99.01%
- Unseen-patient ROC-AUC: 0.9993

## Important limitation

The heart-failure and healthy recordings come from different PhysioNet databases. Performance may partly reflect differences in recording systems, acquisition conditions, or dataset composition rather than only heart-failure physiology.

This repository is an experimental research prototype and is not a medical device or clinically validated diagnostic system.

## Installation

```bash
python -m venv hf_env
hf_env\Scripts\activate
python -m pip install -r requirements.txt
```

## Main scripts

- `02_extract_windows.py`
- `03_extract_healthy_windows.py`
- `04_build_training_dataset.py`
- `05_clean_dataset.py`
- `06_train_baseline.py`
- `07_group_cross_validation.py`
- `08_compare_models.py`
- `09_shap_analysis.py`
- `10_feature_selection.py`
- `11_train_final_model.py`
- `12_test_single_record.py`
- `13_test_unseen_patients.py`
- `14_create_evaluation_figures.py`
- `15_dataset_bias_check.py`

## Disclaimer

For research and educational use only.
