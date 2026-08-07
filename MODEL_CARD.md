# HF-Watch-AI Model Card

## Model

**HF-Watch-AI v1.1 / LR-top20** is an experimental logistic-regression classifier that maps HRV features extracted from 5-minute ECG windows to a CHF-like versus healthy-like research classification.

## Intended use

- Educational and research experimentation with ECG-derived HRV.
- Reproducible evaluation of a heart-failure-related HRV classifier.
- Demonstration of signal processing, feature extraction, explainability, and cross-database validation.

## Not intended for

- Diagnosis, screening, triage, treatment, or ruling out heart failure.
- Estimating a person's true medical probability of heart failure.
- Use as a medical device or as a substitute for clinical evaluation.

## Input

A complete 5-minute WFDB ECG window. The inference pipeline cleans the selected ECG channel, detects R-peaks, calculates R-R intervals and HRV features, and supplies 20 selected HRV features to the final model.

## Output

- CHF-like model score from 0 to 1.
- Binary research classification using a fixed 0.50 threshold.
- ECG/R-peak visualization and HRV features.
- Basic signal-quality indicators.
- Per-feature logistic-regression contributions to the model logit.

The score is a model output, not a calibrated clinical disease probability.

## Development data

- BIDMC Congestive Heart Failure Database.
- MIT-BIH Normal Sinus Rhythm Database.

The classes originate from different source databases, creating a substantial database-source-bias risk.

## External validation

The trained model was evaluated without retraining on long-term RR-interval databases using 5-minute windows:

- Congestive Heart Failure RR Interval Database (`chf2db`): 29 CHF records.
- Normal Sinus Rhythm RR Interval Database (`nsr2db`): 54 healthy records.

Record-level results across 83 records:

- Accuracy: 84.34% (95% bootstrap CI 75.90%–91.57%)
- Sensitivity: 72.41% (95% CI 55.56%–88.00%)
- Specificity: 90.74% (95% CI 82.14%–98.04%)
- Precision: 80.77% (95% CI 64.70%–95.45%)
- F1: 76.36% (95% CI 62.22%–87.50%)
- ROC-AUC: 0.8902 (95% CI 0.7918–0.9661)

The external positive and negative cohorts still originate from different databases, so this does not eliminate source bias.

## Quality controls

The application rejects windows with excessive missing data, too few detected beats, or a very high fraction of physiologically unusual R-R intervals. These checks are engineering safeguards, not validated clinical signal-quality criteria.

## Explainability

The application can display each selected feature's exact contribution to the logistic-regression logit after imputation and scaling. Positive values push the mathematical score toward CHF-like and negative values toward healthy-like. These contributions must not be interpreted as causal physiology.

## Known limitations

- Cross-database source bias remains.
- External RR validation uses provided beat annotations, while raw-ECG inference detects R-peaks algorithmically.
- Long recordings create repeated correlated windows; record-level metrics are therefore more informative than window-level metrics.
- The 0.50 threshold was not tuned on the external evaluation set.
- No prospective clinical validation has been performed.
- No subgroup fairness or calibration study has yet established performance across age, sex, comorbidity, device, acquisition site, or rhythm groups.
- Full-record majority-window aggregation is a research summary rule, not a clinically validated patient-level decision rule.

## Versioning

Current application/model interface: **HF-Watch-AI v1.1 / LR-top20**.
