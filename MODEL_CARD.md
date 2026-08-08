# HF-Watch-AI Model Card

## Model

**HF-Watch-AI v1.1 / LR-top20** is an experimental logistic-regression classifier that maps HRV features extracted from 5-minute ECG windows to a CHF-like versus healthy-like research classification.

## Intended use

- Educational and research experimentation with ECG-derived HRV.
- Reproducible evaluation of a heart-failure-related HRV classifier.
- Demonstration of signal processing, feature extraction, explainability, cross-database validation, and descriptive full-record consistency analysis.

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
- For full-record analysis: window agreement, descriptive pattern consistency, score spread, and score range.

The score is a model output, not a clinical disease probability. Full-record consistency measures agreement among analyzed windows; it does **not** estimate the probability that the model is correct.

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

Record-level confusion counts were 49 true negatives, 5 false positives, 8 false negatives, and 21 true positives.

The external positive and negative cohorts still originate from different databases, so this does not eliminate source bias.

## Development-only robustness experiments

### Regularization

A patient-grouped development benchmark compared logistic-regression `C` values from 0.01 to 10. `C=10` was numerically best, but its accuracy improvement over the deployed `C=1` model was only about 0.08 percentage points. The deployed model was therefore retained rather than changed for a negligible gain.

### Probability calibration

A grouped development-only benchmark compared the current uncalibrated model with sigmoid and isotonic calibration. Lower Brier score and log loss were better.

- Uncalibrated: Brier 0.006144, log loss 0.028100
- Sigmoid: Brier 0.008103, log loss 0.033361
- Isotonic: Brier 0.008215, log loss 0.063078

The uncalibrated model performed best in this benchmark, so no calibration layer was added. This does **not** make the output a clinically calibrated disease probability.

### Record aggregation

Mean score, median score, and majority-window aggregation all achieved 100% record-level performance on the development grouped benchmark. Because no method showed an advantage, the simple majority-window research summary was retained.

## External error and consistency audit

A descriptive audit examined the 13 misclassified external records without changing the model, threshold, features, calibration, or aggregation rule.

- Misclassified external records: 13/83
- False negatives: 8
- False positives: 5

The errors were not explained by a simple threshold issue: some incorrect records were close to 0.50, while others were strongly on the wrong side of the threshold.

The full-record consistency heuristic was then audited descriptively on the external set. Among the 13 misclassified records:

- 10 were `Low / mixed`
- 1 was `Moderate`
- 2 were `High`

Among the 70 correctly classified records, 18 were `Low / mixed`.

This shows that low window agreement can be a useful warning that a record is difficult or internally heterogeneous, but it is **not a correctness detector**. Two false-negative CHF records (`chf201` and `chf219`) were classified incorrectly despite `High` consistency. Therefore:

> **High consistency means the analyzed windows agree with each other. It does not mean the model is correct.**

The High/Moderate/Low labels are presentation heuristics only. They were not optimized on the external test set and are not validated uncertainty or clinical thresholds.

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
- No subgroup fairness study has established performance across age, sex, comorbidity, device, acquisition site, or rhythm groups.
- Development-only calibration experiments do not establish clinical probability calibration.
- Full-record majority-window aggregation is a research summary rule, not a clinically validated patient-level decision rule.
- Pattern consistency reflects within-record agreement, not correctness or clinical confidence.

## Versioning

Current application/model interface: **HF-Watch-AI v1.1 / LR-top20**.
