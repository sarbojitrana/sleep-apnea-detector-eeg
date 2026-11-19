# EEG SVM performance report

This is a working research baseline with **moderate accuracy**, not demonstrated high clinical performance. Scores below were computed from real EEG-derived features; synthetic demos contribute no training or evaluation rows.

A subsequent 252-configuration development-only search showed only a small validation gain. The evaluated model and test scores below are retained. See [EEG_OPTIMIZATION_REPORT.md](EEG_OPTIMIZATION_REPORT.md) for the attempts and reproducibility details.

## Dataset and task

Original source: [MIT-BIH Polysomnographic Database v1.0.0](https://physionet.org/content/slpdb/1.0.0/). This replaces a proposed Kaggle EEG source; the existing Kaggle lifestyle data cannot train an EEG classifier.

All ten recordings with C3–O1 EEG and available apnea annotations were included: slp03, slp04, slp14, slp16, slp48, slp59, slp60, slp61, slp66 and slp67x. The original database describes a small, male-only cohort. Montage selection and exclusion rules were fixed before evaluation. See `models/eeg_dataset_provenance.json` for per-record counts and raw-file hashes.

- Total: 3,376 non-overlapping sleep windows, 30 seconds each at 250 Hz.
- Development: 2644 windows from eight subjects.
- Test: 732 windows from **slp03, slp16**, including 485 normal and 247 apnea windows.
- Ground truth: OA/X/CA/CAA annotations are positive. Sleep without respiratory-event codes is negative. Hypopnea-only, wake, movement and unknown epochs are excluded. Each annotation labels its following 30 seconds; sample timestamps are used directly.
- Startup: the first 30 seconds are excluded where otherwise eligible. The causal filter runs continuously per recording.

## Selection and evaluation

A deterministic subject-grouped split reserves test subjects before fitting. Four-fold subject-grouped development CV searches 144 candidates. The scaler and ANOVA feature selector are fitted inside each training fold. Hyperparameters are selected using mean **development balanced accuracy**, not test accuracy. The saved model retains the development fit; it is not refit on test data.

Selected settings: `{"select__k": 10, "svc__C": 0.1, "svc__class_weight": "balanced", "svc__gamma": 0.01, "svc__kernel": "rbf", "transform__logarithmic": true}`.

Selected features: E_A5, K_A5, E_D5, K_D5, K_D4, M_D3, K_D3, E_D2, M_D2, K_D2.

Development CV balanced accuracy: **62.5%**, fold standard deviation **1.7%**. This is a tuning score, not an independent test estimate.

An initial model without logarithmic transformation was inspected during implementation. The subsequent transform comparison was selected only by development CV. Consequently, this fixed holdout has been viewed during development: treat the reported result as provisional, and obtain a new external cohort for a final unbiased evaluation. No alternate splits were searched to improve the displayed test score.

## Fixed subject-wise holdout

| Metric | Measured result |
|---|---:|
| Accuracy | 74.0% |
| Balanced accuracy | 75.5% |
| Sensitivity / recall | 80.2% |
| Specificity | 70.9% |
| Precision | 58.4% |
| F1 | 67.6% |
| ROC AUC | 82.8% |
| Average precision | 65.7% |
| Majority-class baseline accuracy | 66.3% |

Confusion matrix: actual rows / predicted columns, [normal, apnea]:

```text
[[344, 141], [49, 198]]
```

The model missed **49 of 247** apnea windows and flagged **141 of 485** normal windows. The 66.3% majority-class accuracy is a reference baseline, not an acceptable clinical operating point.

| Subject | Windows | Accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| slp03 | 387 | 78.0% | 13.6% | 86.3% |
| slp16 | 345 | 69.6% | 94.6% | 33.8% |

The subject breakdown is critical: pooled sensitivity masks poor sensitivity for slp03. This model should not be described as consistently accurate across patients. Correlated windows from only two test subjects do not support a precise population-level confidence interval.

## Performance

- Median validated 15-feature-to-prediction latency: **5.05 ms/window**; p95 **5.77 ms** on the training host, over 30 single-window calls.
- Training time: approximately **90.9 seconds** for the full search on this host.
- Support vectors: **2181**. Inference bounds intermediate kernel matrices to 128 windows per chunk.
- Artifact scores were compared against scikit-learn decision scores before saving, with tolerance 1e-7.

Latency includes feature validation, transformation, scaling, selection and the NumPy SVM. It excludes raw EEG processing, FPGA latency, serial capture, CSV parsing and UI rendering. Hardware throughput and numerical accuracy were not measured.

## FPGA compatibility and limits

Training uses an explicit floating-point Python reference: causal Butterworth filtering, five-level Haar DWT with symmetric extension, total energy, population mobility and Pearson kurtosis. The precise coefficients, coefficient counts and units are in `models/eeg_feature_contract.json`. Raw EEG processing is offline only; runtime starts with the FPGA parameters.

**Fixed-point RTL parity has not been established.** The model may be unsuitable for actual FPGA output if the hardware uses different filter coefficients, rounding, scales, windows, wavelet normalization, boundary handling or kurtosis definitions. Compare FPGA output against the reference and retrain on compatible features before hardware validation.

Other limits: small, narrow cohort; no artifact rejection beyond unusable/invalid windows; no clinical severity, AHI or event-duration estimation; no validation of a combined EEG/lifestyle/ECG/SpO₂ model. Hypopneas are outside this classifier's evaluated task. Demos are invented perturbed parameters with no known labels and cannot establish accuracy.

For reproducible download, feature extraction and retraining commands, see [README.md](README.md). Dataset attribution and license are included there and in the provenance manifest.
