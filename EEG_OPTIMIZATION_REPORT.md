# EEG optimization experiments

Additional preprocessing and SVM experiments were run to investigate possible performance improvements. The existing deployed model is retained; its held-out accuracy remains 74.0%, sensitivity 80.2%, specificity 70.9%, precision 58.4% and F1 67.6%.

## What was tested

All experiments use only the original eight development subjects and the same four subject-wise CV folds. The two previously inspected holdout subjects (`slp03`, `slp16`) are removed before validation, transformations, weighting, threshold selection or model fitting. No new holdout scores were used to select an experiment.

1. **144 additional preprocessing/SVM configurations:** relative band energy, log relative energy, centered log band energy, the existing log transform, standard/robust scaling, 5/10/all features, a wider C/gamma range and different class weights.
2. **108 additional configurations with subject/class-balanced fitting:** equal total fitting weight per subject and class, plus energy-only, mobility-only, kurtosis-only, energy/mobility, mobility/kurtosis and all-feature subsets.
3. **161 decision thresholds:** selected from out-of-fold development margins to improve the weakest of accuracy, sensitivity, specificity, precision and F1. This is threshold tuning, not an unbiased performance estimate; the deployed threshold is unchanged.

Relative energy uses only the current window's five energies divided by their sum. It neither uses future windows nor requires extra hardware outputs. Robust scaling is fitted exclusively to training folds. Synthetic demonstrations are never used for these experiments.

## Results

| Development experiment | Best mean subject-wise CV balanced accuracy |
|---|---:|
| Previously selected model | 62.5% |
| Additional preprocessing/SVM search | 62.8% |
| Subject/class weighting and feature subsets | 62.8% |

The best additional configurations reached 62.8% mean subject-wise validation balanced accuracy, compared with 62.5% previously. This small gain does not establish a reliable improvement.

The best preprocessing candidate's pooled out-of-fold predictions had 62.1% accuracy, 86.6% sensitivity, 42.5% specificity, 54.7% precision and 67.0% F1. Raising its threshold improved the balance of sensitivity/specificity but still produced only approximately 62.4% accuracy, 63.2% sensitivity, 61.8% specificity, 57.1% precision and 60.0% F1. These pooled tuning predictions are **not** held-out test metrics and must not replace the UI's evaluated-model results.

The small development-CV gain does not establish a reliable improvement. The deployed model, synthetic-demo source profiles and original holdout metrics are unchanged. No alternative random splits, relabeling, synthetic training data or held-out subject filtering were used to obtain higher displayed scores.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python optimize_eeg_svm.py
```

This saves a development-only audit to `reports/eeg_optimization.json`, including candidate counts, parameters, scores, subject IDs and baseline/data hashes. The script never replaces the deployed model. Each baseline stores its own development subject IDs; subjects outside that list are excluded before experiments. Custom CSVs still need matching subject IDs and the original feature schema.

The existing training/inference module now supports the additional window-local transforms and robust-scaler artifact export for future experiments. Tests verify energy-scale invariance, independence from other windows, portable model agreement and per-subject weight totals. The performance panel displays the measured classification metrics.

## What would support further improvement

These experiments suggest that tuning alone is insufficient with the current ten-subject subset and 15 descriptors. The next useful work is to validate FPGA/software feature agreement and signal quality, obtain a larger independently labeled cohort with compatible channels and units, and evaluate whether additional descriptors or temporal context improve prediction. A claim of improved performance requires evaluation on new subjects after development choices are fixed.

See [EEG_PERFORMANCE_REPORT.md](EEG_PERFORMANCE_REPORT.md) for the deployed model's actual confusion matrix and subject-level limitations.
