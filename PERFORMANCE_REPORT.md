# Lifestyle SVM Performance Report

## Summary

The current application is **not demonstrated to be better than the prior IoT/lifestyle model**. It uses the same six lifestyle inputs and the linked SleepSense dataset. EEG is not part of this SVM and does not change its result.

The numbers below are a reproducible grouped evaluation of the six-input SVM procedure on the linked 374-row dataset. They assess agreement with that dataset's `Sleep Apnea` label, not clinical diagnosis. These metrics do not establish that adding EEG improves performance.

## Dataset and method

- Source: [`Sleep_health_and_lifestyle_dataset.csv`](https://github.com/sarbojitrana/sleep-apnea-detector/blob/main/data/Sleep_health_and_lifestyle_dataset.csv)
- 374 rows, 106 distinct six-input profiles; repeated profiles are kept together across splits.
- Target: `Sleep Apnea` versus the dataset's other labels (`None` and `Insomnia`).
- Inputs: gender, age, sleep duration, BMI category, heart rate, daily steps.
- StandardScaler + class-balanced SVM; linear/RBF kernel grid tuned on F2.
- Fixed grouped holdout: 300 development rows and 74 test rows (22 profiles), selected before model tuning.
- Nested development cross-validation: four outer grouped folds, with model tuning repeated inside each outer training fold.
- Evaluation was run locally using the linked repository's documented split, tuning, and scoring procedure. The results match its published lifestyle-model benchmark.

## Results

| Metric | Fixed grouped holdout | Nested development CV (pooled) |
|---|---:|---:|
| Accuracy | 90.5% | 90.7% |
| Sensitivity / recall | 100.0% | 85.7% |
| Specificity | 88.1% | 92.0% |
| Precision | 68.2% | 74.0% |
| F1 | 81.1% | 79.4% |
| F2 | 91.5% | 83.1% |
| ROC AUC | 98.1% | 91.6% |
| Average precision | 87.2% | 79.4% |

Holdout confusion matrix (actual rows, predicted columns; other label then apnea label):

```text
[[52, 7],
 [ 0,15]]
```

In this holdout, all 15 apnea-labeled rows were flagged, while seven other-labeled rows were also flagged. The holdout has only 15 positive rows, profiles repeat, and some profiles have conflicting labels. A descriptive row-level sensitivity interval reported by the source is approximately 79.6%–100%; row duplication further limits its interpretation. The nested CV missed 9 of 63 positive rows.

## Comparison with the previous application

**No measured performance improvement is established.** This app reuses the same six-input task and source dataset; the EEG feature viewer is separate and contributes no signal to the classifier. There is no paired test on a new, independent cohort comparing the previous model to the current app. The results should therefore be read as a benchmark of the lifestyle model, not as evidence of improvement from EEG.

The linked repository also now documents a separate overnight SpO₂ model evaluated on a different dataset and task. Those SpO₂ results are not directly comparable to this lifestyle model.

## Interpretation and limitations

- This is a screening pattern prediction for labels in a small lifestyle dataset, not a medical diagnosis or an apnea-event detector.
- It cannot calculate AHI, identify individual apnea events, estimate clinical severity, or rule out apnea.
- The source data has no EEG, airflow, SpO₂, or event timestamps. The current model cannot make an EEG-based prediction.
- The bundled artifact was refit on all 374 rows after grouped cross-validation tuning. The fixed-holdout metrics above come from a separate fixed split using the same procedure; they are not a claim that the artifact was evaluated on unseen clinical data.
- The model abstains when a numeric input is outside the training support. Its training support includes ages 27–59, sleep duration 5.8–8.5 hours, heart rate 65–86 bpm, and steps 3,000–10,000/day.

Sources: [SleepSense README and measured lifestyle-model benchmark](https://github.com/sarbojitrana/sleep-apnea-detector#measured-dataset-performance); source dataset linked above.
