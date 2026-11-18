# Sleep Apnea Screening Workbench

A Streamlit application with:

1. A **pre-trained lifestyle SVM** that takes the six inputs used by the linked SleepSense app—gender, age, BMI category, sleep duration, resting heart rate, and daily steps—plus its three optional symptom prompts.
2. One screening interface for optional EEG, ECG, and SpO₂ CSV/FPGA inputs, demo data, a show/hide performance-metrics button, and session-only screening history.

The lifestyle result is based on lifestyle data only. EEG, ECG, and SpO₂ are recorded and summarized for context but are not included in its score; there is no trained multimodal apnea classifier in this release. Demo biosignals are synthetic and are not apnea-labeled. Neither signal data nor demo waveforms affect the SVM prediction.

## Run

Python 3.12 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run ui.py
```

The pre-trained JSON model is bundled at `models/lifestyle_svm.json`. It is a non-executable model representation, so inference does not deserialize a `joblib` or pickle file.

## Lifestyle model and limitations

The linked [SleepSense repository](https://github.com/sarbojitrana/sleep-apnea-detector) uses a Kaggle sleep-health/lifestyle dataset. Its data includes no EEG waveform, airflow, SpO₂, or apnea event timestamps. The bundled six-input SVM was trained from the dataset copy published in that repository, using the same six input fields, grouped cross-validation, an RBF/linear SVM search, and the zero decision-margin threshold. It screens for a pattern associated with the dataset's `Sleep Apnea` label versus its other labels.

The displayed grouped-CV F2 value is used during model selection on the full dataset; it is **not** a held-out test result or a clinical performance estimate. The source dataset has 374 rows and repeated profiles. The model abstains when a submitted value is outside the training support. A lower signal does not rule out apnea. Optional loud-snoring, witnessed-pause, and daytime-sleepiness inputs only prompt follow-up and do not alter the SVM prediction.

The model can be rebuilt with the same source CSV:

```bash
python lifestyle_model.py --data /path/to/Sleep_health_and_lifestyle_dataset.csv
```

The CSV is not bundled separately. The rebuild command writes a portable JSON artifact and does not report its tuning score as held-out validation.

## EEG / FPGA input

The unified form accepts one numeric CSV channel for EEG, ECG, or SpO₂. EEG can also be captured from an FPGA serial port that sends one ASCII decimal sample per newline. The Add example data and Random valid values buttons populate lifestyle fields from the actual model support bounds and select deterministic synthetic previews for EEG, ECG, and SpO₂. Synthetic previews are for testing the UI only; they are not real recordings or apnea labels. Serial port, baud rate, and EEG sample rate are configurable. The FPGA protocol was not provided; binary frames, checksums, and multiplexed channels need a matching parser in `fpga_serial.py`.

The Model and performance tab explains the SVM and dataset and has a Show performance metrics button. The History tab keeps up to 100 screening entries in the current Streamlit session and offers CSV export. It does not store raw signal samples or persist data to disk. See [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) for the full metrics and limitations.

The viewer computes windowed band-pass/DWT features (five EEG bands and six statistics per band) for inspection or export. It does not classify apnea. For that, a separately trained EEG model and appropriate EEG recordings with time-aligned respiratory-event labels are required. The [SHHS polysomnography database](https://physionet.org/content/shhpsgdb/1.0.0/) includes EEG and respiratory-event annotations; access to the larger cohort is through [NSRR](https://sleepdata.org/datasets/shhs) subject to approval and data-use terms. The Kaggle lifestyle labels cannot substitute for EEG event labels.

No artifact rejection, electrode-quality checks, multi-channel processing, AHI calculation, or clinical severity estimate is performed. This is a research screening aid, not a medical device or diagnosis. Uploaded signals are not persisted by the app.
