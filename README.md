# Sleep Apnea Screening Workbench

A Streamlit application with:

1. A **pre-trained lifestyle SVM** that takes the six inputs used by the linked SleepSense app—gender, age, BMI category, sleep duration, resting heart rate, and daily steps—plus its three optional symptom prompts.
2. A separate EEG input and feature viewer for CSV samples or a configurable FPGA serial stream.

The lifestyle result is based on lifestyle data only. EEG features are not included in its score, and there is no trained EEG apnea classifier in this release. The two outputs must not be interpreted as a combined or EEG-based prediction.

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

The EEG viewer accepts one numeric channel as a CSV column or captures from a serial port with one ASCII decimal sample per newline. It also includes a deterministic synthetic signal option for a one-time software demo; this is not real EEG and is not apnea-labeled. Port, baud rate, channel label, sample rate, and analysis-window duration are configurable. The FPGA protocol was not provided; binary frames, checksums, and multiplexed channels need a matching parser in `fpga_serial.py`.

The viewer computes windowed band-pass/DWT features (five EEG bands and six statistics per band) for inspection or export. It does not classify apnea. For that, a separately trained EEG model and appropriate EEG recordings with time-aligned respiratory-event labels are required. The [SHHS polysomnography database](https://physionet.org/content/shhpsgdb/1.0.0/) includes EEG and respiratory-event annotations; access to the larger cohort is through [NSRR](https://sleepdata.org/datasets/shhs) subject to approval and data-use terms. The Kaggle lifestyle labels cannot substitute for EEG event labels.

No artifact rejection, electrode-quality checks, multi-channel processing, AHI calculation, or clinical severity estimate is performed. This is a research screening aid, not a medical device or diagnosis. Uploaded signals are not persisted by the app.
