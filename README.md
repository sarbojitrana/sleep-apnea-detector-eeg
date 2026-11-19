# Sleep Apnea Screening Workbench

A Streamlit workbench with the existing Screening, History, and Model and performance tabs. The lifestyle form and model remain available. EEG now has its own trained SVM that starts **after the FPGA supplies its numerical features**. ECG and SpO₂ remain contextual signals; classifier outputs are displayed separately.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run ui.py
```

Both portable, non-executable JSON models are bundled in `models/`. Inference does not load pickle/joblib artifacts. Select **Add example data** for a reproducible demo or **Random valid values** for fresh lifestyle values and synthetic EEG parameters, then **Run screening**. Synthetic inputs and exports are marked as demonstrations and have no known normal/apnea ground truth. The EEG demos perturb unlabelled development examples to preserve plausible feature scales; they do not contribute to model training or metrics.

## EEG / FPGA input

Each row represents one EEG analysis window, with these 15 **decoded and unnormalized** values:

```text
E_A5,M_A5,K_A5,E_D5,M_D5,K_D5,E_D4,M_D4,K_D4,E_D3,M_D3,K_D3,E_D2,M_D2,K_D2
```

The UI accepts a named feature CSV, manual entry, a synthetic demo, or FPGA serial capture. CSV columns can be reordered because names determine order; all 15 features must be present. Missing, nonnumeric, nonfinite, negative and all-zero-energy windows are rejected rather than silently discarded. Uploads are limited to 10 MB / 1,000 windows. Sample CSVs are in `examples/`; the recorded example contains real EEG-derived feature rows, while the synthetic example contains a provenance marker.

Serial format is **one newline-terminated ASCII CSV line with exactly 15 numbers in the order above**. An exact header is optional. Capture buffers fragmented lines, rejects invalid or incomplete frames and caps frame/window sizes. This replaces the previous one-raw-sample-per-line parser. Binary framing and fixed-point decoding require adaptation to the actual hardware protocol. The port is opened by the machine running Streamlit.

Runtime processing: validate → optional learned-choice `log1p` transform → training-derived StandardScaler → selected features → SVM → **0=normal / 1=apnea**. Positive decision margin means apnea; this is not a probability. Values beyond development ranges are flagged. Window counts are not AHI or a severity estimate.

### Matching the FPGA

See `models/eeg_feature_contract.json` for the complete reference convention, including SOS coefficients. The bundled model uses:

- C3–O1 EEG, 250 Hz, 30-second windows, amplitude in µV.
- Causal continuous Butterworth filtering, 0.5–45 Hz, prototype order 4 / bandpass order 8, initialized once per recording.
- Five-level orthonormal Haar DWT with symmetric boundary extension; A5/D5/D4/D3/D2 coefficient counts are 235/235/469/938/1875.
- Total squared-coefficient energy, population Hjorth mobility without an additional sample-rate factor, and ordinary Pearson population kurtosis (Gaussian value 3).
- Zero-variance bands produce mobility and kurtosis zero. Entire constant windows are rejected by the training reference.

These are explicit software reference choices where the project summary leaves details open. **RTL parity has not been established.** Changes to FPGA filtering, coefficient normalization, window length, units, fixed-point scaling, rounding, overflow, or kurtosis conventions require feature comparisons and potentially retraining. Hardware must send physical feature values, not already standardized features. Do not apply normalization twice.

Raw EEG processing in `prepare_eeg_dataset.py` is an offline training reference, not part of the runtime UI. The actual FPGA preprocessing coefficients were not provided.

## Training data and reproducibility

The bundled lifestyle model retains the original Kaggle-derived Sleep Health and Lifestyle dataset. That dataset has no EEG, so the separate EEG model was trained on the original [MIT-BIH Polysomnographic Database v1.0.0](https://physionet.org/content/slpdb/1.0.0/), rather than an unverified Kaggle mirror. It provides EEG sampled at 250 Hz and aligned sleep/apnea annotations.

The EEG subset has **3,376 real windows from 10 subjects**, all using C3–O1. Other montages and records without apnea annotations are excluded. Sleep epochs with OA/X/CA/CAA are apnea; sleep epochs without respiratory-event codes are normal. Hypopnea-only, wake, movement, unknown, incomplete and filter-startup epochs are excluded. Thus this is an apnea-versus-normal **sleep-window** task, not a detector for every form of sleep-disordered breathing.

```bash
python -m pip install -r requirements-training.txt
python prepare_eeg_dataset.py --download
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python eeg_apnea.py \
  --data data/processed/slpdb/eeg_features.csv \
  --contract data/processed/slpdb/feature_contract.json \
  --source https://physionet.org/content/slpdb/1.0.0/
```

The download is approximately 380 MB; raw recordings and processed training data stay in ignored `data/` directories. Record hashes, source attribution and inclusion/exclusion decisions are recorded in the provenance manifest. `models/eeg_dataset_provenance.json` preserves the manifest for the bundled model.

To use a different labeled EEG dataset, first produce the same 15 features using the actual FPGA-compatible extraction. Supply `subject_id` and binary `label` columns, a complete feature contract JSON, and its source. Subject IDs must identify people consistently across their recordings. At least six subjects and both classes in each split are required. Do not relabel sleep-stage targets, lifestyle labels or ECG-only data as EEG apnea ground truth. Explicit synthetic/demo markers are rejected by training; arbitrary fabricated data cannot be detected automatically.

Scaling and feature selection are fitted inside subject-grouped development CV. The grid compares identity/log transforms, 5/10/all features, linear/RBF kernels, C, gamma and class weighting; it selects by balanced accuracy. The fixed test subjects are not used for fitting/tuning or subsequent all-data refitting. The saved artifact is the evaluated development model. Portable scores are checked against scikit-learn before saving.

## Performance and limits

The bundled EEG model measured **74.0% accuracy, 75.5% balanced accuracy, 80.2% sensitivity and 70.9% specificity** on 732 windows from two held-out subjects. Median full feature-to-prediction latency was approximately **5 ms/window** on the training host. These are provisional development results; performance varies strongly by subject. They do not establish high clinical accuracy or hardware validity. See [EEG_PERFORMANCE_REPORT.md](EEG_PERFORMANCE_REPORT.md) for the confusion matrix, subject breakdown and limitations.

The separate lifestyle model's fixed grouped holdout and nested CV results remain in [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md). Those metrics do not describe the EEG model. Neither model is clinically validated; no combined score is inferred.

An additional 252-configuration development-only search tested relative energies, robust scaling, broader SVM parameters, subject/class weighting and domain feature subsets. The validation gain was small, so the deployed model was retained. The performance panel displays the measured classification metrics. See [EEG_OPTIMIZATION_REPORT.md](EEG_OPTIMIZATION_REPORT.md); reproduce the experiments with `python optimize_eeg_svm.py`. These experiments never replace the deployed model automatically.

History stores up to 100 summary entries in the current Streamlit session, including EEG window counts and synthetic provenance, with CSV export. It does not persist user recordings to disk.

## Dataset attribution

Moody, G. (1999). *MIT-BIH Polysomnographic Database*, version 1.0.0. PhysioNet. [doi:10.13026/C23K5S](https://doi.org/10.13026/C23K5S). Ichimaru Y, Moody GB. *Development of the polysomnographic database on CD-ROM*. Psychiatry and Clinical Neurosciences 53:175–177 (1999). The source database is distributed under the [Open Data Commons Attribution License v1.0](https://physionet.org/content/slpdb/view-license/1.0.0/). The sample feature rows and demonstration anchors are derived from that database. See the source page for its requested PhysioNet citation.

## Verification

```bash
python -m pytest -q
```

Tests cover feature contracts, causal reference processing, label alignment rules, subject separation and scaler isolation, portable/sklearn score agreement, malformed serial frames and fragmented reads, synthetic provenance, and existing lifestyle behavior. Hardware capture still requires an attached FPGA.
