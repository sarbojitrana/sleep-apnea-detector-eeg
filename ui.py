"""Lifestyle screening and FPGA-feature EEG classification with session history."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app_inputs import example_supported_record, random_supported_record, summarize_signal, synthetic_signal
from eeg_apnea import (
    BANDS, FEATURE_NAMES, MODEL_PATH as EEG_MODEL_PATH,
    load_model as load_eeg_model, predict_features,
    synthetic_features, validate_features,
)
from fpga_serial import capture_features
from lifestyle_model import DATASET_URL, SYMPTOMS, load_model, predict

st.set_page_config(page_title="Sleep Apnea Screening", layout="wide")
st.title("Sleep Apnea Screening Workbench")
st.caption(
    "One-page lifestyle screening with EEG window classification, ECG and SpO₂ context, "
    "model evidence, and session history."
)
st.warning(
    "EEG windows are classified separately from the lifestyle screening. "
    "ECG and SpO₂ remain informational. Synthetic inputs are for demonstration only. "
    "This research screening is not a diagnosis."
)


@st.cache_resource
def get_model():
    return load_model()


try:
    screening_model = get_model()
except (FileNotFoundError, ValueError) as exc:
    screening_model = None
    st.error(f"The trained lifestyle model could not be loaded: {exc}")

@st.cache_resource
def get_eeg_model(modification_time: int):
    return load_eeg_model()


try:
    eeg_model = get_eeg_model(EEG_MODEL_PATH.stat().st_mtime_ns) if EEG_MODEL_PATH.exists() else None
except (OSError, ValueError) as exc:
    eeg_model = None
    st.error(f"The EEG model could not be loaded: {exc}")

if "screening_history" not in st.session_state:
    st.session_state.screening_history = []
if "last_screening" not in st.session_state:
    st.session_state.last_screening = None
for key, value in {
    "input_gender": "Female",
    "input_age": 40,
    "input_bmi": "Normal",
    "input_sleep_duration": 7.0,
    "input_heart_rate": 70,
    "input_daily_steps": 7000,
    "input_loud_snoring": False,
    "input_witnessed_pauses": False,
    "input_daytime_sleepiness": False,
    "signal_EEG_mode": "Not supplied",
    "signal_ECG_mode": "Not supplied",
    "signal_SpO2_mode": "Not supplied",
    "signal_EEG_rate": 250.0,
    "eeg_demo_seed": 42,
    "signal_ECG_rate": 250.0,
    "signal_SpO2_rate": 1.0,
    "show_metrics": False,
}.items():
    st.session_state.setdefault(key, value)


def set_sample_values(values: dict) -> None:
    for key in ("gender", "age", "bmi", "sleep_duration", "heart_rate", "daily_steps"):
        st.session_state[f"input_{key}"] = values[key]
    for key in SYMPTOMS:
        st.session_state[f"input_{key}"] = values[key]
    for kind in ("EEG", "ECG", "SpO2"):
        st.session_state[f"signal_{kind}_mode"] = "Synthetic demo"
    st.session_state.eeg_demo_seed = int(np.random.default_rng().integers(0, 2**31))
    st.session_state.last_screening = None


def random_values_callback() -> None:
    if screening_model is not None:
        set_sample_values(random_supported_record(screening_model))


def example_values_callback() -> None:
    set_sample_values(example_supported_record())
    st.session_state.eeg_demo_seed = 42


def toggle_metrics_callback() -> None:
    st.session_state.show_metrics = not st.session_state.show_metrics


screen_tab, history_tab, model_tab = st.tabs(
    ["Screening", "History", "Model and performance"]
)

with screen_tab:
    st.subheader("Inputs")
    st.caption(
        "The lifestyle values below are required by the bundled model. "
        "For a reproducible demo, use Add example data; for fresh values within training "
        "support, use Random valid values."
    )
    example_col, random_col = st.columns(2)
    example_col.button(
        "Add example data",
        on_click=example_values_callback,
        disabled=screening_model is None,
        width="stretch",
    )
    random_col.button(
        "Random valid values",
        on_click=random_values_callback,
        disabled=screening_model is None,
        width="stretch",
    )

    fields = st.columns(3)
    gender = fields[0].selectbox(
        "Gender category",
        ("Female", "Male"),
        key="input_gender",
    )
    age = fields[1].number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        key="input_age",
    )
    bmi = fields[2].selectbox(
        "BMI category",
        ("Normal", "Overweight", "Obese"),
        key="input_bmi",
    )
    sleep_duration = fields[0].number_input(
        "Sleep duration (hours/night)",
        min_value=0.5,
        max_value=16.0,
        step=0.1,
        key="input_sleep_duration",
    )
    heart_rate = fields[1].number_input(
        "Resting heart rate (bpm)",
        min_value=30,
        max_value=220,
        key="input_heart_rate",
    )
    daily_steps = fields[2].number_input(
        "Daily steps",
        min_value=0,
        max_value=100000,
        step=100,
        key="input_daily_steps",
    )

    with st.expander("Optional symptoms"):
        symptom_columns = st.columns(3)
        symptoms = {
            "loud_snoring": symptom_columns[0].checkbox(
                "Loud snoring",
                key="input_loud_snoring",
            ),
            "witnessed_pauses": symptom_columns[1].checkbox(
                "Witnessed breathing pauses",
                key="input_witnessed_pauses",
            ),
            "daytime_sleepiness": symptom_columns[2].checkbox(
                "Daytime sleepiness",
                key="input_daytime_sleepiness",
            ),
        }

    signal_summaries: dict[str, dict] = {}
    eeg_frame = None
    with st.expander("Optional biosignals — EEG parameters, ECG and SpO₂ context"):
        st.markdown(
            "EEG accepts the 15 decoded FPGA parameters per analysis window. "
            "ECG and SpO₂ accept one numeric sample column; SpO₂ uses percent values."
        )
        signal_columns = st.columns(3)
        signal_inputs: dict[str, tuple[str, float, Any, str | None]] = {}
        for column, kind in zip(signal_columns, ("EEG", "ECG", "SpO2")):
            column.markdown(f"**{kind}**")
            modes = ("Not supplied", "Upload CSV", "Synthetic demo")
            if kind == "EEG":
                modes = (*modes, "Manual parameters", "FPGA serial")
            mode = column.selectbox(f"{kind} source", modes, key=f"signal_{kind}_mode")
            if kind == "EEG":
                column.caption(
                    "Energy, mobility and Pearson kurtosis for A5, D5, D4, D3 and D2. "
                    "Provide unnormalized values in the training feature units."
                )
                if mode == "Upload CSV":
                    uploaded = column.file_uploader("EEG 15-feature CSV", type=["csv"], key="signal_EEG_file")
                    if uploaded is not None:
                        try:
                            if uploaded.size > 10_000_000:
                                raise ValueError("EEG CSV must be no larger than 10 MB.")
                            uploaded.seek(0)
                            eeg_frame = validate_features(pd.read_csv(uploaded))
                            if len(eeg_frame) > 1000:
                                raise ValueError("Use at most 1,000 feature windows per screening.")
                        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                            eeg_frame = None
                            column.error(f"Could not read EEG features: {exc}")
                elif mode == "Synthetic demo":
                    eeg_frame = synthetic_features(seed=st.session_state.eeg_demo_seed, model=eeg_model)
                    column.info("Invented feature values for display; no ground-truth labels.")
                elif mode == "Manual parameters":
                    row = {}
                    for band in BANDS:
                        column.markdown(f"**{band}**")
                        for statistic in ("E", "M", "K"):
                            name = f"{statistic}_{band}"
                            row[name] = column.number_input(
                                name, min_value=0.0, value=0.0,
                                format="%.6f", key=f"eeg_manual_{name}",
                            )
                    eeg_frame = pd.DataFrame([row])
                elif mode == "FPGA serial":
                    eeg_frame = st.session_state.get("captured_eeg_features")
                if eeg_frame is not None:
                    column.caption(f"{len(eeg_frame)} EEG feature window(s)")
                    column.dataframe(eeg_frame.head(20), width="stretch", hide_index=True)
                column.download_button(
                    "Download synthetic EEG example CSV",
                    synthetic_features(model=eeg_model).assign(source="synthetic demo").to_csv(index=False).encode(),
                    file_name="synthetic_fpga_features.csv", mime="text/csv",
                )
                signal_inputs[kind] = (mode, 250.0, None, None)
                continue
            rate = column.number_input(
                f"{kind} sample rate (Hz)", min_value=0.1 if kind == "SpO2" else 1.0,
                max_value=4096.0, step=1.0, key=f"signal_{kind}_rate",
            )
            uploaded, selected_column = None, None
            if mode == "Upload CSV":
                uploaded = column.file_uploader(f"{kind} one-channel CSV", type=["csv"], key=f"signal_{kind}_file")
                if uploaded is not None:
                    try:
                        if uploaded.size > 10_000_000:
                            raise ValueError(f"{kind} CSV must be no larger than 10 MB.")
                        uploaded.seek(0)
                        signal_frame = pd.read_csv(uploaded)
                        numeric_columns = [name for name in signal_frame.columns
                                           if pd.to_numeric(signal_frame[name], errors="coerce").notna().any()]
                        if not numeric_columns:
                            raise ValueError(f"{kind} CSV has no numeric sample column.")
                        selected_column = column.selectbox(f"{kind} sample column", numeric_columns, key=f"signal_{kind}_column")
                    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                        column.error(f"Could not read {kind} CSV: {exc}")
            signal_inputs[kind] = (mode, float(rate), uploaded, selected_column)

    eeg_mode = signal_inputs["EEG"][0]
    if eeg_mode == "FPGA serial":
        st.markdown(
            "FPGA format: one ASCII CSV line per window with 15 decoded, unnormalized values "
            "in this order: " + ", ".join(FEATURE_NAMES) + ". End each line with a newline."
        )
        serial_columns = st.columns(3)
        serial_port = serial_columns[0].text_input("Serial port", key="fpga_port", placeholder="/dev/ttyUSB0")
        baud_rate = serial_columns[1].number_input("Baud rate", min_value=1, max_value=4_000_000, value=115200, key="fpga_baud")
        capture_duration = serial_columns[2].number_input(
            "Capture duration (seconds)", min_value=1.5, max_value=600.0, value=35.0, key="fpga_duration",
        )
        if st.button("Capture EEG from FPGA"):
            st.session_state.pop("captured_eeg_features", None)
            eeg_frame = None
            try:
                with st.spinner("Waiting for FPGA feature windows…"):
                    eeg_frame = capture_features(serial_port, int(baud_rate), float(capture_duration))
                st.session_state.captured_eeg_features = eeg_frame
                st.success(f"Captured {len(eeg_frame)} EEG feature windows.")
            except (ValueError, RuntimeError) as exc:
                st.error(f"FPGA capture failed: {exc}")
    if eeg_mode != "Not supplied":
        if eeg_model is None:
            st.info("EEG features can be previewed. Classification requires models/eeg_svm.json; see Model and performance.")
        else:
            contract = eeg_model["contract"]
            st.caption(
                f"EEG model input: {contract['channel']}, {contract['sampling_rate_hz']:g} Hz, "
                f"{contract['window_seconds']:g}-second windows, {contract['amplitude_unit']}. "
                "FPGA feature definitions and scaling must match; full contract is in Model and performance."
            )

    predict_button = st.button(
        "Run screening",
        type="primary",
        disabled=screening_model is None,
        width="stretch",
    )
    if predict_button and screening_model is not None:
        record = {
            "gender": gender,
            "age": age,
            "bmi": bmi,
            "sleep_duration": sleep_duration,
            "heart_rate": heart_rate,
            "daily_steps": daily_steps,
            **symptoms,
        }
        try:
            result = predict(screening_model, record)
            eeg_result = None
            if eeg_mode != "Not supplied":
                if eeg_frame is None:
                    raise ValueError("Supply valid EEG parameters or capture FPGA feature windows first.")
                features = validate_features(eeg_frame)
                predictions = predict_features(eeg_model, features) if eeg_model is not None else None
                # Uploaded synthetic exports retain their provenance marker.
                is_synthetic = eeg_mode == "Synthetic demo"
                if eeg_mode == "Upload CSV":
                    upload = st.session_state.get("signal_EEG_file")
                    if upload is not None:
                        upload.seek(0)
                        original = pd.read_csv(upload)
                        is_synthetic = "source" in original and original["source"].astype(str).str.contains("synthetic|demo", case=False).any()
                if predictions is not None:
                    predictions["source"] = eeg_mode
                    predictions["synthetic"] = bool(is_synthetic)
                eeg_result = {
                    "source": eeg_mode, "synthetic": bool(is_synthetic), "windows": len(features),
                    "features": features.to_dict(orient="records"),
                    "predictions": predictions.to_dict(orient="records") if predictions is not None else [],
                    "apnea_windows": int(predictions["prediction"].sum()) if predictions is not None else None,
                    "model_created_at": eeg_model["created_at"] if eeg_model is not None else None,
                }
            for kind, (mode, rate, uploaded, selected_column) in signal_inputs.items():
                if kind == "EEG" or mode == "Not supplied":
                    continue
                if mode == "Synthetic demo":
                    samples = synthetic_signal(kind, rate, 30.0, seed=st.session_state.eeg_demo_seed)
                    signal_summaries[kind] = summarize_signal(samples, rate, kind)
                    signal_summaries[kind]["source"] = "synthetic demo"
                    continue
                if uploaded is None or selected_column is None:
                    raise ValueError(f"Select a valid {kind} CSV sample column or a demo source.")
                if uploaded.size > 10_000_000:
                    raise ValueError(f"{kind} CSV must be no larger than 10 MB.")
                uploaded.seek(0)
                signal_frame = pd.read_csv(uploaded)
                samples = pd.to_numeric(signal_frame[selected_column], errors="coerce").to_numpy(dtype=np.float64)
                signal_summaries[kind] = summarize_signal(samples, rate, kind)
                signal_summaries[kind]["source"] = "uploaded CSV"
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not run screening: {exc}")
        else:
            entry = {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "inputs": record,
                "result": result,
                "signals": signal_summaries,
                "eeg": eeg_result,
            }
            st.session_state.last_screening = entry
            history_entry = {**entry, "eeg": {
                key: value for key, value in eeg_result.items() if key not in {"features", "predictions"}
            } if eeg_result else None}
            st.session_state.screening_history = [history_entry, *st.session_state.screening_history[:99]]

    last_screening = st.session_state.last_screening
    if last_screening is not None:
        st.divider()
        st.subheader("Latest screening")
        eeg_result = last_screening.get("eeg")
        if eeg_result:
            st.markdown("**EEG window classification**")
            if eeg_result["synthetic"]:
                st.warning("SYNTHETIC DEMONSTRATION — these invented inputs have no measured clinical meaning or known labels.")
            if eeg_result["predictions"]:
                eeg_predictions = pd.DataFrame(eeg_result["predictions"])
                total = eeg_result["windows"]
                positive = eeg_result["apnea_windows"]
                st.write(f"{total} windows: {positive} classified as apnea, {total - positive} as normal.")
                st.dataframe(eeg_predictions, width="stretch", hide_index=True)
                if eeg_predictions["outside_training_range"].any():
                    st.warning("Some feature values are outside the development data ranges. Predictions for those windows may be unreliable.")
                st.caption("Decision margin is an SVM score, not a probability. Window counts do not measure AHI or severity.")
                st.download_button("Download EEG window predictions CSV", eeg_predictions.to_csv(index=False).encode(),
                                   file_name="eeg_predictions.csv", mime="text/csv")
            else:
                st.info("Features accepted; EEG classification is unavailable until the EEG model is trained.")
            with st.expander("EEG FPGA feature values"):
                st.dataframe(pd.DataFrame(eeg_result["features"]), width="stretch", hide_index=True)
        st.markdown("**Lifestyle screening**")
        result = last_screening["result"]
        if result["status"] == "screen_positive":
            st.error("Screen positive — the lifestyle model raised a signal.")
        elif result["status"] == "lower_model_signal":
            st.success("Lower model signal — this does not rule out sleep apnea.")
        else:
            st.warning("Insufficient evidence — inputs are outside model training support.")
        if result["decision_margin"] is not None:
            st.caption(
                f"Decision margin: {result['decision_margin']:.3f}. "
                "This is an SVM score, not a probability."
            )
        st.write(result["next_step"])
        st.caption(result["disclaimer"])
        st.info(
            "The lifestyle score uses lifestyle inputs only. EEG is classified separately; "
            "ECG and SpO₂ summaries are informational."
        )
        if last_screening["signals"]:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            key: value
                            for key, value in summary.items()
                            if key != "frequency_band_features"
                        }
                        for summary in last_screening["signals"].values()
                    ]
                ),
                width="stretch",
            )

with history_tab:
    st.subheader("Screening history")
    st.caption("History is held only in this browser session; it is not saved to disk.")
    history = st.session_state.screening_history
    if history:
        history_rows = []
        for entry in history:
            history_rows.append(
                {
                    "Time": entry["timestamp"],
                    "Result": entry["result"]["status"],
                    "Decision margin": entry["result"]["decision_margin"],
                    "Age": entry["inputs"]["age"],
                    "Gender": entry["inputs"]["gender"],
                    "BMI": entry["inputs"]["bmi"],
                    "Signals supplied": ", ".join([*(entry["signals"]), *(["EEG"] if entry.get("eeg") else [])]) or "None",
                    "EEG windows": (entry.get("eeg") or {}).get("windows"),
                    "EEG apnea windows": (entry.get("eeg") or {}).get("apnea_windows"),
                    "Synthetic EEG": (entry.get("eeg") or {}).get("synthetic", False),
                }
            )
        st.dataframe(pd.DataFrame(history_rows), width="stretch", hide_index=True)
        download = pd.DataFrame(
            [
                {
                    **entry["inputs"],
                    "timestamp": entry["timestamp"],
                    "status": entry["result"]["status"],
                    "decision_margin": entry["result"]["decision_margin"],
                    "signals_supplied": ", ".join([*entry["signals"], *(["EEG"] if entry.get("eeg") else [])]),
                    "eeg_windows": (entry.get("eeg") or {}).get("windows"),
                    "eeg_apnea_windows": (entry.get("eeg") or {}).get("apnea_windows"),
                    "synthetic_eeg": (entry.get("eeg") or {}).get("synthetic", False),
                }
                for entry in history
            ]
        )
        st.download_button(
            "Download history CSV",
            download.to_csv(index=False).encode("utf-8"),
            file_name="screening_history.csv",
            mime="text/csv",
        )
        if st.button("Clear history"):
            st.session_state.screening_history = []
            st.session_state.last_screening = None
            st.rerun()
    else:
        st.info("No screenings in this session yet.")

with model_tab:
    st.subheader("Model, dataset, and method")
    st.markdown(
        f"""
The **lifestyle model** is a pre-trained, class-balanced SVM with a StandardScaler. It uses
the six lifestyle inputs only. The separate **EEG SVM** takes the FPGA's 15 parameters and
returns a normal/apnea label for each window. The two scores are not combined.

Training data: [SleepSense lifestyle Kaggle dataset]({DATASET_URL}). It contains 374 rows and
106 distinct six-input profiles; it does not contain EEG, ECG, SpO₂, airflow, or apnea event
timestamps. The target is `Sleep Apnea` versus the dataset's other labels, not clinically
confirmed diagnosis.

The model was tuned with grouped cross-validation, grouping identical six-input profiles.
The bundled model was then refit on all rows. Performance shown below is from the separate
fixed grouped holdout and nested development CV in the [performance report](PERFORMANCE_REPORT.md);
the tuning CV score is not a test result.
        """
    )
    st.button(
        "Hide performance metrics" if st.session_state.show_metrics else "Show performance metrics",
        key="show_model_metrics",
        on_click=toggle_metrics_callback,
    )
    st.markdown("#### EEG model and FPGA feature interface")
    if eeg_model is None:
        st.info("No trained EEG artifact is installed. Synthetic features demonstrate input handling only.")
        st.code("python prepare_eeg_dataset.py --download\n"
                "python eeg_apnea.py --data data/processed/slpdb/eeg_features.csv "
                "--contract data/processed/slpdb/feature_contract.json "
                "--source https://physionet.org/content/slpdb/1.0.0/", language="bash")
    else:
        report = eeg_model["report"]
        st.markdown(f"Training source: [MIT-BIH Polysomnographic Database]({eeg_model['source']}). "
                    f"{report['train_windows']} development windows and {report['test_windows']} held-out windows.")
        st.write("Selected features: " + ", ".join(FEATURE_NAMES[i] for i in eeg_model["selected_indices"]))
        st.caption("Scaling, feature selection and SVM tuning use development subjects only. Test subjects are retained as unseen data.")
        if st.session_state.show_metrics:
            st.markdown("**EEG subject-wise held-out performance**")
            st.dataframe(pd.DataFrame({
                "Metric": ["Accuracy", "Balanced accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC AUC", "Average precision", "Majority-class baseline accuracy"],
                "Held-out": [f"{report[key]:.1%}" for key in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "average_precision", "majority_baseline_accuracy")],
            }), width="stretch", hide_index=True)
            st.write("Confusion matrix: actual rows / predicted columns [normal, apnea]")
            st.dataframe(pd.DataFrame(report["confusion_matrix"], index=["Actual normal", "Actual apnea"],
                                      columns=["Predicted normal", "Predicted apnea"]))
            st.caption(f"Median feature-to-prediction time: {report['single_window_latency_ms_median']:.2f} ms/window; "
                       f"p95: {report['single_window_latency_ms_p95']:.2f} ms on the training host. "
                       "Excludes FPGA processing and serial waiting time.")
            st.write(report["evaluation_note"])
            if "holdout_subject_metrics" in report:
                st.dataframe(pd.DataFrame(report["holdout_subject_metrics"]).drop(columns="confusion_matrix"),
                             width="stretch", hide_index=True)
            st.json({key: report[key] for key in ("train_subjects", "test_subjects", "test_class_counts", "best_parameters", "development_cv_balanced_accuracy", "development_cv_std")})
        with st.expander("EEG feature definitions and FPGA compatibility"):
            st.json(eeg_model["contract"])
        st.warning("Performance was measured using Python reference features. Numerical agreement with FPGA output is still required before hardware predictions can be trusted.")
    if st.session_state.show_metrics:
        st.markdown("#### Lifestyle model: fixed grouped holdout vs nested development CV")
        metrics = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Sensitivity / recall",
                    "Specificity",
                    "Precision",
                    "F1",
                    "F2",
                    "ROC AUC",
                    "Average precision",
                ],
                "Fixed holdout": [
                    "90.5%",
                    "100.0%",
                    "88.1%",
                    "68.2%",
                    "81.1%",
                    "91.5%",
                    "98.1%",
                    "87.2%",
                ],
                "Nested CV (pooled)": [
                    "90.7%",
                    "85.7%",
                    "92.0%",
                    "74.0%",
                    "79.4%",
                    "83.1%",
                    "91.6%",
                    "79.4%",
                ],
            }
        )
        st.dataframe(metrics, width="stretch", hide_index=True)
        st.caption(
            "Holdout confusion matrix, actual rows / predicted columns (other label, apnea label): "
            "[[52, 7], [0, 15]]. Seven non-apnea-labeled rows were flagged. The holdout contained "
            "only 15 apnea-labeled rows."
        )
        st.warning(
            "These are results on a small lifestyle dataset, not clinical performance. "
            "These lifestyle metrics are separate from the EEG classifier metrics above."
        )
    if screening_model is not None:
        st.markdown("#### Training support of the bundled model")
        st.json(screening_model["metadata"]["feature_bounds"])
    st.markdown(
        f"""
**Method:** scale the six input fields, then apply an RBF-kernel SVM; the tuned model uses
`C=10` and `gamma=0.01`. It reports a signed decision margin, not probability or disease
severity. Outside the observed training support it abstains. Optional symptoms only prompt
follow-up.

**Limitations:** no breathing events or clinical severity are measured; a lower model signal
does not rule out apnea. EEG window predictions come from the separate feature SVM, and
ECG/SpO₂ do not affect either classifier. No combined clinical validation has been performed.
        """
    )
