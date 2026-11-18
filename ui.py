"""Unified lifestyle screening interface with optional signal context and history."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app_inputs import example_supported_record, random_supported_record, summarize_signal, synthetic_signal
from eeg_apnea import extract_features
from fpga_serial import capture_samples
from lifestyle_model import DATASET_URL, SYMPTOMS, load_model, predict

st.set_page_config(page_title="Sleep Apnea Screening", layout="wide")
st.title("Sleep Apnea Screening Workbench")
st.caption(
    "One-page lifestyle screening with optional EEG, ECG, and SpO₂ signal context, "
    "model evidence, and session history."
)
st.warning(
    "The trained SVM uses only gender, age, BMI category, sleep duration, resting heart rate, "
    "and daily steps. EEG, ECG, and SpO₂ are recorded for context and are **not used in the "
    "prediction**. This research screening is not a diagnosis."
)


@st.cache_resource
def get_model():
    return load_model()


try:
    screening_model = get_model()
except (FileNotFoundError, ValueError) as exc:
    screening_model = None
    st.error(f"The trained lifestyle model could not be loaded: {exc}")

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
    "signal_EEG_rate": 125.0,
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
    st.session_state.last_screening = None


def random_values_callback() -> None:
    if screening_model is not None:
        set_sample_values(random_supported_record(screening_model))


def example_values_callback() -> None:
    set_sample_values(example_supported_record())


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
    with st.expander("Optional biosignals — not used by the current SVM"):
        st.markdown(
            "Choose a CSV signal, a clearly synthetic demo waveform, or no signal. "
            "Uploads must contain one numeric sample column. SpO₂ is entered as percent values."
        )
        signal_columns = st.columns(3)
        signal_inputs: dict[str, tuple[str, float, Any, str | None]] = {}
        for column, kind in zip(signal_columns, ("EEG", "ECG", "SpO2")):
            column.markdown(f"**{kind}**")
            modes = ("Not supplied", "Upload CSV", "Synthetic demo")
            if kind == "EEG":
                modes = (*modes, "FPGA serial")
            mode = column.selectbox(
                f"{kind} source",
                modes,
                key=f"signal_{kind}_mode",
            )
            rate = column.number_input(
                f"{kind} sample rate (Hz)",
                min_value=0.1 if kind == "SpO2" else 1.0,
                max_value=4096.0,
                step=1.0,
                key=f"signal_{kind}_rate",
            )
            uploaded = None
            selected_column = None
            if mode == "Upload CSV":
                uploaded = column.file_uploader(
                    f"{kind} one-channel CSV",
                    type=["csv"],
                    key=f"signal_{kind}_file",
                )
                if uploaded is not None:
                    if uploaded.size > 10_000_000:
                        column.error(f"{kind} CSV must be no larger than 10 MB.")
                        signal_inputs[kind] = (mode, float(rate), uploaded, None)
                        continue
                    try:
                        uploaded.seek(0)
                        signal_frame = pd.read_csv(uploaded)
                        numeric_columns = [
                            name
                            for name in signal_frame.columns
                            if pd.to_numeric(signal_frame[name], errors="coerce").notna().any()
                        ]
                        if numeric_columns:
                            selected_column = column.selectbox(
                                f"{kind} sample column",
                                numeric_columns,
                                key=f"signal_{kind}_column",
                            )
                        else:
                            column.error(f"{kind} CSV has no numeric sample column.")
                    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
                        column.error(f"Could not read {kind} CSV: {exc}")
            signal_inputs[kind] = (mode, float(rate), uploaded, selected_column)

    eeg_mode = signal_inputs["EEG"][0]
    if eeg_mode == "FPGA serial":
        st.markdown(
            "FPGA serial format: one ASCII decimal EEG sample followed by a newline. "
            "Binary, framed, and multi-channel protocols need a matching parser."
        )
        serial_columns = st.columns(3)
        serial_port = serial_columns[0].text_input(
            "Serial port",
            key="fpga_port",
            placeholder="/dev/ttyUSB0",
        )
        baud_rate = serial_columns[1].number_input(
            "Baud rate",
            min_value=1,
            max_value=4_000_000,
            value=115200,
            key="fpga_baud",
        )
        capture_duration = serial_columns[2].number_input(
            "Capture duration (seconds)",
            min_value=1.5,
            max_value=600.0,
            value=30.0,
            key="fpga_duration",
        )
        if st.button("Capture EEG from FPGA"):
            try:
                st.session_state.captured_eeg_samples = capture_samples(
                    serial_port,
                    int(baud_rate),
                    float(capture_duration),
                )
                st.success(
                    f"Captured {len(st.session_state.captured_eeg_samples)} EEG samples."
                )
            except (ValueError, RuntimeError) as exc:
                st.error(f"FPGA capture failed: {exc}")

    predict_button = st.button(
        "Predict using lifestyle model",
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
            for kind, (mode, rate, uploaded, selected_column) in signal_inputs.items():
                if mode == "Not supplied":
                    continue
                if mode == "FPGA serial":
                    samples = np.asarray(
                        st.session_state.get("captured_eeg_samples", []),
                        dtype=np.float64,
                    )
                    if not len(samples):
                        raise ValueError("Capture EEG from FPGA before predicting.")
                    signal_summaries[kind] = summarize_signal(samples, rate, kind)
                    signal_summaries[kind]["source"] = "FPGA serial"
                    signal_summaries[kind]["frequency_band_features"] = extract_features(
                        samples,
                        rate,
                    )
                    continue
                if mode == "Synthetic demo":
                    samples = synthetic_signal(kind, rate, 30.0)
                    signal_summaries[kind] = summarize_signal(samples, rate, kind)
                    signal_summaries[kind]["source"] = "synthetic demo"
                    if kind == "EEG":
                        signal_summaries[kind]["frequency_band_features"] = extract_features(
                            samples,
                            rate,
                        )
                    continue
                if uploaded is None or selected_column is None:
                    raise ValueError(f"Select a valid {kind} CSV sample column or a demo source.")
                if uploaded.size > 10_000_000:
                    raise ValueError(f"{kind} CSV must be no larger than 10 MB.")
                uploaded.seek(0)
                signal_frame = pd.read_csv(uploaded)
                samples = pd.to_numeric(
                    signal_frame[selected_column], errors="coerce"
                ).dropna().to_numpy(dtype=np.float64)
                if len(samples) != len(signal_frame):
                    raise ValueError(f"{kind} CSV contains missing or non-numeric samples.")
                signal_summaries[kind] = summarize_signal(samples, rate, kind)
                signal_summaries[kind]["source"] = "uploaded CSV"
                if kind == "EEG":
                    signal_summaries[kind]["frequency_band_features"] = extract_features(
                        samples,
                        rate,
                    )
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not run screening: {exc}")
        else:
            entry = {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "inputs": record,
                "result": result,
                "signals": signal_summaries,
            }
            st.session_state.last_screening = entry
            st.session_state.screening_history = (
                [entry, *st.session_state.screening_history[:99]]
            )

    last_screening = st.session_state.last_screening
    if last_screening is not None:
        st.divider()
        st.subheader("Latest screening")
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
            "EEG, ECG, and SpO₂ were not included in this prediction. "
            "The signal summaries below are informational only."
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
            if "EEG" in last_screening["signals"]:
                with st.expander("EEG frequency-band features (not used by SVM)"):
                    st.json(
                        last_screening["signals"]["EEG"]["frequency_band_features"]
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
                    "Signals supplied": ", ".join(entry["signals"]) or "None",
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
                    "signals_supplied": ", ".join(entry["signals"]),
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
The available model is a **pre-trained, class-balanced SVM** with a StandardScaler. It uses
the six lifestyle inputs only. EEG/ECG/SpO₂ are not classifier features and will not change
the result.

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
    if st.session_state.show_metrics:
        st.markdown("#### Fixed grouped holdout vs nested development CV")
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
            "No improvement from EEG is established because EEG is not part of this model."
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
does not rule out apnea. A genuine EEG/ECG/SpO₂ prediction requires a separately trained and
validated model with corresponding labeled recordings.
        """
    )
