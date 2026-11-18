"""Streamlit lifestyle screening form and independent EEG feature viewer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from eeg_apnea import EEG_BANDS, extract_features, split_signal
from fpga_serial import capture_samples
from lifestyle_model import MODEL_PATH, SYMPTOMS, load_model, predict

st.set_page_config(page_title="Sleep Apnea Screening", layout="wide")
st.title("Sleep Apnea Screening Workbench")
st.caption(
    "Lifestyle-based research screening with a separate EEG signal feature viewer. "
    "This is not a medical device or diagnosis."
)
st.warning(
    "The lifestyle SVM and EEG viewer are separate. The bundled model uses only the six "
    "lifestyle inputs; EEG features do not affect its result. There is no trained EEG model."
)


@st.cache_resource
def get_lifestyle_model():
    return load_model()


try:
    lifestyle_model = get_lifestyle_model()
except (FileNotFoundError, ValueError) as exc:
    lifestyle_model = None
    st.error(f"The pre-trained lifestyle model could not be loaded: {exc}")

screen_tab, eeg_tab, model_tab = st.tabs(
    ["Lifestyle screening", "EEG input and features", "Model information"]
)

with screen_tab:
    st.subheader("Enter the existing six screening inputs")
    st.caption(
        "All six values are required. Symptoms are optional and only affect the follow-up "
        "prompt, not the SVM prediction."
    )
    with st.form("lifestyle_screening"):
        columns = st.columns(3)
        gender = columns[0].selectbox("Gender category", ("Female", "Male"))
        age = columns[1].number_input("Age (years)", min_value=18, max_value=100, value=40)
        bmi = columns[2].selectbox("BMI category", ("Normal", "Overweight", "Obese"))
        sleep_duration = columns[0].number_input(
            "Sleep duration (hours/night)",
            min_value=0.5,
            max_value=16.0,
            value=7.0,
            step=0.1,
        )
        heart_rate = columns[1].number_input(
            "Resting heart rate (bpm)",
            min_value=30.0,
            max_value=220.0,
            value=70.0,
            step=1.0,
        )
        daily_steps = columns[2].number_input(
            "Daily steps",
            min_value=0,
            max_value=100000,
            value=7000,
            step=100,
        )
        st.markdown("**Optional symptoms**")
        symptom_columns = st.columns(3)
        symptoms = {
            "loud_snoring": symptom_columns[0].checkbox("Loud snoring"),
            "witnessed_pauses": symptom_columns[1].checkbox("Witnessed breathing pauses"),
            "daytime_sleepiness": symptom_columns[2].checkbox("Daytime sleepiness"),
        }
        submitted = st.form_submit_button(
            "Screen using lifestyle model",
            type="primary",
            disabled=lifestyle_model is None,
        )

    if submitted and lifestyle_model is not None:
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
            result = predict(lifestyle_model, record)
        except ValueError as exc:
            st.error(f"Could not screen these inputs: {exc}")
        else:
            if result["status"] == "screen_positive":
                st.error("Screen positive — the lifestyle model raised a signal.")
            elif result["status"] == "lower_model_signal":
                st.success("Lower model signal — this does not rule out sleep apnea.")
            else:
                st.warning("Insufficient evidence — inputs are outside the model's training support.")
            if result["decision_margin"] is not None:
                st.caption(
                    "Model decision margin: "
                    f"{result['decision_margin']:.3f} "
                    "(positive values cross the source model's screening threshold)."
                )
            for warning in result["warnings"]:
                st.warning(warning)
            if result["reported_symptoms"]:
                st.info(
                    "Reported symptoms: " + ", ".join(result["reported_symptoms"])
                )
            st.write(result["next_step"])
            st.caption(result["disclaimer"])

with eeg_tab:
    st.subheader("Provide EEG input")
    st.markdown(
        "EEG is handled independently from the lifestyle SVM. You can upload one channel or "
        "capture from a supported FPGA serial stream, then inspect per-window band features. "
        "These features are **not** converted to an apnea prediction."
    )
    source = st.radio(
        "EEG source",
        ("Upload CSV", "FPGA serial", "Synthetic demo (not real EEG)"),
        horizontal=True,
    )
    signal_columns = st.columns(3)
    sampling_rate = signal_columns[0].number_input(
        "Sampling rate (Hz)",
        min_value=91.0,
        max_value=4096.0,
        value=125.0,
        step=1.0,
        help="Must be the actual sampling rate of the provided EEG channel.",
    )
    channel_name = signal_columns[1].text_input("EEG channel name", value="EEG")
    window_seconds = signal_columns[2].number_input(
        "Analysis window (seconds)",
        min_value=1.5,
        max_value=600.0,
        value=30.0,
        step=1.0,
    )
    samples: list[float] | np.ndarray | None = None

    if source == "Upload CSV":
        signal_file = st.file_uploader(
            "EEG samples CSV",
            type=["csv"],
            help="Upload one column of finite numeric samples for one EEG channel.",
            key="eeg_samples_csv",
        )
        if signal_file is not None:
            try:
                signal_frame = pd.read_csv(signal_file)
                if signal_frame.empty or len(signal_frame.columns) == 0:
                    st.error("The EEG CSV has no sample columns.")
                else:
                    sample_column = st.selectbox(
                        "Sample column",
                        signal_frame.columns,
                        key="eeg_sample_column",
                    )
                    values = pd.to_numeric(
                        signal_frame[sample_column], errors="coerce"
                    ).to_numpy(dtype=np.float64)
                    if not np.isfinite(values).all():
                        st.error("The selected EEG column contains missing or invalid values.")
                    else:
                        samples = values
            except (pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(f"Could not read the EEG CSV: {exc}")
    elif source == "FPGA serial":
        st.markdown(
            "Serial contract: one ASCII decimal sample followed by a newline. "
            "Binary, framed, or multi-channel protocols need a matching parser."
        )
        serial_columns = st.columns(3)
        port = serial_columns[0].text_input(
            "FPGA serial port",
            placeholder="/dev/ttyUSB0",
        )
        baud_rate = serial_columns[1].number_input(
            "Baud rate",
            min_value=1,
            max_value=4_000_000,
            value=115_200,
            step=1,
        )
        capture_seconds = serial_columns[2].number_input(
            "Capture duration (seconds)",
            min_value=float(window_seconds),
            max_value=600.0,
            value=float(window_seconds),
            step=1.0,
        )
        if st.button("Capture EEG from FPGA"):
            try:
                captured = capture_samples(port, int(baud_rate), float(capture_seconds))
            except (ValueError, RuntimeError) as exc:
                st.error(f"FPGA capture failed: {exc}")
            else:
                st.session_state.eeg_samples = captured
                st.success(f"Captured {len(captured)} samples from {channel_name}.")
        samples = st.session_state.get("eeg_samples")
    else:
        st.info(
            "This deterministic synthetic waveform is only for checking the feature pipeline. "
            "It is not EEG, is not labeled for apnea, and cannot produce an apnea prediction."
        )
        sample_count = int(round(float(sampling_rate) * float(window_seconds)))
        time_axis = np.arange(sample_count) / float(sampling_rate)
        samples = (
            np.sin(2 * np.pi * 2.0 * time_axis)
            + 0.6 * np.sin(2 * np.pi * 10.0 * time_axis)
            + 0.25 * np.sin(2 * np.pi * 20.0 * time_axis)
            + 0.05 * np.random.default_rng(42).normal(size=sample_count)
        )

    if samples is not None and st.button("Calculate EEG features"):
        try:
            windows = split_signal(
                samples,
                float(sampling_rate),
                float(window_seconds),
                overlap_seconds=0.0,
            )
            feature_rows = [
                extract_features(window, float(sampling_rate))
                for window in windows
            ]
        except ValueError as exc:
            st.error(f"Could not analyze the EEG signal: {exc}")
        else:
            features = pd.DataFrame(feature_rows)
            features.insert(0, "window", np.arange(1, len(features) + 1))
            features.insert(
                1,
                "start_seconds",
                np.arange(len(features)) * float(window_seconds),
            )
            st.caption(
                f"Channel: {channel_name} · Sampling rate: {sampling_rate:g} Hz · "
                f"{len(features)} windows"
            )
            st.dataframe(features, use_container_width=True)
            st.download_button(
                "Download EEG features",
                data=features.to_csv(index=False).encode("utf-8"),
                file_name="eeg_window_features.csv",
                mime="text/csv",
            )
            st.info(
                "No EEG apnea classifier is available. These signal features cannot establish "
                "whether apnea is present."
            )
            st.caption("Bands analyzed: " + ", ".join(EEG_BANDS))

with model_tab:
    st.subheader("Model and limits")
    if lifestyle_model is not None:
        metadata = lifestyle_model["metadata"]
        st.write(
            "A pre-trained, RBF-kernel SVM uses the six existing lifestyle inputs. "
            "The bundled model was trained using the linked repository's dataset and "
            "its six-feature encoding."
        )
        st.json(
            {
                "source_dataset": metadata["source_dataset"],
                "dataset_sha256": metadata["dataset_sha256"],
                "training_rows": metadata["training_rows"],
                "unique_input_profiles": metadata["unique_input_profiles"],
                "selected_parameters": metadata["best_params"],
                "grouped_cv_f2_development_only": metadata["cv_f2_mean"],
                "training_support": metadata["feature_bounds"],
            }
        )
        st.caption(metadata["cv_note"])
    st.markdown(
        """
- The model predicts a pattern associated with the source dataset's **Sleep Apnea** label; it does not detect breathing pauses or diagnose a person.
- An out-of-support input produces **insufficient evidence**, not a negative result.
- Optional symptoms only prompt follow-up; they are not SVM features.
- EEG is not part of this model. Combining it requires a properly labeled EEG dataset and a separately validated combined model.
- Files and readings are kept in the current Streamlit session; no application database is used.
        """
    )
