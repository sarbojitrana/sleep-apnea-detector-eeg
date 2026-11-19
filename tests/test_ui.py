from pathlib import Path
from streamlit.testing.v1 import AppTest

UI_PATH = Path(__file__).resolve().parents[1] / 'ui.py'


def click(app, label):
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception


def test_preserved_tabs_lifestyle_and_random_eeg_demo():
    app = AppTest.from_file(str(UI_PATH)).run(timeout=20)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ['Screening', 'History', 'Model and performance']
    click(app, 'Run screening')
    assert app.session_state.last_screening['eeg'] is None
    click(app, 'Add example data')
    click(app, 'Run screening')
    example = app.session_state.last_screening['eeg']
    assert example['synthetic'] is True
    assert len(example['predictions']) == 6
    assert all(row['synthetic'] for row in example['predictions'])
    assert 'features' not in app.session_state.screening_history[0]['eeg']
    assert 'predictions' not in app.session_state.screening_history[0]['eeg']
    click(app, 'Random valid values')
    click(app, 'Run screening')
    assert example['features'] != app.session_state.last_screening['eeg']['features']
    assert len(app.session_state.screening_history) == 3
    click(app, 'Show performance metrics')
    assert any('EEG subject-wise' in item.value for item in app.markdown)
    assert all('80% target' not in frame.value.columns for frame in app.dataframe)
    assert all('80%' not in message.value for message in app.info)


def test_empty_manual_parameters_do_not_create_a_prediction():
    app = AppTest.from_file(str(UI_PATH)).run(timeout=20)
    next(box for box in app.selectbox if box.label == 'EEG source').select('Manual parameters').run()
    click(app, 'Run screening')
    assert not app.session_state.screening_history
    assert any('zero energy' in error.value for error in app.error)


def test_missing_eeg_model_allows_feature_demo_without_fake_predictions(monkeypatch, tmp_path):
    import eeg_apnea
    monkeypatch.setattr(eeg_apnea, 'MODEL_PATH', tmp_path / 'missing.json')
    app = AppTest.from_file(str(UI_PATH)).run(timeout=20)
    click(app, 'Add example data')
    click(app, 'Run screening')
    result = app.session_state.last_screening['eeg']
    assert result['windows'] == 6
    assert result['predictions'] == []
    assert result['apnea_windows'] is None
