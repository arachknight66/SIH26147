from __future__ import annotations
import os
import pytest
from app.ui.models import UIStateModel
from app.ui.theme import DARK_THEME, EPISTEMIC_COLORS
from app.orchestration.pipeline_config import PresetName, get_preset_config
from app.orchestration.pipeline_runner import run_pipeline
from tests.test_phase6_cases import _make_rec_sig
from scripts.generate_digital_dataset import generate_digital_stream

def test_ui_state_model():
    model = UIStateModel()
    assert model.is_running is False
    assert len(model.log_history) == 0

    logs = []
    model.add_listener(lambda ev, data: logs.append((ev, data)))
    model.add_log("System initialized")
    assert len(logs) == 1
    assert logs[0][0] == "log"

def test_epistemic_theme_colors():
    assert "OBSERVED" in EPISTEMIC_COLORS
    assert "SUPPORTED" in EPISTEMIC_COLORS
    assert "INDEPENDENTLY_VERIFIED" in EPISTEMIC_COLORS
    assert "FALSIFIED" in EPISTEMIC_COLORS
    assert len(DARK_THEME) > 100

def test_qt_ui_smoke():
    # Smoke test headless UI initialization
    try:
        from PySide6.QtWidgets import QApplication
        from app.ui.main_window import MainWindow
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QApplication.instance() or QApplication([])
        win = MainWindow()
        assert win is not None
    except ImportError:
        pytest.skip("PySide6 not available in this test environment")
