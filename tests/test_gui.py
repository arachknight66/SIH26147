import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from signal_analysis.gui import MainWindow, HAS_QT, _get_status_color, format_stage_status
from signal_analysis.models import PipelineStageStatus

def test_status_color_mapping():
    assert _get_status_color(PipelineStageStatus.NOT_ATTEMPTED) == "gray"
    assert _get_status_color(PipelineStageStatus.FAILED) == "red"
    assert _get_status_color(PipelineStageStatus.COMPLETED) == "green"
    
def test_format_stage_status_na_reasoning():
    # Test that FAILED previous stage bubbles up a "stopped upstream" reason
    text1 = format_stage_status(PipelineStageStatus.NOT_ATTEMPTED, PipelineStageStatus.FAILED)
    assert "gray" in text1
    assert "upstream" in text1
    
    text2 = format_stage_status(PipelineStageStatus.NOT_ATTEMPTED, PipelineStageStatus.COMPLETED)
    assert "upstream" not in text2

@pytest.mark.skipif(not HAS_QT, reason="Qt not available")
def test_stereo_wav_heuristic(tmp_path):
    # 1. Correlated dual-real (same signal on both channels)
    import wave
    corr_path = tmp_path / "corr.wav"
    t = np.linspace(0, 1, 4096)
    sig = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    stereo_corr = np.column_stack([sig, sig]) # perfect correlation
    stereo_corr_int16 = (stereo_corr * 32767).astype(np.int16)
    
    with wave.open(str(corr_path), 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(stereo_corr_int16.tobytes())
        
    # 2. Quadrature (I/Q)
    iq_path = tmp_path / "iq.wav"
    i_sig = np.random.randn(4096).astype(np.float32)
    q_sig = np.random.randn(4096).astype(np.float32)
    stereo_iq = np.column_stack([i_sig, q_sig]) # uncorrelated
    stereo_iq_int16 = (stereo_iq * 10000).astype(np.int16)
    
    with wave.open(str(iq_path), 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(stereo_iq_int16.tobytes())
        
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    window = MainWindow()
    
    # Assert heuristic output directly
    corr_guess = window._guess_stereo_mode_heuristic(str(corr_path))
    assert corr_guess == "likely-independent"
    
    iq_guess = window._guess_stereo_mode_heuristic(str(iq_path))
    assert iq_guess == "likely-quadrature"
    
@pytest.mark.skipif(not HAS_QT, reason="Qt not available")
def test_open_file_dialog_wiring_complex_iq(tmp_path):
    import wave
    wav_path = tmp_path / "test.wav"
    with wave.open(str(wav_path), 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(np.zeros((100, 2), dtype=np.int16).tobytes())
        
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    window = MainWindow()
    
    # Monkeypatch QFileDialog.getOpenFileName and QInputDialog.getItem
    with patch("signal_analysis.gui.QFileDialog.getOpenFileName", return_value=(str(wav_path), "")):
        with patch("signal_analysis.gui.QInputDialog.getItem", return_value=("Complex I/Q pair (Ch0=I, Ch1=Q) (stereo_iq)", True)):
            # We also need to patch window.update_plots to avoid actually painting during a headless test
            with patch.object(window, "update_plots"):
                # And capture the recording passed to update_metadata
                with patch.object(window.sidebar, "update_metadata") as mock_update_metadata:
                    window.open_file()
                    mock_update_metadata.assert_called_once()
                    recording = mock_update_metadata.call_args[0][0]
                    assert recording.semantic_type == "complex_iq"
                    # Assert no diagnostic contains the heuristic text
                    assert not any("heuristic" in d.message.lower() for d in recording.diagnostics)
