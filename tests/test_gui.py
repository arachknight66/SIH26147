import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock
from signal_analysis.gui import MainWindow, MetadataSidebar, FrameProfileDialog, HAS_QT, _get_status_color, format_stage_status, GuiPipelineContractError
from signal_analysis.models import (PipelineStageStatus, SignalRecording, SourceFormat,
    MetadataValue, MetadataStatus, PipelineResult)


def _wait_for_analysis(app, window, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while window._active_job is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert window._active_job is None, "analysis job did not finish"

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
    assert corr_guess == "stereo_real"
    
    iq_guess = window._guess_stereo_mode_heuristic(str(iq_path))
    assert iq_guess == "stereo_iq"


@pytest.mark.skipif(not HAS_QT, reason="Qt not available")
def test_frame_profile_dialog_feeds_production_configuration():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    dialog = FrameProfileDialog({
        "header_name": "CCSDS_ASM_32",
        "payload_bytes": 223,
        "crc_name": "CRC-16/CCITT-FALSE",
        "minimum_valid_frames": 2,
    })

    profile = dialog.profile()
    window._frame_profile = profile
    config = window._selected_pipeline_config()

    assert config["frame_profiles"] == [profile]
    assert profile["strict_next_header_boundary"] is True
    
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
        with patch("signal_analysis.gui.QInputDialog.exec", return_value=1):
            with patch("signal_analysis.gui.QInputDialog.textValue", return_value="Complex I/Q pair (Ch0=I, Ch1=Q) (stereo_iq)"):
                with patch.object(window, "update_plots"):
                    with patch.object(window.sidebar, "update_metadata") as mock_update_metadata:
                        window.open_file()
                        _wait_for_analysis(app, window)
                    mock_update_metadata.assert_called_once()
                    recording = mock_update_metadata.call_args[0][0]
                    assert recording.semantic_type == "complex_iq"
                    # Assert no diagnostic contains the heuristic text
                    assert not any("heuristic" in d.message.lower() for d in recording.diagnostics)


@pytest.mark.skipif(not HAS_QT, reason="Qt not available")
def test_uppercase_wav_extension_uses_wav_import_path(tmp_path):
    import wave
    from PySide6.QtWidgets import QApplication

    wav_path = tmp_path / "capture.WAV"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(np.zeros(100, dtype=np.int16).tobytes())
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    with patch("signal_analysis.gui.QFileDialog.getOpenFileName", return_value=(str(wav_path), "")):
        with patch.object(window, "update_plots"):
            with patch.object(window.sidebar, "update_metadata") as update:
                window.open_file()
                _wait_for_analysis(app, window)
    assert update.call_args[0][0].source_format is SourceFormat.WAV


def _minimal_pipeline_tree():
    recording = SignalRecording(np.zeros(8, np.complex64), SourceFormat.RAW_IQ, "complex64", "complex_iq",
        MetadataValue(1.0, "test", MetadataStatus.KNOWN), MetadataValue(None, "test", MetadataStatus.MISSING), {}, [])
    return recording, PipelineResult(recording, PipelineStageStatus.NOT_ATTEMPTED, None, [],
        PipelineStageStatus.NOT_ATTEMPTED, None, PipelineStageStatus.NOT_ATTEMPTED, None, None,
        PipelineStageStatus.NOT_ATTEMPTED, None)


@pytest.mark.skipif(not HAS_QT, reason="Qt not available")
def test_sidebar_contract_guard_accepts_real_dataclasses():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    recording, pipeline = _minimal_pipeline_tree()
    sidebar = MetadataSidebar()
    sidebar.update_metadata(recording, pipeline)
    assert "RAW_IQ" in sidebar.meta_text.text()


@pytest.mark.skipif(not HAS_QT, reason="Qt not available")
def test_sidebar_contract_guard_is_atomic_before_widget_mutation():
    from dataclasses import dataclass
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    recording, pipeline = _minimal_pipeline_tree(); sidebar = MetadataSidebar(); sidebar.meta_text.setText("old consistent state")
    @dataclass(frozen=True)
    class DriftedPipeline: diagnostics: list
    with pytest.raises(GuiPipelineContractError, match="GUI_PIPELINE_CONTRACT.*DriftedPipeline"):
        sidebar.update_metadata(recording, DriftedPipeline([]))
    assert sidebar.meta_text.text() == "old consistent state"


@pytest.mark.skipif(not HAS_QT, reason="Qt not available")
def test_async_completion_surfaces_contract_error_distinctly():
    from signal_analysis.jobs.analysis_job import JobSnapshot, JobState
    from signal_analysis.workflow import ProductionAnalysisResult, WorkflowStatus, AnalysisRequest
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([]); recording, pipeline = _minimal_pipeline_tree(); window = MainWindow()
    outcome = ProductionAnalysisResult(AnalysisRequest(__import__('pathlib').Path("x.iq")), recording, pipeline, WorkflowStatus.COMPLETED, 3, 3)
    class DoneJob:
        def snapshot(self): return JobSnapshot(JobState.COMPLETED, 3, 3, 1.0, None)
        def result(self, timeout=0): return outcome
    window._active_job = DoneJob()
    with patch.object(window, "update_plots"), patch.object(window.sidebar, "update_metadata", side_effect=GuiPipelineContractError("GUI_PIPELINE_CONTRACT: PipelineResult missing field(s): fec_result")), patch("signal_analysis.gui.QMessageBox.critical") as critical:
        window._poll_analysis_job()
    assert "contract mismatch" in window.job_status.text().lower()
    assert "GUI_PIPELINE_CONTRACT" in critical.call_args.args[2]
