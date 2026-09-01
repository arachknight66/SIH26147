import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog
from signal_analysis.gui import MainWindow

def test_gui_native_dialog_race_condition_prevention():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    wav_path = Path("test_clean_qpsk.wav").absolute()
    
    with patch("signal_analysis.gui.QFileDialog.getOpenFileName", return_value=(str(wav_path), "")) as mock_file_dialog:
        def fake_exec(*args, **kwargs):
            dialog = window.findChild(QDialog, "Stereo WAV Detected") or [w for w in window.children() if isinstance(w, QDialog)][0]
            dialog.setTextValue("Complex I/Q pair (Ch0=I, Ch1=Q) (stereo_iq)")
            return QDialog.Accepted
            
        with patch("signal_analysis.gui.QInputDialog.exec", autospec=True, side_effect=fake_exec):
            with patch.object(window, "update_plots"):
                window.open_file()
                
    # CRITICAL: Assert that we explicitly bypass the Windows native dialog to prevent the double-click event leak
    mock_file_dialog.assert_called_once()
    _, kwargs = mock_file_dialog.call_args
    assert "options" in kwargs, "getOpenFileName must specify options to disable native dialog"
    assert kwargs["options"] == QFileDialog.DontUseNativeDialog, "Must use DontUseNativeDialog to prevent race condition"
    
    # Assert pipeline actually ran to completion
    sync_html = window.sidebar.sync_text.text()
    assert "COMPLETED" in sync_html, f"Phase 3 NOT_ATTEMPTED. Text was: {sync_html}"

if __name__ == "__main__":
    test_gui_native_dialog_race_condition_prevention()
