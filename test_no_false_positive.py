import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication, QDialog
from signal_analysis.gui import MainWindow

def test_real_dialog_flow():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    wav_path = Path("test_clean_qpsk.wav").absolute()
    
    # Only mock QFileDialog
    with patch("signal_analysis.gui.QFileDialog.getOpenFileName", return_value=(str(wav_path), "")):
        # To simulate a user selecting "Complex I/Q pair", we can't easily interact with the blocking dialog.
        # BUT we can intercept the exec() call, modify the dialog's state, and return QDialog.Accepted.
        # We can patch QInputDialog.exec to first change the combo box value, then return QDialog.Accepted
        def fake_exec(*args, **kwargs):
            dialog = window.findChild(QDialog, "Stereo WAV Detected") or [w for w in window.children() if isinstance(w, QDialog)][0]
            dialog.setTextValue("Complex I/Q pair (Ch0=I, Ch1=Q) (stereo_iq)")
            return QDialog.Accepted
            
        with patch("signal_analysis.gui.QInputDialog.exec", autospec=True, side_effect=fake_exec):
            # We also patch update_plots to save time
            with patch.object(window, "update_plots"):
                window.open_file()
                
    # Now check if it actually reached COMPLETED
    sync_html = window.sidebar.sync_text.text()
    assert "COMPLETED" in sync_html, f"Failed! Text was: {sync_html}"
    print("Success!")

test_real_dialog_flow()
