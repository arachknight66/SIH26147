from __future__ import annotations
from pathlib import Path
import sys
import threading
from typing import Any

from app.orchestration.cancellation import CancellationToken
from app.orchestration.pipeline_config import PipelineConfig, PresetName, get_preset_config
from app.orchestration.pipeline_runner import PipelineResult, run_pipeline
from app.reporting.artifact_manifest import export_all_artifacts
from app.ui.models import UIStateModel
from app.ui.pages import (
    AssessmentPage,
    DataRecoveryPage,
    DetectionPage,
    DiagnosticsPage,
    FECPage,
    FalsificationPage,
    InputPage,
    LineagePage,
    ModulationPage,
    ParameterPage,
    RecoveryPage,
    SignalPage,
    VerificationPage,
)
from app.ui.theme import DARK_THEME

try:
    from PySide6.QtCore import QObject, QThread, Qt, Signal
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSplitter,
        QStackedWidget,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QMainWindow:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
    class QThread:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
    def Signal(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return None

class PipelineWorker(QThread):
    finished_signal = Signal(object)
    progress_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(self, input_path: str, config: PipelineConfig, token: CancellationToken) -> None:
        super().__init__()
        self.input_path = input_path
        self.config = config
        self.token = token

    def run(self) -> None:
        try:
            res = run_pipeline(
                self.input_path,
                config=self.config,
                cancel_token=self.token,
                progress_callback=lambda p: self.progress_signal.emit(p),
            )
            self.finished_signal.emit(res)
        except Exception as e:
            self.error_signal.emit(str(e))

class MainWindow(QMainWindow):
    """
    SIH26147 Desktop Application Main Window.
    """
    def __init__(self) -> None:
        super().__init__()
        self.state_model = UIStateModel()
        self.current_worker: PipelineWorker | None = None
        self.cancel_token = CancellationToken()

        if HAS_QT:
            self.setWindowTitle("SIH26147 — Scientific Signal Recovery & Verification Engine v0.7.0")
            self.resize(1280, 850)
            self.setStyleSheet(DARK_THEME)

            central = QWidget()
            self.setCentralWidget(central)
            main_layout = QVBoxLayout(central)

            # Top Toolbar
            top_bar = QHBoxLayout()
            self.btn_open = QPushButton("📂 Open Signal Recording")
            self.btn_run = QPushButton("▶ Run Analysis Pipeline")
            self.btn_cancel = QPushButton("⏹ Cancel")
            self.btn_demo = QPushButton("⭐ Judge / Demo Mode")
            self.btn_export = QPushButton("💾 Export Reports")

            self.btn_cancel.setEnabled(False)
            self.btn_demo.setStyleSheet("background-color: #f59e0b; color: #0f172a; font-weight: bold;")

            self.btn_open.clicked.connect(self._open_file)
            self.btn_run.clicked.connect(self._run_analysis)
            self.btn_cancel.clicked.connect(self._cancel_analysis)
            self.btn_demo.clicked.connect(self._run_judge_demo)
            self.btn_export.clicked.connect(self._export_reports)

            top_bar.addWidget(self.btn_open)
            top_bar.addWidget(self.btn_run)
            top_bar.addWidget(self.btn_cancel)
            top_bar.addWidget(self.btn_demo)
            top_bar.addStretch()
            top_bar.addWidget(self.btn_export)
            main_layout.addLayout(top_bar)

            # Middle Workspace: Left Nav Rail + Stacked Pages
            workspace_splitter = QSplitter(Qt.Horizontal)

            # Left Navigation Rail
            self.nav_list = QListWidget()
            self.nav_list.setFixedWidth(210)
            nav_items = [
                "01. Input Metadata",
                "02. Signal Spectrum",
                "03. Signal Detection",
                "04. Parameters",
                "05. Modulation",
                "06. Recovery & Sync",
                "07. Data Reconstruction",
                "08. FEC Modifications",
                "09. Verification Matrix",
                "10. Falsification Log",
                "11. Final Assessment",
                "12. Forensic Lineage",
                "13. Diagnostics",
            ]
            for item in nav_items:
                self.nav_list.addItem(item)

            self.nav_list.currentRowChanged.connect(self._switch_page)
            workspace_splitter.addWidget(self.nav_list)

            # Stacked Pages
            self.pages_stack = QStackedWidget()
            self.page_input = InputPage()
            self.page_signal = SignalPage()
            self.page_detection = DetectionPage()
            self.page_parameter = ParameterPage()
            self.page_modulation = ModulationPage()
            self.page_recovery = RecoveryPage()
            self.page_data = DataRecoveryPage()
            self.page_fec = FECPage()
            self.page_verification = VerificationPage()
            self.page_falsification = FalsificationPage()
            self.page_assessment = AssessmentPage()
            self.page_lineage = LineagePage()
            self.page_diagnostics = DiagnosticsPage()

            self.pages_stack.addWidget(self.page_input)
            self.pages_stack.addWidget(self.page_signal)
            self.pages_stack.addWidget(self.page_detection)
            self.pages_stack.addWidget(self.page_parameter)
            self.pages_stack.addWidget(self.page_modulation)
            self.pages_stack.addWidget(self.page_recovery)
            self.pages_stack.addWidget(self.page_data)
            self.pages_stack.addWidget(self.page_fec)
            self.pages_stack.addWidget(self.page_verification)
            self.pages_stack.addWidget(self.page_falsification)
            self.pages_stack.addWidget(self.page_assessment)
            self.pages_stack.addWidget(self.page_lineage)
            self.pages_stack.addWidget(self.page_diagnostics)

            workspace_splitter.addWidget(self.pages_stack)
            workspace_splitter.setStretchFactor(1, 4)
            main_layout.addWidget(workspace_splitter)

            # Status Bar
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)

            self.progress_bar = QProgressBar()
            self.progress_bar.setFixedWidth(200)
            self.progress_bar.setVisible(False)
            self.lbl_status = QLabel("Ready")

            self.status_bar.addWidget(self.lbl_status, 1)
            self.status_bar.addPermanentWidget(self.progress_bar)

            self.nav_list.setCurrentRow(0)

    def _switch_page(self, index: int) -> None:
        if HAS_QT and index >= 0:
            self.pages_stack.setCurrentIndex(index)

    def _open_file(self) -> None:
        if not HAS_QT:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Signal Recording",
            "",
            "Signal Files (*.iq *.raw *.bin *.wav *.sigmf-meta);;All Files (*)",
        )
        if path:
            self.state_model.current_recording_path = path
            self.lbl_status.setText(f"Loaded: {Path(path).name}")

    def _run_analysis(self) -> None:
        if not HAS_QT:
            return
        path = self.state_model.current_recording_path
        if not path:
            QMessageBox.warning(self, "No Input", "Please open a signal recording file first.")
            return

        self._start_pipeline(path, get_preset_config(PresetName.STANDARD_ANALYSIS))

    def _run_judge_demo(self) -> None:
        if not HAS_QT:
            return
        # Generate a demonstration synthetic dataset recording if not present
        from scripts.generate_digital_dataset import generate_digital_stream
        import numpy as np
        from app.models.signal import SignalRecording, SourceFormat
        from tests.test_phase6_cases import _make_rec_sig

        rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
        rec = _make_rec_sig(rx, soft)

        self.lbl_status.setText("Running Judge / Demo Mode on Protocol A Stream...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(20)

        # Run synchronously for demonstration
        res = run_pipeline(rec, config=get_preset_config(PresetName.STANDARD_ANALYSIS))
        self._on_pipeline_finished(res)

    def _start_pipeline(self, path: str, config: PipelineConfig) -> None:
        if not HAS_QT:
            return
        self.cancel_token = CancellationToken()
        self.current_worker = PipelineWorker(path, config, self.cancel_token)
        self.current_worker.progress_signal.connect(self._on_progress)
        self.current_worker.finished_signal.connect(self._on_pipeline_finished)
        self.current_worker.error_signal.connect(self._on_pipeline_error)

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Starting pipeline...")
        self.current_worker.start()

    def _cancel_analysis(self) -> None:
        self.cancel_token.cancel()
        self.lbl_status.setText("Cancelling analysis...")

    def _on_progress(self, update: Any) -> None:
        if not HAS_QT:
            return
        pct = int(update.progress_fraction * 100)
        self.progress_bar.setValue(pct)
        eta_str = f"ETA: {update.estimated_remaining_seconds:.1f}s" if update.estimated_remaining_seconds is not None else "ETA: N/A"
        self.lbl_status.setText(f"[{update.phase_name}] {update.operation} ({eta_str})")

    def _on_pipeline_finished(self, result: PipelineResult) -> None:
        if not HAS_QT:
            return
        self.state_model.set_result(result)
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)

        # Update all pages
        self.page_input.update_data(result)
        self.page_signal.update_data(result)
        self.page_detection.update_data(result)
        self.page_parameter.update_data(result)
        self.page_modulation.update_data(result)
        self.page_recovery.update_data(result)
        self.page_data.update_data(result)
        self.page_fec.update_data(result)
        self.page_verification.update_data(result)
        self.page_falsification.update_data(result)
        self.page_assessment.update_data(result)
        self.page_lineage.update_data(result)

        stat = "INDEPENDENTLY VERIFIED" if result.is_verified else ("FAILED" if result.failure else "UNVERIFIED")
        self.lbl_status.setText(f"Completed in {result.total_duration_seconds:.2f}s | Result: {stat}")
        self.nav_list.setCurrentRow(10) # Jump to Final Assessment Page

    def _on_pipeline_error(self, err_msg: str) -> None:
        if not HAS_QT:
            return
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"Error: {err_msg}")
        QMessageBox.critical(self, "Pipeline Error", f"An error occurred during analysis:\n\n{err_msg}")

    def _export_reports(self) -> None:
        if not HAS_QT:
            return
        res = self.state_model.current_result
        if not res:
            QMessageBox.warning(self, "No Results", "No completed analysis results to export.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if out_dir:
            artifacts = export_all_artifacts(res, out_dir)
            QMessageBox.information(
                self,
                "Export Successful",
                f"Artifacts exported successfully to:\n{out_dir}\n\n"
                f"• report.html\n• report.json\n• frames.csv\n• parameters.csv\n• manifest.json",
            )

def launch_gui() -> int:
    """Launch the SIH26147 Desktop Application."""
    if not HAS_QT:
        print("Error: PySide6 is required to launch the GUI.")
        return 1
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
