from __future__ import annotations
from typing import Any
from app.orchestration.pipeline_runner import PipelineResult
from app.ui.widgets.epistemic_badge import EpistemicBadge

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class InputPage(QWidget):
    """
    Input Metadata and Forensic Diagnostics Page.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            layout = QVBoxLayout(self)

            # Metadata Box
            gb_meta = QGroupBox("Signal Recording Metadata & Provenance")
            grid = QGridLayout(gb_meta)

            self.lbl_path = QLabel("Source: None")
            self.lbl_format = QLabel("Format: Unknown")
            self.lbl_samples = QLabel("Samples: 0")
            self.lbl_dtype = QLabel("Data Type: Unknown")
            self.lbl_rate = QLabel("Sample Rate: UNKNOWN (No metadata)")
            self.lbl_freq = QLabel("Center Freq: UNKNOWN (Baseband assumed)")

            grid.addWidget(self.lbl_path, 0, 0, 1, 2)
            grid.addWidget(self.lbl_format, 1, 0)
            grid.addWidget(self.lbl_samples, 1, 1)
            grid.addWidget(self.lbl_dtype, 2, 0)
            grid.addWidget(self.lbl_rate, 2, 1)
            grid.addWidget(self.lbl_freq, 3, 0, 1, 2)
            layout.addWidget(gb_meta)

            # Forensics Table
            gb_diag = QGroupBox("Input Forensics & Diagnostics")
            v_diag = QVBoxLayout(gb_diag)
            self.diag_table = QTableWidget(0, 3)
            self.diag_table.setHorizontalHeaderLabels(["Severity", "Code", "Message"])
            self.diag_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            v_diag.addWidget(self.diag_table)
            layout.addWidget(gb_diag)

    def update_data(self, result: PipelineResult) -> None:
        if not HAS_QT:
            return
        rec = result.input_recording
        self.lbl_path.setText(f"<b>Source:</b> {result.input_path or 'In-Memory Synthetic Stream'}")
        if rec:
            self.lbl_format.setText(f"<b>Format:</b> {rec.source_format.value}")
            self.lbl_samples.setText(f"<b>Sample Count:</b> {len(rec.samples):,}")
            self.lbl_dtype.setText(f"<b>Data Type:</b> {rec.original_dtype}")
            self.lbl_rate.setText("<b>Sample Rate:</b> UNKNOWN (Metadata absent, normalized time used)")
            self.lbl_freq.setText("<b>Center Freq:</b> 0.0 Hz (Baseband IQ representation)")

            diags = rec.diagnostics
            self.diag_table.setRowCount(len(diags))
            for r, d in enumerate(diags):
                self.diag_table.setItem(r, 0, QTableWidgetItem(d.severity.value.upper()))
                self.diag_table.setItem(r, 1, QTableWidgetItem(d.code))
                self.diag_table.setItem(r, 2, QTableWidgetItem(d.message))
