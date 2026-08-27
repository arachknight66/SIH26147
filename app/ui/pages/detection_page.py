from __future__ import annotations
from typing import Any
from app.orchestration.pipeline_runner import PipelineResult

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QLabel
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class DetectionPage(QWidget):
    """
    Signal Region Detection & Bounding Box Inspection Page.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("<h2>Detected Signal Regions & Energy Bursts</h2>"))

            self.table = QTableWidget(0, 7)
            self.table.setHorizontalHeaderLabels([
                "Region ID", "Start Sample", "End Sample", "Center Freq", "Bandwidth", "Estimated SNR", "Detection Score"
            ])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            layout.addWidget(self.table)

    def update_data(self, result: PipelineResult) -> None:
        if not HAS_QT:
            return
        p2 = result.phase2_result.output if (result.phase2_result and result.phase2_result.output) else None
        if not p2:
            return
        regions = p2.detected_regions
        self.table.setRowCount(len(regions))
        for r, reg in enumerate(regions):
            self.table.setItem(r, 0, QTableWidgetItem(f"Region {reg.region_id}"))
            self.table.setItem(r, 1, QTableWidgetItem(str(reg.start_sample)))
            self.table.setItem(r, 2, QTableWidgetItem(str(reg.end_sample)))
            self.table.setItem(r, 3, QTableWidgetItem(f"{reg.center_freq_normalized:.4f}"))
            self.table.setItem(r, 4, QTableWidgetItem(f"{reg.bandwidth_normalized:.4f}"))
            self.table.setItem(r, 5, QTableWidgetItem(f"{reg.estimated_snr_db:.1f} dB"))
            self.table.setItem(r, 6, QTableWidgetItem(f"{reg.detection_score:.2f}"))
