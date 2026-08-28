from __future__ import annotations
from typing import Any, Sequence
from app.data_recovery.models import FrameCandidate

try:
    from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QTableWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class FrameTableWidget(QTableWidget):
    """
    Interactive table for inspecting recovered digital frames.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            self.setColumnCount(8)
            self.setHorizontalHeaderLabels([
                "Frame #", "Start Bit", "End Bit", "Length", "Sequence", "CRC Valid", "FEC Corrected", "Payload Bytes"
            ])
            self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.setSelectionBehavior(QTableWidget.SelectRows)

    def populate(self, frames: Sequence[FrameCandidate]) -> None:
        if not HAS_QT:
            return
        self.setRowCount(len(frames))
        for r, fr in enumerate(frames):
            self.setItem(r, 0, QTableWidgetItem(str(fr.frame_index)))
            self.setItem(r, 1, QTableWidgetItem(str(fr.start_bit)))
            self.setItem(r, 2, QTableWidgetItem(str(fr.end_bit)))
            self.setItem(r, 3, QTableWidgetItem(str(len(fr.raw_bits))))
            self.setItem(r, 4, QTableWidgetItem(str(fr.sequence_number if fr.sequence_number is not None else "N/A")))
            crc_item = QTableWidgetItem("PASS" if fr.is_crc_valid else "FAIL")
            self.setItem(r, 5, crc_item)
            self.setItem(r, 6, QTableWidgetItem("YES" if fr.is_fec_corrected else "NO"))
            self.setItem(r, 7, QTableWidgetItem(str(len(fr.decoded_payload) if fr.decoded_payload else 0)))
