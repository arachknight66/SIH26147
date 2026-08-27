from __future__ import annotations
from typing import Any
from app.orchestration.pipeline_runner import PipelineResult
from app.ui.widgets.bitstream_view import BitstreamViewer
from app.ui.widgets.frame_table import FrameTableWidget

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class DataRecoveryPage(QWidget):
    """
    Data Reconstruction, Framing, Bitstream, and CRC Inspection Page.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("<h2>Recovered Digital Bitstream & Frames</h2>"))

            tabs = QTabWidget()
            self.frame_table = FrameTableWidget()
            self.bitstream_viewer = BitstreamViewer()

            tabs.addTab(self.frame_table, "Frame Hierarchy")
            tabs.addTab(self.bitstream_viewer, "Raw Bitstream Inspector")
            layout.addWidget(tabs)

    def update_data(self, result: PipelineResult) -> None:
        if not HAS_QT:
            return
        p5 = result.phase5_result.output if (result.phase5_result and result.phase5_result.output) else None
        if not p5 or not p5.selected_candidate:
            return
        sel = p5.selected_candidate
        self.frame_table.populate(sel.frames)
        if sel.bit_hypothesis and sel.bit_hypothesis.bitstream:
            self.bitstream_viewer.set_bits(sel.bit_hypothesis.bitstream.hard_bits)
