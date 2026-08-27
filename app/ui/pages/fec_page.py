from __future__ import annotations
from typing import Any
from app.orchestration.pipeline_runner import PipelineResult
from app.ui.widgets.fec_correction_view import FECCorrectionViewer

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class FECPage(QWidget):
    """
    Forward Error Correction & Bit Modification Mask Inspection Page.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("<h2>Forward Error Correction (FEC) & Bit Modifications</h2>"))
            self.viewer = FECCorrectionViewer()
            layout.addWidget(self.viewer)

    def update_data(self, result: PipelineResult) -> None:
        if not HAS_QT:
            return
        p5 = result.phase5_result.output if (result.phase5_result and result.phase5_result.output) else None
        fec_dec = p5.selected_candidate.fec_decode if (p5 and p5.selected_candidate) else None
        self.viewer.set_fec_result(fec_dec)
