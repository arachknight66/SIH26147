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

class ModulationPage(QWidget):
    """
    Ranked Modulation Hypotheses & Evidence Strength Page.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("<h2>Ranked Modulation Hypotheses</h2>"))

            self.table = QTableWidget(0, 5)
            self.table.setHorizontalHeaderLabels([
                "Rank", "Modulation Scheme", "Score", "Family", "Evidence Strength"
            ])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            layout.addWidget(self.table)

    def update_data(self, result: PipelineResult) -> None:
        if not HAS_QT:
            return
        p3 = result.phase3_result.output if (result.phase3_result and result.phase3_result.output) else None
        if not p3:
            return
        hyps = p3.hypotheses
        self.table.setRowCount(len(hyps))
        for r, h in enumerate(hyps):
            self.table.setItem(r, 0, QTableWidgetItem(str(r + 1)))
            self.table.setItem(r, 1, QTableWidgetItem(h.label))
            self.table.setItem(r, 2, QTableWidgetItem(f"{h.score:.4f}"))
            self.table.setItem(r, 3, QTableWidgetItem(h.family.value))
            ev_str = "STRONG" if h.score > 0.60 else ("MODERATE" if h.score > 0.30 else "WEAK")
            self.table.setItem(r, 4, QTableWidgetItem(ev_str))
