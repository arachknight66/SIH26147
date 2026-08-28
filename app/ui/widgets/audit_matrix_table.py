from __future__ import annotations
from typing import Any, Sequence
from app.verification.models import VerificationClaim

try:
    from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
    from PySide6.QtGui import QColor
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QTableWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class AuditMatrixTable(QTableWidget):
    """
    Interactive table for the 7-Claim Scientific Verification Matrix.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            self.setColumnCount(5)
            self.setHorizontalHeaderLabels([
                "Claim ID", "Claim Description", "Status", "Confidence", "Independence"
            ])
            self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.setSelectionBehavior(QTableWidget.SelectRows)

    def populate(self, claims: Sequence[VerificationClaim]) -> None:
        if not HAS_QT:
            return
        self.setRowCount(len(claims))
        for r, c in enumerate(claims):
            self.setItem(r, 0, QTableWidgetItem(f"Claim {c.claim_id}"))
            self.setItem(r, 1, QTableWidgetItem(c.claim_text))

            item_status = QTableWidgetItem(c.status.value.upper())
            if c.status.value == "supported":
                item_status.setForeground(QColor("#10b981"))
            elif c.status.value == "falsified":
                item_status.setForeground(QColor("#ef4444"))
            else:
                item_status.setForeground(QColor("#f59e0b"))

            self.setItem(r, 2, item_status)
            self.setItem(r, 3, QTableWidgetItem(f"{c.confidence:.2f}"))
            self.setItem(r, 4, QTableWidgetItem(c.independence_level.value))
