from __future__ import annotations
from typing import Any
from app.orchestration.pipeline_runner import PipelineResult
from app.ui.widgets.audit_matrix_table import AuditMatrixTable

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QGridLayout
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class VerificationPage(QWidget):
    """
    Phase 6 7-Claim Scientific Verification Matrix & Error Budget Page.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("<h2>Independent Scientific Verification & Falsification Matrix</h2>"))

            gb_budget = QGroupBox("Composite Uncertainty Error Budget")
            grid = QGridLayout(gb_budget)
            self.lbl_carrier = QLabel("Carrier Uncertainty: 0.0000")
            self.lbl_timing = QLabel("Timing Uncertainty: 0.0000")
            self.lbl_ber = QLabel("BER Proxy: 0.0000")
            self.lbl_total = QLabel("Total Composite Uncertainty: 0.0000")

            grid.addWidget(self.lbl_carrier, 0, 0)
            grid.addWidget(self.lbl_timing, 0, 1)
            grid.addWidget(self.lbl_ber, 1, 0)
            grid.addWidget(self.lbl_total, 1, 1)
            layout.addWidget(gb_budget)

            self.matrix_table = AuditMatrixTable()
            layout.addWidget(self.matrix_table)

    def update_data(self, result: PipelineResult) -> None:
        if not HAS_QT:
            return
        p6 = result.phase6_result.output if (result.phase6_result and result.phase6_result.output) else None
        if not p6:
            return
        if p6.error_budget:
            eb = p6.error_budget
            self.lbl_carrier.setText(f"<b>Carrier Uncertainty:</b> {eb.carrier_uncertainty:.4f}")
            self.lbl_timing.setText(f"<b>Timing Uncertainty:</b> {eb.timing_uncertainty:.4f}")
            self.lbl_ber.setText(f"<b>BER Proxy:</b> {eb.bit_error_rate_proxy:.4f}")
            self.lbl_total.setText(f"<b>Total Uncertainty:</b> {eb.total_composite_uncertainty:.4f}")

        self.matrix_table.populate(p6.claims)
