from __future__ import annotations
from typing import Any
from app.orchestration.pipeline_runner import PipelineResult
from app.ui.widgets.epistemic_badge import EpistemicBadge

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QLabel
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class ParameterPage(QWidget):
    """
    Extracted Physical and Digital Parameter Summary Page with Epistemic Statuses.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("<h2>Extracted Quantitative Parameters</h2>"))

            self.table = QTableWidget(0, 5)
            self.table.setHorizontalHeaderLabels([
                "Parameter", "Value", "Unit", "Epistemic Status", "Uncertainty / Evidence"
            ])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            layout.addWidget(self.table)

    def update_data(self, result: PipelineResult) -> None:
        if not HAS_QT:
            return
        p2 = result.phase2_result.output if (result.phase2_result and result.phase2_result.output) else None
        p3 = result.phase3_result.output if (result.phase3_result and result.phase3_result.output) else None
        p4 = result.phase4_result.output if (result.phase4_result and result.phase4_result.output) else None
        p5 = result.phase5_result.output if (result.phase5_result and result.phase5_result.output) else None

        rows = [
            ("SNR Estimate", f"{p2.snr_candidates[0].snr_db:.2f}" if (p2 and p2.snr_candidates) else "N/A", "dB", "ESTIMATED", "PSD / M2M4 ratio"),
            ("Noise Floor", f"{p2.noise_estimate.noise_floor_db:.2f}" if p2 else "N/A", "dB", "ESTIMATED", "25th percentile Welch PSD"),
            ("Modulation Family", p3.selected_hypothesis.label if (p3 and p3.selected_hypothesis) else "UNKNOWN", "class", "INFERRED", f"Score: {p3.selected_hypothesis.score:.3f}" if (p3 and p3.selected_hypothesis) else "N/A"),
            ("Samples Per Symbol", f"{p4.recovered_signal.samples_per_symbol:.2f}" if (p4 and p4.recovered_signal) else "UNKNOWN", "samples/sym", "INFERRED", "Gardner TED"),
            ("EVM RMS", f"{p4.recovered_signal.evm_percent:.2f}" if (p4 and p4.recovered_signal) else "UNKNOWN", "%", "MEASURED", "1-SPS Constellation distance"),
            ("Residual CFO", f"{p4.recovered_signal.cfo_normalized:.6f}" if (p4 and p4.recovered_signal) else "UNKNOWN", "norm", "MEASURED", "Costas loop locked"),
            ("FEC Scheme", p5.selected_candidate.fec.code_name if (p5 and p5.selected_candidate and p5.selected_candidate.fec) else "NONE", "code", "INFERRED", "Viterbi Trellis"),
            ("Integrity CRC", p5.selected_candidate.integrity.crc_results[0].crc_name if (p5 and p5.selected_candidate and p5.selected_candidate.integrity and p5.selected_candidate.integrity.crc_results) else "NONE", "crc", "INFERRED", "Syndrome validation"),
        ]

        self.table.setRowCount(len(rows))
        for r, (param, val, unit, ep, ev) in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(param))
            self.table.setItem(r, 1, QTableWidgetItem(val))
            self.table.setItem(r, 2, QTableWidgetItem(unit))
            self.table.setItem(r, 3, QTableWidgetItem(ep))
            self.table.setItem(r, 4, QTableWidgetItem(ev))
