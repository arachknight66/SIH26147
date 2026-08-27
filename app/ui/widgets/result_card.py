from __future__ import annotations
from typing import Any
from app.orchestration.pipeline_runner import PipelineResult
from .epistemic_badge import EpistemicBadge

try:
    from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout
    from PySide6.QtCore import Qt
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QFrame:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
        def setStyleSheet(self, s: str) -> None: pass

class ResultCard(QFrame):
    """
    Compact Executive Result Card summarizing the recovered signal parameters.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            self.setStyleSheet(
                "QFrame { background-color: #1e293b; border: 2px solid #38bdf8; border-radius: 10px; padding: 12px; }"
            )
            self._layout = QVBoxLayout(self)
            self._title = QLabel("<b>SIGNAL RECOVERY & VERIFICATION CARD</b>")
            self._title.setStyleSheet("color: #38bdf8; font-size: 14px;")
            self._layout.addWidget(self._title)

            self._grid = QGridLayout()
            self._layout.addLayout(self._grid)

            self._lbl_mod = QLabel("Modulation: UNKNOWN")
            self._lbl_sps = QLabel("Symbol Rate: UNKNOWN")
            self._lbl_fec = QLabel("FEC Code: NONE")
            self._lbl_crc = QLabel("Integrity: NONE")
            self._lbl_status = QLabel("Overall: UNKNOWN")

            self._grid.addWidget(self._lbl_mod, 0, 0)
            self._grid.addWidget(self._lbl_sps, 0, 1)
            self._grid.addWidget(self._lbl_fec, 1, 0)
            self._grid.addWidget(self._lbl_crc, 1, 1)
            self._grid.addWidget(self._lbl_status, 2, 0, 1, 2)

    def update_result(self, result: PipelineResult) -> None:
        if not HAS_QT:
            return
        p3 = result.phase3_result.output if (result.phase3_result and result.phase3_result.output) else None
        p4 = result.phase4_result.output if (result.phase4_result and result.phase4_result.output) else None
        p5 = result.phase5_result.output if (result.phase5_result and result.phase5_result.output) else None
        p6 = result.phase6_result.output if (result.phase6_result and result.phase6_result.output) else None
        sel_cand = p5.selected_candidate if p5 else None

        mod_str = p3.selected_hypothesis.label if (p3 and p3.selected_hypothesis) else "UNKNOWN"
        sps_str = f"{p4.recovered_signal.samples_per_symbol:.2f} SPS" if (p4 and p4.recovered_signal) else "UNKNOWN"
        fec_str = sel_cand.fec.code_name if (sel_cand and sel_cand.fec) else "NONE"
        crc_str = sel_cand.integrity.crc_results[0].crc_name if (sel_cand and sel_cand.integrity and sel_cand.integrity.crc_results) else "NONE"
        stat_str = p6.status.value.upper() if p6 else ("FAILED" if result.failure else "UNKNOWN")

        self._lbl_mod.setText(f"<b>Modulation:</b> {mod_str}")
        self._lbl_sps.setText(f"<b>Rate:</b> {sps_str}")
        self._lbl_fec.setText(f"<b>FEC:</b> {fec_str}")
        self._lbl_crc.setText(f"<b>Integrity:</b> {crc_str}")
        self._lbl_status.setText(f"<b>Verification:</b> <span style='color: #10b981;'>{stat_str}</span>" if result.is_verified else f"<b>Verification:</b> <span style='color: #ef4444;'>{stat_str}</span>")
