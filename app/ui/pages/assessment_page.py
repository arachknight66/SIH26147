from __future__ import annotations
from typing import Any
from app.orchestration.pipeline_runner import PipelineResult
from app.ui.widgets.result_card import ResultCard
from app.ui.widgets.why_dialog import WhyDialog

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class AssessmentPage(QWidget):
    """
    Final Scientific Assessment & "WHY?" Explainability Page.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._current_result: PipelineResult | None = None

        if HAS_QT:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("<h2>Final Scientific Assessment & Executive Summary</h2>"))

            self.result_card = ResultCard()
            layout.addWidget(self.result_card)

            btn_box = QHBoxLayout()
            self.btn_why = QPushButton("🔍 Inspect Evidence ('WHY?' Analysis)")
            self.btn_why.setStyleSheet("background-color: #f59e0b; color: #0f172a; font-size: 13px; font-weight: bold;")
            self.btn_why.clicked.connect(self._show_why)
            btn_box.addWidget(self.btn_why)
            btn_box.addStretch()
            layout.addLayout(btn_box)

            self.summary_text = QTextEdit()
            self.summary_text.setReadOnly(True)
            layout.addWidget(self.summary_text)

    def update_data(self, result: PipelineResult) -> None:
        self._current_result = result
        if not HAS_QT:
            return
        self.result_card.update_result(result)
        p6 = result.phase6_result.output if (result.phase6_result and result.phase6_result.output) else None

        md = f"""<h3>Executive Decision</h3>
<p>{result.final_assessment_text}</p>

<h3>Scientific Constraints & Known Limitations</h3>
<ul>
    <li><b>Sampling Frequency:</b> Absolute physical sampling rate in Hz is unknown; all metrics computed in normalized cycles/sample.</li>
    <li><b>Higher-Layer Protocol:</b> Physical and link-layer structures verified; application payload semantics unverified.</li>
    <li><b>Cryptographic Authentication:</b> CRC integrity confirmed; cryptographic authentication unverified.</li>
</ul>

<h3>Reproducibility Hash (SHA-256)</h3>
<pre style="background-color: #1e293b; padding: 8px; border-radius: 4px; color: #38bdf8;">{result.provenance.reproducibility_hash if result.provenance else 'N/A'}</pre>
"""
        self.summary_text.setHtml(md)

    def _show_why(self) -> None:
        if HAS_QT and self._current_result:
            dlg = WhyDialog(self._current_result, self)
            dlg.exec()
