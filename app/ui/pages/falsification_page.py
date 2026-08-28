from __future__ import annotations
from typing import Any
from app.orchestration.pipeline_runner import PipelineResult

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class FalsificationPage(QWidget):
    """
    Falsification & Adversarial Hypothesis Disproof View.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("<h2>Adversarial Falsification & Disproof Log</h2>"))

            self.text_view = QTextEdit()
            self.text_view.setReadOnly(True)
            layout.addWidget(self.text_view)

    def update_data(self, result: PipelineResult) -> None:
        if not HAS_QT:
            return
        p6 = result.phase6_result.output if (result.phase6_result and result.phase6_result.output) else None
        if not p6:
            self.text_view.setPlainText("No verification analysis available.")
            return

        fals_audit = p6.falsification_audit
        tests = []
        for c in p6.claims:
            for t in c.tests:
                tests.append(t)

        failed_tests = [t for t in tests if t.status.value in ("FAIL", "falsified")]
        passed_tests = [t for t in tests if t.status.value in ("PASS", "supported", "WEAK_PASS")]

        md = f"""<h3>Falsification Summary</h3>
<p><b>Total Tests Evaluated:</b> {len(tests)} | <b>Surviving Falsification Tests:</b> {len(passed_tests)} | <b>Failed / Counter-Evidence:</b> {len(failed_tests)}</p>

<h3>Active Falsification Probes & Outcomes</h3>
<ul>
    <li><b>Boundary Perturbation Test:</b> Evaluated shifts &plusmn;1, &plusmn;2, &plusmn;4, &plusmn;8 bits. Status: <b>{'PASS (Structural Collapse Verified)' if (p6.frame_audit and p6.frame_audit.boundary_perturbation_passed) else 'FAIL'}</b></li>
    <li><b>Leave-One-Out Frame Stability:</b> Tested frame subset variance. Status: <b>{'PASS (Stable Periodic Grid)' if (p6.robustness_audit and p6.robustness_audit.leave_one_out_stable) else 'FAIL'}</b></li>
    <li><b>Held-Out FEC Validation (70/30):</b> Validated Viterbi trellis on 30% unobserved stream. Status: <b>{'PASS' if (p6.fec_audit and p6.fec_audit.held_out_validation_passed) else 'FAIL'}</b></li>
    <li><b>Random Noise Null Model Comparison:</b> CRC false discovery p-value = {p6.integrity_audit.multiple_testing_corrected_p_value if p6.integrity_audit else 1.0:.2e}</li>
</ul>
"""
        self.text_view.setHtml(md)
