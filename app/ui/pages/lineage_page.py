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

class LineagePage(QWidget):
    """
    Forensic Data Lineage Graph View.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("<h2>Forensic Data Transformation Lineage</h2>"))

            self.text_view = QTextEdit()
            self.text_view.setReadOnly(True)
            layout.addWidget(self.text_view)

    def update_data(self, result: PipelineResult) -> None:
        if not HAS_QT:
            return
        p2 = result.phase2_result.output if (result.phase2_result and result.phase2_result.output) else None
        p3 = result.phase3_result.output if (result.phase3_result and result.phase3_result.output) else None
        p4 = result.phase4_result.output if (result.phase4_result and result.phase4_result.output) else None
        p5 = result.phase5_result.output if (result.phase5_result and result.phase5_result.output) else None
        p6 = result.phase6_result.output if (result.phase6_result and result.phase6_result.output) else None
        sel = p5.selected_candidate if p5 else None

        md = f"""<h3>End-to-End Forensic Lineage Graph</h3>
<pre style="background-color: #1e293b; color: #38bdf8; padding: 16px; border-radius: 8px; font-family: monospace; font-size: 13px;">
[01. Raw Recording]
     ↓ (SHA-256: {result.input_sha256[:16]}...)
[02. Physical Measurements]
     ↓ (SNR: {p2.snr_candidates[0].snr_db if (p2 and p2.snr_candidates) else 0.0:.1f} dB, Noise: {p2.noise_estimate.noise_floor_db if p2 else 0.0:.1f} dB)
[03. Modulation Hypothesis]
     ↓ (Selected: {p3.selected_hypothesis.label if (p3 and p3.selected_hypothesis) else 'None'})
[04. Recovered Symbols]
     ↓ (1-SPS Constellation, EVM: {p4.recovered_signal.evm_percent if (p4 and p4.recovered_signal) else 0.0:.2f}%)
[05. Bitstream & Framing]
     ↓ (Frames: {len(sel.frames) if sel else 0}, Alignment: MSB_FIRST)
[06. Error Correction & Descrambler]
     ↓ (FEC: {sel.fec.code_name if (sel and sel.fec) else 'None'}, CRC: {sel.integrity.crc_results[0].crc_name if (sel and sel.integrity and sel.integrity.crc_results) else 'None'})
[07. Independent Verification]
     ↳ Status: {p6.status.value.upper() if p6 else 'UNKNOWN'} (Reproducibility: {result.provenance.reproducibility_hash[:16] if result.provenance else 'N/A'}...)
</pre>
"""
        self.text_view.setHtml(md)
