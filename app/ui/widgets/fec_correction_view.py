from __future__ import annotations
from typing import Any
import numpy as np
from app.data_recovery.models import FECDecodeResult

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class FECCorrectionViewer(QWidget):
    """
    Detailed visualization of bit-level FEC modifications and correction mask.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            layout = QVBoxLayout(self)
            self.lbl_stats = QLabel("FEC Statistics: N/A")
            layout.addWidget(self.lbl_stats)

            self.text_view = QTextEdit()
            self.text_view.setReadOnly(True)
            self.text_view.setStyleSheet("font-family: monospace; font-size: 12px;")
            layout.addWidget(self.text_view)

    def set_fec_result(self, fec_dec: FECDecodeResult | None) -> None:
        if not HAS_QT:
            return
        if fec_dec is None:
            self.lbl_stats.setText("No Forward Error Correction applied.")
            self.text_view.setPlainText("Signal unencoded or raw bits used.")
            return

        corr_count = fec_dec.corrected_bit_count
        corr_frac = fec_dec.correction_fraction
        self.lbl_stats.setText(
            f"<b>Corrected Bits:</b> {corr_count} | <b>Modification Fraction:</b> {corr_frac:.2%} "
            f"| <b>Safety Budget:</b> &le; 10.0%"
        )

        mask = fec_dec.correction_mask
        mask_str = "".join("X" if b else "." for b in mask[:512])
        formatted = "\n".join(mask_str[i:i+64] for i in range(0, len(mask_str), 64))

        self.text_view.setPlainText(
            f"--- BIT-LEVEL MODIFICATION MASK (X = bit flipped by decoder, . = unchanged) ---\n\n{formatted}"
        )
