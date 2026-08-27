from __future__ import annotations
from typing import Any
import numpy as np

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel, QRadioButton, QButtonGroup
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class BitstreamViewer(QWidget):
    """
    Multi-mode digital stream inspector: Binary, Hex, and ASCII view.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._raw_bits: np.ndarray = np.array([], dtype=np.uint8)

        if HAS_QT:
            layout = QVBoxLayout(self)
            bar = QHBoxLayout()

            self.rb_hex = QRadioButton("HEX View")
            self.rb_bin = QRadioButton("Binary View")
            self.rb_asc = QRadioButton("ASCII View")
            self.rb_hex.setChecked(True)

            self.bg = QButtonGroup(self)
            self.bg.addButton(self.rb_hex)
            self.bg.addButton(self.rb_bin)
            self.bg.addButton(self.rb_asc)

            self.rb_hex.toggled.connect(self._render)
            self.rb_bin.toggled.connect(self._render)
            self.rb_asc.toggled.connect(self._render)

            bar.addWidget(QLabel("<b>Display Mode:</b>"))
            bar.addWidget(self.rb_hex)
            bar.addWidget(self.rb_bin)
            bar.addWidget(self.rb_asc)
            bar.addStretch()

            layout.addLayout(bar)

            self.text_edit = QTextEdit()
            self.text_edit.setReadOnly(True)
            self.text_edit.setStyleSheet("font-family: monospace; font-size: 12px;")
            layout.addWidget(self.text_edit)

    def set_bits(self, bits: np.ndarray) -> None:
        self._raw_bits = bits
        self._render()

    def _render(self) -> None:
        if not HAS_QT or len(self._raw_bits) == 0:
            return

        if self.rb_bin.isChecked():
            # Format bits in groups of 8
            s = "".join(str(b) for b in self._raw_bits[:2048])
            formatted = " ".join(s[i:i+8] for i in range(0, len(s), 8))
            self.text_edit.setPlainText(formatted)
        elif self.rb_hex.isChecked():
            raw_bytes = np.packbits(self._raw_bits)
            hex_str = " ".join(f"{b:02X}" for b in raw_bytes[:1024])
            self.text_edit.setPlainText(hex_str)
        else: # ASCII
            raw_bytes = np.packbits(self._raw_bits)
            ascii_chars = [chr(b) if 32 <= b <= 126 else "." for b in raw_bytes[:1024]]
            self.text_edit.setPlainText("".join(ascii_chars))
