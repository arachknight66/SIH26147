from __future__ import annotations
from typing import Any
from app.ui.theme import EPISTEMIC_COLORS

try:
    from PySide6.QtWidgets import QLabel
    from PySide6.QtCore import Qt
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QLabel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
        def setText(self, t: str) -> None: pass
        def setStyleSheet(self, s: str) -> None: pass

class EpistemicBadge(QLabel):
    """
    Color-coded epistemic badge displaying OBSERVED, INFERRED, ASSUMED, VERIFIED, etc.
    """
    def __init__(self, status: str = "UNKNOWN", parent: Any = None) -> None:
        super().__init__(parent)
        self.set_status(status)

    def set_status(self, status: str) -> None:
        s_upper = status.upper()
        color = EPISTEMIC_COLORS.get(s_upper, "#64748b")
        if HAS_QT:
            self.setText(f" {s_upper} ")
            self.setStyleSheet(
                f"background-color: {color}; color: #0f172a; font-weight: bold; "
                f"border-radius: 4px; padding: 2px 6px; font-size: 11px;"
            )
