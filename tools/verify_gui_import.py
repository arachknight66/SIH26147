"""Cross-platform CI smoke check for the optional Qt GUI dependencies."""

import PySide6
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow
import pyqtgraph

from signal_analysis.gui import HAS_QT


if not HAS_QT:
    raise RuntimeError("PySide6 and pyqtgraph imported, but signal_analysis.gui reports HAS_QT=False")

print(f"PySide6 {PySide6.__version__}; pyqtgraph {pyqtgraph.__version__}; HAS_QT={HAS_QT}")
