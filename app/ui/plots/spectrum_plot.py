from __future__ import annotations
from typing import Any
import numpy as np

try:
    import pyqtgraph as pg
    from PySide6.QtWidgets import QWidget, QVBoxLayout
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class SpectrumPlotWidget(QWidget):
    """PyQtGraph widget for Welch PSD and FFT spectrum."""
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_PYQTGRAPH:
            layout = QVBoxLayout(self)
            self.plot_widget = pg.PlotWidget(title="Power Spectral Density (Welch PSD)")
            self.plot_widget.setBackground("#0f172a")
            self.plot_widget.setLabel("left", "Power Spectral Density", units="dB/Hz")
            self.plot_widget.setLabel("bottom", "Normalized Frequency", units="cycles/sample")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.curve = self.plot_widget.plot(pen=pg.mkPen("#38bdf8", width=1.5))
            layout.addWidget(self.plot_widget)

    def set_data(self, freqs: np.ndarray, psd_db: np.ndarray) -> None:
        if HAS_PYQTGRAPH and hasattr(self, "curve"):
            self.curve.setData(freqs, psd_db)
