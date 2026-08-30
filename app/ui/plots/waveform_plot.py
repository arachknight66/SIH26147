from __future__ import annotations
from typing import Any
import numpy as np
from .plot_manager import downsample_signal

try:
    import pyqtgraph as pg
    from PySide6.QtWidgets import QWidget, QVBoxLayout
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class WaveformPlotWidget(QWidget):
    """PyQtGraph widget for Time-Domain I/Q Waveform."""
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_PYQTGRAPH:
            layout = QVBoxLayout(self)
            self.plot_widget = pg.PlotWidget(title="Time-Domain Signal Waveform")
            self.plot_widget.setBackground("#0f172a")
            self.plot_widget.setLabel("left", "Amplitude")
            self.plot_widget.setLabel("bottom", "Sample Index")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.curve_i = self.plot_widget.plot(pen=pg.mkPen("#38bdf8", width=1.2), name="I (In-Phase)")
            self.curve_q = self.plot_widget.plot(pen=pg.mkPen("#f59e0b", width=1.2), name="Q (Quadrature)")
            layout.addWidget(self.plot_widget)

    def set_samples(self, samples: np.ndarray) -> None:
        if not HAS_PYQTGRAPH or not hasattr(self, "curve_i"):
            return
        ds = downsample_signal(samples, max_points=5000)
        t = np.arange(len(ds))
        self.curve_i.setData(t, np.real(ds))
        self.curve_q.setData(t, np.imag(ds))
