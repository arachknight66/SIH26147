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

class ConstellationPlotWidget(QWidget):
    """PyQtGraph widget for 1-SPS Constellation & Decision Centroids."""
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_PYQTGRAPH:
            layout = QVBoxLayout(self)
            self.plot_widget = pg.PlotWidget(title="1-SPS Constellation Diagram (I/Q)")
            self.plot_widget.setBackground("#0f172a")
            self.plot_widget.setLabel("left", "Quadrature (Q)")
            self.plot_widget.setLabel("bottom", "In-Phase (I)")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setAspectLocked(True)

            self.scatter = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None), brush=pg.mkBrush("#38bdf8", alpha=180))
            self.centroids = pg.ScatterPlotItem(size=12, pen=pg.mkPen("#ef4444", width=2), brush=pg.mkBrush("#f59e0b"))
            self.plot_widget.addItem(self.scatter)
            self.plot_widget.addItem(self.centroids)
            layout.addWidget(self.plot_widget)

    def set_symbols(self, symbols: np.ndarray, ideal_centroids: np.ndarray | None = None) -> None:
        if not HAS_PYQTGRAPH or not hasattr(self, "scatter"):
            return
        # Sample points
        i_pts = np.real(symbols[:2000])
        q_pts = np.imag(symbols[:2000])
        self.scatter.setData(i_pts, q_pts)

        if ideal_centroids is not None and len(ideal_centroids) > 0:
            c_i = np.real(ideal_centroids)
            c_q = np.imag(ideal_centroids)
            self.centroids.setData(c_i, c_q)
        else:
            self.centroids.setData([], [])
