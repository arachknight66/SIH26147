from __future__ import annotations
from typing import Any
import numpy as np
from app.orchestration.pipeline_runner import PipelineResult
from app.ui.plots.spectrum_plot import SpectrumPlotWidget
from app.ui.plots.waveform_plot import WaveformPlotWidget

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter
    from PySide6.QtCore import Qt
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class SignalPage(QWidget):
    """
    Time-domain Waveform and Power Spectral Density Visualization Page.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            layout = QVBoxLayout(self)
            splitter = QSplitter(Qt.Vertical)

            self.waveform_plot = WaveformPlotWidget()
            self.spectrum_plot = SpectrumPlotWidget()

            splitter.addWidget(self.waveform_plot)
            splitter.addWidget(self.spectrum_plot)
            layout.addWidget(splitter)

    def update_data(self, result: PipelineResult) -> None:
        if not HAS_QT:
            return
        rec = result.input_recording
        if rec and len(rec.samples) > 0:
            self.waveform_plot.set_samples(rec.samples)

        p2 = result.phase2_result.output if (result.phase2_result and result.phase2_result.output) else None
        if p2 and p2.psd:
            self.spectrum_plot.set_data(p2.psd.frequencies_normalized, p2.psd.psd_db)
