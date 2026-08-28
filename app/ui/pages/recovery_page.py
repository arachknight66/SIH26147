from __future__ import annotations
from typing import Any
from app.orchestration.pipeline_runner import PipelineResult
from app.ui.plots.constellation_plot import ConstellationPlotWidget

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QGridLayout
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class RecoveryPage(QWidget):
    """
    Constellation Reconstruction, Carrier Sync, and Demodulation Inspection Page.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            layout = QVBoxLayout(self)

            gb_stats = QGroupBox("Receiver Synchronization Metrics")
            grid = QGridLayout(gb_stats)
            self.lbl_lock = QLabel("Carrier/Timing Lock: UNLOCKED")
            self.lbl_evm = QLabel("EVM: N/A")
            self.lbl_cfo = QLabel("Residual CFO: N/A")
            self.lbl_sps = QLabel("Recovered SPS: N/A")

            grid.addWidget(self.lbl_lock, 0, 0)
            grid.addWidget(self.lbl_evm, 0, 1)
            grid.addWidget(self.lbl_cfo, 1, 0)
            grid.addWidget(self.lbl_sps, 1, 1)
            layout.addWidget(gb_stats)

            self.const_plot = ConstellationPlotWidget()
            layout.addWidget(self.const_plot)

    def update_data(self, result: PipelineResult) -> None:
        if not HAS_QT:
            return
        p4 = result.phase4_result.output if (result.phase4_result and result.phase4_result.output) else None
        if not p4 or not p4.recovered_signal:
            return
        rec_sig = p4.recovered_signal
        self.lbl_lock.setText("<b>Lock Status:</b> <span style='color: #10b981;'>LOCKED</span>")
        self.lbl_evm.setText(f"<b>1-SPS EVM:</b> {rec_sig.evm_percent:.2f}%")
        self.lbl_cfo.setText(f"<b>Residual CFO:</b> {rec_sig.cfo_normalized:.6f} cycles/sample")
        self.lbl_sps.setText(f"<b>SPS:</b> {rec_sig.samples_per_symbol:.2f}")

        self.const_plot.set_symbols(rec_sig.symbols)
