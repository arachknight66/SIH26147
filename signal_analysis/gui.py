import sys
import numpy as np
from typing import Optional

HAS_QT = True
try:
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                   QHBoxLayout, QLabel, QPushButton, QFileDialog,
                                   QDialog, QComboBox, QFormLayout, QLineEdit, QDialogButtonBox,
                                   QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QTextEdit)
    from PySide6.QtCore import Qt
    import pyqtgraph as pg
except ImportError:
    try:
        from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                     QHBoxLayout, QLabel, QPushButton, QFileDialog,
                                     QDialog, QComboBox, QFormLayout, QLineEdit, QDialogButtonBox,
                                     QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QTextEdit)
        from PyQt6.QtCore import Qt
        import pyqtgraph as pg
    except ImportError:
        HAS_QT = False

from .models import SignalRecording, MetadataStatus, ModulationHypothesis
from .loaders import RawIQConfig, RawIQReader, WavReader, read_sigmf
from .measurements import compute_psd, compute_spectrogram
from .features import extract_all_features
from .classifier import compute_classical_scores
from .hypotheses import evaluate_and_rank_hypotheses, check_temporal_consistency
from .demodulation import attempt_synchronization_multi_hypothesis

if HAS_QT:
    class RawIQDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Raw IQ Configuration")
            self.layout = QFormLayout(self)
            
            self.dtype_cb = QComboBox()
            self.dtype_cb.addItems(["complex64", "float32", "int16", "int8", "uint8"])
            self.layout.addRow("Data Type:", self.dtype_cb)
            
            self.iq_order_cb = QComboBox()
            self.iq_order_cb.addItems(["IQ", "QI"])
            self.layout.addRow("I/Q Order:", self.iq_order_cb)
            
            self.endian_cb = QComboBox()
            self.endian_cb.addItems(["little", "big"])
            self.layout.addRow("Endianness:", self.endian_cb)
            
            self.sr_edit = QLineEdit()
            self.layout.addRow("Sample Rate (Hz, optional):", self.sr_edit)
            
            self.cf_edit = QLineEdit()
            self.layout.addRow("Center Freq (Hz, optional):", self.cf_edit)
            
            self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            self.buttons.accepted.connect(self.accept)
            self.buttons.rejected.connect(self.reject)
            self.layout.addRow(self.buttons)
            
        def get_config(self) -> RawIQConfig:
            sr_text = self.sr_edit.text().strip()
            cf_text = self.cf_edit.text().strip()
            return RawIQConfig(
                dtype=self.dtype_cb.currentText(),
                iq_order=self.iq_order_cb.currentText(),
                endian=self.endian_cb.currentText(),
                sample_rate_hz=float(sr_text) if sr_text else None,
                center_frequency_hz=float(cf_text) if cf_text else None
            )

    class MetadataSidebar(QScrollArea):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWidgetResizable(True)
            self.content = QWidget()
            self.layout = QVBoxLayout(self.content)
            self.setWidget(self.content)
            
            self.labels = {}
            for key in ["Source Format", "Sample Count", "DType", "Semantic Type", "Sample Rate", "Center Frequency"]:
                lbl = QLabel(f"{key}: N/A")
                self.layout.addWidget(lbl)
                self.labels[key] = lbl
                
            # Phase 2 UI elements
            self.layout.addWidget(QLabel("--- Phase 2 Analysis ---"))
            
            self.symbol_rate_lbl = QLabel("Symbol Rate: N/A")
            self.layout.addWidget(self.symbol_rate_lbl)
            
            self.hyp_table = QTableWidget(0, 3)
            self.hyp_table.setHorizontalHeaderLabels(["Modulation", "Score", "Tier"])
            self.hyp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.hyp_table.itemSelectionChanged.connect(self.on_hyp_selected)
            self.layout.addWidget(self.hyp_table)
            
            self.evidence_lbl = QLabel("Evidence Breakdown:\nSelect a hypothesis")
            self.evidence_lbl.setWordWrap(True)
            self.layout.addWidget(self.evidence_lbl)
            
            self.diagnostics_label = QLabel("--- Diagnostics ---")
            self.layout.addWidget(self.diagnostics_label)
            
            self.layout.addWidget(QLabel("--- Phase 3 Sync ---"))
            self.sync_status_lbl = QLabel("Sync Status: N/A")
            self.layout.addWidget(self.sync_status_lbl)
            self.cfo_lbl = QLabel("CFO: N/A")
            self.layout.addWidget(self.cfo_lbl)
            self.evm_lbl = QLabel("EVM: N/A")
            self.layout.addWidget(self.evm_lbl)
            
            self.bitstream_text = QTextEdit()
            self.bitstream_text.setReadOnly(True)
            self.bitstream_text.setMaximumHeight(80)
            self.layout.addWidget(QLabel("Bitstream (Hex):"))
            self.layout.addWidget(self.bitstream_text)
            
            self.llr_plot = pg.PlotWidget(title="LLR Histogram")
            self.llr_plot.setMaximumHeight(150)
            self.layout.addWidget(self.llr_plot)

            self.layout.addStretch()
            
            self.current_hypotheses = []
            
        def update_metadata(self, recording: SignalRecording):
            self.labels["Source Format"].setText(f"Source Format: {recording.source_format.value}")
            self.labels["Sample Count"].setText(f"Sample Count: {len(recording.samples)}")
            self.labels["DType"].setText(f"DType: {recording.original_dtype}")
            self.labels["Semantic Type"].setText(f"Semantic Type: {recording.semantic_type}")
            
            sr = recording.sample_rate_hz
            self.labels["Sample Rate"].setText(f"Sample Rate: {sr.value if sr.value else 'N/A'} [{sr.status.value}]")
            
            cf = recording.center_frequency_hz
            self.labels["Center Frequency"].setText(f"Center Frequency: {cf.value if cf.value else 'N/A'} [{cf.status.value}]")
            
            # Clear old diagnostics
            while self.layout.count() > 11:
                item = self.layout.takeAt(11)
                if item.widget():
                    item.widget().deleteLater()
                    
            for diag in recording.diagnostics:
                self.add_diag(diag)
                
            # Phase 2: Hypothesis and Feature extraction
            if recording.samples.ndim > 1:
                import dataclasses
                rec_1d = dataclasses.replace(recording, samples=recording.samples[:, 0])
            else:
                rec_1d = recording

            fv = extract_all_features(rec_1d)
            c_scores = compute_classical_scores(fv)
            
            # Fake SNR estimate based on dynamic range / validity?
            snr_est = 20.0 # Placeholder for GUI MVP
            
            hypotheses, selected, is_ambig, is_unk = evaluate_and_rank_hypotheses(fv, c_scores, snr_est, {}, rec_1d)
            self.current_hypotheses = hypotheses
            
            cons_score, cyc_diag = check_temporal_consistency(rec_1d, {})
            
            # Phase 3 Synchronization
            self.sync_results = attempt_synchronization_multi_hypothesis(rec_1d, hypotheses, {})
            self.display_sync_result(self.sync_results[0] if self.sync_results else None)

            if cyc_diag:
                self.add_diag(cyc_diag)
                
            # Update Table
            self.hyp_table.setRowCount(0)
            for h in hypotheses:
                row = self.hyp_table.rowCount()
                self.hyp_table.insertRow(row)
                
                label_item = QTableWidgetItem(h.label)
                if selected and h.label == selected.label:
                    label_item.setBackground(Qt.GlobalColor.green)
                elif is_ambig and h.status.value == "AMBIGUOUS":
                    label_item.setBackground(Qt.GlobalColor.yellow)
                    
                self.hyp_table.setItem(row, 0, label_item)
                self.hyp_table.setItem(row, 1, QTableWidgetItem(f"{h.score:.2f}"))
                self.hyp_table.setItem(row, 2, QTableWidgetItem(h.quality_tier))
                
            # Update Symbol Rate
            if hypotheses and hypotheses[0].candidate_parameters.symbol_rate:
                cp = hypotheses[0].candidate_parameters
                self.symbol_rate_lbl.setText(f"Symbol Rate: {cp.symbol_rate:.4e} {cp.symbol_rate_unit}")
            else:
                self.symbol_rate_lbl.setText("Symbol Rate: N/A")
                
        def on_hyp_selected(self):
            sel = self.hyp_table.selectedItems()
            if not sel: return
            row = sel[0].row()
            h = self.current_hypotheses[row]
            
            ev_text = f"Status: {h.status.value}\n\nEvidence:\n"
            for k, v in h.evidence.items():
                ev_text += f"- {k}: {v:.2f}\n"
            
            if h.contradictions:
                ev_text += "\nContradictions:\n"
                for c in h.contradictions:
                    ev_text += f"- {c}\n"
                    
            self.evidence_lbl.setText(ev_text)
            
        def add_diag(self, diag):
            lbl = QLabel(f"{diag.severity.value}: {diag.message}")
            if diag.severity.value == "ERROR":
                lbl.setStyleSheet("color: red;")
            elif diag.severity.value == "WARNING":
                lbl.setStyleSheet("color: orange;")
            self.layout.insertWidget(self.layout.count() - 1, lbl)

    
        def display_sync_result(self, res):
            if not res:
                self.sync_status_lbl.setText("Sync Status: No Attempts")
                return
                
            sync = res.sync_result
            status = "LOCKED" if res.hypothesis_confirmed else "FAILED"
            self.sync_status_lbl.setText(f"Sync Status: {status} (Clk: {sync.symbol_clock_locked}, Carrier: {sync.carrier_locked})\nLQ: {sync.lock_quality_metric:.4f}")
            self.cfo_lbl.setText(f"CFO: {sync.cfo_estimate:.2e} {sync.cfo_unit}")
            self.evm_lbl.setText(f"EVM: {sync.evm_percent:.2f}%")
            
            if len(res.hard_bits) > 0:
                # convert to hex
                hex_str = np.packbits(res.hard_bits, bitorder='big').tobytes().hex()
                self.bitstream_text.setText(hex_str)
                
                # plot LLR histogram
                self.llr_plot.clear()
                y, x = np.histogram(res.soft_llrs, bins=50)
                bg = pg.BarGraphItem(x0=x[:-1], x1=x[1:], height=y, brush='b')
                self.llr_plot.addItem(bg)
                
            # We should also tell the main window to update the constellation to synced symbols
            if hasattr(self, "parent_window") and self.parent_window:
                self.parent_window.update_synced_constellation(res)

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Signal Analysis MVP - Phase 3")
            self.resize(1200, 800)
            
            self.central = QWidget()
            self.setCentralWidget(self.central)
            
            self.layout = QHBoxLayout(self.central)
            
            # Left panel with plots
            self.plot_splitter = QSplitter(Qt.Orientation.Vertical)
            self.layout.addWidget(self.plot_splitter, 3)
            
            self.waveform_plot = pg.PlotWidget(title="Waveform (I/Q)")
            self.waveform_plot.addLegend()
            self.plot_splitter.addWidget(self.waveform_plot)
            
            self.psd_plot = pg.PlotWidget(title="PSD")
            self.plot_splitter.addWidget(self.psd_plot)
            
            # Bottom plots split horizontally
            self.bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
            self.plot_splitter.addWidget(self.bottom_splitter)
            
            self.waterfall_plot = pg.PlotWidget(title="Waterfall")
            self.waterfall_img = pg.ImageItem()
            self.waterfall_plot.addItem(self.waterfall_img)
            self.bottom_splitter.addWidget(self.waterfall_plot)
            
            self.constellation_plot = pg.PlotWidget(title="Constellation")
            self.constellation_scatter = pg.ScatterPlotItem(size=3, pen=pg.mkPen(None), brush=pg.mkBrush(0, 150, 255, 120))
            self.constellation_plot.addItem(self.constellation_scatter)
            self.constellation_plot.setAspectLocked(True)
            self.constellation_plot.addLine(x=0, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine))
            self.constellation_plot.addLine(y=0, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine))
            self.bottom_splitter.addWidget(self.constellation_plot)
            
            # Right panel
            self.sidebar_layout = QVBoxLayout()
            self.layout.addLayout(self.sidebar_layout, 1)
            
            self.open_btn = QPushButton("Open File...")
            self.open_btn.clicked.connect(self.open_file)
            self.sidebar_layout.addWidget(self.open_btn)
            
            self.sidebar = MetadataSidebar()
            self.sidebar.parent_window = self
            self.sidebar_layout.addWidget(self.sidebar)
            
        def open_file(self):
            path, _ = QFileDialog.getOpenFileName(self, "Open Signal File", "", "All Files (*);;WAV (*.wav);;SigMF (*.sigmf-meta)")
            if not path:
                return
                
            try:
                if path.endswith(".wav"):
                    reader = WavReader(path, mode="unresolved")
                    recording = reader.read()
                elif path.endswith(".sigmf-meta"):
                    recording = read_sigmf(path)
                else:
                    dialog = RawIQDialog(self)
                    if dialog.exec():
                        config = dialog.get_config()
                        reader = RawIQReader(path, config)
                        recording = reader.read()
                    else:
                        return
                        
                self.update_plots(recording)
                self.sidebar.update_metadata(recording)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error loading file: {e}")
                
        
        def update_synced_constellation(self, res):
            self.constellation_plot.clear()
            
            # redraw axes
            self.constellation_plot.addLine(x=0, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine))
            self.constellation_plot.addLine(y=0, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine))
            
            if res and len(res.symbol_decisions) > 0:
                c_data = res.symbol_decisions
                
                # Decision boundaries based on label
                label = res.source_hypothesis_label
                if label == "BPSK":
                    self.constellation_plot.addLine(x=0, pen=pg.mkPen('y'))
                elif label == "QPSK":
                    self.constellation_plot.addLine(x=0, pen=pg.mkPen('y'))
                    self.constellation_plot.addLine(y=0, pen=pg.mkPen('y'))
                
                scatter = pg.ScatterPlotItem(size=3, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 150))
                scatter.setData(x=c_data.real, y=c_data.imag)
                self.constellation_plot.addItem(scatter)
            else:
                self.constellation_plot.addItem(self.constellation_scatter) # Re-add original scatter?
                # or just leave empty / fallback

        def update_plots(self, recording: SignalRecording):
            self.waveform_plot.clear()
            
            max_points = 5000
            n_samples = len(recording.samples)
            if n_samples > max_points:
                step = n_samples // max_points
                indices = np.arange(0, n_samples, step)
                plot_data = recording.samples[indices]
            else:
                indices = np.arange(n_samples)
                plot_data = recording.samples
                
            if plot_data.ndim > 1:
                self.waveform_plot.plot(indices, plot_data[:, 0].real, pen='b', name='Ch0')
                self.waveform_plot.plot(indices, plot_data[:, 1].real, pen='r', name='Ch1')
            else:
                self.waveform_plot.plot(indices, plot_data.real, pen='b', name='I / Real')
                if recording.semantic_type == "complex_iq":
                    self.waveform_plot.plot(indices, plot_data.imag, pen='r', name='Q / Imag')
                    
            psd_result = compute_psd(recording)
            self.psd_plot.clear()
            self.psd_plot.plot(psd_result.frequencies, 10 * np.log10(psd_result.psd + 1e-12), pen='g')
            self.psd_plot.setLabel('bottom', "Frequency", units=psd_result.freq_unit)
            self.psd_plot.setLabel('left', "Magnitude", units="dB")
            
            spec_result = compute_spectrogram(recording)
            self.waterfall_img.setImage(10 * np.log10(spec_result.Sxx.T + 1e-12), autoLevels=True)
            
            if len(spec_result.times) > 0 and len(spec_result.frequencies) > 0:
                t0 = spec_result.times[0]
                t_range = spec_result.times[-1] - spec_result.times[0]
                if t_range == 0: t_range = 1.0
                f0 = spec_result.frequencies[0]
                f_range = spec_result.frequencies[-1] - spec_result.frequencies[0]
                if f_range == 0: f_range = 1.0
                self.waterfall_img.setRect(t0, f0, t_range, f_range)
                
            self.waterfall_plot.setLabel('bottom', "Time", units="s")
            self.waterfall_plot.setLabel('left', "Frequency", units=spec_result.freq_unit)
            
            # Constellation sub-sampled to ~2000 points
            max_const = 2000
            if n_samples > max_const:
                c_step = max(1, n_samples // max_const)
                c_data = recording.samples[::c_step]
            else:
                c_data = recording.samples
                
            if c_data.ndim > 1:
                self.constellation_scatter.setData(x=c_data[:, 0].real, y=c_data[:, 1].real)
            else:
                self.constellation_scatter.setData(x=c_data.real, y=c_data.imag)

def run_app():
    if not HAS_QT:
        print("PySide6/PyQt6 or pyqtgraph not found. Cannot run GUI.")
        sys.exit(1)
        
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
