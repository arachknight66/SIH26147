import sys
import numpy as np
from typing import Optional

HAS_QT = True
try:
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                   QHBoxLayout, QLabel, QPushButton, QFileDialog, QInputDialog,
                                   QDialog, QComboBox, QFormLayout, QLineEdit, QDialogButtonBox,
                                   QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QTextEdit)
    from PySide6.QtCore import Qt
    import pyqtgraph as pg
except ImportError:
    try:
        from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                     QHBoxLayout, QLabel, QPushButton, QFileDialog, QInputDialog,
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
        def __init__(self):
            super().__init__()
            self.setWidgetResizable(True)
            content = QWidget()
            self.layout = QVBoxLayout(content)
            self.setWidget(content)
            
            self.labels = {}
            fields = ["Source Format", "Sample Count", "DType", "Semantic Type", "Sample Rate", "Center Frequency"]
            for f in fields:
                lbl = QLabel(f"{f}: ")
                self.layout.addWidget(lbl)
                self.labels[f] = lbl
                
            self.layout.addWidget(QLabel("--- Pipeline Status ---"))
            self.stage_lbls = {}
            for s in ["Hypothesis", "Sync", "FEC", "Framing"]:
                lbl = QLabel(f"{s}: PENDING")
                self.layout.addWidget(lbl)
                self.stage_lbls[s] = lbl
                
            self.layout.addWidget(QLabel("--- Analysis Details ---"))
            self.details_lbl = QLabel("Details...")
            self.details_lbl.setWordWrap(True)
            self.layout.addWidget(self.details_lbl)
            
            self.llr_plot = pg.PlotWidget(title="LLR Histogram")
            self.llr_plot.setMaximumHeight(120)
            self.layout.addWidget(self.llr_plot)
            
            self.layout.addWidget(QLabel("Recovered Bits (Hex):"))
            self.final_bitstream_text = QTextEdit()
            self.final_bitstream_text.setReadOnly(True)
            self.final_bitstream_text.setMaximumHeight(80)
            self.layout.addWidget(self.final_bitstream_text)
            
        def add_diag(self, diag):
            lbl = QLabel(f"[{diag.severity.value}] {diag.code}: {diag.message}")
            if diag.severity.value == "ERROR":
                lbl.setStyleSheet("color: red")
            elif diag.severity.value == "WARNING":
                lbl.setStyleSheet("color: orange")
            self.layout.insertWidget(len(self.labels), lbl)

        def update_metadata(self, recording):
            # Update basics
            self.labels["Source Format"].setText(f"Source Format: {recording.source_format.value}")
            self.labels["Sample Count"].setText(f"Sample Count: {len(recording.samples)}")
            self.labels["DType"].setText(f"DType: {recording.original_dtype}")
            self.labels["Semantic Type"].setText(f"Semantic Type: {recording.semantic_type}")
            
            sr = recording.sample_rate_hz
            self.labels["Sample Rate"].setText(f"Sample Rate: {sr.value if sr.value else 'N/A'} [{sr.status.value}]")
            cf = recording.center_frequency_hz
            self.labels["Center Frequency"].setText(f"Center Frequency: {cf.value if cf.value else 'N/A'} [{cf.status.value}]")
            
            from .pipeline import run_full_pipeline
            pipe_res = run_full_pipeline(recording)
            
            # Display Pipeline Stage Statuses
            self.stage_lbls["Hypothesis"].setText(f"Hypothesis: {pipe_res.hypothesis_status.value}")
            self.stage_lbls["Sync"].setText(f"Sync: {pipe_res.sync_status.value}")
            self.stage_lbls["FEC"].setText(f"FEC: {pipe_res.fec_status.value}")
            self.stage_lbls["Framing"].setText(f"Framing: {pipe_res.framing_status.value}")
            
            details = ""
            if pipe_res.top_hypothesis:
                h = pipe_res.top_hypothesis
                details += f"Top Hyp: {h.label} [{h.status.value}] (Score {h.score:.2f})\n"
                
            if pipe_res.demod_result:
                s = pipe_res.demod_result.sync_result
                details += f"Sync EVM: {s.evm_percent:.1f}% \n" + \
                           f"CFO: {s.cfo_estimate:.2f} {s.cfo_unit}\n"
                
                # Plot LLRs
                self.llr_plot.clear()
                llrs = pipe_res.demod_result.soft_llrs
                if len(llrs) > 0:
                    y, x = np.histogram(llrs, bins=50)
                    bg = pg.BarGraphItem(x0=x[:-1], x1=x[1:], height=y, brush='b')
                    self.llr_plot.addItem(bg)
                    
                if hasattr(self, 'parent_window') and self.parent_window:
                    self.parent_window.update_synced_constellation(pipe_res.demod_result)
                    
            if pipe_res.fec_result:
                r = pipe_res.fec_result
                details += f"FEC: {r.codec_name} -> {r.decode_success}\n" + \
                           f"Corrected: {r.corrected_bit_count} bits\n"
                if r.diagnostics:
                    details += f"FEC Diag: {r.diagnostics[0].message}\n"
                    
            if pipe_res.frame_structure:
                fs = pipe_res.frame_structure
                details += f"Framing Status: [{fs.status.value}]\n"
                details += f"Sync Word: {fs.header_match.pattern.name} at {fs.header_match.bit_offset}\n"
                details += f"Periodicity: {fs.header_match.periodicity_consistent}\n"
                if fs.crc_candidate:
                    details += f"CRC: {fs.crc_candidate.polynomial_name} verified={fs.crc_candidate.verified}\n"
                    
            self.details_lbl.setText(details)
            
            final_bits = None
            if pipe_res.fec_result:
                final_bits = pipe_res.fec_result.decoded_bits
            elif pipe_res.demod_result:
                final_bits = pipe_res.demod_result.hard_bits
                
            if final_bits is not None and len(final_bits) > 0:
                import numpy as np
                # Pad to 8
                pad = (8 - len(final_bits) % 8) % 8
                if pad > 0:
                    final_bits = np.pad(final_bits, (0, pad))
                hex_str = np.packbits(final_bits, bitorder='big').tobytes().hex()
                self.final_bitstream_text.setText(hex_str)
            else:
                self.final_bitstream_text.setText("N/A")

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Signal Analysis MVP - Phase 4")
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
                import wave
                with wave.open(path, 'rb') as wf:
                    channels = wf.getnchannels()
                
                mode = "unresolved"
                if channels == 2:
                    items = [
                        "Two independent real channels (stereo_real)",
                        "Complex I/Q pair (Ch0=I, Ch1=Q) (stereo_iq)",
                        "Auto-detect is unavailable — I'm not sure"
                    ]
                    item, ok = QInputDialog.getItem(self, "Stereo WAV Detected", "Select semantic type for 2-channel WAV:", items, 0, False)
                    if ok and item:
                        if "stereo_real" in item: mode = "stereo_real"
                        elif "stereo_iq" in item: mode = "stereo_iq"
                        
                reader = WavReader(path, mode=mode)
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
