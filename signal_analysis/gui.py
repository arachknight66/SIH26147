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

from .models import (SignalRecording, SourceFormat, MetadataValue, 
                     MetadataStatus, PipelineResult, PipelineStageStatus, FeatureValidity)
from .loaders import WavReader, RawIQReader, RawIQConfig, read_sigmf
from .measurements import compute_psd, compute_spectrogram
from .pipeline import run_full_pipeline

if HAS_QT:
    class RawIQDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Raw I/Q Import Parameters")
            layout = QFormLayout(self)
            
            self.dtype_combo = QComboBox()
            self.dtype_combo.addItems(["int8", "int16", "float32", "complex64", "uint8"])
            layout.addRow("Data Type:", self.dtype_combo)
            
            self.sr_edit = QLineEdit("1000000")
            layout.addRow("Sample Rate (Hz):", self.sr_edit)
            
            self.cf_edit = QLineEdit("0")
            layout.addRow("Center Freq (Hz):", self.cf_edit)
            
            self.order_combo = QComboBox()
            self.order_combo.addItems(["IQ", "QI"])
            layout.addRow("I/Q Order:", self.order_combo)

            self.endian_combo = QComboBox()
            self.endian_combo.addItems(["little", "big"])
            layout.addRow("Endianness:", self.endian_combo)
            
            self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            self.buttons.accepted.connect(self.accept)
            self.buttons.rejected.connect(self.reject)
            layout.addRow(self.buttons)
            
        def get_config(self) -> RawIQConfig:
            sr = float(self.sr_edit.text()) if self.sr_edit.text() else None
            cf = float(self.cf_edit.text()) if self.cf_edit.text() else None
            return RawIQConfig(
                dtype=self.dtype_combo.currentText(),
                iq_order=self.order_combo.currentText(),
                endian=self.endian_combo.currentText(),
                sample_rate_hz=sr,
                center_frequency_hz=cf
            )

    class MetadataSidebar(QScrollArea):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWidgetResizable(True)
            self.container = QWidget()
            self.layout = QVBoxLayout(self.container)
            self.setWidget(self.container)
            self.parent_window = None
            
            self.meta_label = QLabel("<b>Metadata & Physical Layer</b>")
            self.layout.addWidget(self.meta_label)
            
            self.meta_text = QLabel("No file loaded.")
            self.meta_text.setWordWrap(True)
            self.layout.addWidget(self.meta_text)
            
            line1 = QFrame()
            line1.setFrameShape(QFrame.Shape.HLine)
            self.layout.addWidget(line1)
            
            self.hyp_label = QLabel("<b>Hypothesis Ranking (Phase 2)</b>")
            self.layout.addWidget(self.hyp_label)
            
            self.hyp_table = QTableWidget(0, 3)
            self.hyp_table.setHorizontalHeaderLabels(["Label", "Score", "Tier"])
            self.hyp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.layout.addWidget(self.hyp_table)
            
            line2 = QFrame()
            line2.setFrameShape(QFrame.Shape.HLine)
            self.layout.addWidget(line2)
            
            self.sync_label = QLabel("<b>Demodulation & Sync (Phase 3)</b>")
            self.layout.addWidget(self.sync_label)
            
            self.sync_text = QLabel("N/A")
            self.sync_text.setWordWrap(True)
            self.layout.addWidget(self.sync_text)
            
            line3 = QFrame()
            line3.setFrameShape(QFrame.Shape.HLine)
            self.layout.addWidget(line3)
            
            self.fec_label = QLabel("<b>FEC & Interleaver (Phase 4)</b>")
            self.layout.addWidget(self.fec_label)
            
            self.fec_text = QLabel("N/A")
            self.fec_text.setWordWrap(True)
            self.layout.addWidget(self.fec_text)

            line4 = QFrame()
            line4.setFrameShape(QFrame.Shape.HLine)
            self.layout.addWidget(line4)
            
            self.framing_label = QLabel("<b>Framing & Payloads (Phase 5)</b>")
            self.layout.addWidget(self.framing_label)
            
            self.framing_text = QLabel("N/A")
            self.framing_text.setWordWrap(True)
            self.layout.addWidget(self.framing_text)
            
            self.final_bitstream_label = QLabel("<b>Payload Bitstream (Hex)</b>")
            self.layout.addWidget(self.final_bitstream_label)
            
            self.final_bitstream_text = QTextEdit()
            self.final_bitstream_text.setReadOnly(True)
            self.final_bitstream_text.setMaximumHeight(100)
            self.layout.addWidget(self.final_bitstream_text)

            self.layout.addStretch()

        def update_metadata(self, recording: SignalRecording):
            pipe_res = run_full_pipeline(recording)
            
            # Check for NON_COMPLEX_PIPELINE
            has_non_complex = any(d.code == "NON_COMPLEX_PIPELINE" for d in pipe_res.diagnostics)
            
            # Metadata display
            details = ""
            if has_non_complex:
                details += "<p style='color: red; font-weight: bold;'>[WARNING] Real-valued signal. Phase/Cumulant features unavailable. Hypothesis max quality = LOW.</p>"
                
            details += f"<b>File:</b> {recording.source_format.name}<br>"
            details += f"<b>Type:</b> {recording.semantic_type}<br>"
            details += f"<b>Samples:</b> {len(recording.samples):,}<br>"
            
            sr = recording.sample_rate_hz
            if sr.value is not None:
                details += f"<b>Sample Rate:</b> {sr.value/1e6:.3f} MHz [{sr.status.value}]<br>"
            else:
                details += f"<b>Sample Rate:</b> Unknown [{sr.status.value}]<br>"
            
            cf = recording.center_frequency_hz
            if cf.value is not None:
                details += f"<b>Center Freq:</b> {cf.value/1e6:.3f} MHz [{cf.status.value}]<br>"
            else:
                details += f"<b>Center Freq:</b> Unknown [{cf.status.value}]<br>"
            
            if recording.diagnostics:
                details += "<b>Diagnostics:</b><br>"
                for d in recording.diagnostics:
                    details += f" - [{d.severity.value}] {d.code}: {d.message}<br>"
            self.meta_text.setText(details)
            
            # Hypotheses
            self.hyp_table.setRowCount(0)
            for h in pipe_res.all_hypotheses:
                r = self.hyp_table.rowCount()
                self.hyp_table.insertRow(r)
                self.hyp_table.setItem(r, 0, QTableWidgetItem(f"{h.label} [{h.status.value}]"))
                self.hyp_table.setItem(r, 1, QTableWidgetItem(f"{h.score:.2f}"))
                self.hyp_table.setItem(r, 2, QTableWidgetItem(str(h.quality_tier)))
                
            # Sync / Demod
            if pipe_res.sync_status == PipelineStageStatus.COMPLETED and pipe_res.demod_result:
                dm = pipe_res.demod_result
                sync = dm.sync_result
                self.sync_text.setText(
                    f"<b>Status:</b> COMPLETED ({dm.source_hypothesis_label})<br>"
                    f"<b>Locked:</b> {dm.hypothesis_confirmed}<br>"
                    f"<b>CFO:</b> {sync.cfo_estimate:.1f} {sync.cfo_unit}<br>"
                    f"<b>Lock Quality:</b> {sync.lock_quality_metric:.2f}<br>"
                    f"<b>EVM:</b> {sync.evm_percent:.1f}%<br>"
                    f"<b>Bit Count:</b> {len(dm.hard_bits)}"
                )
                if self.parent_window:
                    self.parent_window.update_synced_constellation(dm)
            else:
                self.sync_text.setText(f"Status: {pipe_res.sync_status.value}")
                if self.parent_window:
                    self.parent_window.update_synced_constellation(None)
                    
            # FEC / Deinterleaving
            if pipe_res.fec_status == PipelineStageStatus.COMPLETED:
                deint = pipe_res.deint_result
                fec = pipe_res.fec_result
                self.fec_text.setText(
                    f"<b>Interleaver:</b> {deint.hypothesis.family.name}<br>"
                    f"<b>FEC Scheme:</b> {fec.scheme_name}<br>"
                    f"<b>FEC Corrected Bits:</b> {fec.corrected_bit_count} ({fec.corrected_bit_fraction*100:.1f}%)<br>"
                    f"<b>FEC Success:</b> {fec.success}"
                )
            else:
                self.fec_text.setText(f"Status: {pipe_res.fec_status.value}")

            # Framing
            if pipe_res.framing_status == PipelineStageStatus.COMPLETED and pipe_res.frame_structure:
                fs = pipe_res.frame_structure
                self.framing_text.setText(
                    f"<b>Status:</b> COMPLETED<br>"
                    f"<b>Sync Word:</b> {fs.sync_word_name}<br>"
                    f"<b>Frame Length:</b> {fs.frame_length_bits} bits<br>"
                    f"<b>Payload Length:</b> {fs.payload_length_bits} bits<br>"
                    f"<b>CRC Type:</b> {fs.crc_type}<br>"
                    f"<b>Valid Frames:</b> {fs.valid_frames_count}/{fs.total_frames_found}"
                )
            else:
                self.framing_text.setText(f"Status: {pipe_res.framing_status.value}")
                
            # Bitstream
            final_bits = None
            if pipe_res.frame_structure and len(pipe_res.frame_structure.payloads) > 0:
                final_bits = np.concatenate(pipe_res.frame_structure.payloads)
            elif pipe_res.fec_result is not None and len(pipe_res.fec_result.decoded_bits) > 0:
                final_bits = pipe_res.fec_result.decoded_bits
            elif pipe_res.demod_result is not None:
                final_bits = pipe_res.demod_result.hard_bits
                
            if final_bits is not None and len(final_bits) > 0:
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
            self.setWindowTitle("Signal Analysis MVP")
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
                            "Auto-detect is unavailable - I'm not sure"
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
                self.constellation_plot.addItem(self.constellation_scatter)

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
