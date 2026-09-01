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

def _format_hz(hz: float) -> str:
    if hz is None: return "Unknown"
    if hz >= 1e6: return f"{hz/1e6:.3f} MHz"
    elif hz >= 1e3: return f"{hz/1e3:.3f} kHz"
    else: return f"{hz:.1f} Hz"

def _get_status_color(status: PipelineStageStatus) -> str:
    if status == PipelineStageStatus.NOT_ATTEMPTED: return "gray"
    if status == PipelineStageStatus.FAILED: return "red"
    if status == PipelineStageStatus.COMPLETED: return "green"
    return "white"

def format_stage_status(status: PipelineStageStatus, previous_status: PipelineStageStatus = None) -> str:
    color = _get_status_color(status)
    text = f"<span style='color: {color}; font-weight: bold;'>{status.value}</span>"
    if status == PipelineStageStatus.NOT_ATTEMPTED:
        if previous_status == PipelineStageStatus.FAILED:
            text += f"<br><span style='color: gray'>N/A — pipeline stopped upstream</span>"
        else:
            text += f"<br><span style='color: gray'>N/A</span>"
    return text

if HAS_QT:
    class CollapsibleSection(QWidget):
        def __init__(self, title, parent=None):
            super().__init__(parent)
            self.layout = QVBoxLayout(self)
            self.layout.setContentsMargins(0, 0, 0, 0)
            self.layout.setSpacing(0)
            
            self.toggle_btn = QPushButton(title)
            self.toggle_btn.setCheckable(True)
            self.toggle_btn.setChecked(True)
            self.toggle_btn.setStyleSheet("text-align: left; font-weight: bold; padding: 5px; background-color: #333; border: 1px solid #555;")
            self.toggle_btn.toggled.connect(self._on_toggle)
            
            self.content_area = QWidget()
            self.content_layout = QVBoxLayout(self.content_area)
            self.content_layout.setContentsMargins(10, 5, 0, 10)
            
            self.layout.addWidget(self.toggle_btn)
            self.layout.addWidget(self.content_area)
            
        def _on_toggle(self, checked):
            self.content_area.setVisible(checked)
            if checked:
                self.toggle_btn.setText(self.toggle_btn.text().replace("► ", "▼ "))
            else:
                self.toggle_btn.setText(self.toggle_btn.text().replace("▼ ", "► "))
            
        def addWidget(self, widget):
            self.content_layout.addWidget(widget)
            
        def setExpanded(self, expanded):
            self.toggle_btn.setChecked(expanded)
            self._on_toggle(expanded)

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
            
            self.meta_section = CollapsibleSection("▼ Metadata & Physical Layer")
            self.layout.addWidget(self.meta_section)
            self.meta_text = QLabel("No file loaded.")
            self.meta_text.setWordWrap(True)
            self.meta_section.addWidget(self.meta_text)
            
            self.hyp_section = CollapsibleSection("▼ Hypothesis Ranking (Phase 2)")
            self.layout.addWidget(self.hyp_section)
            self.hyp_table = QTableWidget(0, 3)
            self.hyp_table.setHorizontalHeaderLabels(["Label", "Score", "Tier"])
            self.hyp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.hyp_section.addWidget(self.hyp_table)
            
            self.sync_section = CollapsibleSection("▼ Demodulation & Sync (Phase 3)")
            self.layout.addWidget(self.sync_section)
            self.sync_text = QLabel("N/A")
            self.sync_text.setWordWrap(True)
            self.sync_section.addWidget(self.sync_text)
            
            self.fec_section = CollapsibleSection("▼ FEC & Interleaver (Phase 4)")
            self.layout.addWidget(self.fec_section)
            self.fec_text = QLabel("N/A")
            self.fec_text.setWordWrap(True)
            self.fec_section.addWidget(self.fec_text)
            
            self.framing_section = CollapsibleSection("▼ Framing & Payloads (Phase 5)")
            self.layout.addWidget(self.framing_section)
            self.framing_text = QLabel("N/A")
            self.framing_text.setWordWrap(True)
            self.framing_section.addWidget(self.framing_text)
            
            self.bitstream_section = CollapsibleSection("▼ Payload Bitstream (Hex)")
            self.layout.addWidget(self.bitstream_section)
            self.final_bitstream_text = QTextEdit()
            self.final_bitstream_text.setReadOnly(True)
            self.final_bitstream_text.setMaximumHeight(100)
            self.bitstream_section.addWidget(self.final_bitstream_text)

            self.layout.addStretch()

        def update_metadata(self, recording: SignalRecording, pipe_res: PipelineResult = None):
            if pipe_res is None:
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
            details += f"<b>Sample Rate:</b> {_format_hz(sr.value)} [{sr.status.value}]<br>"
            
            cf = recording.center_frequency_hz
            details += f"<b>Center Freq:</b> {_format_hz(cf.value)} [{cf.status.value}]<br>"
            
            if recording.diagnostics:
                details += "<b>Diagnostics:</b><br>"
                for d in recording.diagnostics:
                    color = "red" if d.severity.name == "ERROR" else "orange" if d.severity.name == "WARNING" else "gray"
                    details += f" - <span style='color: {color}'>[{d.severity.value}]</span> {d.code}: {d.message}<br>"
            self.meta_text.setText(details)
            
            # Hypotheses
            self.hyp_table.setRowCount(0)
            if has_non_complex:
                self.hyp_table.setStyleSheet("QTableWidget { color: #888888; }")
            else:
                self.hyp_table.setStyleSheet("")
                
            for h in pipe_res.all_hypotheses:
                r = self.hyp_table.rowCount()
                self.hyp_table.insertRow(r)
                self.hyp_table.setItem(r, 0, QTableWidgetItem(f"{h.label} [{h.status.value}]"))
                self.hyp_table.setItem(r, 1, QTableWidgetItem(f"{h.score:.2f}"))
                self.hyp_table.setItem(r, 2, QTableWidgetItem(str(h.quality_tier)))
                
            self.hyp_section.setExpanded(pipe_res.hypothesis_status != PipelineStageStatus.NOT_ATTEMPTED)
                
            # Sync / Demod
            if pipe_res.sync_status == PipelineStageStatus.COMPLETED and pipe_res.demod_result:
                dm = pipe_res.demod_result
                sync = dm.sync_result
                self.sync_text.setText(
                    format_stage_status(pipe_res.sync_status, pipe_res.hypothesis_status) + "<br>" +
                    f"<b>Locked:</b> {dm.hypothesis_confirmed}<br>"
                    f"<b>CFO:</b> {sync.cfo_estimate:.1f} {sync.cfo_unit}<br>"
                    f"<b>Lock Quality:</b> {sync.lock_quality_metric:.2f}<br>"
                    f"<b>EVM:</b> {sync.evm_percent:.1f}%<br>"
                    f"<b>Bit Count:</b> {len(dm.hard_bits)}"
                )
                if self.parent_window:
                    self.parent_window.update_synced_constellation(dm, has_non_complex)
            else:
                self.sync_text.setText(format_stage_status(pipe_res.sync_status, pipe_res.hypothesis_status))
                if self.parent_window:
                    self.parent_window.update_synced_constellation(None, has_non_complex)
            
            self.sync_section.setExpanded(pipe_res.sync_status != PipelineStageStatus.NOT_ATTEMPTED)
                    
            # FEC / Deinterleaving
            if pipe_res.fec_status == PipelineStageStatus.COMPLETED:
                deint = pipe_res.deint_result
                fec = pipe_res.fec_result
                self.fec_text.setText(
                    format_stage_status(pipe_res.fec_status, pipe_res.sync_status) + "<br>" +
                    f"<b>Interleaver:</b> {deint.hypothesis.family.name}<br>"
                    f"<b>FEC Scheme:</b> {fec.codec_name}<br>"
                    f"<b>FEC Corrected Bits:</b> {fec.corrected_bit_count} ({fec.corrected_bit_fraction*100:.1f}%)<br>"
                    f"<b>FEC Success:</b> {fec.decode_success}"
                )
            else:
                self.fec_text.setText(format_stage_status(pipe_res.fec_status, pipe_res.sync_status))

            self.fec_section.setExpanded(pipe_res.fec_status != PipelineStageStatus.NOT_ATTEMPTED)

            # Framing
            if pipe_res.framing_status == PipelineStageStatus.COMPLETED and pipe_res.frame_structure:
                fs = pipe_res.frame_structure
                sync_name = fs.header_match.pattern.name if fs.header_match.pattern else "Unknown"
                crc_name = fs.crc_candidate.polynomial_name if fs.crc_candidate else "None"
                flen = fs.header_length_bits + (fs.payload_length_bits or 0)
                self.framing_text.setText(
                    format_stage_status(pipe_res.framing_status, pipe_res.fec_status) + "<br>" +
                    f"<b>Sync Word:</b> {sync_name}<br>"
                    f"<b>Frame Length:</b> {flen} bits<br>"
                    f"<b>Payload Length:</b> {fs.payload_length_bits} bits<br>"
                    f"<b>CRC Type:</b> {crc_name}<br>"
                    f"<b>Valid Frames:</b> N/A"
                )
            else:
                self.framing_text.setText(format_stage_status(pipe_res.framing_status, pipe_res.fec_status))
                
            self.framing_section.setExpanded(pipe_res.framing_status != PipelineStageStatus.NOT_ATTEMPTED)
                
            # Bitstream
            final_bits = None
            if pipe_res.frame_structure and pipe_res.fec_result is not None and len(pipe_res.fec_result.decoded_bits) > 0:
                start = pipe_res.frame_structure.payload_start_bit
                end = start + (pipe_res.frame_structure.payload_length_bits or len(pipe_res.fec_result.decoded_bits))
                final_bits = pipe_res.fec_result.decoded_bits[start:end]
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
                self.bitstream_section.setExpanded(True)
            else:
                self.final_bitstream_text.setText("N/A — pipeline stopped upstream" if pipe_res.framing_status in [PipelineStageStatus.FAILED, PipelineStageStatus.NOT_ATTEMPTED] else "N/A — no payload bits")
                self.bitstream_section.setExpanded(False)

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Signal Analysis MVP")
            self.resize(1200, 800)
            
            self._last_wav_mode_idx = 0
            self._last_raw_config = None
            
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
            self.constellation_text = pg.TextItem("", color=(255, 100, 100), anchor=(0.5, 0))
            self.constellation_plot.addItem(self.constellation_text)
            self.constellation_text.setPos(0, 0)
            self.constellation_text.hide()
            self.bottom_splitter.addWidget(self.constellation_plot)
            
            # Right panel
            self.sidebar_layout = QVBoxLayout()
            self.layout.addLayout(self.sidebar_layout, 1)
            
            self.open_btn = QPushButton("Open File...")
            self.open_btn.clicked.connect(self.open_file)
            self.sidebar_layout.addWidget(self.open_btn)
            
            self.demo_btn = QPushButton("Demo Mode...")
            self.demo_btn.clicked.connect(self.show_demo_mode)
            self.sidebar_layout.addWidget(self.demo_btn)
            
            self.sidebar = MetadataSidebar()
            self.sidebar.parent_window = self
            self.sidebar_layout.addWidget(self.sidebar)
            
        def show_demo_mode(self):
            class DemoDialog(QDialog):
                def __init__(self, parent=None):
                    super().__init__(parent)
                    self.setWindowTitle("Demo Mode")
                    self.resize(500, 300)
                    self.layout = QVBoxLayout(self)
                    self.list = QComboBox()
                    self.list.addItem("Clean QPSK (High SNR)", "demo_clean_qpsk.wav")
                    self.list.addItem("Concatenated FEC (RS + BPSK)", "demo_concatenated.wav")
                    self.list.addItem("Low SNR QPSK", "demo_low_snr_qpsk.wav")
                    self.list.addItem("OFDM (Out of Scope)", "demo_ofdm_out_of_scope.wav")
                    self.list.addItem("Real Valued (Audio)", "demo_real_valued_gate.wav")
                    self.layout.addWidget(QLabel("Select Demo Fixture:"))
                    self.layout.addWidget(self.list)
                    self.desc = QTextEdit()
                    self.desc.setReadOnly(True)
                    self.layout.addWidget(QLabel("Narration:"))
                    self.layout.addWidget(self.desc)
                    self.run_btn = QPushButton("Run Selected")
                    self.run_btn.clicked.connect(self.accept)
                    self.layout.addWidget(self.run_btn)
                    self.list.currentIndexChanged.connect(self.update_desc)
                    self.update_desc()
                def update_desc(self):
                    idx = self.list.currentIndex()
                    if idx == 0:
                        self.desc.setText("Clean QPSK, high SNR, unencoded. Demonstrates Phase 2 confident classification + Phase 3 clean sync/demod. You'll see high hypothesis scores and lock quality.")
                    elif idx == 1:
                        self.desc.setText("Concatenated simulation. Shows framing and CRC detection on a BPSK signal. Demonstrates Phase 4/5 recovering HDLC frames successfully out of the payload.")
                    elif idx == 2:
                        self.desc.setText("Low SNR QPSK. Demonstrates how the pipeline degrades gracefully under noise, reflecting lower confidence in classification and lock metrics.")
                    elif idx == 3:
                        self.desc.setText("OFDM signal. Demonstrates correct rejection at Phase 2 (hypothesis status UNKNOWN) due to out-of-scope bimodal frequency distribution.")
                    elif idx == 4:
                        self.desc.setText("Real-valued audio. Demonstrates rejection at Phase 2 because the MVP is explicitly for complex I/Q baseband.")
            
            dlg = DemoDialog(self)
            if dlg.exec():
                import os
                fixture_file = dlg.list.currentData()
                path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures", "demo", fixture_file)
                if os.path.exists(path):
                    self.open_file(override_path=path, force_stereo_iq=True)
                else:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "Missing Fixture", f"Fixture not found at {path}")

        def _guess_stereo_mode_heuristic(self, path: str) -> str:
            import wave
            import numpy as np
            try:
                with wave.open(path, 'rb') as wf:
                    n_frames = min(wf.getnframes(), 4096)
                    if n_frames == 0: return "unable to analyze"
                    raw = wf.readframes(n_frames)
                    sw = wf.getsampwidth()
                    if sw not in [1, 2, 4]: return "unable to analyze"
                    dt = np.uint8 if sw == 1 else np.int16 if sw == 2 else np.float32
                    data = np.frombuffer(raw, dtype=dt).reshape(-1, 2)
                    ch0 = data[:, 0].astype(np.float32)
                    ch1 = data[:, 1].astype(np.float32)
                    p0 = np.mean(ch0**2)
                    p1 = np.mean(ch1**2)
                    if p0 < 1e-6 and p1 < 1e-6:
                        return "unable to analyze"
                    
                    ratio = p0 / p1 if p1 > 1e-9 else 0
                    if ratio > 1: ratio = 1 / ratio
                    
                    ch0_c = ch0 - np.mean(ch0)
                    ch1_c = ch1 - np.mean(ch1)
                    var0 = np.var(ch0_c)
                    var1 = np.var(ch1_c)
                    
                    frames = wf.readframes(1024)
                    if wf.getsampwidth() == 2:
                        samples = np.frombuffer(frames, dtype=np.int16).reshape(-1, 2)
                        # If channels are nearly identical, probably real audio
                        if np.allclose(samples[:, 0], samples[:, 1], atol=10):
                            return "stereo_real"
                        return "stereo_iq"
            except Exception:
                return "unable to analyze"

        def open_file(self, override_path: str = None, force_stereo_iq: bool = False):
            if override_path:
                path = override_path
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, "Open Signal File", "", "All Files (*);;WAV (*.wav);;SigMF (*.sigmf-meta)",
                    options=QFileDialog.DontUseNativeDialog
                )
            if not path:
                return
                
            try:
                if path.endswith(".wav"):
                    import wave
                    with wave.open(path, 'rb') as wf:
                        channels = wf.getnchannels()
                    
                    mode = "unresolved"
                    if force_stereo_iq:
                        mode = "stereo_iq"
                    elif channels == 2:
                        heuristic = self._guess_stereo_mode_heuristic(path)
                        hint = f"Heuristic: channels appear {heuristic} (not a determination — verify against file provenance)."
                        items = [
                            "Two independent real channels (stereo_real)",
                            "Complex I/Q pair (Ch0=I, Ch1=Q) (stereo_iq)",
                            "Auto-detect is unavailable - I'm not sure"
                        ]
                        dialog = QInputDialog(self)
                        dialog.setWindowTitle("Stereo WAV Detected")
                        dialog.setLabelText(f"Select semantic type for 2-channel WAV:\n\n{hint}")
                        dialog.setComboBoxItems(items)
                        dialog.setOption(QInputDialog.UseListViewForComboBoxItems, False)
                        dialog.setComboBoxEditable(False)
                        dialog.setWindowModality(Qt.ApplicationModal)
                        
                        QApplication.processEvents()
                        
                        if dialog.exec() == QDialog.Accepted:
                            item = dialog.textValue()
                            if item in items:
                                self._last_wav_mode_idx = items.index(item)
                            if "stereo_real" in item: mode = "stereo_real"
                            elif "stereo_iq" in item: mode = "stereo_iq"
                            
                    reader = WavReader(path, mode=mode)
                    recording = reader.read()
                elif path.endswith(".sigmf-meta"):
                    recording = read_sigmf(path)
                else:
                    dialog = RawIQDialog(self)
                    if self._last_raw_config:
                        # Pre-fill (simple version)
                        dialog.sr_edit.setText(str(self._last_raw_config.sample_rate_hz or 1000000))
                        dialog.cf_edit.setText(str(self._last_raw_config.center_frequency_hz or 0))
                    
                    if dialog.exec():
                        config = dialog.get_config()
                        self._last_raw_config = config
                        reader = RawIQReader(path, config)
                        recording = reader.read()
                    else:
                        return
                        
                self.update_plots(recording)
                self.sidebar.update_metadata(recording)
            except (OSError, ValueError) as e:
                # File IO or parsing errors
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "File Error", f"Failed to open file:\n{e}")
            except Exception as e:
                # Unexpected crashes should surface visibly as a Diagnostic
                import traceback
                tb_str = traceback.format_exc()
                print(tb_str)
                from PySide6.QtWidgets import QMessageBox
                from .models import Diagnostic, Severity, PipelineResult, PipelineStageStatus
                
                QMessageBox.critical(self, "Pipeline Crash", f"An unexpected pipeline error occurred:\n{type(e).__name__}: {e}")
                
                # If we have a recording, forcibly render the error state in the sidebar
                if 'recording' in locals():
                    err_diag = Diagnostic(Severity.ERROR, "UNHANDLED_EXCEPTION", str(e), tb_str)
                    crash_res = PipelineResult(
                        recording=recording,
                        hypothesis_status=PipelineStageStatus.FAILED,
                        top_hypothesis=None,
                        all_hypotheses=[],
                        sync_status=PipelineStageStatus.FAILED,
                        demod_result=None,
                        fec_status=PipelineStageStatus.FAILED,
                        deint_result=None,
                        fec_result=None,
                        framing_status=PipelineStageStatus.FAILED,
                        frame_structure=None,
                        diagnostics=[err_diag]
                    )
                    # Pass the pre-rendered result
                    self.sidebar.update_metadata(recording, crash_res)
                
        def update_synced_constellation(self, res, has_non_complex=False):
            self.constellation_plot.clear()
            self.constellation_plot.addItem(self.constellation_text)
            
            # redraw axes
            self.constellation_plot.addLine(x=0, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine))
            self.constellation_plot.addLine(y=0, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine))
            
            if has_non_complex:
                self.constellation_text.setText("Real-valued input — constellation not physically meaningful")
                self.constellation_text.show()
            else:
                self.constellation_text.hide()
            
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
