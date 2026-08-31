new_sidebar = '''    class MetadataSidebar(QScrollArea):
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
                details += f"Top Hyp: {h.label} [{h.status.value}] (Score {h.score:.2f})\\n"
                
            if pipe_res.demod_result:
                s = pipe_res.demod_result.sync_result
                details += f"Sync EVM: {s.evm_percent:.1f}% \\n" + \\
                           f"CFO: {s.cfo_estimate:.2f} {s.cfo_unit}\\n"
                
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
                details += f"FEC: {r.codec_name} -> {r.decode_success}\\n" + \\
                           f"Corrected: {r.corrected_bit_count} bits\\n"
                if r.diagnostics:
                    details += f"FEC Diag: {r.diagnostics[0].message}\\n"
                    
            if pipe_res.frame_structure:
                fs = pipe_res.frame_structure
                details += f"Framing Status: [{fs.status.value}]\\n"
                details += f"Sync Word: {fs.header_match.pattern.name} at {fs.header_match.bit_offset}\\n"
                details += f"Periodicity: {fs.header_match.periodicity_consistent}\\n"
                if fs.crc_candidate:
                    details += f"CRC: {fs.crc_candidate.polynomial_name} verified={fs.crc_candidate.verified}\\n"
                    
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
'''

import re
with open('signal_analysis/gui.py', 'r') as f:
    content = f.read()

start_idx = content.find('    class MetadataSidebar(QScrollArea):')
end_idx = content.find('    class MainWindow(QMainWindow):')

if start_idx != -1 and end_idx != -1:
    new_content = content[:start_idx] + new_sidebar + '\n' + content[end_idx:]
    with open('signal_analysis/gui.py', 'w') as f:
        f.write(new_content)
