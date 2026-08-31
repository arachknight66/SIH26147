import re

with open('signal_analysis/gui.py', 'r') as f:
    content = f.read()

# Change Title to Phase 4
content = content.replace('self.setWindowTitle("Signal Analysis MVP - Phase 3")', 'self.setWindowTitle("Signal Analysis MVP - Phase 4")')

sidebar_additions = """\g<1>
            
            self.layout.addWidget(QLabel("--- Phase 4 Deint & FEC ---"))
            self.deint_lbl = QLabel("Deinterleaver: N/A")
            self.layout.addWidget(self.deint_lbl)
            self.fec_lbl = QLabel("FEC: N/A")
            self.layout.addWidget(self.fec_lbl)
            
            self.final_bitstream_text = QTextEdit()
            self.final_bitstream_text.setReadOnly(True)
            self.final_bitstream_text.setMaximumHeight(80)
            self.layout.addWidget(QLabel("Final Recovered (Hex):"))
            self.layout.addWidget(self.final_bitstream_text)
"""

sidebar_init_search = r'(self\.llr_plot = pg\.PlotWidget\(title="LLR Histogram"\)\n\s+self\.llr_plot\.setMaximumHeight\(150\)\n\s+self\.layout\.addWidget\(self\.llr_plot\))'

content = re.sub(sidebar_init_search, sidebar_additions, content)


# Add Phase 4 execution in update_metadata
update_meta_additions = """\g<1>
            
            # Phase 4 Deinterleave & FEC
            from .fec_concatenated import decode_concatenated
            from .models import DeinterleavingResult, DeinterleaverHypothesis, HypothesisStatus, DeinterleaverFamily
            
            # Simple run on best sync result
            best_sync = self.sync_results[0] if self.sync_results else None
            if best_sync and best_sync.hypothesis_confirmed:
                vit_res, rs_res, deint_res = decode_concatenated(best_sync)
                self.display_fec_result(vit_res, rs_res, deint_res)
            else:
                self.deint_lbl.setText("Deinterleaver: None (No Sync)")
                self.fec_lbl.setText("FEC: None (No Sync)")
"""

update_meta_search = r'(self\.display_sync_result\(self\.sync_results\[0\] if self\.sync_results else None\))'
content = re.sub(update_meta_search, update_meta_additions, content)

display_fec_method = """
        def display_fec_result(self, vit_res, rs_res, deint_res):
            dh = deint_res.hypothesis
            self.deint_lbl.setText(f"Deinterleaver: {dh.family.value}\\nParams: {dh.parameters}\\nScore: {dh.score:.2f} (CV: {deint_res.cross_validation_score:.2f})")
            
            # Warnings
            fec_status = "SUCCESS" if rs_res.decode_success else "FAIL"
            fec_str = f"FEC: {rs_res.codec_name} -> {fec_status}\\n"
            fec_str += f"Inner Margin: {vit_res.pre_correction_metric:.2f}\\n"
            fec_str += f"Outer Corrected: {rs_res.corrected_bit_count} bits ({rs_res.corrected_bit_fraction*100:.1f}%)"
            
            for diag in rs_res.diagnostics:
                if diag.severity.value == "WARNING":
                    fec_str += f"\\n[WARNING: {diag.message}]"
            
            self.fec_lbl.setText(fec_str)
            
            hex_str = np.packbits(rs_res.decoded_bits, bitorder='big').tobytes().hex()
            self.final_bitstream_text.setText(hex_str)
"""

content = content.replace("class MainWindow(QMainWindow):", display_fec_method + "\n    class MainWindow(QMainWindow):")

with open('signal_analysis/gui.py', 'w') as f:
    f.write(content)
print("Patched gui.py")
