import re

with open('signal_analysis/gui.py', 'r') as f:
    content = f.read()

# Change Title to Phase 3
content = content.replace('self.setWindowTitle("Signal Analysis MVP - Phase 2")', 'self.setWindowTitle("Signal Analysis MVP - Phase 3")')

# Add UI elements to MetadataSidebar
sidebar_init_search = r'(self\.diagnostics_label = QLabel\("--- Diagnostics ---"\)\n\s+self\.layout\.addWidget\(self\.diagnostics_label\))'

sidebar_additions = """\g<1>
            
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
"""

content = re.sub(sidebar_init_search, sidebar_additions, content)

# Modify update_metadata to call Phase 3
update_meta_search = r'(self\.current_hypotheses = hypotheses\n\s+cons_score, cyc_diag = check_temporal_consistency\(rec_1d, \{\}\))'

update_meta_additions = """\g<1>
            
            # Phase 3 Synchronization
            self.sync_results = attempt_synchronization_multi_hypothesis(rec_1d, hypotheses, {})
            self.display_sync_result(self.sync_results[0] if self.sync_results else None)
"""
content = re.sub(update_meta_search, update_meta_additions, content)

# Add display_sync_result method to MetadataSidebar
display_sync_method = """
        def display_sync_result(self, res):
            if not res:
                self.sync_status_lbl.setText("Sync Status: No Attempts")
                return
                
            sync = res.sync_result
            status = "LOCKED" if res.hypothesis_confirmed else "FAILED"
            self.sync_status_lbl.setText(f"Sync Status: {status} (Clk: {sync.symbol_clock_locked}, Carrier: {sync.carrier_locked})\\nLQ: {sync.lock_quality_metric:.4f}")
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
"""

# Find the end of MetadataSidebar class (before MainWindow) and insert display_sync_result
content = content.replace("class MainWindow(QMainWindow):", display_sync_method + "\n    class MainWindow(QMainWindow):")

# Link sidebar to parent window so it can call update_synced_constellation
content = content.replace("self.sidebar = MetadataSidebar()", "self.sidebar = MetadataSidebar()\n            self.sidebar.parent_window = self")

# Add update_synced_constellation to MainWindow
update_sync_const = """
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
"""
content = content.replace("def update_plots(self, recording: SignalRecording):", update_sync_const + "\n        def update_plots(self, recording: SignalRecording):")

with open('signal_analysis/gui.py', 'w') as f:
    f.write(content)
print("Patched gui.py")
