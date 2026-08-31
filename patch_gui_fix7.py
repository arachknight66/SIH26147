import re
with open('signal_analysis/gui.py', 'r') as f:
    content = f.read()

content = content.replace('QFileDialog,', 'QFileDialog, QInputDialog,')

open_file_patch = '''    def open_file(self):
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
                recording = reader.read()'''

content = re.sub(r'    def open_file\(self\):.*?recording = reader\.read\(\)', open_file_patch, content, flags=re.DOTALL)

update_metadata_patch = '''    def update_metadata(self, result):
        # Check for NON_COMPLEX_PIPELINE
        has_non_complex = any(d.code == "NON_COMPLEX_PIPELINE" for d in result.diagnostics)
        
        details = ""
        if has_non_complex:
            details += "<p style='color: red; font-weight: bold;'>[WARNING] Real-valued signal. Phase/Cumulant features unavailable. Hypothesis max quality = LOW.</p>"
            
        details += f"<b>File:</b> {result.recording.source_format.name}<br>"'''

content = content.replace('    def update_metadata(self, result):\n        details = f"<b>File:</b> {result.recording.source_format.name}<br>"', update_metadata_patch)

with open('signal_analysis/gui.py', 'w') as f:
    f.write(content)
