with open('signal_analysis/gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

idx1 = content.find('        def open_file(self):')
idx2 = content.find('        def update_synced_constellation(self, res):')

correct_open_file = '''    def open_file(self):
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
            
    '''

content = content[:idx1] + correct_open_file + content[idx2:]

with open('signal_analysis/gui.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated open_file with 4-space method indentation")
