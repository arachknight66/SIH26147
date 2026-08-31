import re
with open('signal_analysis/gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

gui_patch = '''            from .pipeline import run_full_pipeline
            pipe_res = run_full_pipeline(recording)
            
            # Prominent diagnostic for non-complex IQ
            for diag in pipe_res.diagnostics:
                if diag.code == "NON_COMPLEX_PIPELINE":
                    warn_lbl = QLabel(f"[WARNING] {diag.message}")
                    warn_lbl.setStyleSheet("color: red; font-weight: bold; background-color: #440000; padding: 5px;")
                    self.layout.insertWidget(len(self.labels), warn_lbl)
'''
content = content.replace('            from .pipeline import run_full_pipeline\n            pipe_res = run_full_pipeline(recording)', gui_patch)

with open('signal_analysis/gui.py', 'w', encoding='utf-8') as f:
    f.write(content)
