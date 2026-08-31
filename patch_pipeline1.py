import re
with open('signal_analysis/pipeline.py', 'r') as f:
    content = f.read()

patch = '''    # --- Stage 2: Hypothesis ---
    res = PipelineResult(**{**res.__dict__, 'hypothesis_status': PipelineStageStatus.COMPLETED})
    
    if recording.semantic_type != "complex_iq":
        from .models import Diagnostic, Severity
        diag = Diagnostic("NON_COMPLEX_PIPELINE", f"Pipeline ran on {recording.semantic_type}. Phase/cumulant metrics are compromised.", Severity.WARNING)
        res = PipelineResult(**{**res.__dict__, 'diagnostics': res.diagnostics + [diag]})
'''

content = content.replace('    # --- Stage 2: Hypothesis ---\n    res = PipelineResult(**{**res.__dict__, \'hypothesis_status\': PipelineStageStatus.COMPLETED})', patch)

with open('signal_analysis/pipeline.py', 'w') as f:
    f.write(content)
