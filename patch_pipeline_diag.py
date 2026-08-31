import re
with open('signal_analysis/pipeline.py', 'r') as f:
    content = f.read()

content = content.replace(
    'diag = Diagnostic("NON_COMPLEX_PIPELINE", f"Pipeline ran on {recording.semantic_type}. Phase/cumulant metrics are compromised.", Severity.WARNING)',
    'diag = Diagnostic(Severity.WARNING, "NON_COMPLEX_PIPELINE", f"Pipeline ran on {recording.semantic_type}. Phase/cumulant metrics are compromised.", "")'
)

with open('signal_analysis/pipeline.py', 'w') as f:
    f.write(content)
