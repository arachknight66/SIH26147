import re
with open('signal_analysis/features.py', 'r') as f:
    content = f.read()

if 'from dataclasses import dataclass, field' not in content:
    content = content.replace('from dataclasses import dataclass', 'from dataclasses import dataclass, field\nfrom typing import List\nfrom .models import Diagnostic, Severity')

if 'diagnostics: List[Diagnostic]' not in content:
    content = content.replace(
        '    cyclostationary: CyclostationaryFeatures',
        '    cyclostationary: CyclostationaryFeatures\n    diagnostics: List[Diagnostic] = field(default_factory=list)'
    )

truncation_patch = '''    max_samples = DEFAULT_MAX_ANALYSIS_SAMPLES
    process_samples = recording.samples[:max_samples]
    
    diagnostics = []
    if len(recording.samples) > max_samples:
        frac = max_samples / len(recording.samples)
        diagnostics.append(Diagnostic(
            code="TRUNCATED_ANALYSIS",
            message=f"Analyzed first {max_samples} of {len(recording.samples)} samples ({frac*100:.2f}% of file)",
            severity=Severity.INFO
        ))
'''
content = re.sub(r'    max_samples = DEFAULT_MAX_ANALYSIS_SAMPLES\n    process_samples = recording\.samples\[:max_samples\]', truncation_patch, content, flags=re.DOTALL)

content = re.sub(r'        cyclostationary=cyclo\n    \)', '        cyclostationary=cyclo,\n        diagnostics=diagnostics\n    )', content)

with open('signal_analysis/features.py', 'w') as f:
    f.write(content)
