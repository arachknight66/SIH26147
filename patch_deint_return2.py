import re
with open('signal_analysis/deinterleaving.py', 'r') as f:
    content = f.read()

result_patch = '''    diags = []
    if best_hyp.family == DeinterleaverFamily.NONE and any("Search space exhausted" in ev for ev in best_hyp.falsification_evidence):
        from .models import Diagnostic, Severity
        diags.append(Diagnostic(Severity.INFO, "DEINTERLEAVER_SEARCH_EXHAUSTED", best_hyp.falsification_evidence[0], ""))

    res = DeinterleavingResult(
        bits=out_bits,
        llrs_reordered=out_llrs,
        hypothesis=best_hyp,
        cross_validation_score=cv_score,
        diagnostics=diags
    )
    return res, hypotheses
'''

# Use a safer string replacement instead of regex
idx = content.find('    res = DeinterleavingResult(')
if idx != -1:
    end_idx = content.find('    return res, hypotheses', idx) + len('    return res, hypotheses')
    if end_idx != -1:
        content = content[:idx] + result_patch + content[end_idx:]

with open('signal_analysis/deinterleaving.py', 'w') as f:
    f.write(content)
