import re
with open('signal_analysis/deinterleaving.py', 'r') as f:
    content = f.read()

result_patch = '''    diags = []
    if best_hyp.family == DeinterleaverFamily.NONE and any("Search space exhausted" in ev for ev in best_hyp.falsification_evidence):
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

content = re.sub(r'    res = DeinterleavingResult\(\n        bits=out_bits,\n        llrs_reordered=out_llrs,\n        hypothesis=best_hyp,\n        cross_validation_score=cv_score\n    \)\n    return res, hypotheses', result_patch, content, flags=re.DOTALL)

with open('signal_analysis/deinterleaving.py', 'w') as f:
    f.write(content)
