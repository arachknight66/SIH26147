import re
with open('signal_analysis/deinterleaving.py', 'r') as f:
    content = f.read()

# Add config param to search
content = content.replace(
    'def search_interleaver_hypotheses(demod_result: DemodulationResult) -> List[DeinterleaverHypothesis]:',
    'from typing import Optional, Dict, Any\nfrom .models import Diagnostic, Severity\n\ndef search_interleaver_hypotheses(demod_result: DemodulationResult, config: Optional[Dict[str, Any]] = None) -> List[DeinterleaverHypothesis]:\n    config = config or {}'
)

# test_dims parameter
content = content.replace(
    'test_dims = [8, 12, 16, 32, 64, 128, 255]',
    'test_dims = config.get("deinterleaver_test_dims", [8, 12, 16, 32, 64, 128, 255])'
)

# And if best_block_score < score_none
block_fallback = '''    
    if best_block_score <= score_none:
        # No candidate exceeded baseline
        hypotheses[0] = DeinterleaverHypothesis(
            family=DeinterleaverFamily.NONE,
            parameters={},
            score=score_none,
            falsification_evidence=[f"Search space exhausted: no BLOCK interleaver in {test_dims} exceeded NONE baseline ({score_none:.4f})"],
            status=HypothesisStatus.HYPOTHESIS_UNVERIFIED
        )
'''
# Actually we can just let it fall through, the hypotheses list is sorted.
# Let's see how hypotheses is sorted.
content = content.replace(
    '    hypotheses.sort(key=lambda h: h.score, reverse=True)\n    return hypotheses',
    block_fallback + '\n    hypotheses.sort(key=lambda h: h.score, reverse=True)\n    return hypotheses'
)

# Add config to attempt_deinterleaving
content = content.replace(
    'def attempt_deinterleaving(demod_result: DemodulationResult) -> Tuple[DeinterleavingResult, List[DeinterleaverHypothesis]]:',
    'def attempt_deinterleaving(demod_result: DemodulationResult, config: Optional[Dict[str, Any]] = None) -> Tuple[DeinterleavingResult, List[DeinterleaverHypothesis]]:'
)
content = content.replace(
    '    hypotheses = search_interleaver_hypotheses(demod_result)',
    '    hypotheses = search_interleaver_hypotheses(demod_result, config)'
)

# Extract diagnostics from NONE fallback evidence
result_patch = '''    # Extract diagnostic if NONE won by exhaustion
    diags = []
    if best_hyp.family == DeinterleaverFamily.NONE and any("Search space exhausted" in ev for ev in best_hyp.falsification_evidence):
        diags.append(Diagnostic(Severity.INFO, "DEINTERLEAVER_SEARCH_EXHAUSTED", best_hyp.falsification_evidence[0], ""))
        
    return DeinterleavingResult(
        bits=np.array(deint_bits, dtype=np.uint8),
        llrs_reordered=np.array(deint_llrs, dtype=np.float32),
        hypothesis=best_hyp,
        cross_validation_score=cv_score,
        diagnostics=diags
    ), hypotheses'''

content = content.replace(
    '    return DeinterleavingResult(\n        bits=np.array(deint_bits, dtype=np.uint8),\n        llrs_reordered=np.array(deint_llrs, dtype=np.float32),\n        hypothesis=best_hyp,\n        cross_validation_score=cv_score\n    ), hypotheses',
    result_patch
)

with open('signal_analysis/deinterleaving.py', 'w') as f:
    f.write(content)
