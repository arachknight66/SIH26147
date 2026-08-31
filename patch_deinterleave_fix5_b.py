import re
with open('signal_analysis/deinterleaving.py', 'r') as f:
    content = f.read()

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
content = content.replace(
    '    hypotheses.sort(key=lambda x: x.score, reverse=True)',
    block_fallback + '\n    hypotheses.sort(key=lambda x: x.score, reverse=True)'
)
with open('signal_analysis/deinterleaving.py', 'w') as f:
    f.write(content)
