import re
with open('signal_analysis/classifier.py', 'r') as f:
    content = f.read()

cap_patch = '''    has_unavail = (feature_vector.cumulant.validity.value == "UNAVAILABLE" or feature_vector.phase.validity.value == "UNAVAILABLE")
    
    import dataclasses
    for i, h in enumerate(hypotheses):
        if has_unavail and h.quality_tier != "UNKNOWN":
            # Cap quality at LOW
            new_ev = list(h.evidence)
            for diag in feature_vector.diagnostics:
                if diag.code == "COMPLEX_FEATURES_UNAVAILABLE":
                    new_ev.append(f"WARNING: {diag.message}")
            hypotheses[i] = dataclasses.replace(h, quality_tier="LOW", evidence=new_ev)
'''
# We need to replace the bad patch we just applied, or just revert and reapply.
# Let's see what was added.
content = re.sub(r'    has_unavail = .*?    # Check for ambiguity margin', cap_patch + '\n    # Check for ambiguity margin', content, flags=re.DOTALL)

with open('signal_analysis/classifier.py', 'w') as f:
    f.write(content)
