import re
with open('signal_analysis/classifier.py', 'r') as f:
    content = f.read()

cap_patch = '''    has_unavail = (feature_vector.cumulant.validity.value == "UNAVAILABLE" or feature_vector.phase.validity.value == "UNAVAILABLE")
    
    for h in hypotheses:
        if has_unavail and h.quality_tier.value != "UNKNOWN":
            # Cap quality at LOW
            from .models import QualityTier
            h.quality_tier = QualityTier.LOW
            # Propagate the diagnostic
            for diag in feature_vector.diagnostics:
                if diag.code == "COMPLEX_FEATURES_UNAVAILABLE":
                    h.evidence.append(f"WARNING: {diag.message}")
'''
content = content.replace('    # Check for ambiguity margin', cap_patch + '\n    # Check for ambiguity margin')

with open('signal_analysis/classifier.py', 'w') as f:
    f.write(content)
