import re
with open('signal_analysis/fec_reed_solomon.py', 'r') as f:
    content = f.read()

verify_patch = '''        # Verification pass
        syndromes_check = self.calc_syndromes(corrected)
        if sum(syndromes_check) != 0:
            diagnostics.append(Diagnostic(Severity.ERROR, "RS_DECODE_FAILED", "Syndromes non-zero after correction", ""))
            return msg[:self.k], 0, False, diagnostics
            
        # FIX 3: Independent re-encode verification
        reencoded = self.encode(corrected[:self.k])
        if reencoded[self.k:] != corrected[self.k:]:
            diagnostics.append(Diagnostic(Severity.ERROR, "RS_REENCODE_MISMATCH", "Re-encoded parity symbols do not match corrected codeword parity symbols", ""))
            return msg[:self.k], 0, False, diagnostics
'''

content = content.replace('''        # Verification pass
        syndromes_check = self.calc_syndromes(corrected)
        if sum(syndromes_check) != 0:
            diagnostics.append(Diagnostic(Severity.ERROR, "RS_DECODE_FAILED", "Syndromes non-zero after correction", ""))
            return msg[:self.k], 0, False, diagnostics''', verify_patch)

with open('signal_analysis/fec_reed_solomon.py', 'w') as f:
    f.write(content)
