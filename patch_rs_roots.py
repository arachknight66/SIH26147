import re
with open('signal_analysis/fec_reed_solomon.py', 'r') as f:
    content = f.read()

chien_patch = '''    def find_roots_chien(self, locator: List[int]) -> List[int]:
        roots = []
        for i in range(1, self.gf.size):
            # locator is Lambda(x) = 1 + L1 x + L2 x^2 + ...
            # Evaluate Lambda at x = alpha^(-X_j) = inv(i)
            val = self.gf.poly_eval(locator[::-1], self.gf.inv(i))
            if val == 0:
                roots.append(self.gf.log[i]) # This is X_j
        return roots'''
content = re.sub(r'    def find_roots_chien.*?return roots', chien_patch, content, flags=re.DOTALL)

forney_patch = '''    def forney_magnitudes(self, syndromes: List[int], locator: List[int], roots: List[int]) -> List[int]:
        # Error evaluator polynomial Omega = S * Lambda mod x^(n-k)
        omega = self.gf.poly_mul(syndromes, locator)[:self.n - self.k]
        
        # Formal derivative of locator Lambda'
        lambda_prime = [0] * (len(locator) - 1)
        for i in range(1, len(locator)):
            # In GF(2^m), derivative of x^i is i*x^(i-1) mod 2 -> 1 if i is odd, 0 if even
            if i % 2 == 1:
                lambda_prime[i - 1] = locator[i]
                
        magnitudes = []
        for X_j in roots:
            x_inv = self.gf.inv(self.gf.power(2, X_j))
            
            num = self.gf.poly_eval(omega[::-1], x_inv)
            den = self.gf.poly_eval(lambda_prime[::-1], x_inv)
            if den == 0:
                raise ValueError("Forney denominator is 0 (duplicate root)")
                
            mag = self.gf.div(num, den)
            # multiply by X_j since b=0
            x_j = self.gf.power(2, X_j)
            mag = self.gf.mul(mag, x_j)
            magnitudes.append(mag)
            
        return magnitudes'''
content = re.sub(r'    def forney_magnitudes.*?return magnitudes', forney_patch, content, flags=re.DOTALL)

decode_patch = '''        # 5. Forney
        mags = self.forney_magnitudes(syndromes, locator, roots)
        
        # Correct
        corrected = list(msg)
        for X_j, m in zip(roots, mags):
            idx = len(msg) - 1 - X_j
            if idx < 0 or idx >= len(msg):
                diagnostics.append(Diagnostic(Severity.ERROR, "RS_OUT_OF_BOUNDS", f"Correction index {idx} out of bounds", ""))
                return msg[:self.k], 0, False, diagnostics
            corrected[idx] = self.gf.add(corrected[idx], m)'''
content = re.sub(r'        # 5\. Forney.*?corrected\[idx\] = self\.gf\.add\(corrected\[idx\], m\)', decode_patch, content, flags=re.DOTALL)

with open('signal_analysis/fec_reed_solomon.py', 'w') as f:
    f.write(content)
