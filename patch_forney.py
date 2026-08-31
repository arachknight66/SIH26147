import re
with open('signal_analysis/fec_reed_solomon.py', 'r') as f:
    content = f.read()

forney_patch = '''            num = self.gf.poly_eval(omega[::-1], x_inv)
            den = self.gf.poly_eval(lambda_prime[::-1], x_inv)
            if den == 0:
                raise ValueError("Forney denominator is 0 (duplicate root)")
                
            mag = self.gf.div(num, den)
            # multiply by X_j since b=0
            x_j = self.gf.power(2, pos)
            mag = self.gf.mul(mag, x_j)
            magnitudes.append(mag)'''

content = re.sub(r'            num = self.gf.poly_eval\(omega\[::-1\], x_inv\).*?magnitudes\.append\(mag\)', forney_patch, content, flags=re.DOTALL)
with open('signal_analysis/fec_reed_solomon.py', 'w') as f:
    f.write(content)
