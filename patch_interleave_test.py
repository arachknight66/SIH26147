import re
with open('tests/test_deint_fixes.py', 'r') as f:
    content = f.read()

patch = '''    # Interleave manually (correctly per block)
    interleaved = bits.copy()
    block_size = 7 * 11
    for b in range(2):
        for r in range(7):
            for c in range(11):
                i_idx = b * block_size + r * 11 + c
                o_idx = b * block_size + c * 7 + r
                interleaved[o_idx] = bits[i_idx]
'''
content = re.sub(r'    # Interleave manually\n    interleaved = bits\.copy\(\)\n    for i in range\(len\(bits\)\):\n        r = i // 11\n        c = i % 11\n        idx = c \* 7 \+ r\n        interleaved\[idx\] = bits\[i\]', patch, content, flags=re.DOTALL)

with open('tests/test_deint_fixes.py', 'w') as f:
    f.write(content)
