import re
with open('tests/test_deint_fixes.py', 'r') as f:
    content = f.read()

patch = '''    demod = DemodulationResult(
        hard_bits=interleaved,
        soft_llrs=llrs,
        sync_result=SynchronizationResult(0.0, 'Hz', 0.0, True, True, 1.0, 1.0, []),
        bits_per_symbol=1,
        symbol_decisions=interleaved,
        source_hypothesis_label='BPSK',
        hypothesis_confirmed=True
    )'''

content = re.sub(r'    demod = DemodulationResult\(\n        hard_bits=interleaved,\n        soft_llrs=llrs,\n        sync_result=SynchronizationResult\(0.0, \'Hz\', 0.0, True, True, 1.0, 1.0, \[\]\)\n    \)', patch, content, flags=re.DOTALL)

with open('tests/test_deint_fixes.py', 'w') as f:
    f.write(content)
