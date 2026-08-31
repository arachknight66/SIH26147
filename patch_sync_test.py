import re
with open('tests/test_deint_fixes.py', 'r') as f:
    content = f.read()

content = content.replace("sync_result=SynchronizationResult(True, 0, 'Hz', 1.0, 1.0, HypothesisStatus.VERIFIED)", "sync_result=SynchronizationResult(0.0, 'Hz', 1.0, 1.0, [])")

with open('tests/test_deint_fixes.py', 'w') as f:
    f.write(content)
