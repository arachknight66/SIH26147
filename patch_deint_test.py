import re
with open('tests/test_deinterleaving.py', 'r') as f:
    content = f.read()

test_patch = '''def test_block_deinterleave_exact_recovery():
    import numpy as np
    from signal_analysis.models import DemodulationResult, SynchronizationResult, HypothesisStatus, DeinterleaverFamily
    from signal_analysis.deinterleaving import attempt_deinterleaving
    
    # Construct a signal with a known structure (periodic sync word + random payload)
    rng = np.random.RandomState(42)
    sync_word = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=np.uint8)
    
    # 128 frames, each frame is 8 bit sync + 24 bit random payload = 32 bits
    frames = []
    for _ in range(128):
        payload = rng.randint(0, 2, 24).astype(np.uint8)
        frames.append(np.concatenate([sync_word, payload]))
        
    message = np.concatenate(frames)
    
    # Message length is 128 * 32 = 4096
    # Interleave it by writing rows, reading cols
    r, c = 32, 128
    block = message.reshape((c, r)).T.flatten()
    
    demod = DemodulationResult(
        hard_bits=block,
        soft_llrs=np.where(block == 1, 10.0, -10.0).astype(np.float32),
        bits_per_symbol=1,
        symbol_decisions=np.array([]),
        sync_result=SynchronizationResult(0, "Hz", 0, True, True, 0, 0, []),
        source_hypothesis_label="BPSK",
        hypothesis_confirmed=True
    )
    
    res, hyps = attempt_deinterleaving(demod)
    
    assert res.hypothesis.family == DeinterleaverFamily.BLOCK
    
    # Check that bits are recovered
    assert np.array_equal(res.bits, message)'''

content = re.sub(r'def test_block_deinterleave_exact_recovery\(\):.*?assert np\.array_equal\(res\.bits, message\)', test_patch, content, flags=re.DOTALL)
with open('tests/test_deinterleaving.py', 'w') as f:
    f.write(content)
