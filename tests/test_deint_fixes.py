
def test_fix5_deinterleaver_search_exhausted():
    import numpy as np
    from signal_analysis.models import DemodulationResult, SynchronizationResult, CandidateParameters, HypothesisStatus
    from signal_analysis.deinterleaving import attempt_deinterleaving
    
    # We create a BLOCK interleaved sequence with dimensions that are explicitly
    # NOT in the default test_dims grid, say 7 x 11.
    bits = np.random.randint(0, 2, 7*11*5).astype(np.uint8)
    # The structure must have some periodicity to actually score well if found
    for i in range(len(bits)//7):
        bits[i*7] = 1
        bits[i*7+1] = 0
        
    # Interleave manually (correctly per block)
    interleaved = bits.copy()
    block_size = 7 * 11
    for b in range(5):
        for r in range(7):
            for c in range(11):
                i_idx = b * block_size + r * 11 + c
                o_idx = b * block_size + c * 7 + r
                interleaved[o_idx] = bits[i_idx]


    # Add noise to soft LLRs
    llrs = np.where(interleaved == 1, 5.0, -5.0) + np.random.randn(len(interleaved))
    
    demod = DemodulationResult(
        hard_bits=interleaved,
        soft_llrs=llrs,
        sync_result=SynchronizationResult(0.0, 'Hz', 0.0, True, True, 1.0, 1.0, []),
        bits_per_symbol=1,
        symbol_decisions=interleaved,
        source_hypothesis_label='BPSK',
        hypothesis_confirmed=True
    )
    
    deint_res, hyps = attempt_deinterleaving(demod, {"deinterleaver_test_dims": [17, 19]})
    assert deint_res.hypothesis.family.name == "NONE"
    assert any(d.code == "DEINTERLEAVER_SEARCH_EXHAUSTED" for d in deint_res.diagnostics)
    
    # Run with custom config where it WILL find it
    deint_res2, hyps2 = attempt_deinterleaving(demod, {"deinterleaver_test_dims": [7, 11, 16]})
    assert deint_res2.hypothesis.family.name == "BLOCK"
    assert not any(d.code == "DEINTERLEAVER_SEARCH_EXHAUSTED" for d in deint_res2.diagnostics)
