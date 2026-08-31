import numpy as np
import pytest
from signal_analysis.models import (
    DemodulationResult,
    SynchronizationResult,
    HypothesisStatus,
    DeinterleaverFamily
)
from signal_analysis.deinterleaving import attempt_deinterleaving, _deinterleave_block

def test_block_deinterleave_exact_recovery():
    import numpy as np
    from signal_analysis.models import DemodulationResult, SynchronizationResult, HypothesisStatus, DeinterleaverFamily
    from signal_analysis.deinterleaving import attempt_deinterleaving
    
    # Construct a signal with a known structure (periodic sync word + random payload)
    rng = np.random.RandomState(42)
    sync_word = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=np.uint8)
    
    # 128 frames, each frame is 8 bit sync + 24 bit random payload = 32 bits
    frames = []
    for _ in range(32):
        payload = rng.randint(0, 2, 24).astype(np.uint8)
        frames.append(np.concatenate([sync_word, payload]))
        
    message = np.concatenate(frames)
    
    # Message length is 128 * 32 = 4096
    # Interleave it by writing rows, reading cols
    r, c = 8, 32
    from signal_analysis.deinterleaving import _deinterleave_block
    block = _deinterleave_block(message, r, c, read_by_row=False)
    
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
    assert np.array_equal(res.bits, message)

def test_deinterleave_falsification_downgrade():
    # Construct a pure random sequence (flat response surface)
    # The autocorrelation will be flat, perturbations won't drop the score significantly
    rng = np.random.RandomState(42)
    random_bits = rng.randint(0, 2, 800).astype(np.uint8)
    
    demod = DemodulationResult(
        hard_bits=random_bits,
        soft_llrs=np.where(random_bits == 1, 2.0, -2.0).astype(np.float32),
        bits_per_symbol=1,
        symbol_decisions=np.array([]),
        sync_result=SynchronizationResult(0, "Hz", 0, True, True, 0, 0, []),
        source_hypothesis_label="BPSK",
        hypothesis_confirmed=True
    )
    
    res, hyps = attempt_deinterleaving(demod)
    
    if res.hypothesis.family == DeinterleaverFamily.BLOCK:
        # If it picked BLOCK, it MUST have downgraded it to AMBIGUOUS because it's noise
        assert res.hypothesis.status == HypothesisStatus.AMBIGUOUS
    else:
        # Or it correctly stuck with NONE
        assert res.hypothesis.family == DeinterleaverFamily.NONE

def test_pseudo_random_non_goal():
    demod = DemodulationResult(
        hard_bits=np.zeros(100, dtype=np.uint8),
        soft_llrs=np.zeros(100, dtype=np.float32),
        bits_per_symbol=1,
        symbol_decisions=np.array([]),
        sync_result=SynchronizationResult(0, "Hz", 0, True, True, 0, 0, []),
        source_hypothesis_label="BPSK",
        hypothesis_confirmed=True
    )
    
    res, hyps = attempt_deinterleaving(demod)
    
    pr_hyp = next(h for h in hyps if h.family == DeinterleaverFamily.PSEUDO_RANDOM)
    assert pr_hyp.status == HypothesisStatus.INSUFFICIENT_EVIDENCE
    assert "unfalsifiable" in pr_hyp.falsification_evidence[0].lower()
