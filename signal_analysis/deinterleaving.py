import numpy as np
from typing import List, Tuple, Dict, Any
from .models import (
    DeinterleaverFamily, 
    DeinterleaverHypothesis, 
    DeinterleavingResult, 
    HypothesisStatus,
    DemodulationResult
)

def _deinterleave_block(data: np.ndarray, rows: int, cols: int, read_by_row: bool = True) -> np.ndarray:
    """
    Block de-interleaver. Assumes data can be reshaped to (rows, cols).
    If read_by_row is True, it fills by column and reads by row, else fills by row reads by col.
    """
    total = rows * cols
    if len(data) < total:
        return data  # Too short to apply full block, ignore or pad? For MVP, return as is.
    
    # Truncate to multiple of block size
    n_blocks = len(data) // total
    out = np.zeros_like(data[:n_blocks * total])
    
    for b in range(n_blocks):
        block = data[b*total : (b+1)*total]
        if read_by_row:
            # Filled by column, read by row (transpose)
            reshaped = block.reshape((cols, rows)).T
        else:
            # Filled by row, read by column
            reshaped = block.reshape((rows, cols)).T
        out[b*total : (b+1)*total] = reshaped.flatten()
        
    return out

def structural_payoff_score(bits: np.ndarray) -> float:
    """
    Computes a structural payoff score based on periodic autocorrelation.
    High score implies a repeating sync word or strong frame structure.
    """
    if len(bits) < 200:
        return 0.0
        
    # Convert bits to bipolar to avoid DC bias dominating correlation
    bipolar = np.where(bits == 1, 1, -1)
    
    # Fast correlation using FFT
    N = len(bipolar)
    F = np.fft.fft(bipolar, n=2*N)
    R = np.fft.ifft(F * np.conj(F)).real
    
    # Ignore the lag=0 peak and immediately adjacent lags (e.g. up to lag 10)
    # We look for periodic peaks at lag >= 16 (typical minimum frame sizes)
    R = R[:N]
    R[:16] = 0
    
    if np.max(np.abs(R)) == 0:
        return 0.0
        
    # Score is the ratio of max peak to the median of the absolute autocorrelation (to normalize)
    median_R = np.median(np.abs(R[16:]))
    if median_R < 1e-9:
        return 0.0
        
    score = np.max(np.abs(R)) / median_R
    return float(score)

def search_interleaver_hypotheses(demod_result: DemodulationResult) -> List[DeinterleaverHypothesis]:
    bits = demod_result.hard_bits
    llrs = demod_result.soft_llrs
    
    hypotheses = []
    
    # NONE candidate
    score_none = structural_payoff_score(bits)
    hypotheses.append(DeinterleaverHypothesis(
        family=DeinterleaverFamily.NONE,
        parameters={},
        score=score_none,
        falsification_evidence=[],
        status=HypothesisStatus.HYPOTHESIS_UNVERIFIED
    ))
    
    # BLOCK candidate search
    # We search small typical interleaver dimensions: 8, 16, 32, 64, 128, 255 etc.
    # For a block, N = rows * cols. We try pairs (r, c)
    test_dims = [8, 12, 16, 32, 64, 128, 255]
    best_block_score = -1.0
    best_block_params = {}
    
    for r in test_dims:
        for c in test_dims:
            if r * c > len(bits) // 4: 
                continue # Need at least 4 blocks to get meaningful autocorrelation
                
            # Try read by row
            d_bits1 = _deinterleave_block(bits, r, c, True)
            s1 = structural_payoff_score(d_bits1)
            if s1 > best_block_score:
                best_block_score = s1
                best_block_params = {'rows': r, 'cols': c, 'read_by_row': True}
                
            # Try read by col
            d_bits2 = _deinterleave_block(bits, r, c, False)
            s2 = structural_payoff_score(d_bits2)
            if s2 > best_block_score:
                best_block_score = s2
                best_block_params = {'rows': r, 'cols': c, 'read_by_row': False}
                
    if best_block_score > 0:
        hypotheses.append(DeinterleaverHypothesis(
            family=DeinterleaverFamily.BLOCK,
            parameters=best_block_params,
            score=best_block_score,
            falsification_evidence=[],
            status=HypothesisStatus.HYPOTHESIS_UNVERIFIED
        ))
        
    # Explicit Non-Goal: Pseudo-Random blind recovery.
    hypotheses.append(DeinterleaverHypothesis(
        family=DeinterleaverFamily.PSEUDO_RANDOM,
        parameters={'seed': 'UNKNOWN'},
        score=0.0,
        falsification_evidence=["Blind pseudo-random interleaver recovery is mathematically unfalsifiable without seed. Marked as explicit non-goal."],
        status=HypothesisStatus.INSUFFICIENT_EVIDENCE
    ))
    
    # Sort by score descending
    hypotheses.sort(key=lambda x: x.score, reverse=True)
    return hypotheses

def falsify_and_cross_validate(bits: np.ndarray, hyp: DeinterleaverHypothesis) -> DeinterleaverHypothesis:
    """
    Perturb the parameters slightly and confirm score drops.
    Also compute held-out cross-validation.
    """
    if hyp.family in [DeinterleaverFamily.NONE, DeinterleaverFamily.PSEUDO_RANDOM]:
        # Trivial or unfalsifiable
        return hyp
        
    if hyp.family == DeinterleaverFamily.BLOCK:
        # Cross validation split
        half = len(bits) // 2
        bits_train = bits[:half]
        bits_test = bits[half:]
        
        r = hyp.parameters['rows']
        c = hyp.parameters['cols']
        read_row = hyp.parameters['read_by_row']
        
        # Test baseline on full and split
        base_score_test = structural_payoff_score(_deinterleave_block(bits_test, r, c, read_row))
        base_score_full = hyp.score
        
        # Perturbation: flip read direction, or shift boundaries (r+1)
        # 1. Flip read_row
        pert_bits_1 = _deinterleave_block(bits, r, c, not read_row)
        score_pert_1 = structural_payoff_score(pert_bits_1)
        
        # 2. Add offset (simulate out-of-sync)
        if len(bits) > r*c + 1:
            pert_bits_2 = _deinterleave_block(bits[1:], r, c, read_row)
            score_pert_2 = structural_payoff_score(pert_bits_2)
        else:
            score_pert_2 = 0
            
        evidence = list(hyp.falsification_evidence)
        status = hyp.status
        
        # Check flat response surface
        # If perturbation doesn't significantly drop score (e.g. within 5%), it's ambiguous
        margin_1 = base_score_full - score_pert_1
        margin_2 = base_score_full - score_pert_2
        
        if margin_1 < (0.05 * base_score_full) and margin_2 < (0.05 * base_score_full):
            evidence.append(f"Falsification failed: Perturbations yielded margins ({margin_1:.2f}, {margin_2:.2f}) < 5% of base score. Unconstrained parameters.")
            status = HypothesisStatus.AMBIGUOUS
        else:
            evidence.append(f"Falsification passed: Score dropped significantly on perturbations (Margins: {margin_1:.2f}, {margin_2:.2f}).")
            
        return DeinterleaverHypothesis(
            family=hyp.family,
            parameters=hyp.parameters,
            score=hyp.score,
            falsification_evidence=evidence,
            status=status
        )
        
    return hyp

def attempt_deinterleaving(demod_result: DemodulationResult) -> Tuple[DeinterleavingResult, List[DeinterleaverHypothesis]]:
    """
    Search, falsify, and apply best deinterleaver.
    """
    hypotheses = search_interleaver_hypotheses(demod_result)
    
    # Take top hypothesis, attempt falsification
    best_hyp = hypotheses[0]
    best_hyp = falsify_and_cross_validate(demod_result.hard_bits, best_hyp)
    hypotheses[0] = best_hyp
    
    bits = demod_result.hard_bits
    llrs = demod_result.soft_llrs
    
    # Calculate cross-validation on half
    half = len(bits) // 2
    
    if best_hyp.family == DeinterleaverFamily.BLOCK:
        r = best_hyp.parameters['rows']
        c = best_hyp.parameters['cols']
        read_row = best_hyp.parameters['read_by_row']
        out_bits = _deinterleave_block(bits, r, c, read_row)
        out_llrs = _deinterleave_block(llrs, r, c, read_row)
        
        out_test = _deinterleave_block(bits[half:], r, c, read_row)
        cv_score = structural_payoff_score(out_test)
    else:
        out_bits = bits
        out_llrs = llrs
        cv_score = structural_payoff_score(bits[half:])
        
    res = DeinterleavingResult(
        bits=out_bits,
        llrs_reordered=out_llrs,
        hypothesis=best_hyp,
        cross_validation_score=cv_score
    )
    
    return res, hypotheses
