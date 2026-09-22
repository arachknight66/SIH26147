import numpy as np
from typing import List, Tuple, Dict, Any
from .models import (
    DeinterleaverFamily, 
    DeinterleaverHypothesis, 
    DeinterleavingResult, 
    HypothesisStatus,
    DemodulationResult
)
from .native import deinterleave as native_deinterleave, require_native


def apply_configured_deinterleaver(
    bits: np.ndarray, llrs: np.ndarray, profile: Dict[str, Any]
) -> DeinterleavingResult:
    """Apply an explicit native interleaver profile without blind seed guessing."""
    native = require_native()
    family_name = str(profile.get("family", "NONE")).upper().replace("-", "_")
    try:
        family = getattr(native.InterleaverFamily, family_name)
    except AttributeError as exc:
        raise ValueError(f"unsupported interleaver family {family_name}") from exc
    config = native.NativeInterleaverConfig()
    config.family = family
    config.rows = int(profile.get("rows", 0))
    config.columns = int(profile.get("columns", profile.get("cols", 0)))
    config.read_by_row = bool(profile.get("read_by_row", True))
    config.branches = int(profile.get("branches", 0))
    config.delay = int(profile.get("delay", 0))
    config.seed = int(profile.get("seed", 0))
    if "permutation" in profile:
        config.permutation = [int(value) for value in profile["permutation"]]
    native_result = native_deinterleave(
        np.ascontiguousarray(bits, dtype=np.uint8),
        np.ascontiguousarray(llrs, dtype=np.float32),
        config=config,
    )
    diagnostics = [
        Diagnostic(getattr(Severity, item.severity.name), item.code, item.message, item.evidence)
        for item in native_result.diagnostics
    ]
    hypothesis = DeinterleaverHypothesis(
        family=DeinterleaverFamily[family_name],
        parameters=dict(profile),
        score=float(profile.get("score", 0.0)),
        falsification_evidence=["Explicit configured profile; no blind seed or permutation inference was attempted."],
        status=HypothesisStatus.HYPOTHESIS_UNVERIFIED,
    )
    return DeinterleavingResult(
        bits=np.asarray(native_result.bits),
        llrs_reordered=np.asarray(native_result.llrs),
        hypothesis=hypothesis,
        cross_validation_score=0.0,
        diagnostics=diagnostics,
    )

def _deinterleave_block(data: np.ndarray, rows: int, cols: int, read_by_row: bool = True) -> np.ndarray:
    """
    Block de-interleaver. Assumes data can be reshaped to (rows, cols).
    If read_by_row is True, it fills by column and reads by row, else fills by row reads by col.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("block interleaver rows and cols must be positive")
    if data.ndim != 1:
        raise ValueError("interleaver input must be one-dimensional")
    native = require_native()
    # The legacy helper's two transpose directions are inverses.  The native
    # API is named from the receiver's perspective, so select the inverse
    # direction here while retaining the public helper's historical semantics.
    config = native.NativeInterleaverConfig()
    config.family = native.InterleaverFamily.BLOCK
    config.rows = int(rows)
    config.columns = int(cols)
    config.read_by_row = not bool(read_by_row)
    if data.dtype == np.uint8:
        bits = np.ascontiguousarray(data)
        llrs = np.where(bits == 1, 1.0, -1.0).astype(np.float32)
        return np.asarray(native_deinterleave(bits, llrs, config=config).bits)
    values = np.ascontiguousarray(data, dtype=np.float32)
    bits = (values > 0).astype(np.uint8)
    return np.asarray(native_deinterleave(bits, values, config=config).llrs, dtype=data.dtype)

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

from typing import Optional, Dict, Any
from .models import Diagnostic, Severity

def search_interleaver_hypotheses(demod_result: DemodulationResult, config: Optional[Dict[str, Any]] = None) -> List[DeinterleaverHypothesis]:
    """
        Search for block interleaver dimensions.

        Parameters
        ----------
        demod_result : DemodulationResult
            Demodulation output containing hard bits.
        config : Optional[Dict[str, Any]]
            Configuration for test dimensions.

        Returns
        -------
        List[DeinterleaverHypothesis]
            Ranked interleaver hypotheses.
        """
    config = config or {}
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
    test_dims = config.get("deinterleaver_test_dims", [8, 12, 16, 32, 64, 128, 255])
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
    
    if best_block_score <= score_none:
        # No candidate exceeded baseline
        hypotheses[0] = DeinterleaverHypothesis(
            family=DeinterleaverFamily.NONE,
            parameters={},
            score=score_none,
            falsification_evidence=[f"Search space exhausted: no BLOCK interleaver in {test_dims} exceeded NONE baseline ({score_none:.4f})"],
            status=HypothesisStatus.HYPOTHESIS_UNVERIFIED
        )

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
        
        # Short random streams can produce modest autocorrelation changes under
        # a transpose.  Treat low absolute payoff as ambiguous even if a local
        # perturbation happens to reduce it.
        if base_score_full < 10.0 or (margin_1 < (0.05 * base_score_full) and margin_2 < (0.05 * base_score_full)):
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

def attempt_deinterleaving(demod_result: DemodulationResult, config: Optional[Dict[str, Any]] = None) -> Tuple[DeinterleavingResult, List[DeinterleaverHypothesis]]:
    """
    Search, falsify, and apply best deinterleaver.
    """
    config = config or {}
    explicit = config.get("interleaver_profile")
    if explicit:
        result = apply_configured_deinterleaver(demod_result.hard_bits, demod_result.soft_llrs, explicit)
        return result, [result.hypothesis]
    hypotheses = search_interleaver_hypotheses(demod_result, config)
    
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
        
    diags = []
    if best_hyp.family == DeinterleaverFamily.NONE and any("Search space exhausted" in ev for ev in best_hyp.falsification_evidence):
        from .models import Diagnostic, Severity
        diags.append(Diagnostic(Severity.INFO, "DEINTERLEAVER_SEARCH_EXHAUSTED", best_hyp.falsification_evidence[0], ""))

    res = DeinterleavingResult(
        bits=out_bits,
        llrs_reordered=out_llrs,
        hypothesis=best_hyp,
        cross_validation_score=cv_score,
        diagnostics=diags
    )
    return res, hypotheses
