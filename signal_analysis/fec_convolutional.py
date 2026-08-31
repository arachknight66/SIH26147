import numpy as np
from typing import List, Tuple
from .models import FECDecodeResult, Diagnostic, Severity, DeinterleavingResult

# Standard NASA/CCSDS Convolutional Code K=7, Rate 1/2
POLY_1 = 0o171
POLY_2 = 0o133
K = 7

def viterbi_decode_soft(deint: DeinterleavingResult, traceback_depth: int = 35) -> FECDecodeResult:
    """
    Soft-decision Viterbi decoding using LLRs from Phase 3.
    POLY_1 = 171 (octal) -> 1111001 (binary)
    POLY_2 = 133 (octal) -> 1011011 (binary)
    LLR > 0 means bit 1 is more likely.
    """
    llrs = deint.llrs_reordered
    if len(llrs) % 2 != 0:
        llrs = np.append(llrs, 0.0) # pad to even
        
    num_symbols = len(llrs) // 2
    
    # State space
    num_states = 1 << (K - 1)
    
    # Precompute outputs for state transitions
    outputs = np.zeros((num_states, 2, 2), dtype=int)
    for state in range(num_states):
        for input_bit in [0, 1]:
            # Shift register: input bit is newest (leftmost), old bits shift right
            reg = (input_bit << (K - 1)) | state
            
            # Poly 1
            out1 = bin(reg & POLY_1).count('1') % 2
            # Poly 2
            out2 = bin(reg & POLY_2).count('1') % 2
            
            # Map 0 -> -1, 1 -> 1 for correlation with LLRs
            outputs[state, input_bit, 0] = 1 if out1 == 1 else -1
            outputs[state, input_bit, 1] = 1 if out2 == 1 else -1

    # Path metrics
    # Initialize with -inf, except state 0
    path_metrics = np.full(num_states, -1e9, dtype=np.float32)
    path_metrics[0] = 0.0
    
    # Traceback table
    traceback = np.zeros((num_symbols, num_states), dtype=np.uint8)
    
    # Forward pass
    for i in range(num_symbols):
        l1 = llrs[2*i]
        l2 = llrs[2*i + 1]
        
        new_metrics = np.full(num_states, -1e9, dtype=np.float32)
        
        for state in range(num_states):
            if path_metrics[state] < -1e8: continue
            
            for input_bit in [0, 1]:
                next_state = (state >> 1) | (input_bit << (K - 2))
                
                # Branch metric: correlation between LLR and expected output.
                # LLR > 0 means 1 is likely. Output 1 maps to 1, Output 0 maps to -1.
                # Branch metric = l1 * out1 + l2 * out2
                bm = l1 * outputs[state, input_bit, 0] + l2 * outputs[state, input_bit, 1]
                
                metric = path_metrics[state] + bm
                if metric > new_metrics[next_state]:
                    new_metrics[next_state] = metric
                    traceback[i, next_state] = state
                    
        path_metrics = new_metrics
        
    # Margin calculation
    sorted_metrics = np.sort(path_metrics)[::-1]
    margin = float(sorted_metrics[0] - sorted_metrics[1]) if len(sorted_metrics) > 1 else 0.0
    
    # Traceback
    best_state = np.argmax(path_metrics)
    decoded = np.zeros(num_symbols, dtype=np.uint8)
    
    curr_state = best_state
    for i in range(num_symbols - 1, -1, -1):
        prev_state = traceback[i, curr_state]
        # Which input bit caused this transition?
        # The newest bit is at position K-2 in curr_state
        input_bit = (curr_state >> (K - 2)) & 1
        decoded[i] = input_bit
        curr_state = prev_state
        
    diags = []
    if margin < 1.0:
        diags.append(Diagnostic(Severity.WARNING, "VITERBI_LOW_MARGIN", "Path metric margin is dangerously low", f"Margin: {margin:.2f}"))
        
    return FECDecodeResult(
        decoded_bits=decoded,
        corrected_bit_count=0, # Hard to quantify exactly in soft-Viterbi without hard-decision diff
        corrected_bit_fraction=0.0,
        decode_success=True,
        codec_name=f"Convolutional(K={K}, R=1/2)",
        pre_correction_metric=margin,
        diagnostics=diags
    )
