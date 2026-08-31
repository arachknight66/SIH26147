import numpy as np
import pytest
from signal_analysis.models import (
    SyncWordPattern, 
    HeaderMatch,
    FrameStructure,
    CRCMatch,
    HypothesisStatus,
    DeinterleavingResult,
    DeinterleaverFamily
)
from signal_analysis.correlation import correlate_sync_words, BUILTIN_SYNC_WORDS
from signal_analysis.crc_search import search_crcs, COMMON_CRCS, compute_crc_bitwise
from signal_analysis.framing import assemble_frames

def test_sync_word_correlation_exact_match():
    # Insert known pattern in random bits
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, 1000).astype(np.uint8)
    
    pattern = BUILTIN_SYNC_WORDS[0] # HDLC 01111110
    bits[100:108] = pattern.bit_pattern
    
    llrs = np.where(bits == 1, 5.0, -5.0).astype(np.float32)
    
    matches = correlate_sync_words(bits, llrs, [pattern])
    
    # Should find it at offset 100 with hamming distance 0
    assert any(m.bit_offset == 100 and m.hamming_distance == 0 for m in matches)

def test_sync_word_correlation_bit_errors():
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, 1000).astype(np.uint8)
    pattern = BUILTIN_SYNC_WORDS[1] # 32 bit CCSDS
    pat = pattern.bit_pattern.copy()
    
    # Insert with 2 errors (should be within 15% of 32 = 4 errors)
    pat[5] ^= 1
    pat[15] ^= 1
    bits[200:232] = pat
    llrs = np.where(bits == 1, 5.0, -5.0).astype(np.float32)
    
    matches = correlate_sync_words(bits, llrs, [pattern])
    assert any(m.bit_offset == 200 and m.hamming_distance == 2 for m in matches)
    
    # Insert with 10 errors (exceeds threshold)
    pat2 = pattern.bit_pattern.copy()
    for i in range(10): pat2[i] ^= 1
    bits[400:432] = pat2
    
    matches2 = correlate_sync_words(bits, llrs, [pattern])
    assert not any(m.bit_offset == 400 for m in matches2)

def test_periodicity_detection():
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, 1000).astype(np.uint8)
    pattern = BUILTIN_SYNC_WORDS[0]
    
    # Insert at 100, 200, 300 (spacing 100)
    bits[100:108] = pattern.bit_pattern
    bits[200:208] = pattern.bit_pattern
    bits[300:308] = pattern.bit_pattern
    
    # Isolated match at 800
    bits[800:808] = pattern.bit_pattern
    
    llrs = np.where(bits == 1, 5.0, -5.0).astype(np.float32)
    
    matches = correlate_sync_words(bits, llrs, [pattern])
    
    m100 = next(m for m in matches if m.bit_offset == 100)
    m200 = next(m for m in matches if m.bit_offset == 200)
    m800 = next(m for m in matches if m.bit_offset == 800)
    
    assert m100.periodicity_consistent == True
    assert m200.periodicity_consistent == True
    assert m800.periodicity_consistent == False

def test_match_confidence_weighting():
    # Identical hamming distance, but one is on high LLRs (Phase 4 flagged clean), 
    # one is on low LLRs (highly corrected/uncertain)
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, 1000).astype(np.uint8)
    pattern = BUILTIN_SYNC_WORDS[0]
    
    bits[100:108] = pattern.bit_pattern
    bits[500:508] = pattern.bit_pattern
    
    # Base LLRs around 10
    llrs = np.where(bits == 1, 10.0, -10.0).astype(np.float32)
    
    # Make 500:508 very uncertain (magnitude ~ 0.1)
    llrs[500:508] = np.where(bits[500:508] == 1, 0.1, -0.1)
    
    matches = correlate_sync_words(bits, llrs, [pattern])
    
    m100 = next(m for m in matches if m.bit_offset == 100)
    m500 = next(m for m in matches if m.bit_offset == 500)
    
    # Both have 0 hamming distance
    assert m100.hamming_distance == 0
    assert m500.hamming_distance == 0
    
    # But clean bits should have much higher confidence
    assert m100.match_confidence > m500.match_confidence

def test_crc_search():
    # CRC-16/CCITT-FALSE
    alg = COMMON_CRCS[1]
    
    payload = np.array([1, 0, 1, 0, 1, 0, 1, 0]*4, dtype=np.uint8) # 32 bits
    crc_val = compute_crc_bitwise(payload, alg)
    
    # Pack crc_val into 16 bits (MSB first for CCITT-FALSE)
    crc_bits = []
    for i in range(16):
        crc_bits.append((crc_val >> (15 - i)) & 1)
        
    full_frame = np.concatenate([payload, crc_bits])
    
    # Embed in random bitstream
    bits = np.concatenate([np.zeros(100, dtype=np.uint8), full_frame, np.zeros(100, dtype=np.uint8)])
    
    matches = search_crcs(bits, start_idx=100)
    
    assert any(m.verified and m.polynomial_name == alg.name for m in matches)

def test_framing_unknown_fallback():
    # Pure noise, no sync word
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, 50).astype(np.uint8)
    llrs = np.where(bits == 1, 5.0, -5.0).astype(np.float32)
    
    matches = correlate_sync_words(bits, llrs, BUILTIN_SYNC_WORDS)
    frames = assemble_frames(bits, matches)
    
    assert len(frames) == 1
    assert frames[0].status == HypothesisStatus.UNKNOWN
