import numpy as np
import pytest
from signal_analysis.models import SignalRecording, SourceFormat, MetadataValue, MetadataStatus
from signal_analysis.pipeline import run_full_pipeline, PipelineStageStatus

def test_pipeline_early_termination_unknown_signal():
    # Noise -> Phase 2 UNKNOWN -> Early terminate
    rng = np.random.RandomState(42)
    noise = rng.randn(1000) + 1j * rng.randn(1000)
    recording = SignalRecording(
        samples=noise.astype(np.complex64),
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        semantic_type="complex_iq",
        sample_rate_hz=MetadataValue(1e6, "test", MetadataStatus.KNOWN),
        center_frequency_hz=MetadataValue(0, "test", MetadataStatus.KNOWN),
        provenance={},
        diagnostics=[]
    )
    
    res = run_full_pipeline(recording)
    
    # Hypothesis stage ran
    assert res.hypothesis_status == PipelineStageStatus.COMPLETED
    
    # Sync and downstream should NOT have run
    assert res.sync_status == PipelineStageStatus.NOT_ATTEMPTED
    assert res.fec_status == PipelineStageStatus.NOT_ATTEMPTED
    assert res.framing_status == PipelineStageStatus.NOT_ATTEMPTED

def test_pipeline_synthetic_end_to_end():
    # Construct a valid signal: Header + Payload + CRC -> encoded -> interleaved -> modulated
    # For speed in test, we just provide BPSK with a sync word to see it through to Framing.
    rng = np.random.RandomState(42)
    
    from signal_analysis.correlation import BUILTIN_SYNC_WORDS
    pattern = BUILTIN_SYNC_WORDS[0].bit_pattern
    
    # 5 frames of payload
    bits = []
    for _ in range(5):
        bits.extend(pattern)
        bits.extend(rng.randint(0, 2, 64)) # payload
        
    bits = np.array(bits, dtype=np.uint8)
    
    # Modulate BPSK
    symbols = np.where(bits == 1, 1.0, -1.0)
    sps = 4
    tx = np.repeat(symbols, sps)
    # Add noise
    rx = tx + rng.randn(len(tx)) * 0.5
    
    recording = SignalRecording(
        samples=rx.astype(np.complex64),
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        semantic_type="complex_iq",
        sample_rate_hz=MetadataValue(1e6, "test", MetadataStatus.KNOWN),
        center_frequency_hz=MetadataValue(0, "test", MetadataStatus.KNOWN),
        provenance={},
        diagnostics=[]
    )
    
    # We provide a config with known parameters so hypothesis passes easily
    # Actually, BPSK is easily detected.
    
    # Mock decode_concatenated because the signal isn't actually RS/Viterbi encoded!
    import signal_analysis.pipeline
    original_decode = signal_analysis.pipeline.decode_concatenated
    
    def fake_decode(demod):
        from signal_analysis.models import FECDecodeResult, DeinterleavingResult, DeinterleaverHypothesis, HypothesisStatus, DeinterleaverFamily
        import numpy as np
        none_hyp = DeinterleaverHypothesis(DeinterleaverFamily.NONE, {}, 1.0, [], HypothesisStatus.HYPOTHESIS_UNVERIFIED)
        deint_res = DeinterleavingResult(demod.hard_bits, demod.soft_llrs, none_hyp, 1.0)
        rs_res = FECDecodeResult(demod.hard_bits, 0, 0.0, True, 'NONE', 1.0, [])
        return rs_res, rs_res, deint_res
        
    signal_analysis.pipeline.decode_concatenated = fake_decode
    
    try:
        res = run_full_pipeline(recording)
    finally:
        signal_analysis.pipeline.decode_concatenated = original_decode

    
    assert res.hypothesis_status == PipelineStageStatus.COMPLETED
    assert res.top_hypothesis.label == "BPSK"
    
    assert res.sync_status == PipelineStageStatus.COMPLETED
    
    # FEC may be None/NONE interleaved, but it's completed
    assert res.fec_status == PipelineStageStatus.COMPLETED
    
    assert res.framing_status == PipelineStageStatus.COMPLETED
    assert res.frame_structure is not None
    assert res.frame_structure.header_match.pattern.name == "HDLC_FLAG"
