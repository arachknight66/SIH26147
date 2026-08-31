
def test_fix1_real_valued_gate():
    import numpy as np
    from signal_analysis.models import SignalRecording, SourceFormat, MetadataValue, MetadataStatus, FeatureValidity
    from signal_analysis.features import extract_all_features
    
    rng = np.random.RandomState(42)
    real_samples = rng.randn(1000).astype(np.complex64)
    recording = SignalRecording(
        samples=real_samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype='float32',
        semantic_type='mono_real',
        sample_rate_hz=MetadataValue(1e6, 'test', MetadataStatus.KNOWN),
        center_frequency_hz=MetadataValue(0, 'test', MetadataStatus.KNOWN),
        provenance={},
        diagnostics=[]
    )
    
    fv = extract_all_features(recording)
    assert fv.phase.validity.value == 'UNAVAILABLE'
    assert fv.cumulant.validity.value == 'UNAVAILABLE'
    assert any(d.code == 'COMPLEX_FEATURES_UNAVAILABLE' for d in fv.diagnostics)
    
def test_fix4_ofdm_plausibility():
    import numpy as np
    from signal_analysis.models import SignalRecording, SourceFormat, MetadataValue, MetadataStatus, FeatureValidity
    from signal_analysis.features import extract_all_features
    
    rng = np.random.RandomState(42)
    # Synthetic OFDM: repeated 64-sample FFT with 16-sample CP
    syms = []
    for i in range(200):
        sym = rng.randn(64) + 1j*rng.randn(64)
        cp = sym[-16:]
        syms.append(cp)
        syms.append(sym)
    ofdm_samples = np.concatenate(syms)

        
    recording = SignalRecording(
        samples=ofdm_samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype='complex64',
        semantic_type='complex_iq',
        sample_rate_hz=MetadataValue(1e6, 'test', MetadataStatus.KNOWN),
        center_frequency_hz=MetadataValue(0, 'test', MetadataStatus.KNOWN),
        provenance={},
        diagnostics=[]
    )
    
    fv = extract_all_features(recording)
    assert any(d.code == 'OFDM_PLAUSIBILITY' for d in fv.diagnostics)
    
    # Check that clean BPSK doesn't trigger it
    bpsk_bits = rng.randint(0, 2, 10000)
    bpsk_syms = np.where(bpsk_bits == 1, 1.0, -1.0).astype(np.complex64)
    bpsk_rec = SignalRecording(
        samples=bpsk_syms,
        source_format=SourceFormat.RAW_IQ,
        original_dtype='complex64',
        semantic_type='complex_iq',
        sample_rate_hz=MetadataValue(1e6, 'test', MetadataStatus.KNOWN),
        center_frequency_hz=MetadataValue(0, 'test', MetadataStatus.KNOWN),
        provenance={},
        diagnostics=[]
    )
    fv_bpsk = extract_all_features(bpsk_rec)
    assert not any(d.code == 'OFDM_PLAUSIBILITY' for d in fv_bpsk.diagnostics)

def test_fix6_truncation_transparency():
    import numpy as np
    from signal_analysis.models import SignalRecording, SourceFormat, MetadataValue, MetadataStatus
    from signal_analysis.features import extract_all_features
    from signal_analysis.constants import DEFAULT_MAX_ANALYSIS_SAMPLES
    
    rng = np.random.RandomState(42)
    long_samples = rng.randn(DEFAULT_MAX_ANALYSIS_SAMPLES + 100).astype(np.complex64)
    recording = SignalRecording(
        samples=long_samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype='float32',
        semantic_type='complex_iq',
        sample_rate_hz=MetadataValue(1e6, 'test', MetadataStatus.KNOWN),
        center_frequency_hz=MetadataValue(0, 'test', MetadataStatus.KNOWN),
        provenance={},
        diagnostics=[]
    )
    
    fv = extract_all_features(recording)
    assert any(d.code == 'TRUNCATED_ANALYSIS' for d in fv.diagnostics)
    
    short_samples = rng.randn(DEFAULT_MAX_ANALYSIS_SAMPLES - 100).astype(np.complex64)
    recording2 = SignalRecording(
        samples=short_samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype='float32',
        semantic_type='complex_iq',
        sample_rate_hz=MetadataValue(1e6, 'test', MetadataStatus.KNOWN),
        center_frequency_hz=MetadataValue(0, 'test', MetadataStatus.KNOWN),
        provenance={},
        diagnostics=[]
    )
    fv2 = extract_all_features(recording2)
    assert not any(d.code == 'TRUNCATED_ANALYSIS' for d in fv2.diagnostics)
