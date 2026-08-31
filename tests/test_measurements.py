import pytest
import numpy as np
from signal_analysis.models import SignalRecording, SourceFormat, MetadataValue, MetadataStatus, Severity
from signal_analysis.measurements import compute_psd, detect_clipping

def create_dummy_recording(samples, dtype="complex64", sr=None):
    return SignalRecording(
        samples=samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype=dtype,
        semantic_type="complex_iq",
        sample_rate_hz=MetadataValue(sr, "test", MetadataStatus.KNOWN if sr else MetadataStatus.MISSING),
        center_frequency_hz=MetadataValue(None, "test", MetadataStatus.MISSING),
        provenance={},
        diagnostics=[]
    )

def test_psd_peak_frequency():
    # Known synthetic complex sinusoid at a known normalized frequency
    # e.g., f = 0.2 cycles/sample
    N = 1024
    n = np.arange(N)
    f0 = 0.2
    samples = np.exp(1j * 2 * np.pi * f0 * n).astype(np.complex64)
    rec = create_dummy_recording(samples)
    
    res = compute_psd(rec, nperseg=N)
    
    assert res.freq_unit == "cycles/sample"
    peak_idx = np.argmax(res.psd)
    peak_f = res.frequencies[peak_idx]
    
    # Assert peak within one bin's resolution
    bin_res = 1.0 / N
    assert abs(peak_f - f0) <= bin_res

def test_psd_axis_unit():
    N = 1024
    samples = np.zeros(N, dtype=np.complex64)
    
    # Missing sample rate -> cycles/sample
    rec_missing = create_dummy_recording(samples, sr=None)
    res_missing = compute_psd(rec_missing)
    assert res_missing.freq_unit == "cycles/sample"
    
    # Known sample rate -> Hz
    rec_known = create_dummy_recording(samples, sr=1e6)
    res_known = compute_psd(rec_known)
    assert res_known.freq_unit == "Hz"
    
def test_clipping_detector_int16():
    N = 10000
    samples = np.zeros(N, dtype=np.complex64)
    # saturate 6% of samples
    samples[:600] = 32767 + 0j
    rec = create_dummy_recording(samples, dtype="int16")
    
    frac, diag = detect_clipping(rec)
    assert pytest.approx(frac, abs=1e-4) == 0.06
    assert diag is not None
    assert diag.severity == Severity.ERROR
    assert diag.code == "CLIPPING_DETECTED"
    
    # saturate 0.5% of samples
    samples = np.zeros(N, dtype=np.complex64)
    samples[:50] = 32767 + 0j
    rec = create_dummy_recording(samples, dtype="int16")
    
    frac, diag = detect_clipping(rec)
    assert pytest.approx(frac, abs=1e-4) == 0.005
    assert diag is not None
    assert diag.severity == Severity.WARNING
