import numpy as np
import pytest
from app.analysis.analyzer import analyze_signal
from app.models.metadata import MetadataSource, MetadataStatus, MetadataValue
from app.models.signal import SignalRecording, SourceFormat
from app.modulation.analyzer import analyze_modulation
from app.modulation.models import ModulationFamily
from app.recovery.analyzer import recover_candidate, recover_signal
from app.recovery.models import RecoveryQualityLevel, RecoveryStatus
from scripts.generate_modulated_dataset import generate_modulated_signal

def _make_rec(samples: np.ndarray) -> SignalRecording:
    meta_sr = MetadataValue(
        value=100000.0,
        source=MetadataSource.USER_INPUT,
        status=MetadataStatus.KNOWN,
        confidence=1.0,
        evidence=["test"],
    )
    return SignalRecording(
        samples=samples.astype(np.complex64),
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
        sample_rate_hz=meta_sr,
    )

def test_recovery_pipeline_qpsk_end_to_end():
    samples, _ = generate_modulated_signal("QPSK", n_symbols=512, samples_per_symbol=8, snr_db=25.0, seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    
    rec_an = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert rec_an.is_recovered is True
    assert rec_an.selected_candidate is not None
    assert rec_an.selected_candidate.family == ModulationFamily.PSK
    assert rec_an.selected_candidate.order == 4
    assert rec_an.recovered_signal is not None
    assert len(rec_an.recovered_signal.hard_bits) > 0

def test_recover_candidate_direct():
    samples, _ = generate_modulated_signal("BPSK", n_symbols=512, samples_per_symbol=8, snr_db=25.0, seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    
    cand = recover_candidate(rec, an, mod_an.hypotheses[0])
    assert cand.status == RecoveryStatus.RECOVERED
    assert cand.family == mod_an.hypotheses[0].family
