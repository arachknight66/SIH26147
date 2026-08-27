from __future__ import annotations
import numpy as np
import pytest
from app.orchestration.pipeline_config import PresetName, get_preset_config
from app.orchestration.pipeline_runner import run_pipeline
from app.models.signal import SignalRecording, SourceFormat
from tests.test_phase6_cases import _make_rec_sig
from scripts.generate_digital_dataset import generate_digital_stream

# 1. Clean Protocols A through E via Orchestrator
def test_case_1_orchestrator_protocol_a():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))
    assert res.is_verified is True

def test_case_2_orchestrator_protocol_b():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_B", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))
    assert res.is_verified is True

def test_case_3_orchestrator_protocol_c():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_C", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))
    assert res.is_verified is True

def test_case_4_orchestrator_protocol_d():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_D", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))
    assert res.is_verified is True

def test_case_5_orchestrator_protocol_e():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_E", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))
    assert res.is_verified is True

# 2. Impaired Signals
def test_case_6_orchestrator_fec_under_noise():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_C", num_frames=5, ber=0.005, seed=42)
    rec = _make_rec_sig(rx, soft)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))
    assert res.is_verified is True

def test_case_7_orchestrator_burst_errors():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_C", num_frames=5, burst_error_len=6, seed=42)
    rec = _make_rec_sig(rx, soft)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))
    assert res.is_verified is True

# 3. Adversarial and OOD Signals
def test_case_8_orchestrator_ood_random_rejection():
    rx, soft, _ = generate_digital_stream(protocol="OOD_RANDOM", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))
    assert res.is_verified is False

def test_case_9_orchestrator_pure_noise_rejection():
    noise = np.random.randint(0, 2, 1024, dtype=np.uint8)
    rec = _make_rec_sig(noise)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))
    assert res.is_verified is False

def test_case_10_orchestrator_all_zeros_rejection():
    zeros = np.zeros(512, dtype=np.uint8)
    rec = _make_rec_sig(zeros)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))
    assert res.is_verified is False

# 4. Provenance & Reproducibility
def test_case_11_orchestrator_reproducibility():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    cfg = get_preset_config(PresetName.FAST_SCREENING)
    res1 = run_pipeline(rec, config=cfg)
    res2 = run_pipeline(rec, config=cfg)
    assert res1.provenance.reproducibility_hash == res2.provenance.reproducibility_hash
