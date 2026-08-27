from __future__ import annotations
import numpy as np
import pytest
from app.orchestration.pipeline_config import PresetName, get_preset_config
from app.orchestration.pipeline_runner import run_pipeline
from app.models.signal import SignalRecording, SourceFormat
from tests.test_phase6_cases import _make_rec_sig
from scripts.generate_digital_dataset import generate_digital_stream

def test_run_pipeline_protocol_a():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    cfg = get_preset_config(PresetName.FAST_SCREENING)
    res = run_pipeline(rec, config=cfg)

    assert res.is_success is True
    assert res.is_verified is True
    assert res.phase1_result.success is True
    assert res.phase2_result.success is True
    assert res.phase3_result.success is True
    assert res.phase4_result.success is True
    assert res.phase5_result.success is True
    assert res.phase6_result.success is True
    assert res.provenance is not None
    assert len(res.provenance.reproducibility_hash) == 64

def test_run_pipeline_pure_noise_rejected():
    noise = np.random.randint(0, 2, 1024, dtype=np.uint8)
    rec = _make_rec_sig(noise)
    cfg = get_preset_config(PresetName.FAST_SCREENING)
    res = run_pipeline(rec, config=cfg)

    assert res.is_verified is False
