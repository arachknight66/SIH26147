from __future__ import annotations
import pytest
from app.orchestration.pipeline_config import (
    PipelineConfig,
    PresetName,
    get_preset_config,
)

def test_preset_standard():
    cfg = get_preset_config(PresetName.STANDARD_ANALYSIS)
    assert cfg.preset == PresetName.STANDARD_ANALYSIS
    assert cfg.random_seed == 42
    assert cfg.analysis.fft_size == 4096

def test_preset_fast_screening():
    cfg = get_preset_config(PresetName.FAST_SCREENING)
    assert cfg.preset == PresetName.FAST_SCREENING
    assert cfg.verification.temporal_windows == 4

def test_preset_deep_analysis():
    cfg = get_preset_config(PresetName.DEEP_ANALYSIS)
    assert cfg.preset == PresetName.DEEP_ANALYSIS
    assert cfg.verification.temporal_windows == 16
    assert cfg.analysis.fft_size == 8192

def test_preset_forensic_analysis():
    cfg = get_preset_config("forensic_analysis")
    assert cfg.preset == PresetName.FORENSIC_ANALYSIS
    assert len(cfg.verification.boundary_perturbation_offsets) >= 8

def test_config_serialization_and_hash():
    cfg1 = get_preset_config(PresetName.STANDARD_ANALYSIS)
    cfg2 = get_preset_config(PresetName.STANDARD_ANALYSIS)
    cfg3 = get_preset_config(PresetName.FAST_SCREENING)

    h1 = cfg1.compute_hash()
    h2 = cfg2.compute_hash()
    h3 = cfg3.compute_hash()

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64

def test_measurement_config_respects_pipeline_settings():
    cfg = get_preset_config(PresetName.FAST_SCREENING)
    measurement = cfg.to_measurement_config()
    assert measurement.fft_size == cfg.analysis.fft_size
    assert measurement.psd_segment_length == cfg.analysis.welch_segment_length
    assert measurement.max_samples_for_analysis == cfg.limits.max_analysis_samples

def test_modulation_config_is_deterministic_and_respects_candidate_limit():
    cfg = get_preset_config(PresetName.STANDARD_ANALYSIS)
    modulation = cfg.to_modulation_config()
    assert modulation.enable_ml is False
    assert modulation.max_hypotheses == cfg.modulation.max_hypotheses

def test_recovery_config_respects_pipeline_controls():
    cfg = get_preset_config(PresetName.FAST_SCREENING)
    recovery = cfg.to_recovery_config()
    assert recovery.max_candidates == cfg.recovery.max_candidates
    assert recovery.loop_bandwidth == cfg.recovery.loop_bandwidth
    assert recovery.max_recovery_samples == cfg.limits.max_analysis_samples

def test_data_recovery_config_respects_pipeline_controls():
    cfg = get_preset_config(PresetName.STANDARD_ANALYSIS)
    data = cfg.to_data_recovery_config()
    assert data.enable_viterbi is cfg.data_recovery.enable_viterbi
    assert data.enable_descrambler is cfg.data_recovery.enable_descrambler
    assert data.evaluate_all_bit_offsets is cfg.data_recovery.eval_all_bit_offsets
