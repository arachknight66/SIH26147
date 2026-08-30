from __future__ import annotations
import numpy as np
import pytest
from app.orchestration.pipeline_config import PresetName, get_preset_config
from app.orchestration.pipeline_runner import run_pipeline
from app.reporting.json_report import build_json_report

def test_provenance_and_data_truth_architecture():
    """Verify that JSON report adheres strictly to the Data-Truth Architecture with explicit provenance."""
    result = run_pipeline("examples/clean_qpsk.iq", config=get_preset_config(PresetName.FAST_SCREENING))
    report = build_json_report(result)

    assert report["schema_version"] == "1.0"
    assert report["is_success"] is True

    # 1. Source Provenance
    inp = report["input"]
    assert inp["source_path"] == "examples/clean_qpsk.iq"
    assert inp["format"] in ("raw_iq", "iq", "complex_iq")
    assert inp["is_calibrated_power"] is False
    assert "dBFS" in inp["power_unit"]
    assert inp["is_simulation"] is True

    # 2. Physical Measurements
    p2 = report["phase2_physical"]
    assert p2["snr_provenance"] in ("ESTIMATED", "CALCULATED")
    assert p2["noise_floor_provenance"] == "ESTIMATED"
    assert isinstance(p2["rms_amplitude"], float)
    assert isinstance(p2["crest_factor_db"], float)

    # 3. Modulation Inference
    p3 = report["phase3_modulation"]
    assert p3["winner"] is not None
    for hyp in p3["hypotheses"]:
        assert "evidence" in hyp
        assert "score" in hyp

    # 4. Synchronization & Demodulation
    p4 = report["phase4_recovery"]
    assert p4["lock_status"] != "unknown"
    assert p4["evm_percent"] is not None

    # 5. Spectrogram Matrix
    plots = report["plots"]
    spectro = plots["spectrogram"]
    assert spectro["available"] is True
    assert len(spectro["matrix"]) > 0
    assert len(spectro["matrix"][0]) > 0
    assert spectro["min_dbfs"] <= spectro["max_dbfs"]

    # 6. Verification Claims
    p6 = report["phase6_verification"]
    assert len(p6["claims"]) > 0
    assert len(p6["tests"]) > 0
    for claim in p6["claims"]:
        assert "claim_id" in claim
        assert "confidence" in claim
        assert "independence" in claim

    # 7. Explicit Limitations
    assert len(report["limitations"]) >= 2
    assert any("uncalibrated" in lim.lower() for lim in report["limitations"])

def test_empty_uncomputed_truthfulness():
    """Verify that uncomputed parameters return None/UNAVAILABLE rather than fabricated values."""
    # Run on pure noise where carrier lock and CRC do not succeed
    result = run_pipeline("examples/pure_noise.iq", config=get_preset_config(PresetName.FAST_SCREENING))
    report = build_json_report(result)

    # Recovery and verification should honestly report unverified / unlocked state
    assert report["is_verified"] is False
    assert report["phase6_verification"]["status"] in ("unknown", "unverified", "rejected", "inconclusive")

    p5 = report["phase5_data"]
    # Should not fabricate valid CRC frames on random noise
    assert len(p5["frames_list"]) == 0 or all(not f["is_crc_valid"] for f in p5["frames_list"])
