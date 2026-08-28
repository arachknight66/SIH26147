from __future__ import annotations
from pathlib import Path
import json
import pytest
from app.orchestration.pipeline_config import PresetName, get_preset_config
from app.orchestration.pipeline_runner import run_pipeline
from app.reporting.artifact_manifest import export_all_artifacts
from app.reporting.html_report import build_html_report
from app.reporting.json_report import build_json_report
from tests.test_phase6_cases import _make_rec_sig
from scripts.generate_digital_dataset import generate_digital_stream

def test_json_report_structure(tmp_path: Path):
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))

    data = build_json_report(res)
    assert data["schema_version"] == "1.0"
    assert "input" in data
    assert "phase2_physical" in data
    assert "phase3_modulation" in data
    assert "phase6_verification" in data
    assert data["is_verified"] is True

def test_html_report_generation(tmp_path: Path):
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))

    html = build_html_report(res)
    assert "<!DOCTYPE html>" in html
    assert "SIH26147 Scientific Signal Recovery" in html
    assert "Independent 7-Claim Verification Matrix" in html

def test_export_all_artifacts(tmp_path: Path):
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))

    out_map = export_all_artifacts(res, tmp_path)
    assert Path(out_map["json_report"]).exists()
    assert Path(out_map["html_report"]).exists()
    assert Path(out_map["frames_csv"]).exists()
    assert Path(out_map["parameters_csv"]).exists()
    assert Path(out_map["manifest"]).exists()
