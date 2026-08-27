from __future__ import annotations
from pathlib import Path
import pytest
from app.orchestration.pipeline_config import PresetName, get_preset_config
from app.orchestration.pipeline_runner import run_pipeline
from app.replay.experiment import load_experiment, save_experiment
from app.replay.runner import replay_experiment
from app.replay.comparator import DifferentialStatus, compare_runs
from tests.test_phase6_cases import _make_rec_sig
from scripts.generate_digital_dataset import generate_digital_stream

def test_save_and_load_experiment(tmp_path: Path):
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))

    exp_file = tmp_path / "experiment.json"
    save_experiment(res, exp_file)
    assert exp_file.exists()

    bundle = load_experiment(exp_file)
    assert bundle.reproducibility_hash == res.provenance.reproducibility_hash

def test_differential_compare_identical():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    cfg = get_preset_config(PresetName.FAST_SCREENING)

    run_a = run_pipeline(rec, config=cfg)
    run_b = run_pipeline(rec, config=cfg)

    comp = compare_runs(run_a, run_b)
    assert comp.overall_status == DifferentialStatus.IDENTICAL
    assert comp.first_divergent_stage is None
