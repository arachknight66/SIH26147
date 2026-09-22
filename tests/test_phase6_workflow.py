import wave

import numpy as np

from signal_analysis.demo import get_demo, reveal_ground_truth
from signal_analysis.native import new_cancellation_token
from signal_analysis.workflow import AnalysisRequest, WorkflowStatus, run_production_analysis


def _write_iq_wav(path):
    n = 4096
    t = np.arange(n, dtype=np.float32)
    samples = np.column_stack((np.cos(0.12 * t), np.sin(0.12 * t)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes((samples * 20_000).astype(np.int16).tobytes())


def test_production_workflow_loads_and_analyzes_wav(tmp_path):
    capture = tmp_path / "capture.wav"
    _write_iq_wav(capture)

    result = run_production_analysis(AnalysisRequest(capture, wav_stereo_mode="stereo_iq"))

    assert result.execution_status is WorkflowStatus.COMPLETED
    assert result.recording is not None
    assert result.pipeline_result is not None
    assert result.recording.semantic_type == "complex_iq"
    assert result.processed_stages == result.total_stages == 3


def test_renaming_capture_does_not_change_production_analysis(tmp_path):
    original = tmp_path / "ordinary_capture.wav"
    renamed = tmp_path / "looks_like_a_demo_fixture.wav"
    _write_iq_wav(original)
    renamed.write_bytes(original.read_bytes())

    first = run_production_analysis(AnalysisRequest(original, wav_stereo_mode="stereo_iq"))
    second = run_production_analysis(AnalysisRequest(renamed, wav_stereo_mode="stereo_iq"))

    assert first.pipeline_result.hypothesis_status == second.pipeline_result.hypothesis_status
    assert first.pipeline_result.sync_status == second.pipeline_result.sync_status
    assert first.pipeline_result.top_hypothesis.label == second.pipeline_result.top_hypothesis.label


def test_cancellation_before_load_has_no_analysis_result(tmp_path):
    token = new_cancellation_token()
    token.cancel()

    result = run_production_analysis(AnalysisRequest(tmp_path / "unused.wav"), cancellation=token)

    assert result.execution_status is WorkflowStatus.CANCELLED
    assert result.recording is None
    assert result.pipeline_result is None


def test_demo_request_contains_no_ground_truth_and_uses_shared_importer():
    fixture = get_demo("structured_bpsk")
    request = fixture.analysis_request()
    truth = reveal_ground_truth(fixture.identifier)

    assert request.origin == "demo"
    assert request.path == fixture.path
    assert request.wav_stereo_mode == "stereo_iq"
    assert "truth" not in request.pipeline_config
    assert truth["evaluation_only"] is True
    assert truth["validated_concatenated"] is False
