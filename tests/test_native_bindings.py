from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import pytest

from signal_analysis.jobs import AnalysisJob, JobState
from signal_analysis.native import (
    decode_viterbi_k7_r12,
    new_cancellation_token,
    new_progress_state,
    runtime_info,
)


EXPECTED_BITS = np.array([1, 0, 1, 1, 0, 0, 1, 1], dtype=np.uint8)
ENCODED_BITS = np.array(
    [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0], dtype=np.uint8
)


def test_native_runtime_and_fixed_golden_vector():
    assert runtime_info().available
    assert runtime_info().version == "0.2.0.dev1"
    assert runtime_info().api_version == 5
    llrs = np.where(ENCODED_BITS == 1, 8.0, -8.0).astype(np.float32)
    progress = new_progress_state()
    result = decode_viterbi_k7_r12(llrs, progress=progress)

    assert result.execution_status.name == "COMPLETED"
    assert result.decode_validity.name == "UNVERIFIED"
    assert result.consumed_llrs == len(llrs)
    assert result.residual_llrs == 0
    np.testing.assert_array_equal(result.decoded_bits, EXPECTED_BITS)
    assert progress.snapshot == (len(EXPECTED_BITS), len(EXPECTED_BITS))


def test_native_diagnostic_contract_is_typed():
    from signal_analysis.native import require_native

    native = require_native()
    diagnostic = native.Diagnostic()
    diagnostic.severity = native.Severity.WARNING
    diagnostic.code = "PILOT_UNVERIFIED"
    diagnostic.message = "Decoder completion does not verify the candidate."
    diagnostic.evidence = "No frame checksum was supplied."
    assert diagnostic.severity == native.Severity.WARNING
    assert diagnostic.code == "PILOT_UNVERIFIED"


def test_native_orchestration_imports_without_gui_toolkits():
    script = """
import json
import sys
import signal_analysis.native
import signal_analysis.jobs
print(json.dumps(sorted(name for name in sys.modules if name.startswith(('PySide', 'PyQt', 'pyqtgraph')))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_result_array_keeps_native_storage_alive():
    llrs = np.where(ENCODED_BITS == 1, 8.0, -8.0).astype(np.float32)
    result = decode_viterbi_k7_r12(llrs)
    decoded = result.decoded_bits
    del result
    np.testing.assert_array_equal(decoded, EXPECTED_BITS)
    assert decoded.flags.owndata is False


@pytest.mark.parametrize(
    "value, exception",
    [
        (np.ones(16, dtype=np.float64), TypeError),
        (np.ones((8, 2), dtype=np.float32), ValueError),
        (np.ones(32, dtype=np.float32)[::2], ValueError),
        (np.ones(15, dtype=np.float32), ValueError),
        (np.array([1.0, np.nan], dtype=np.float32), ValueError),
    ],
)
def test_invalid_arrays_are_rejected_without_implicit_conversion(value, exception):
    with pytest.raises(exception):
        decode_viterbi_k7_r12(value)


def test_pre_cancelled_decode_reports_unconsumed_input():
    llrs = np.ones(4096, dtype=np.float32)
    cancellation = new_cancellation_token()
    cancellation.cancel()
    result = decode_viterbi_k7_r12(llrs, cancellation=cancellation)
    assert result.execution_status.name == "CANCELLED"
    assert result.consumed_llrs == 0
    assert result.residual_llrs == len(llrs)
    assert result.decoded_bits.size == 0


def test_background_job_progress_and_completion():
    llrs = np.where(np.tile(ENCODED_BITS, 5000) == 1, 3.0, -3.0).astype(np.float32)
    job = AnalysisJob(
        lambda cancellation, progress: decode_viterbi_k7_r12(
            llrs, cancellation=cancellation, progress=progress
        )
    ).start()
    result = job.result(timeout=10)
    snapshot = job.snapshot()
    assert result.execution_status.name == "COMPLETED"
    assert snapshot.state is JobState.COMPLETED
    assert snapshot.progress_fraction == 1.0


def test_native_call_releases_gil():
    llrs = np.ones(2_000_000, dtype=np.float32)
    job = AnalysisJob(
        lambda cancellation, progress: decode_viterbi_k7_r12(
            llrs, cancellation=cancellation, progress=progress
        )
    ).start()
    deadline = time.monotonic() + 2.0
    python_iterations = 0
    observed_native_progress = False
    while job.snapshot().state is JobState.RUNNING and time.monotonic() < deadline:
        python_iterations += 1
        if job.snapshot().completed_units > 0:
            observed_native_progress = True
            job.cancel()
            break
    result = job.result(timeout=5)
    assert python_iterations > 0
    assert observed_native_progress
    assert result.execution_status.name == "CANCELLED"
    assert result.residual_llrs > 0
