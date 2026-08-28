from __future__ import annotations
import pytest
from app.orchestration.failure_recovery import (
    FailureCategory,
    classify_stage_failure,
)

def test_classify_low_snr():
    err = ValueError("SNR below detection threshold in noise estimation")
    fail = classify_stage_failure("phase2", err)
    assert fail.category == FailureCategory.LOW_SNR

def test_classify_sync_failure():
    err = RuntimeError("Carrier Costas loop failed to acquire frequency lock")
    fail = classify_stage_failure("phase4", err)
    assert fail.category == FailureCategory.SYNCHRONIZATION_FAILURE

def test_classify_crc_failure():
    err = ValueError("CRC checksum validation failed across all candidate frames")
    fail = classify_stage_failure("phase5", err)
    assert fail.category == FailureCategory.CRC_FAILURE

def test_classify_timeout():
    err = TimeoutError("Stage execution exceeded maximum time limit")
    fail = classify_stage_failure("phase6", err)
    assert fail.category == FailureCategory.TIMEOUT
