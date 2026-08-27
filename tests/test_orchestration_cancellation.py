from __future__ import annotations
import pytest
from app.orchestration.cancellation import (
    CancellationToken,
    PipelineCancelledError,
)

def test_cancellation_token_default():
    token = CancellationToken()
    assert token.is_cancelled is False
    token.check() # Should not raise

def test_cancellation_token_cancel():
    token = CancellationToken()
    called = []
    token.register_callback(lambda: called.append(True))
    assert len(called) == 0

    token.cancel()
    assert token.is_cancelled is True
    assert len(called) == 1

    with pytest.raises(PipelineCancelledError):
        token.check()

def test_cancellation_callback_after_cancel():
    token = CancellationToken()
    token.cancel()
    called = []
    token.register_callback(lambda: called.append(True))
    assert len(called) == 1
