from __future__ import annotations
import pytest
from app.orchestration.state_machine import (
    InvalidStateTransitionError,
    PipelineState,
    PipelineStateMachine,
)

def test_valid_state_transitions():
    sm = PipelineStateMachine()
    assert sm.current_state == PipelineState.IDLE

    sm.transition_to(PipelineState.LOADING)
    assert sm.current_state == PipelineState.LOADING

    sm.transition_to(PipelineState.VALIDATING)
    sm.transition_to(PipelineState.ANALYZING)
    sm.transition_to(PipelineState.CLASSIFYING)
    sm.transition_to(PipelineState.SYNCHRONIZING)
    sm.transition_to(PipelineState.DEMODULATING)
    sm.transition_to(PipelineState.RECONSTRUCTING)
    sm.transition_to(PipelineState.CORRECTING)
    sm.transition_to(PipelineState.VERIFYING)
    sm.transition_to(PipelineState.REPORTING)
    sm.transition_to(PipelineState.COMPLETED)

    assert sm.current_state == PipelineState.COMPLETED
    assert len(sm.history) == 12

def test_invalid_state_transition_raises():
    sm = PipelineStateMachine()
    with pytest.raises(InvalidStateTransitionError):
        sm.transition_to(PipelineState.COMPLETED)

def test_cancel_transition():
    sm = PipelineStateMachine()
    sm.transition_to(PipelineState.LOADING)
    sm.transition_to(PipelineState.CANCELLED)
    assert sm.current_state == PipelineState.CANCELLED

def test_reset_state_machine():
    sm = PipelineStateMachine()
    sm.transition_to(PipelineState.LOADING)
    sm.reset()
    assert sm.current_state == PipelineState.IDLE
    assert len(sm.history) == 1
