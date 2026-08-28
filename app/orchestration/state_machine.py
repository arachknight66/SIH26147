from __future__ import annotations
from enum import Enum
from typing import Set

class PipelineState(str, Enum):
    IDLE = "IDLE"
    LOADING = "LOADING"
    VALIDATING = "VALIDATING"
    ANALYZING = "ANALYZING"
    DETECTING = "DETECTING"
    CLASSIFYING = "CLASSIFYING"
    SYNCHRONIZING = "SYNCHRONIZING"
    DEMODULATING = "DEMODULATING"
    RECONSTRUCTING = "RECONSTRUCTING"
    CORRECTING = "CORRECTING"
    VERIFYING = "VERIFYING"
    REPORTING = "REPORTING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"

# Explicit graph of legal pipeline transitions
LEGAL_TRANSITIONS: dict[PipelineState, Set[PipelineState]] = {
    PipelineState.IDLE: {PipelineState.LOADING, PipelineState.CANCELLED, PipelineState.FAILED},
    PipelineState.LOADING: {PipelineState.VALIDATING, PipelineState.CANCELLED, PipelineState.FAILED},
    PipelineState.VALIDATING: {PipelineState.ANALYZING, PipelineState.CANCELLED, PipelineState.FAILED},
    PipelineState.ANALYZING: {PipelineState.DETECTING, PipelineState.CLASSIFYING, PipelineState.CANCELLED, PipelineState.FAILED},
    PipelineState.DETECTING: {PipelineState.CLASSIFYING, PipelineState.CANCELLED, PipelineState.FAILED},
    PipelineState.CLASSIFYING: {PipelineState.SYNCHRONIZING, PipelineState.DEMODULATING, PipelineState.RECONSTRUCTING, PipelineState.VERIFYING, PipelineState.REPORTING, PipelineState.CANCELLED, PipelineState.FAILED},
    PipelineState.SYNCHRONIZING: {PipelineState.DEMODULATING, PipelineState.RECONSTRUCTING, PipelineState.VERIFYING, PipelineState.REPORTING, PipelineState.CANCELLED, PipelineState.FAILED},
    PipelineState.DEMODULATING: {PipelineState.RECONSTRUCTING, PipelineState.CORRECTING, PipelineState.VERIFYING, PipelineState.REPORTING, PipelineState.CANCELLED, PipelineState.FAILED},
    PipelineState.RECONSTRUCTING: {PipelineState.CORRECTING, PipelineState.VERIFYING, PipelineState.REPORTING, PipelineState.CANCELLED, PipelineState.FAILED},
    PipelineState.CORRECTING: {PipelineState.VERIFYING, PipelineState.REPORTING, PipelineState.CANCELLED, PipelineState.FAILED},
    PipelineState.VERIFYING: {PipelineState.REPORTING, PipelineState.COMPLETED, PipelineState.CANCELLED, PipelineState.FAILED},
    PipelineState.REPORTING: {PipelineState.COMPLETED, PipelineState.CANCELLED, PipelineState.FAILED},
    PipelineState.COMPLETED: {PipelineState.IDLE, PipelineState.LOADING},
    PipelineState.CANCELLED: {PipelineState.IDLE, PipelineState.LOADING},
    PipelineState.FAILED: {PipelineState.IDLE, PipelineState.LOADING},
}

class InvalidStateTransitionError(Exception):
    pass

class PipelineStateMachine:
    def __init__(self, initial_state: PipelineState = PipelineState.IDLE):
        self._current_state = initial_state
        self._history: list[PipelineState] = [initial_state]

    @property
    def current_state(self) -> PipelineState:
        return self._current_state

    @property
    def history(self) -> tuple[PipelineState, ...]:
        return tuple(self._history)

    def transition_to(self, new_state: PipelineState) -> PipelineState:
        allowed = LEGAL_TRANSITIONS.get(self._current_state, set())
        if new_state not in allowed:
            raise InvalidStateTransitionError(
                f"Illegal pipeline state transition: {self._current_state.value} -> {new_state.value}. "
                f"Allowed transitions from {self._current_state.value} are: {[s.value for s in allowed]}"
            )
        self._current_state = new_state
        self._history.append(new_state)
        return self._current_state

    def reset(self) -> None:
        self._current_state = PipelineState.IDLE
        self._history = [PipelineState.IDLE]
