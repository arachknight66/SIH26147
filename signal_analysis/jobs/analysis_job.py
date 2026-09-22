"""Threaded orchestration without Qt or Python callbacks from native code."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Callable, Generic, TypeVar

from signal_analysis.native import new_cancellation_token, new_progress_state

ResultT = TypeVar("ResultT")
Worker = Callable[[Any, Any], ResultT]


class JobState(Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class JobSnapshot:
    state: JobState
    completed_units: int
    total_units: int
    progress_fraction: float
    error: str | None


class AnalysisJob(Generic[ResultT]):
    """One-shot background job using native cancellation and polled progress."""

    def __init__(self, worker: Worker[ResultT]) -> None:
        self._worker = worker
        self._cancellation = new_cancellation_token()
        self._progress = new_progress_state()
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[ResultT] | None = None
        self._state = JobState.CREATED
        self._error: str | None = None
        self._lock = Lock()

    def start(self) -> "AnalysisJob[ResultT]":
        with self._lock:
            if self._state is not JobState.CREATED:
                raise RuntimeError("AnalysisJob is one-shot and has already been started")
            self._state = JobState.RUNNING
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sih-native")
            self._future = self._executor.submit(self._run)
        return self

    def _run(self) -> ResultT:
        try:
            result = self._worker(self._cancellation, self._progress)
        except BaseException as exc:
            with self._lock:
                self._state = JobState.FAILED
                self._error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            with self._lock:
                status = getattr(getattr(result, "execution_status", None), "name", None)
                self._state = JobState.CANCELLED if status == "CANCELLED" else JobState.COMPLETED
            return result
        finally:
            executor = self._executor
            if executor is not None:
                executor.shutdown(wait=False)

    def cancel(self) -> None:
        self._cancellation.cancel()

    def snapshot(self) -> JobSnapshot:
        completed, total = self._progress.snapshot
        with self._lock:
            return JobSnapshot(
                state=self._state,
                completed_units=int(completed),
                total_units=int(total),
                progress_fraction=float(self._progress.fraction),
                error=self._error,
            )

    def result(self, timeout: float | None = None) -> ResultT:
        future = self._future
        if future is None:
            raise RuntimeError("AnalysisJob has not been started")
        return future.result(timeout=timeout)

    @property
    def done(self) -> bool:
        """Whether the worker has reached a terminal state."""
        future = self._future
        return future is not None and future.done()
