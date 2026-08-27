from __future__ import annotations
import threading
from typing import Callable

class CancellationToken:
    """Thread-safe cooperative cancellation token."""
    def __init__(self) -> None:
        self._is_cancelled = threading.Event()
        self._callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    def cancel(self) -> None:
        self._is_cancelled.set()
        with self._lock:
            for cb in self._callbacks:
                try:
                    cb()
                except Exception:
                    pass

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled.is_set()

    def check(self) -> None:
        """Raise CancellationError if cancellation was requested."""
        if self._is_cancelled.is_set():
            raise PipelineCancelledError("Operation cancelled by user.")

    def register_callback(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if self._is_cancelled.is_set():
                try:
                    cb()
                except Exception:
                    pass
            else:
                self._callbacks.append(cb)

class PipelineCancelledError(Exception):
    """Raised when a pipeline operation is cooperatively cancelled."""
    pass
