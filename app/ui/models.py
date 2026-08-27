from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
from app.orchestration.pipeline_config import PipelineConfig, PresetName, get_preset_config
from app.orchestration.pipeline_runner import PipelineResult

try:
    from PySide6.QtCore import QObject, Signal
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False
    class QObject:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
    def Signal(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return None

class UIStateModel(QObject):
    """
    Central observable state model for the SIH26147 UI.
    """
    if HAS_PYSIDE6:
        result_updated = Signal(object)
        progress_updated = Signal(object)
        log_message = Signal(str)
        status_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.current_recording_path: str | None = None
        self.config: PipelineConfig = get_preset_config(PresetName.STANDARD_ANALYSIS)
        self.current_result: PipelineResult | None = None
        self.is_running: bool = False
        self.log_history: list[str] = []
        self._listeners: list[Callable[[str, Any], None]] = []

    def set_result(self, result: PipelineResult) -> None:
        self.current_result = result
        self.is_running = False
        if HAS_PYSIDE6 and self.result_updated:
            self.result_updated.emit(result)
        self._notify("result", result)

    def add_log(self, msg: str) -> None:
        self.log_history.append(msg)
        if HAS_PYSIDE6 and self.log_message:
            self.log_message.emit(msg)
        self._notify("log", msg)

    def add_listener(self, cb: Callable[[str, Any], None]) -> None:
        self._listeners.append(cb)

    def _notify(self, event: str, payload: Any) -> None:
        for cb in self._listeners:
            try:
                cb(event, payload)
            except Exception:
                pass
