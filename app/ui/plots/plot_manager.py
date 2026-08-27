from __future__ import annotations
from typing import Any
import numpy as np

try:
    import pyqtgraph as pg
    from PySide6.QtWidgets import QWidget, QVBoxLayout
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

def downsample_signal(samples: np.ndarray, max_points: int = 10_000) -> np.ndarray:
    """Downsample large 1D sample arrays safely for UI plotting."""
    n = len(samples)
    if n <= max_points:
        return samples
    step = int(np.ceil(n / max_points))
    return samples[::step]
