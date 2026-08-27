from __future__ import annotations
from dataclasses import dataclass
import importlib.util
import os
import platform
import sys
from typing import Any

@dataclass(frozen=True)
class DependencyInfo:
    name: str
    is_installed: bool
    version: str
    is_required: bool
    description: str

def check_dependency(name: str, is_required: bool, desc: str) -> DependencyInfo:
    spec = importlib.util.find_spec(name)
    installed = spec is not None
    ver = "N/A"
    if installed:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "installed")
        except Exception:
            ver = "installed"
    return DependencyInfo(
        name=name,
        is_installed=installed,
        version=ver,
        is_required=is_required,
        description=desc,
    )

def get_system_environment() -> dict[str, Any]:
    deps = [
        check_dependency("numpy", True, "Numerical computing foundation"),
        check_dependency("scipy", True, "Scientific signal processing and filtering"),
        check_dependency("PySide6", False, "Desktop GUI framework"),
        check_dependency("pyqtgraph", False, "High-performance real-time plotting"),
        check_dependency("matplotlib", False, "Static publication plotting"),
        check_dependency("torch", False, "Optional deep learning acceleration"),
        check_dependency("gnuradio", False, "Optional hardware SDR integration"),
    ]

    return {
        "python_version": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
        "dependencies": [d.__dict__ for d in deps],
        "all_required_present": all(d.is_installed for d in deps if d.is_required),
    }
