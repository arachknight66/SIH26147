from __future__ import annotations
from .package_info import __version__, __schema_version__, __author__, __project_name__
from .environment import get_system_environment
from .diagnostics import run_self_diagnostics

__all__ = [
    "__version__",
    "__schema_version__",
    "__author__",
    "__project_name__",
    "get_system_environment",
    "run_self_diagnostics",
]
