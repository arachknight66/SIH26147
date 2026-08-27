from __future__ import annotations
import pytest
from app.deployment.environment import get_system_environment
from app.deployment.diagnostics import run_self_diagnostics
from app.deployment.package_info import __version__, __schema_version__

def test_package_info():
    assert __version__ == "0.7.0"
    assert __schema_version__ == "1.0"

def test_system_environment_diagnostics():
    env = get_system_environment()
    assert "python_version" in env
    assert "platform" in env
    assert "dependencies" in env
    assert env["all_required_present"] is True

def test_self_diagnostics_healthy():
    diag = run_self_diagnostics()
    assert "overall_health" in diag
    assert diag["overall_health"] in ("HEALTHY", "DEGRADED")
    assert diag.get("phase2_measurement") == "PASS"
