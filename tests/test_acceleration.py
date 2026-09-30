import pytest

from signal_analysis.acceleration import GPUBackendUnavailableError, detect_gpu_capability, require_supported_backend


def test_cpu_backend_is_always_available():
    capability = require_supported_backend("cpu")
    assert isinstance(capability.available, bool)


def test_forced_gpu_never_silently_uses_cpu():
    capability = detect_gpu_capability()
    if capability.available:
        assert require_supported_backend("gpu").available
    else:
        with pytest.raises(GPUBackendUnavailableError):
            require_supported_backend("gpu")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="cpu, gpu, or auto"):
        require_supported_backend("tpu")
