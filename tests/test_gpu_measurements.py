import numpy as np
import pytest

from signal_analysis.measurements import compute_psd, compute_spectrogram
from signal_analysis.models import MetadataStatus, MetadataValue, SignalRecording, SourceFormat


@pytest.fixture(scope="module")
def gpu_recording():
    pytest.importorskip("cupy")
    import cupy
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device unavailable")
    samples = np.exp(2j * np.pi * 0.125 * np.arange(8192)).astype(np.complex64)
    return SignalRecording(samples, SourceFormat.RAW_IQ, "complex64", "complex_iq",
        MetadataValue(None, "test", MetadataStatus.MISSING), MetadataValue(None, "test", MetadataStatus.MISSING), {}, [])


def test_gpu_psd_matches_cpu_peak(gpu_recording):
    cpu = compute_psd(gpu_recording, 1024, backend="cpu")
    gpu = compute_psd(gpu_recording, 1024, backend="gpu")
    assert gpu.frequencies[np.argmax(gpu.psd)] == pytest.approx(cpu.frequencies[np.argmax(cpu.psd)])
    assert gpu.psd[np.argmax(gpu.psd)] == pytest.approx(cpu.psd[np.argmax(cpu.psd)], rel=0.2)


def test_gpu_spectrogram_returns_bounded_host_image(gpu_recording):
    result = compute_spectrogram(gpu_recording, 256, backend="gpu")
    assert isinstance(result.Sxx, np.ndarray)
    assert result.Sxx.shape[0] == len(result.frequencies)
    assert result.processed_samples == len(gpu_recording.samples)
