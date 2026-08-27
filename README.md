# SIH26147 — Signal Input Foundation

Phase 1 provides faithful, testable ingestion of WAV, raw IQ, and common SigMF recordings into a canonical `numpy.complex64` `SignalRecording`.

## Setup and use

```powershell
python -m pip install -e ".[dev]"
pytest
python -m scripts.inspect_signal recording.iq --dtype int16 --iq-order IQ --endian little
python -m scripts.inspect_signal recording.wav --stereo-iq
python -m scripts.generate_test_data datasets/test
```

For metadata-free raw IQ, pass an explicit interpretation. Without one, the CLI reports deterministic *format plausibility candidates*, not format identification.

Raw waveform samples alone do not generally determine an absolute physical sampling frequency or RF center frequency. Both remain `None` unless provided by file metadata or explicit user input.

See [Phase 1 documentation](docs/phase1.md).
