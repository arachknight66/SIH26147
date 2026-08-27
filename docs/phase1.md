# Phase 1: scientific input contract

`SignalRecording.samples` is always `numpy.complex64`. Real WAV modes produce an all-zero imaginary component. This provides one downstream contract without asserting that every input is IQ. Conversion preserves order and amplitude; it never normalizes, filters, clips, resamples, or discards channels. Integer values are represented exactly as float32 where possible.

Every physical field is a `MetadataValue(value, source, status, confidence, evidence)`. `confidence` describes confidence in that stated value under its stated basis; it is not a probability that an unsupported inference is physically true. Missing raw-IQ sample rate and RF frequency are explicitly `None`.

Supported input modes:

- Raw: `complex64`, interleaved `float32`, `int8`, `int16`, and `uint8`, with explicit I/Q order and endian.
- WAV: uncompressed PCM 8/16/24/32-bit mono or stereo. Stereo is `stereo_real` unless `stereo_iq` is explicitly selected.
- SigMF: `cf32_le`, `ci16_le`, `ci8`, `cu8` with a `.sigmf-meta` companion file.

The raw reader exposes `sample_count` and `read_chunk(start, count)` using a memory map. `read()` is intended for moderate files; later streaming DSP can use chunking without changing file-specific logic.

Raw forensics uses file-size compatibility, finite ratio, and variance to produce deterministic format-plausibility scores. These are rankings only: ties or close candidates must remain ambiguous. It does not infer sample rate, RF frequency, modulation, or physical certainty.

The provenance record includes source path, file size, loader, and conversion. SHA-256 is optional for raw IQ (`RawIQConfig(compute_hash=True)`) to avoid imposing latency on large recordings.

Tests generate deterministic temporary recordings and cover integer round trips, endian/IQ ordering, malformed scalar counts, chunk reading, WAV semantic policy, and deterministic forensics. Quantized values should be compared with tolerances when generated from floating-point signals; no universal unit-power normalization is applied.
