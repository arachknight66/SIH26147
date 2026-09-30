# SIH26147 Signal Analysis

An offline signal-recording analysis prototype for complex IQ, WAV, and SigMF
files. The application combines a Python/Qt interface and orchestration layer
with a C++20/pybind11 DSP and decoding core. Analysis produces ranked modulation
hypotheses, synchronization/demodulation evidence, configured FEC results, and
frame candidates. A completed stage or receiver lock does not by itself prove
the modulation, bit mapping, or payload is correct.

Current implementation evidence and remaining gates are tracked in
[progress.md](progress.md). The native migration and beta acceptance criteria
are described separately in [plan.md](plan.md) and [prd.md](prd.md).

## Install

Use Python 3.10+ and [uv](https://docs.astral.sh/uv/). The optional extras add
the GUI, tests, and experimental ML trainer:

```bash
uv sync --extra gui --extra test --extra ml
```

Normal GUI, CLI, and Demo imports also require a GNU Radio Python runtime. GNU
Radio is deliberately installed separately from the application environment;
install a compatible GNU Radio distribution for the host and confirm its
Python can import `gnuradio`. If runtime discovery does not find it, point the
application at that interpreter:

```bash
SIH_GNURADIO_PYTHON=/path/to/gnuradio/python uv run sih26147 capture.sigmf-meta
```

The application routes imports through a GNU Radio File Source → optional
frequency translation/filter/resampler → File Sink flowgraph before analysis.
With no configured transform it is a pass-through graph. GNU Radio is required
for ordinary processing; the native reader/source APIs remain available for
their direct bounded-source workflows.

On CUDA 13 systems, install the optional GPU extra with `uv sync --extra gpu`.
The CPU backend remains the default. See [GPU acceleration](docs/gpu_acceleration.md)
for the supported kernels and CPU boundaries.

## Run

Launch the GUI:

```bash
uv run sih26147-gui
```

Run the headless CLI on a WAV or SigMF metadata file:

```bash
uv run sih26147 test_clean_qpsk.wav --wav-stereo-mode stereo_iq --output json
uv run sih26147 capture.sigmf-meta --output json
```

Two-channel WAVs must be explicitly interpreted as real stereo or complex I/Q.
Raw IQ imports require an explicit dtype, byte order, and I/Q order. The sample
rate defaults to 10 kS/s and is marked `ASSUMED`; mark a rate `known` only when
acquisition metadata supports it:

```bash
uv run sih26147 capture.iq --raw-dtype int16
uv run sih26147 capture.cf32 --raw-dtype complex64 --sample-rate-hz 2000000 \
  --sample-rate-status known --gnuradio-preprocess \
  --gnuradio-output-rate-hz 10000 --gnuradio-lowpass-cutoff-hz 4500
```

For more CLI options, run `uv run sih26147 --help`. JSON output includes
versioned run metadata and coverage/execution information; it omits sample
buffers and Demo ground truth.

## Experimental ML training

The optional ML workflow is a standalone experiment; it does not replace the
production hand-scored classifier. It trains on generated BPSK, QPSK, 8PSK,
16QAM, 64QAM, and 2FSK windows. Its held-out score is synthetic-data evidence,
not real-world multi-class accuracy:

```bash
uv run --extra ml python tools/train_modulation_ml.py --per-class 1200
```

The committed model and report are under `data/dataset_batches/ml/`. Current
metrics and interpretation are in [the training report](data/dataset_batches/ml/training_report.json).

## Dataset WAV files

WAV assets store I and Q in the left and right channels. One recording has an
unknown source sample rate; its WAV header uses a documented 10 kS/s placeholder
for compatibility. See [dataset assets](docs/dataset_assets.md) for sources,
licenses, encodings, sample-rate status, and label limitations. These WAVs are
signal data, not ordinary audio recordings.

## Development checks

```bash
uv run --extra test pytest tests/test_native_bindings.py tests/test_phase2_native.py
uv run --extra test pytest tests/test_phase3_native.py tests/test_phase4_native.py tests/test_phase5_native.py
cmake -S . -B build/native-tests -DSIH_BUILD_PYTHON=OFF -DSIH_BUILD_TESTS=ON
cmake --build build/native-tests --config Release
ctest --test-dir build/native-tests -C Release --output-on-failure
```

The focused commands above do not imply the full release gates have passed.
The frozen 10,000-window suite's historical zero-confirmed-frame gate failed;
post-mitigation S6/full-suite reruns, Windows execution, representative real-RF
validation, and full ordinary-workflow 1-GB processing are still outstanding.
See [progress.md](progress.md) and [known limitations](KNOWN_LIMITATIONS.md).
