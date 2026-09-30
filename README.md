# Signal Analysis MVP

This project is a 6-layer forensic signal analysis pipeline for terrestrial HF/VHF/UHF `.iq` and `.wav` recordings. It sequentially handles file format ingestion, statistical feature extraction, blind modulation classification, time/phase synchronization, de-interleaving and forward error correction (FEC), and final frame structure recovery. The system is designed around strict epistemic discipline—explicitly gating downstream assumptions based on upstream certainty, rather than silently guessing through ambiguities.

## Installation

The project requires Python 3.10+.

```bash
# Clone the repository
# git clone <repo>
# cd SIH26147

# Install dependencies
pip install -r requirements.txt
```

### Native Phase 1–7 development build

The repository now contains a C++20 core and pybind11 extension. Native code
provides the fixed K=7, rate-1/2 soft Viterbi decoder; recording, preprocessing,
statistics, PSD/STFT, and spectral-band primitives; measured parameter and
modulation analysis; configured synchronization/demodulation receivers; and
configured bitstream transforms, Viterbi, RS, LDPC, correlation, and CRC. Python
keeps profile/orchestration and frame presentation. The native codec paths are
configured decoders, not blind recovery of arbitrary interleavers or code matrices.

```bash
uv sync --extra test
uv run pytest tests/test_native_bindings.py
uv run pytest tests/test_phase2_native.py
uv run pytest tests/test_phase3_native.py tests/test_phase4_native.py
uv run pytest tests/test_phase5_native.py

cmake -S . -B build/native-tests -DSIH_BUILD_PYTHON=OFF -DSIH_BUILD_TESTS=ON
cmake --build build/native-tests --config Release
ctest --test-dir build/native-tests -C Release --output-on-failure

# Reproducible release-gate transcript and portable microbenchmark
python tools/run_validation.py --quick
python benchmarks/native_benchmark.py --samples 65536 --iterations 5 --output build/benchmark.json
```

Install the `gui` extra when the Qt dependencies are not already available:

```bash
uv sync --extra test --extra gui
```

Use `signal_analysis.sources` for bounded native input. Integer PCM/IQ is
centered where required and scaled by its negative full-scale magnitude;
floating-point recordings retain their amplitude. A missing raw-IQ sampling
rate remains missing, so spectral output uses `cycles/sample`. Preview results
include processed/source sample coverage, and full mode streams fixed-size
chunks instead of allocating the complete recording.

### Dependency Notes
- **Core Pipeline (Headless CLI):** Requires `numpy` and `scipy`.
- **GUI Application:** Requires `PySide6` and `pyqtgraph`. 

**Graceful Degradation:** The pipeline is designed to run completely headlessly if GUI dependencies are missing. If `PySide6` is not installed, the `HAS_QT` flag safely disables the GUI paths, allowing the CLI (`cli.py`) to process files and output results as JSON or plain text with zero loss of analytic capability.

## Quickstart

A synthetic test fixture (a clean QPSK WAV file) is provided to quickly test the pipeline.

### Running the GUI

Launch the interactive inspection application:

```bash
python run_gui.py
```
*Note: Once open, click "Open File" and select a `.wav` or `.sigmf-meta` file to process. For stereo WAVs, you will be prompted to clarify if the channels represent left/right audio (`stereo_real`) or complex I/Q (`stereo_iq`).*

### Running the CLI

Run the pipeline in a headless automation mode, outputting structured JSON data for downstream ingestion:

```bash
python -m signal_analysis.cli test_clean_qpsk.wav --wav-stereo-mode stereo_iq --output json
```

JSON output includes a versioned `run_metadata` object containing the native API
version, selected profiles, input coverage, and execution status. It excludes
sample buffers, local paths, and Demo ground truth.

Raw IQ requires an explicit data representation. The sample-rate field defaults
to **10 kS/s**, but is recorded as an **assumption** unless you mark it known
from acquisition metadata:

```bash
python -m signal_analysis.cli capture.iq --raw-dtype int16
python -m signal_analysis.cli capture.iq --raw-dtype int16 --sample-rate-hz 1000000 --sample-rate-status known
```

### GNU Radio preprocessing

Every ordinary GUI, CLI, and Demo import passes through GNU Radio before the
analysis pipeline. The application normalizes the selected analysis channel to
`complex64`, then runs GNU Radio File Source → optional frequency translator /
low-pass filter / rational resampler → File Sink. With no explicit transform
settings it is an auditable pass-through graph. GNU Radio is therefore a
required runtime dependency for normal processing; the sidecar preserves
compatibility with the application's Python virtual environment.

For common SDR raw captures declared as little-endian `complex64` or `int16`
I/Q, the File Source reads the capture directly (and GNU Radio converts `int16`
with an explicit unity scale). This avoids a full-size input staging copy; other
declared representations are converted explicitly before entering the same
flowgraph.

```bash
python -m signal_analysis.cli capture.cf32 --raw-dtype complex64 \
  --sample-rate-hz 2000000 --sample-rate-status known \
  --gnuradio-preprocess --gnuradio-output-rate-hz 10000 \
  --gnuradio-lowpass-cutoff-hz 4500
```

The output rate is configuration-derived and remains labelled `ASSUMED`; it is
not inferred from the samples. WAV and SigMF imports use the same GNU Radio
pass-through path, retaining their imported rate metadata unless a configured
rate conversion is requested.

Frame confirmation is intentionally profile-driven: exploratory sync matches
remain hypotheses. To verify a payload CRC, provide a JSON list with the
expected sync marker, fixed payload size, and CRC. For example:

```json
[
  {
    "header_name": "HDLC_FLAG",
    "payload_bytes": 64,
    "crc_name": "CRC-16/CCITT-FALSE"
  }
]
```

```bash
python -m signal_analysis.cli capture.iq --raw-dtype int16 --sample-rate-hz 1000000 --frame-profiles profiles.json
```

A frame is marked `CONFIRMED` only after multiple distinct fixed-boundary CRC
observations and a verified receiver bit mapping; otherwise it remains an
explicit hypothesis.

The GUI offers the same configured verification flow through **Configure frame
profile…**. Raw files are accepted as `.iq`, `.raw`, or any selected file; the
import dialog requires the acquisition dtype, I/Q ordering, endianness, and
sample rate rather than guessing them from bytes.

Use `--compute-backend auto` to use optional CUDA/CuPy kernels where available,
or `--compute-backend gpu` to require them. CUDA covers PSD/STFT, post-lock
PSK/QAM and configured FSK decision/LLR generation, fixed K=7 Viterbi,
zero-syndrome RS screening, and bitstream correlation. Carrier/timing
acquisition, RS correction, LDPC, framing, and unsupported profiles remain
explicit CPU stages. See [GPU acceleration](docs/gpu_acceleration.md).

For CUDA 13 systems, install the optional CUDA accelerator with:

```bash
uv sync --extra gpu
```

When a CUDA device is available, the GUI automatically runs its PSD and
waterfall/STFT computation on the GPU. CLI/pipeline GPU stages are selected
with `--compute-backend auto` or `gpu`; their CPU boundaries are documented in
[GPU acceleration](docs/gpu_acceleration.md).
