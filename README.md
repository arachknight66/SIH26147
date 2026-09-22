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

Raw IQ requires explicit import parameters, for example:

```bash
python -m signal_analysis.cli capture.iq --raw-dtype int16 --raw-iq-order iq --sample-rate-hz 1000000
```
