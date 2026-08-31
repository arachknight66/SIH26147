# Signal Analysis MVP

This project is a 6-layer forensic signal analysis pipeline for terrestrial HF/VHF/UHF `.iq` and `.wav` recordings. It sequentially handles file format ingestion, statistical feature extraction, blind modulation classification, time/phase synchronization, de-interleaving and forward error correction (FEC), and final frame structure recovery. The system is designed around strict epistemic discipline—explicitly gating downstream assumptions based on upstream certainty, rather than silently guessing through ambiguities.

## Installation

The project requires Python 3.8+ (Python 3.11 recommended).

```bash
# Clone the repository
# git clone <repo>
# cd SIH26147

# Install dependencies
pip install -r requirements.txt
```

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
