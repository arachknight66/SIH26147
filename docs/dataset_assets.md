# Dataset assets

The following generated/converted artifacts are versioned in this repository
because they are directly used by the current data and training work. Their
recordings remain source data, not project-created captures.

| Asset | Source / license | Encoding and rate | Label status |
| --- | --- | --- | --- |
| `data/dataset_batches/indoor-jamming/selected/w1_nojamming_70000000.iq` | [Zenodo 7119040](https://zenodo.org/records/7119040), CC BY 4.0 | complex float32 LE, rate unknown | Source documents BPSK |
| `data/dataset_batches/indoor-jamming/selected/w1_nojamming_70000000.wav` | Same source | stereo float32; I/Q channels; WAV rate is an explicit 10 kS/s placeholder because source rate is unknown | Source documents BPSK |
| `data/dataset_batches/satnogs-danuri-2022-08-07/camras-2022_08_07_08_31_34_2260.830MHz_0.5Msps_ci16_le.wav` | [CAMRAS archive](https://data.camras.nl/satellites/raw/), CC BY 4.0; © Stichting CAMRAS | stereo PCM16; I/Q channels; 500 kS/s | Modulation unknown |
| `data/real/camras-danuri-2022-08-17/camras-2022_08_17_11_36_24_2260.830MHz_0.5Msps_ci16_le.wav` | [CAMRAS archive](https://data.camras.nl/satellites/raw/), CC BY 4.0; © Stichting CAMRAS | stereo PCM16; I/Q channels; 500 kS/s | Modulation unknown |
| `data/zenodo-13371136-intelsat37e/gr4-packet-modem-intelsat37e-test.wav` | [Zenodo 13371136](https://zenodo.org/records/13371136), CC BY 4.0 | stereo float32; I/Q channels; 500 samples/s per source metadata | Modulation unknown |
| `data/dataset_batches/ml/synthetic_modulation_extratrees.joblib` and `training_report.json` | Generated locally by `tools/train_modulation_ml.py` | ExtraTrees model and synthetic holdout report | Synthetic-only multi-class training |

For each WAV, channel 1 stores I and channel 2 stores Q. The corresponding
`.wav.json` sidecar records the encoding and sample-rate status. WAV readers may
play these as audio; they are intended to be read as two-channel signal data.
The real BPSK rate is unknown, so consumers should prefer the `.iq` file when
they can accept a complex-float stream with an externally supplied rate.

The much larger `w1.mat` download is incomplete and intentionally excluded.
CAMRAS and Intelsat files do not have verified modulation labels and are not
included in model accuracy metrics. The committed model is an experimental
synthetic-trained artifact; it is not wired into the production classifier.
