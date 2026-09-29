# Representative capture guide

This corpus is evidence, not a fixture generator. Record each capture lawfully, preserve the original bytes, calculate its SHA-256, and put ground truth only in `corpus/truth/*.json`—never in filenames or import settings.

## Recording checklist

For RTL-SDR or HackRF: record raw complex IQ with a documented sample rate, center frequency, gain/AGC state, antenna/front-end, UTC time, receiver/software version, and any filters or frequency correction. For a WAV workflow, record whether channels are mono, stereo-IQ, or a real passband. For SigMF, retain the original `.sigmf-meta` and `.sigmf-data`; do not fill omitted metadata with guesses. Note licensing, redistribution permission, and transmitter/receiver conditions in the manifest.

Use a noise-only capture made with the same receiver settings and no intended transmission for both noise characterization and optional T1 impairment input. It is an unsupported negative, not a representative supported signal.

## Ground truth

Mark each field `KNOWN` only from transmitter/protocol documentation, and `MEASURED_INDEPENDENTLY` only with the method named (for example, an externally decoded known protocol or calibrated counter). Otherwise use `UNKNOWN`. Do not use this engine's SNR, carrier estimate, or label as its own truth.

Potential in-scope signal families are PSK, QAM, and FSK, where the transmission mode is independently documented. Useful unsupported negatives include ADS-B/PPM, APRS/AFSK, POCSAG/FSK, DMR/4FSK, NOAA APT, DVB/ATSC, broadcast AM/FM, and OFDM such as DAB, DVB-T, or LTE. Unsupported does not mean the file is known-good: record the evidence and legal status.

Do not assume a public recording's content or licence. Before adding one, verify its terms and actual signal independently, then put both findings in the manifest.

## Minimum useful T2 collection

Capture independent recordings for each supported family across documented low/mid/high SNR strata, plus at least 368 independently verified unsupported/noise negatives for a zero-failure *two-sided* 95% Clopper--Pearson upper bound below 1% (about 299 for a one-sided bound). Collect roughly 100 held-out supported cases per family/stratum before interpreting a percentage; more may be required by the desired confidence interval. Preserve calibration and held-out allocation from the manifest hash—never redraw it.
