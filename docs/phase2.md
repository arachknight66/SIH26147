# SIH26147 — Phase 2: Scientific Signal Visualization, Detection & Parameter Extraction

## 1. Executive Summary & Objective

Phase 2 transforms a validated Phase 1 canonical `SignalRecording` into a structured, quantitatively characterized `SignalAnalysis` object. It provides research-grade measurements, spectral estimations, time-frequency characterizations, candidate region detections, and parameter extraction with rigorous uncertainty quantification and complete provenance tracking.

Phase 2 adheres strictly to the architectural boundary:
> **Phase 2 measures. Phase 3 hypothesizes. Phase 4 recovers. Phase 5 corrects. Phase 6 verifies.**

Phase 2 is **not** a modulation classifier, **not** a demodulator, and **not** a protocol decoder.

---

## 2. Core Scientific Requirements & Epistemic Taxonomies

To prevent ungrounded inferences from propagating downstream, every measurement and estimation explicitly distinguishes its epistemic status:

* **`OBSERVATION` / `MEASURED`**: Directly measured deterministic physical or sample properties (e.g. sample count, DC offset $\mu_I, \mu_Q$, peak power, dynamic range).
* **`ESTIMATED`**: Quantities derived from mathematical estimation algorithms (e.g. Welch PSD, noise floor, 99% OBW, spectral SNR, M2M4 SNR).
* **`INFERRED`**: Quantities derived through structural deduction from metadata (e.g. duration from frame count and sample rate).
* **`ASSUMED`**: Properties provided as explicit external assertions (e.g. user-supplied raw-IQ sample rate).
* **`AMBIGUOUS` / `CANDIDATE`**: Parameter estimates with multiple plausible interpretations (e.g. preliminary symbol rates, multiple emission candidates).
* **`UNAVAILABLE` / `UNKNOWN`**: Quantities missing due to fundamental physical limits or lack of metadata.

Every estimate retains:
```text
value:               Quantitative value or None
unit:                Physical (Hz, s) or normalized (cycles/sample, samples/symbol)
method:              Exact mathematical algorithm used
quality_score:       Confidence/prominence metric in [0.0, 1.0]
uncertainty:         Analytically or empirically bounded error spread
evidence:            Measurable supporting data
assumptions:         Explicit physical and statistical assumptions
provenance:          Execution environment and parameter configuration
status:              Epistemic status enum
```

---

## 3. Fundamental Physical Limitation: Frequency Normalization

In metadata-free raw IQ recordings, waveform samples alone cannot determine physical sampling rate ($F_s$) or RF center frequency ($f_c$).

Therefore, the engine distinguishes:
* **Physical Frequency ($f_{\text{Hz}}$)**: $f_{\text{Hz}} = f_{\text{norm}} \cdot F_s$ (reported **only** when $F_s$ is provided via WAV/SigMF metadata or explicit user configuration).
* **Normalized Frequency ($f_{\text{norm}}$)**: Expressed in $\text{cycles/sample}$, bounded in $[-0.5, +0.5)$ for complex baseband IQ and $[0, 0.5]$ for real signals.
* **Normalized Symbol Rate ($R_{s,\text{norm}}$)**: Expressed in $\text{symbols/sample}$, bounded in $(0, 0.5]$.
* **Samples per Symbol ($\text{SPS}$)**: $\text{SPS} = 1 / R_{s,\text{norm}}$.

If $F_s$ is missing, no Hz value is manufactured, and frequency axes/diagnostics are labeled in `cycles/sample`.

---

## 4. Architecture & Module Structure

```text
app/
├── models/
│   ├── metadata.py        # MetadataValue, Diagnostic, MetadataStatus, DiagnosticSeverity
│   ├── signal.py          # SignalRecording, SourceFormat, IQOrder, Endian
│   └── analysis.py        # SignalAnalysis, TimeStatistics, SpectrumResult, PSDResult,
│                          # SpectrogramResult, NoiseEstimate, DetectedRegion, BandwidthEstimate,
│                          # SNREstimate, FrequencyEstimate, SymbolRateCandidate, ClippingDiagnostics
├── dsp/
│   ├── __init__.py        # Public DSP API exports
│   ├── windowing.py       # Controlled window generation & coherent/noise power gains
│   ├── spectrum.py        # Complex/Real FFT with centered frequency axes and dB floor
│   ├── psd.py             # Welch PSD estimation with two-sided baseband support
│   ├── spectrogram.py     # STFT time-frequency waterfall analysis
│   ├── noise.py           # Robust noise-floor estimation (iterative sigma-clipping, median)
│   ├── detection.py       # Spectral energy detector & time-domain envelope burst detector
│   ├── bandwidth.py       # Multi-method OBW (99% power containment & threshold methods)
│   ├── snr.py             # Multi-method SNR (spectral integration & decision-independent M2M4)
│   ├── frequency.py       # Sub-bin quadratic peak interpolation & phase progression
│   ├── statistics.py      # Time-domain moments, DC offset, IQ balance, clipping
│   ├── autocorrelation.py # FFT-based normalized autocorrelation R_xx[k]
│   ├── rate_estimation.py # Preliminary symbol-rate candidates (cyclostationary transition spectrum)
│   └── pipeline.py        # Core DSP coordinator & structured diagnostics synthesis
└── analysis/
    ├── __init__.py        # Public analysis API
    └── analyzer.py        # analyze_signal(recording, config)
```

---

## 5. Mathematical Definitions & Algorithmic Principles

### 5.1 Time-Domain Moments & DC Offset
For complex discrete samples $x[n] = I[n] + j Q[n]$:
$$\mu_I = \frac{1}{N}\sum_{n=0}^{N-1} I[n], \quad \mu_Q = \frac{1}{N}\sum_{n=0}^{N-1} Q[n], \quad \mu_x = \mu_I + j \mu_Q$$
$$\sigma_I^2 = \frac{1}{N}\sum_{n=0}^{N-1} (I[n] - \mu_I)^2, \quad \sigma_Q^2 = \frac{1}{N}\sum_{n=0}^{N-1} (Q[n] - \mu_Q)^2$$
$$\text{Cov}(I, Q) = \frac{1}{N}\sum_{n=0}^{N-1} (I[n] - \mu_I)(Q[n] - \mu_Q), \quad \rho_{IQ} = \frac{\text{Cov}(I, Q)}{\sigma_I \sigma_Q}$$

DC offset is measured and reported as an observation without destructive filtering.

### 5.2 Phase Statistics & Masking
Phase $\theta[n] = \text{atan2}(Q[n], I[n])$ exhibits high variance and numerical instability when amplitude $|x[n]| \to 0$. To prevent noise singularities from skewing phase statistics, samples with $|x[n]| < 0.05 \cdot \text{RMS}(|x|)$ are masked out. Circular mean and circular variance are computed on the valid subset:
$$\bar{R} = \frac{1}{M}\sum_{m \in \text{valid}} e^{j\theta[m]}, \quad \bar{\theta} = \text{angle}(\bar{R}), \quad \text{Var}_{\text{circ}} = 1 - |\bar{R}| \in [0, 1]$$

### 5.3 Windowing & Normalization
Window $w[n]$ for $n=0, \dots, N-1$:
* **Coherent Gain ($S_1$)**: $S_1 = \frac{1}{N}\sum_{n=0}^{N-1} w[n]$ (used for coherent sinusoidal amplitude scaling).
* **Noise Power Gain ($S_2$)**: $S_2 = \frac{1}{N}\sum_{n=0}^{N-1} w[n]^2$ (used for noise bandwidth normalization).

### 5.4 Welch Power Spectral Density (PSD)
Signal $x[n]$ is partitioned into $K$ segments of length $L$ with overlap $D$. Each segment is windowed and periodogrammed:
$$P_k(f) = \frac{1}{L S_2} \left| \sum_{n=0}^{L-1} x_k[n] w[n] e^{-j 2\pi f n} \right|^2, \quad \hat{P}_{\text{Welch}}(f) = \frac{1}{K}\sum_{k=0}^{K-1} P_k(f)$$
Welch averaging reduces the variance of the spectral estimate by approximately $1/K$ relative to a single raw periodogram, at the cost of frequency resolution $\Delta f \approx 1/L$.

For complex baseband IQ, the two-sided spectrum spanning $[-0.5, +0.5)$ is preserved via `fftshift`.

### 5.5 Robust Noise-Floor Estimation
To avoid assuming that the minimum spectral bin represents pure noise, the engine uses **iterative sigma-clipping with Median Absolute Deviation (MAD)**:
1. Compute median $M = \text{median}(\text{PSD}_{\text{dB}})$ and $\text{MAD} = \text{median}(|\text{PSD}_{\text{dB}} - M|)$.
2. Calculate normal-equivalent standard deviation $\sigma_{\text{MAD}} = 1.4826 \cdot \text{MAD}$.
3. Reject spectral peaks exceeding $M + 3\sigma_{\text{MAD}}$.
4. Iterate until convergence (max 5 iterations) and compute the linear mean of the remaining unrejected noise bins:
   $$N_{\text{floor, lin}} = \frac{1}{|U|} \sum_{k \in U} \text{PSD}[k], \quad N_{\text{floor, dB}} = 10 \log_{10}(N_{\text{floor, lin}})$$
If $<20\%$ of bins remain, the spectrum is flagged with `is_signal_dominated = True` and diagnostic `NOISE_ESTIMATE_UNCERTAIN`.

### 5.6 Signal Detection
* **Spectral Domain**: Bins satisfying $\text{PSD}_{\text{dB}}[k] \ge N_{\text{floor, dB}} + \Delta_{\text{thresh}}$ (default $\Delta = 10\text{ dB}$) are grouped into contiguous runs. Gaps $\le 2$ bins are merged. Runs shorter than 3 bins are filtered as transient noise spikes.
* **Time Domain**: Moving average smoothed power envelope is compared against quiescent background floor to detect burst intervals $[t_{\text{start}}, t_{\text{end}}]$.

### 5.7 Occupied Bandwidth (OBW)
Multi-method estimation provides cross-checking:
* **Method A (99% Power Containment)**: Finds $[f_{\text{low}}, f_{\text{high}}]$ containing 99% of total cumulative PSD power via sub-bin linear interpolation.
* **Method B (Noise Threshold)**: Measures total contiguous frequency span exceeding $N_{\text{floor}} + 6\text{ dB}$.

### 5.8 Signal-to-Noise Ratio (SNR)
* **Spectral Noise Floor SNR**:
  $$P_{\text{total}} = \int_{-0.5}^{0.5} \text{PSD}(f) df = \text{mean}(\text{PSD}), \quad P_{\text{noise}} = N_{\text{floor, lin}} \cdot 1.0$$
  $$P_{\text{signal}} = \max(0, P_{\text{total}} - P_{\text{noise}}), \quad \text{SNR}_{\text{spectral}} = 10 \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right)$$
* **Decision-Independent M2M4 Moment Estimator**:
  For zero-mean circular complex Gaussian noise and constant-modulus signals ($k_s = 1$):
  $$M_2 = E[|x|^2] = S + N, \quad M_4 = E[|x|^4] = 2 M_2^2 - S^2 \implies S = \sqrt{\max(0, 2 M_2^2 - M_4)}, \quad N = M_2 - S$$
  $$\text{SNR}_{M2M4} = 10 \log_{10}(S / N)$$

### 5.9 Sub-Bin Peak Frequency Interpolation
For peak bin $k_{\max}$ with neighbors $\alpha = P_{\text{dB}}[k_{\max}-1]$, $\beta = P_{\text{dB}}[k_{\max}]$, $\gamma = P_{\text{dB}}[k_{\max}+1]$:
$$\delta = \frac{1}{2} \frac{\alpha - \gamma}{\alpha - 2\beta + \gamma}, \quad f_{\text{peak}} = f_{k_{\max}} + \delta \cdot \Delta f$$
Achieves sub-bin frequency accuracy ($< 10^{-5}\text{ cycles/sample}$).

### 5.10 Cyclostationary Symbol-Rate Candidates
Periodic symbol transitions create cyclostationary spectral lines in the derivative transition energy $e[n] = |x[n] - x[n-1]|^2$. Prominent lines in $\text{FFT}(e[n])$ identify candidate symbol rates $R_{s,\text{norm}}$ and $\text{SPS} = 1 / R_{s,\text{norm}}$. All rates are marked `preliminary` / `ambiguous`.

---

## 6. Structured Diagnostics

| Diagnostic Code | Severity | Trigger Condition |
| :--- | :--- | :--- |
| `MISSING_SAMPLE_RATE` | `WARNING` | Absolute sample rate is unavailable in metadata. |
| `MISSING_CENTER_FREQUENCY` | `WARNING` | RF center frequency is unavailable in metadata. |
| `NO_SIGNAL_DETECTED` | `INFO` | No spectral bins exceeded detection threshold above noise floor. |
| `MULTIPLE_SIGNAL_REGIONS` | `INFO` | More than 1 distinct spectral emission detected. |
| `NOISE_ESTIMATE_UNCERTAIN` | `WARNING` | Spectrum is signal-dominated (>80% occupied bins). |
| `BANDWIDTH_ESTIMATE_UNCERTAIN` | `INFO` | 99% power OBW and threshold OBW differ by >20%. |
| `LOW_SIGNAL_TO_NOISE` | `WARNING` | Estimated SNR < 3.0 dB. |
| `SHORT_RECORDING` | `WARNING` | Recording length < 64 samples. |
| `CLIPPING_DETECTED` | `WARNING`/`ERROR` | >0.1% of samples near datatype extrema. |
| `DC_OFFSET_DETECTED` | `INFO` | DC offset magnitude > 5% of RMS amplitude. |
| `IQ_IMBALANCE_INDICATOR` | `INFO` | I/Q power ratio differs by > 3.0 dB from unity. |
| `SYMBOL_RATE_CANDIDATES_AVAILABLE` | `INFO` | 1 or more preliminary symbol rate candidates generated. |

---

## 7. Known Physical Limitations (Explicitly Declared)

Phase 2 **does NOT guarantee**:
1. Universal absolute sample rate recovery from metadata-free raw IQ.
2. Universal RF carrier frequency recovery.
3. Blind source separation of co-channel interfering emissions.
4. Definitive modulation identification (Phase 3 responsibility).
5. Symbol synchronization, constellation recovery, or carrier tracking (Phase 4 responsibility).
6. FEC error correction or framing recovery (Phase 5 responsibility).

---

## 8. Benchmark Results & Verification

Empirical results from 100 Monte Carlo synthetic trials (`python -m scripts.run_phase2_benchmark`):
* **Frequency Estimation Error**: Median error $2.74 \times 10^{-6}\text{ cycles/sample}$, 95th percentile $3.94 \times 10^{-6}\text{ cycles/sample}$.
* **SNR Estimation Bias**:
  * $0.0\text{ dB}$ Target: Mean Error $+0.01\text{ dB}$, Std Dev $0.06\text{ dB}$
  * $5.0\text{ dB}$ Target: Mean Error $-0.00\text{ dB}$, Std Dev $0.05\text{ dB}$
  * $10.0\text{ dB}$ Target: Mean Error $+0.00\text{ dB}$, Std Dev $0.04\text{ dB}$
  * $20.0\text{ dB}$ Target: Mean Error $+0.01\text{ dB}$, Std Dev $0.04\text{ dB}$
* **Detection Performance**: $P_d = 100.0\%$ at $5\text{ dB}$ SNR; $P_{fa} = 0.0\%$ on pure AWGN.
* **Execution Speed (100k samples)**: Welch PSD $7.2\text{ ms}$, FFT $0.6\text{ ms}$, full analysis pipeline $68.2\text{ ms}$.
* **Test Suite**: 63/63 tests passing with 100% deterministic reproducibility.
