# SIH26147 — Phase 3: Scientific Modulation Identification, Feature Analysis & Hypothesis Generation

## 1. Executive Summary & Objective

Phase 3 transforms the validated canonical `SignalRecording` and quantitative `SignalAnalysis` (from Phases 1 and 2) into ranked, evidence-backed modulation hypotheses.

Phase 3 strictly enforces the architectural separation:
> **Phase 2 measures. Phase 3 hypothesizes. Phase 4 recovers. Phase 5 corrects. Phase 6 verifies.**

Phase 3 is **not** a demodulator, **not** a symbol synchronizer, and **not** an FEC decoder. It does not output raw bit streams or commit to a single uncalibrated "truth". Instead, it outputs a structured `ModulationHypothesis` collection with comprehensive physical evidence vectors, uncertainty quantification, temporal window consistency metrics, and explicit rejection outcomes (`UNKNOWN`, `AMBIGUOUS`, `INSUFFICIENT_EVIDENCE`, `UNSUPPORTED`).

---

## 2. Supported Modulation Families & Orders

The Phase 3 engine supports discrimination across the following families and orders:

| Family | Order | Notation | Constellation Geometry | Key Physical Signatures |
| :--- | :--- | :--- | :--- | :--- |
| `FSK` | 2 | BFSK / 2-FSK | Continuous frequency shift ($\pm \Delta f$) | Bimodal instantaneous frequency distribution, constant envelope, $C_{42} \approx 1$. |
| `PSK` | 2 | BPSK | Antipodal real 1D line ($\pm 1$) | $f_{20} = \|C_{20}\| / C_{21} \approx 1.0$, $f_{40} \approx 2.0$, $x^2$ phase collapse. |
| `PSK` | 4 | QPSK | 4-point circular ring ($e^{j(\pi/4 + k\pi/2)}$) | $f_{20} \approx 0.0$, $f_{40} \approx 1.0$, $f_{42} \approx 1.0$, $x^4$ phase collapse, positive excess kurtosis with RRC. |
| `PSK` | 8 | 8-PSK | 8-point circular ring ($e^{j k\pi/4}$) | $f_{20} \approx 0.0$, $f_{40} \approx 0.0$, $f_{42} \approx 1.0$, $x^8$ phase collapse. |
| `QAM` | 16 | 16-QAM | $4 \times 4$ square multi-ring lattice | Multi-ring envelope variation ($R_A > 0.05$), negative envelope excess kurtosis ($\kappa_A \approx -0.70$), $f_{40} \approx 0.68$, $f_{42} \approx 0.55$. |
| `QAM` | 64 | 64-QAM | $8 \times 8$ square multi-ring lattice | Extended multi-ring envelope variation, negative kurtosis, $f_{40} \approx 0.62$, $f_{42} \approx 0.50$. |

---

## 3. Scientific Boundary & Epistemic Taxonomies

To prevent ungrounded inferences from propagating downstream, Phase 3 strictly differentiates its terminology:

* **Observation**: A directly measured sample or spectral property (e.g. envelope variance $\sigma_A^2$, PSD peak count, phase increment histogram).
* **Feature**: A derived mathematical or statistical quantity used for discrimination (e.g. cumulants $C_{20}, C_{40}, C_{42}$, excess kurtosis $\kappa_A$, bimodal prominence).
* **Hypothesis**: A proposed, unverified modulation candidate with explicit evidence and assumptions (e.g. QPSK with score 0.91).
* **Score**: A relative ranking metric in $[0.0, 1.0]$.
* **Quality**: An explicit reliability category (`HIGH`, `MODERATE`, `LOW`) derived from fused evidence, SNR, and sample support.
* **Status**: Explicit epistemic status (`HYPOTHESIS_UNVERIFIED`, `AMBIGUOUS`, `UNKNOWN`, `INSUFFICIENT_EVIDENCE`, `UNSUPPORTED`).

All outputs carry the explicit invariant:
```text
Status: HYPOTHESIS — NOT YET VERIFIED
```

---

## 4. Modulation Feature Extraction Engine

The engine extracts 25 normalized features across 6 independent domains:

### 4.1 Amplitude Features (`app/modulation/amplitude.py`)
For complex discrete samples $x[n]$ with envelope $A[n] = |x[n]|$:
* **Mean & RMS**: $\mu_A = \frac{1}{N}\sum A[n], \quad \text{RMS}_A = \sqrt{\frac{1}{N}\sum A[n]^2}$
* **Normalized Variance**: $R_A = \frac{\sigma_A^2}{\text{RMS}_A^2}$ (discriminates constant envelope PSK/FSK from multi-ring QAM).
* **Envelope Excess Kurtosis**: $\kappa_A = \frac{E[(A - \mu_A)^4]}{\sigma_A^4} - 3.0$ (square QAM constellations have negative excess kurtosis $\kappa_A \approx -0.70$; pulse-shaped PSK has $\kappa_A > 0$).
* **Coefficient of Variation**: $\gamma_A = \sigma_A / \mu_A$.
* **Peak-to-RMS Ratio (Crest Factor)**: $C_A = \max(A) / \text{RMS}_A$.

### 4.2 Phase Features (`app/modulation/phase.py`)
Samples with $A[n] < 0.05 \cdot \text{RMS}_A$ are masked to eliminate noise singularities in $\text{atan2}$.
* **Phase Increments**: $\Delta\phi[n] = \angle(x[n] x^*[n-1])$.
* **$M$-th Power Phase Collapse Variance**:
  Raising complex samples to the $M$-th power multiplies constellation angles by $M$, collapsing symmetric $M$-PSK constellations to a single unmodulated carrier line:
  $$\text{Var}_{\text{circ}}(\angle(x^2)) = 1 - |\text{mean}(e^{j \angle(x^2)})| \quad (\text{collapses BPSK})$$
  $$\text{Var}_{\text{circ}}(\angle(x^4)) = 1 - |\text{mean}(e^{j \angle(x^4)})| \quad (\text{collapses QPSK})$$
  $$\text{Var}_{\text{circ}}(\angle(x^8)) = 1 - |\text{mean}(e^{j \angle(x^8)})| \quad (\text{collapses 8-PSK})$$

### 4.3 Instantaneous Frequency Features (`app/modulation/frequency.py`)
* **Instantaneous Frequency**: $f_{\text{inst}}[n] = \frac{1}{2\pi}\angle(x[n] x^*[n-1])$.
* **FSK Bimodal Histogram Clustering**: Identifies dominant histogram peak pairs in $f_{\text{inst}}$:
  * State separation: $\Delta f_{\text{state}} = |f_2 - f_1|$.
  * State prominence: $P_{\text{bimodal}} = \min(P_1, P_2) / \max(\text{counts})$.
  * State occupancy ratio: $R_{\text{state}} = \min(N_1, N_2) / \max(N_1, N_2)$.

### 4.4 Higher-Order Complex Cumulants (`app/modulation/cumulants.py`)
Numerically stable 2nd and 4th order cumulants for zero-mean complex signals:
* $C_{20} = E[x^2]$ (unconjugated 2nd moment).
* $C_{21} = E[|x|^2]$ (total power $= S + N$).
* $C_{40} = E[x^4] - 3 C_{20}^2$.
* $C_{41} = E[x^3 x^*] - 3 C_{20} C_{21}$.
* $C_{42} = E[|x|^4] - |C_{20}|^2 - 2 C_{21}^2$.

Normalized scale-invariant ratios:
* $f_{20} = |C_{20}| / C_{21}$ (1.0 for BPSK, 0.0 for circular QPSK/8PSK/QAM/FSK).
* $f_{40} = |C_{40}| / C_{21}^2$ (2.0 for BPSK, 0.85–1.0 for QPSK, 0.0 for 8PSK, 0.68 for 16QAM).
* $f_{42} = |C_{42}| / C_{21}^2$ (2.0 for BPSK, 0.8–1.0 for QPSK, 0.8–1.0 for 8PSK, 0.55–0.68 for 16QAM).

### 4.5 Spectral & Cyclostationary Features (`app/modulation/spectral_features.py`, `app/modulation/cyclostationary.py`)
* Reuses Phase 2 Welch PSD, spectral spread, spectral kurtosis, spectral flatness, peak count, and cyclostationary transition line prominence.

---

## 5. Multi-Source Evidence & Fusion Scoring

The ranking engine combines deterministic classical physical evidence with calibrated lightweight ML evidence:

$$S(h) = \left(w_c S_{\text{classical}}(h) + w_{\text{ml}} S_{\text{ML}}(h)\right) \cdot q_{\text{SNR}} - P_{\text{invalid}} - P_{\text{contra}}$$

Where:
* $w_c = 0.55$, $w_{\text{ml}} = 0.45$.
* $q_{\text{SNR}} = \text{clip}((\text{SNR}_{\text{dB}} - 2.0) / 10.0, 0.20, 1.0)$.
* $P_{\text{invalid}}$: Penalty for missing or unreliable feature subsets.
* $P_{\text{contra}}$: Contradiction penalty ($0.15$ per contradictory physical observation, e.g. bimodal frequency states on a candidate PSK signal, or positive kurtosis on candidate 16-QAM).

---

## 6. Ambiguity Logic & Out-of-Distribution (OOD) Rejection

1. **Unknown / OOD Rule**:
   $$\text{If } \max_{h} S(h) < \text{unknown\_threshold} \quad (0.45) \implies \text{Status} = \text{UNKNOWN}$$
   Signals outside target modulations (e.g. AM, FM, GMSK, OFDM, pure AWGN, or severe noise) are rejected cleanly rather than falsely forced into a modulation class.

2. **Ambiguity Rule**:
   $$\text{If } S(h_1) - S(h_2) < \text{ambiguity\_margin} \quad (0.08) \implies \text{Status} = \text{AMBIGUOUS}$$
   When independent evidence does not cleanly separate two competing candidates (e.g. QPSK vs 8-PSK under high noise), both are retained and flagged `AMBIGUOUS`.

---

## 7. Multi-Window Temporal Consistency

The engine partitions the signal into $K=4$ temporal sub-windows and evaluates modulation features across each window. It computes:
$$\text{Window Consistency} = \frac{1}{K}\sum_{k=1}^K \mathbb{I}(\text{Winner}_k == \text{Winner}_{\text{global}})$$
If consistency falls below 60%, the engine emits a `TEMPORAL_NONSTATIONARITY` diagnostic indicating possible burst boundaries, fading variations, or time-varying modulations.

---

## 8. Phase 3 Exit Contract for Phase 4 Handoff

Each `ModulationHypothesis` contains complete candidate parameters for receiver synchronization in Phase 4:
* `modulation_family`: `"FSK"`, `"PSK"`, `"QAM"`
* `modulation_order`: 2, 4, 8, 16
* `candidate_symbol_rate_hz` & `candidate_symbol_rate_normalized`
* `candidate_samples_per_symbol`
* `candidate_center_frequency_hz` & `candidate_center_frequency_normalized`
* `candidate_bandwidth_hz`
* `quality`: `"HIGH"`, `"MODERATE"`, `"LOW"`
* `evidence`: Full 8-domain evidence vector + supporting notes

---

## 9. Explicit Known Physical Limitations

Phase 3 **does NOT guarantee**:
1. Blind identification of unknown spread-spectrum or highly chirped modulations without prior models.
2. Separation of overlapping co-channel signals within the same sub-band (flagged as `POSSIBLE_MULTI_SIGNAL_REGION`).
3. Reliable high-order classification ($M \ge 16$) at low SNR ($< 5\text{ dB}$).
4. Final symbol decoding or bit recovery (reserved for Phase 4 & 5).
