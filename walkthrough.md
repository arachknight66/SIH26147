# SIH26147 — Phase 4 Implementation Walkthrough

## Scientific Carrier Recovery, Timing Synchronization, Demodulation & Signal Recovery Engine

---

## 1. Executive Summary

Phase 4 of the SIH26147 signal intelligence prototype has been implemented and scientifically validated. Following the foundational principle:

> **Phase 2 measures. Phase 3 hypothesizes. Phase 4 recovers. Phase 5 corrects. Phase 6 verifies.**

Phase 4 acts as an empirical receiver test bench. It treats Phase 3 hypotheses as prior receiver configurations, executes non-destructive matched filtering, symbol-timing recovery (Gardner TED), carrier frequency/phase acquisition (Decision-Directed / Costas loops), constellation analysis (EVM, decision margin, cluster analysis), and demodulation (hard slicing, Gray decoding, soft LLR generation).

Importantly, Phase 4 evaluates physical receiver lock quality and promotes superior candidates when Phase 3 hypotheses are suboptimal, while rejecting out-of-distribution signals (AM, FM, GMSK, OFDM, pure noise) as `RECOVERY_INCONCLUSIVE`.

---

## 2. Architecture & Modules Implemented

```
app/recovery/
├── __init__.py               # Public API exports
├── models.py                 # Dataclasses & typed enums
├── preprocessing.py          # Gain normalization & non-destructive conditioning
├── matched_filter.py         # Root Raised Cosine (RRC) FIR design & ISI check
├── fractional_delay.py       # Linear and 4-point Cubic Hermite interpolators
├── frequency_sync.py         # M-th power non-linear CFO estimation & BFSK tracker
├── timing_sync.py            # Gardner TED with 2nd-order PI loop filter (Bn*T, zeta)
├── carrier_sync.py           # Costas / Decision-Directed carrier phase PLL
├── constellation.py          # 1-SPS normalization, EVM %, dB, margin & cluster analysis
├── demodulation.py           # Slicers, Gray mapping, and soft LLR generators
├── quality.py                # Multi-metric composite scoring & windowed temporal stability
├── fsk_receiver.py           # BFSK dual-tone matched correlation receiver
├── psk_receiver.py           # PSK receiver coordinator (BPSK, QPSK, 8-PSK)
├── qam_receiver.py           # 16-QAM decision-directed grid receiver coordinator
├── candidate_search.py       # Candidate extractor & local SPS search grid
├── ranking.py                # Evidence fusion & Wrong Phase 3 Hypothesis Promotion
└── analyzer.py               # recover_signal, recover_candidate, recover_all_regions
```

---

## 3. Scientific Validation & Benchmark Results

### 3.1 Test Suite Status

```text
pytest: 153 passed in 15.62s
```
* **Phase 1 Tests**: 9 passed
* **Phase 2 Tests**: 54 passed
* **Phase 3 Tests**: 41 passed
* **Phase 4 Tests**: 49 passed (Models, Matched filter, Interpolators, Frequency sync, Carrier sync, Timing sync, Constellation/EVM, PSK demod, QAM demod, FSK demod, Quality, Pipeline, 20 Quality Gate Cases)
* **Total Regressions**: **0**

### 3.2 Monte Carlo Scientific Benchmark (`run_phase4_benchmark.py`)

| Experiment | Target Modulation / Impairment | Success Rate | Mean EVM | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Exp A: Clean Recovery** | BPSK | **10/10 (100%)** | **5.0%** | Optimal 2-cluster decision |
| | QPSK | **10/10 (100%)** | **4.9%** | 4-quadrant Gray slicing |
| | 8-PSK | **10/10 (100%)** | **4.9%** | 8 phase sectors |
| | 2-FSK / BFSK | **10/10 (100%)** | **52.6%** | Dual-tone matched correlation |
| | 16-QAM | **10/10 (100%)** | **17.5%** | 16-point grid recovery |
| **Exp B: CFO Sweep** | QPSK ($\Delta f \in [0.0, 0.015]$) | **4/4 (100%)** | **< 6.0%** | Precise $M$-th power estimation |
| **Exp C: Timing Offset** | QPSK ($\tau \in [0.0, 0.65]$ sps) | **4/4 (100%)** | **< 6.5%** | Gardner fractional strobe lock |
| **Exp D: SNR Sweep** | QPSK ($0 \text{ dB} \to 25 \text{ dB}$) | **Monotonic** | Monotonic | Degrades gracefully below 12 dB |
| **Exp H: OOD Rejection** | AM, FM, GMSK, OFDM, NOISE | **5/5 (100%)** | N/A | **100% Inconclusive / Rejected** |
| **Exp I: Hypothesis Promotion** | 16-QAM (favoring QPSK prior) | **100%** | **16.0%** | **Promoted 16-QAM over QPSK prior** |
| **Speed Benchmark** | 16,384 samples end-to-end | N/A | N/A | **862 ms full receiver pipeline** |

---

## 4. CLI Verification

Command line invocation:
```bash
python -m scripts.recover_signal recording.iq --dtype complex64 --dump-bits --dump-symbols
```

Produces structured, audit-ready scientific recovery summaries conforming to Section 85 of the Phase 4 specification.
