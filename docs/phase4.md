# SIH26147 — Phase 4 Technical Specification & Architecture

## Scientific Carrier Recovery, Timing Synchronization, Demodulation & Signal Recovery Engine

---

## 1. Overview & Architectural Role

Phase 4 bridges quantitative measurement and hypothesis generation (Phases 1–3) and downstream error correction / protocol verification (Phases 5–6):

```text
Phase 1 → Canonical Ingestion & Validation
Phase 2 → Quantitative Measurement (PSD, SNR, Bandwidth, Detection)
Phase 3 → Modulation Hypothesis Generation (Candidate ranking, Multi-evidence fusion)
Phase 4 → Synchronization & Signal Recovery (Carrier lock, Timing lock, Demodulation, EVM)
Phase 5 → Structural Recovery & FEC Correction (Framing, Descrambling, Viterbi/LDPC)
Phase 6 → Cryptographic & Semantic Verification
```

### Governing Architectural Rules
1. **Phase 2 measures. Phase 3 hypothesizes. Phase 4 recovers. Phase 5 corrects. Phase 6 verifies.**
2. **A successful synchronization lock is not equivalent to a correct demodulation.**
3. **Phase 3 hypotheses are priors, not ground truth.** Phase 4 treats candidate hypotheses as receiver configurations to be tested empirically.
4. **Preserve rotational and polarity ambiguities.** Ambiguities (e.g. $\{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$ for QPSK) are retained explicitly for Phase 5 protocol framing rather than arbitrarily chosen.
5. **Measurable Failure States**: When synchronization fails or signals are OOD, the engine emits explicit diagnostics (`CARRIER_UNLOCKED`, `TIMING_UNLOCKED`, `RECOVERY_INCONCLUSIVE`) instead of fabricating plausible-looking bits.

---

## 2. Mathematical Formulations & Loop Physics

### 2.1 Root Raised Cosine (RRC) Matched Filter
For oversampling factor $SPS$ and roll-off $\alpha \in [0.1, 0.9]$:
$$h(t) = \begin{cases} 
1 - \alpha + \frac{4\alpha}{\pi}, & t = 0 \\
\frac{\alpha}{\sqrt{2}} \left[ \left(1 + \frac{2}{\pi}\right)\sin\left(\frac{\pi}{4\alpha}\right) + \left(1 - \frac{2}{\pi}\right)\cos\left(\frac{\pi}{4\alpha}\right) \right], & |t| = \frac{1}{4\alpha} \\
\frac{\sin(\pi t (1-\alpha)) + 4\alpha t \cos(\pi t (1+\alpha))}{\pi t (1 - (4\alpha t)^2)}, & \text{otherwise}
\end{cases}$$
Normalized to unit energy: $\sum_n |h[n]|^2 = 1.0$.

### 2.2 Coarse CFO Estimation ($M$-th Power Non-linear Method)
For $M$-ary PSK candidates ($M=2$ for BPSK, $M=4$ for QPSK, $M=8$ for 8-PSK):
$$y[n] = \left(\frac{x[n]}{|x[n]|}\right)^M$$
$$\Delta\phi[n] = \angle(y[n] \cdot y^*[n-1])$$
$$\hat{f}_{\text{CFO}} = \frac{\operatorname{median}(\Delta\phi)}{2\pi M}$$
Spectral line refinement is obtained via parabolic interpolation on the FFT peak of $y[n]$.

### 2.3 Gardner Timing Error Detector (TED)
Operates at 2 samples per symbol ($T/2$ strobe $x(t_k - T/2)$ and $T$ symbol samples $x(t_k - T), x(t_k)$):
$$e_k = \Re\left\{ x(t_k - T/2) \left[ x^*(t_k - T) - x^*(t_k) \right] \right\}$$
Loop Filter (2nd-order proportional-integral with normalized loop bandwidth $B_n T$ and damping factor $\zeta = 0.707$):
$$\theta = \frac{B_n T}{\zeta + \frac{1}{4\zeta}}, \quad d = 1 + 2\zeta \theta + \theta^2$$
$$K_p = \frac{4 \zeta \theta}{d}, \quad K_i = \frac{4 \theta^2}{d}$$
Timing Lock Criterion: $\operatorname{Var}(e_k) < 0.18$ and eye opening metric $> 0.30$.

### 2.4 Carrier Recovery (Decision-Directed Costas Loop)
Phase error detector on symbol decisions $\hat{a}_k = \operatorname{slice}(z_k)$:
$$e_{\theta, k} = \Im\left\{ z_k \cdot \hat{a}_k^* \right\}$$
2nd-order digital loop filter tracks residual CFO $\Delta f_{\text{res}}$ and carrier phase $\hat{\theta}_k$.
Carrier Lock Criterion: $\operatorname{Var}(e_{\theta, k}) < 0.08 \text{ rad}^2$ and constellation cluster concentration $> 65\%$.

### 2.5 Constellation Extraction, Normalization & EVM
1-SPS extracted symbols normalized by root mean square energy:
$$z_k = \frac{r_k}{\sqrt{\frac{1}{N} \sum_{n=1}^N |r_n|^2}}$$
Error Vector Magnitude against ideal constellation reference points $\{s_k\}$:
$$\text{EVM}_{\text{RMS}} = \sqrt{\frac{\sum_{k=1}^N |z_k - s_k|^2}{\sum_{k=1}^N |s_k|^2}}$$
Reported in linear, percentage ($\text{EVM} \times 100\%$), and dB ($20\log_{10}(\text{EVM})$).

---

## 3. Demodulators & Slicing

| Family | Order | Slicing Rule | Bits/Symbol | Mapping | Ambiguity Set |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BPSK** | 2 | $\Re(z_k) > 0 \to 1$, else $0$ | 1 | Natural | $\{0^\circ, 180^\circ\}$ |
| **QPSK** | 4 | Quadrant slicing on $I, Q$ | 2 | Gray | $\{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$ |
| **8-PSK** | 8 | 8 phase sectors ($45^\circ$) | 3 | Gray | $\{0^\circ, 45^\circ, \dots, 315^\circ\}$ |
| **16-QAM**| 16 | 4-level grid $\{-3, -1, +1, +3\}/\sqrt{10}$ | 4 | Gray | $\{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$ |
| **2-FSK** | 2 | Dual-tone matched filter correlation | 1 | Natural | $\{0^\circ\}$ |

---

## 4. Wrong Phase 3 Hypothesis Promotion

Phase 4 validates Phase 3 candidates empirically.
If Candidate A (highest Phase 3 score) fails to achieve receiver lock ($\text{EVM} > 35\%$, timing loop unlocked), while Candidate B (lower Phase 3 score) achieves clean lock ($\text{EVM} < 15\%$, timing locked), Phase 4 automatically promotes Candidate B as the recovery winner and logs `POSSIBLE_WRONG_MODULATION_HYPOTHESIS`.

---

## 5. Phase 5 & Phase 6 Handoff Interface

Phase 4 outputs `RecoveredSignal` containing:
* `symbols`: 1-SPS complex64 constellation points.
* `hard_bits`: 1D uint8 binary stream.
* `soft_bits`: 1D float32 LLR / distance confidence metrics.
* `symbol_indices`: 1D int32 constellation indices.
* `sample_indices`: Sample timestamps/strobes.
* `rotational_ambiguities_deg`: Exact ambiguity set.
* `bit_polarity_status`: `"unresolved"`.
* `fec_status`: `"not_applied"`.

Phase 5 consumes this output for sync word detection, deframing, descrambling, and FEC error correction.
Phase 6 consumes intermediate receiver metrics for auditability and verification.
