# Phase 6: Independent Scientific Verification, Falsification, Uncertainty Quantification & Reproducibility Engine

---

## 1. Overview & Epistemic Framework

Phase 6 provides sovereign, independent scientific verification and falsification for the entire SIH26147 signal processing pipeline:

```text
Phase 1 → Signal ingestion and canonical validation
Phase 2 → Quantitative physical measurement
Phase 3 → Modulation hypothesis generation
Phase 4 → Carrier/Timing synchronization & Demodulation
Phase 5 → Data-stream reconstruction, Framing, Scrambler & FEC Correction
Phase 6 → Independent verification, falsification, uncertainty quantification and reproducibility
```

### Governing Architectural Axioms
1. **Phase 2 measures. Phase 3 hypothesizes. Phase 4 recovers. Phase 5 corrects. Phase 6 verifies.**
2. **Phase 6 must be capable of rejecting the conclusions of Phases 1–5.**
3. **Epistemic Classification Flow**:
   $$\text{OBSERVED} \longrightarrow \text{INDEPENDENTLY TESTED} \longrightarrow \text{CONSISTENT} \longrightarrow \text{ROBUST} \longrightarrow \text{FALSIFICATION TESTED} \longrightarrow \text{VERIFIED / REJECTED / AMBIGUOUS}$$

---

## 2. Mathematical Formulations

### 2.1 Constellation EVM & 4th-Power Symmetry
For 1-SPS recovered symbols $z_k$:
$$\text{EVM}_{\text{RMS}} = \sqrt{ \frac{\sum_{k=1}^N |z_k - s_k|^2}{\sum_{k=1}^N |s_k|^2} } \times 100\%$$
For $M$-ary PSK, the 4th-power phase concentration metric is:
$$\Gamma_4 = \left| \frac{1}{N} \sum_{k=1}^N \left( \frac{z_k}{\sqrt{P_{\text{avg}}}} \right)^4 \right|$$

### 2.2 Temporal Window Consistency
The signal is partitioned into $N_w$ non-overlapping temporal windows $W_1, \dots, W_{N_w}$. A window $W_j$ passes if $\text{EVM}(W_j) \le 30\%$:
$$f_{\text{window}} = \frac{1}{N_w} \sum_{j=1}^{N_w} \mathbb{I}(\text{EVM}(W_j) \le 30\%) \ge 0.80$$

### 2.3 Frame Interval Stability & Boundary Perturbation
For frame start indices $t_1, t_2, \dots, t_K$, the interval series is $\Delta_i = t_{i+1} - t_i$:
$$\mu_{\Delta} = \frac{1}{K-1} \sum_{i=1}^{K-1} \Delta_i, \quad \sigma_{\Delta} = \sqrt{\frac{1}{K-1} \sum_{i=1}^{K-1} (\Delta_i - \mu_{\Delta})^2}, \quad c_v = \frac{\sigma_{\Delta}}{\mu_{\Delta}} \le 0.05$$
**Boundary Perturbation Falsification**: Deliberate shifts $\delta \in \{\pm 1, \pm 2, \pm 4\}$ bits must cause CRC validity to collapse to 0:
$$\sum_{i=1}^K \text{CRC}(f_i + \delta) = 0, \quad \forall \delta \neq 0$$

### 2.4 FEC Information Gain & Held-Out Cross-Validation
$$\text{Information Gain} = \text{BER}_{\text{before}} - \text{BER}_{\text{after}} \ge 0$$
Frames are split into 70% selection ($S_{\text{sel}}$) and 30% held-out validation ($S_{\text{val}}$). The decoder must independently correct errors on $S_{\text{val}}$.

### 2.5 Multiple-Testing Corrected Null Model Significance
Under a random null model with $M = N_{\text{presets}} \times N_{\text{offsets}} = 64$ tested hypotheses:
$$P_{\text{accidental}} = \sum_{j=k}^N \binom{N}{j} (2^{-W})^j (1 - 2^{-W})^{N-j}$$
$$P_{\text{corrected}} = \min(1.0, M \times P_{\text{accidental}}) < \alpha = 0.01$$

---

## 3. Decision Matrix & Critical Invariants

| Status | Required Conditions |
| :--- | :--- |
| `INDEPENDENTLY_VERIFIED` | 0 Critical Falsifications, EVM $\le 25\%$, $f_{\text{window}} \ge 0.80$, $c_v \le 0.05$, Held-out CRC $p < 0.01$, Boundary perturbation passed, Reproducibility hash matched. |
| `STRONGLY_SUPPORTED` | No critical failures, strong framing / CRC support, minor non-critical warnings ($\le 2$ weak passes). |
| `PARTIALLY_VERIFIED` | Stable framing ($c_v \le 0.05$) without CRC confirmation. |
| `AMBIGUOUS` | Competing modulation or framing hypotheses with comparable evidence. |
| `INSUFFICIENT_EVIDENCE` | Out-of-distribution random bits, pure noise, or $< 16$ bits. |
| `FALSIFIED` / `REJECTED` | Any critical invariant failure (e.g. non-finite samples, boundary perturbation failure, FEC degradation). |
