# SIH26147 — Phase 5 Technical Specification & Architecture

## Scientific Data-Stream Reconstruction, Framing, Scrambling Analysis, FEC Identification & Error Correction Engine

---

## 1. Overview & Architectural Role

Phase 5 transforms the raw symbol/bit stream recovered by Phase 4 into a structurally interpretable, error-corrected digital stream:

```text
Phase 1 → Signal ingestion and canonical validation
Phase 2 → Quantitative physical measurement (PSD, SNR, Bandwidth)
Phase 3 → Modulation hypothesis generation
Phase 4 → Carrier/Timing synchronization & Demodulation
Phase 5 → Data-stream reconstruction, Framing, Scrambler & FEC Correction
Phase 6 → Cryptographic, Semantic & Independent Verification
```

### Governing Architectural Rules
1. **Phase 2 measures. Phase 3 hypothesizes. Phase 4 recovers. Phase 5 corrects. Phase 6 verifies.**
2. **Phase 5 must correct errors and reconstruct structure, not invent information.**
3. **A syntactically valid payload is not necessarily the correct payload.**
4. **Epistemic Invariant**: Phase 5 distinguishes `OBSERVED`, `INFERRED`, `CORRECTED`, `ASSUMED`, `VERIFIED`, and `UNKNOWN`. Phase 5 never labels output as `VERIFIED` (verification is strictly the domain of Phase 6); Phase 5 outputs `STRUCTURALLY_SUPPORTED`, `CORRECTED`, or `INTEGRITY_SUPPORTED`.
5. **Reversibility**: Every corrected bit, framing boundary, and descrambling operation maintains exact provenance and an explicit boolean `correction_mask`.

---

## 2. Mathematical Formulations & Algorithms

### 2.1 Digital Statistical Characterization
Given bitstream $\mathbf{b} = (b_0, b_1, \dots, b_{N-1}) \in \{0, 1\}^N$:
* **Bit Balance**: $\mu_b = \frac{1}{N} \sum_{i=0}^{N-1} b_i$
* **Transition Probability**: $P_{\text{trans}} = \frac{1}{N-1} \sum_{i=1}^{N-1} [b_i \neq b_{i-1}]$
* **Byte Entropy**: For non-overlapping octets $w_k \in \{0, \dots, 255\}$:
  $$H_{\text{byte}} = -\sum_{v=0}^{255} p(v) \log_2 p(v)$$
* **Conditional Entropy**:
  $$H(X_i | X_{i-1}) = H(X_{i-1}, X_i) - H(X_{i-1})$$

### 2.2 Rotational Ambiguity Resolution
Phase 4 preserves constellation rotational ambiguities. Phase 5 systematically evaluates bounded candidate rotations:
* **BPSK**: $\{0^\circ, 180^\circ\}$ (Normal vs Inverted polarity $b_i \to 1 - b_i, LLR \to -LLR$).
* **QPSK**: $\{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$ quadrant transformations on paired bits $(b_I, b_Q)$.

### 2.3 Synchronization & Preamble Detection
Given candidate pattern $\mathbf{p} = (p_0, \dots, p_{L-1})$:
* **Sliding Hamming Distance**: $d_H(k) = \sum_{j=0}^{L-1} [b_{k+j} \neq p_j]$.
* **Interval Periodicity**: Frame spacing $\mu_{\text{spacing}} = \operatorname{mean}(\Delta k)$, variance $\sigma_{\text{spacing}}^2 = \operatorname{var}(\Delta k)$.

### 2.4 Berlekamp–Massey Linear Complexity
Computes the minimal length $L(\mathbf{s})$ of a Linear Feedback Shift Register (LFSR) that generates sequence $\mathbf{s}$.
For random binary sequences, $E[L(\mathbf{s})] \approx N/2$. A significant reduction in $L(\mathbf{s})$ indicates underlying algebraic structure or synchronous scrambling.

### 2.5 Configurable Parameterized CRC Engine
Supports arbitrary polynomial widths ($W \in \{8, 16, 24, 32\}$), generator polynomials $G(x)$, initial values $I$, final XOR masks $X$, and bit reflections:
* **Multi-Frame Binomial $p$-value**:
  $$P(\ge k \text{ accidental matches out of } N) = \sum_{j=k}^N \binom{N}{j} (2^{-W})^j (1 - 2^{-W})^{N-j}$$

### 2.6 Soft-Decision Viterbi Forward Error Correction
For rate $1/2$ convolutional codes ($K=7$ with $G_1=133_8, G_2=171_8$):
* **Soft Branch Metric**: Given received LLRs $(y_{2t}, y_{2t+1})$ and candidate codeword bits $(c_0, c_1)$:
  $$BM(c_0, c_1) = (y_{2t} - (2 c_0 - 1))^2 + (y_{2t+1} - (2 c_1 - 1))^2$$
* **Traceback & Correction Mask**: Decoded bits $\hat{\mathbf{u}}$ are re-encoded to obtain $\hat{\mathbf{c}} = \operatorname{encode}(\hat{\mathbf{u}})$, yielding the exact correction mask:
  $$\mathbf{m}_{\text{corr}} = [\mathbf{r} \neq \hat{\mathbf{c}}]$$
* **Over-Correction Budget**: If correction fraction $\rho = \frac{1}{N} \sum m_i > \rho_{\text{max}} = 0.10$, the decoder logs `FEC_OVER_CORRECTION` and rejects the hypothesis.

---

## 3. Occam's Razor Complexity & Candidate Ranking

Candidate reconstructions are scored via multi-evidence fusion penalized by model complexity:
$$S_{\text{recon}} = 0.35 S_{\text{framing}} + 0.35 S_{\text{integrity}} + 0.15 S_{\text{FEC}} + 0.15 S_{\text{scrambler}} - C_{\text{complexity}}$$
where:
$$C_{\text{complexity}} = 0.03 \cdot \mathbf{1}_{\text{inverted}} + 0.02 \cdot \mathbf{1}_{\text{offset} > 0} + 0.05 \cdot \mathbf{1}_{\text{line\_code}} + 0.05 \cdot \mathbf{1}_{\text{scrambler}} + (0.05 + 0.15 \rho_{\text{corr}}) \cdot \mathbf{1}_{\text{FEC}}$$

This ensures simpler, direct explanations are preferred over complex cascading hypotheses that achieve comparable metrics only by altering excessive bits.

---

## 4. Phase 6 Verification Handoff Contract

Phase 5 outputs `DataRecoveryAnalysis` containing `Phase6Handoff`:
* `raw_bits`: Unmodified recovered binary stream.
* `corrected_bits`: Post-FEC error corrected binary stream.
* `payload_bytes`: Extracted data payload.
* `correction_masks`: Exact boolean arrays of modified bit indices.
* `crc_parameters`: Polynomial, width, initial value, XOR out, reflections.
* `structural_evidence`: Frame counts, intervals, and $p$-values.
* `assumptions & uncertainties`: Explicitly documented assumptions for Phase 6 independent cryptographic and protocol verification.
