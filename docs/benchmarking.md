# SIH26147 — Benchmarking Guide

## Comprehensive Benchmark & Replay Methodology

---

## 1. Running the System Benchmark Suite

```bash
python -m scripts.run_full_benchmark
```

### Benchmark Metric Groups:
1. **Clean Protocol Recovery Rate**: Verification accuracy across Protocols A through E.
2. **Adversarial & OOD Rejection**: Rejection rates on pure noise, constant signals, and randomized streams.
3. **Latency Profile**: Execution duration across `FAST_SCREENING`, `STANDARD_ANALYSIS`, and `DEEP_ANALYSIS` presets.

---

## 2. Generating Demonstration Datasets

```bash
python -m scripts.generate_demo_dataset
```

Outputs sample recordings to `examples/`:
* `clean_qpsk.iq`
* `noisy_qpsk_fec.iq`
* `scrambled_frame.iq`
* `pure_noise.iq`
* `adversarial_random.iq`
