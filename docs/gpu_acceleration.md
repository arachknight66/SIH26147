# GPU acceleration

CUDA acceleration is optional and uses the `gpu` dependency extra. The default
`cpu` backend keeps all work in the native C++ engine; `auto` uses CUDA where a
validated kernel exists and otherwise retains the explicit CPU stage. A forced
`gpu` request fails if no CUDA/CuPy device can be initialized—there is no silent
CPU downgrade.

```bash
uv sync --extra gpu
python -m signal_analysis.cli capture.iq --raw-dtype int16 --sample-rate-hz 1000000 --compute-backend gpu
```

## Current CUDA coverage

| Pipeline operation | CUDA work | CPU work retained |
|---|---|---|
| GUI PSD/waterfall | windowing, FFT, power reduction | plot rendering and host result display |
| PSK/QAM receiver | post-lock nearest-constellation decisions and max-log LLRs | DC removal, carrier/timing acquisition, matched filtering and lock decision |
| 2/4/8-FSK and MSK receiver | post-lock tone clustering, decisions and LLRs | instantaneous-frequency extraction and timing acquisition |
| Convolutional FEC | K=7, R=1/2 `(171,133)` add-compare-select and traceback | profile selection, cancellation orchestration, result/provenance assembly |
| RS FEC | zero-syndrome screening for complete profiles with up to 32 parity symbols | Berlekamp–Massey/error correction of nonzero-syndrome words |
| Bitstream correlation | sliding Hamming and local-LLR scoring | periodicity evidence and frame/CRC validation |

The GPU and CPU paths share the same public `uint8` hard-bit and `float32`
LLR contracts; positive LLR still means bit 1. GPU receiver decisions use the
native receiver's normalized post-lock symbols, so acquisition evidence remains
the native receiver's evidence. RS words with nonzero syndromes are always
corrected and validated by the existing native decoder; the GPU screen alone is
never treated as a correction result.

## Deliberate limits

The following remain CPU-native: blind/modulation inference, carrier/timing
acquisition, differential PSK decision profiles, GMSK/GFSK, deinterleaving,
RS error correction, LDPC min-sum, CRC verification, framing, and report
serialization. GPU execution does not change validity thresholds, framing
confirmation rules, or coverage bounds. CUDA timing is workload- and transfer-
dependent, so performance claims require a recorded benchmark on the target
capture/device rather than a generic speedup assertion.
