"""CUDA implementations for numerically equivalent bounded DSP stages.

This module is intentionally optional: importing it never makes CuPy a base
dependency.  Public functions return NumPy arrays at the Python/native
boundary, preserving the rest of the pipeline's contracts and provenance.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np


def _cp():
    try:
        import cupy  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised on non-CUDA hosts
        raise RuntimeError("CuPy is required for GPU DSP execution") from exc
    return cupy


def _integer_bits(value: int, width: int) -> list[int]:
    return [(value >> (width - bit - 1)) & 1 for bit in range(width)]


def _gray(value: int) -> int:
    return value ^ (value >> 1)


def constellation(modulation: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact native receiver constellation and bit labels."""
    name = modulation.upper()
    if name == "BPSK":
        return np.array([-1.0 + 0j, 1.0 + 0j], np.complex64), np.array([[0], [1]], np.uint8)
    if name in {"QPSK", "OQPSK"}:
        scale = np.float32(1.0 / np.sqrt(2.0))
        return scale * np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], np.complex64), np.array(
            [[1, 1], [0, 1], [0, 0], [1, 0]], np.uint8
        )
    if name == "8PSK":
        points = np.exp(2j * np.pi * np.arange(8, dtype=np.float32) / 8).astype(np.complex64)
        return points, np.array([_integer_bits(_gray(index), 3) for index in range(8)], np.uint8)
    side_by_name = {"16-QAM": 4, "64-QAM": 8, "256-QAM": 16}
    side = side_by_name.get(name)
    if side is None:
        raise ValueError(f"GPU constellation demapper does not support {modulation}")
    axis_bits = int(np.log2(side))
    points: list[complex] = []
    labels: list[list[int]] = []
    for q_row in range(side):
        for i_column in range(side):
            points.append(complex(2 * i_column - side + 1, side - 1 - 2 * q_row))
            q_bits = _integer_bits(_gray(q_row), axis_bits)
            i_bits = _integer_bits(_gray(i_column), axis_bits)
            labels.append([item for pair in zip(q_bits, i_bits) for item in pair])
    point_array = np.asarray(points, np.complex64)
    point_array /= np.sqrt(np.mean(np.abs(point_array) ** 2)).astype(np.float32)
    return point_array, np.asarray(labels, np.uint8)


def gpu_demap_symbols(symbols: np.ndarray, modulation: str, noise_variance: float) -> tuple[np.ndarray, np.ndarray]:
    """Run maximum-likelihood hard decisions and max-log LLRs on CUDA.

    Acquisition/timing and carrier estimates remain those produced by the
    validated native receiver.  This kernel is the decision portion of that
    receiver and uses its normalized post-lock symbols.
    """
    cp = _cp()
    points, labels = constellation(modulation)
    values = cp.asarray(np.ascontiguousarray(symbols, dtype=np.complex64))
    device_points = cp.asarray(points)
    device_labels = cp.asarray(labels)
    # Squared distance avoids a needless sqrt and keeps LLR units identical to
    # the native max-log calculation.
    distance = cp.abs(values[:, None] - device_points[None, :]) ** 2
    nearest = cp.argmin(distance, axis=1)
    hard = device_labels[nearest].reshape(-1)
    variance = np.float32(max(float(noise_variance), 1.0e-8))
    llrs = []
    for bit in range(labels.shape[1]):
        zero = cp.min(distance[:, device_labels[:, bit] == 0], axis=1)
        one = cp.min(distance[:, device_labels[:, bit] == 1], axis=1)
        llrs.append((zero - one) / variance)
    return cp.asnumpy(hard).astype(np.uint8, copy=False), cp.asnumpy(cp.stack(llrs, axis=1).reshape(-1)).astype(np.float32, copy=False)


def gpu_demap_fsk(metrics: np.ndarray, states: int, noise_variance: float) -> tuple[np.ndarray, np.ndarray]:
    """CUDA tone clustering/demapping for the configured FSK receiver output."""
    cp = _cp()
    values = cp.asarray(np.ascontiguousarray(np.real(metrics), dtype=np.float32))
    if values.size == 0:
        return np.zeros(0, np.uint8), np.zeros(0, np.float32)
    # Same quantile initialization and Lloyd nearest-centre refinement used by
    # the native path, but with data-parallel assignment/reduction.
    sorted_values = cp.sort(values)
    centers = sorted_values[cp.asarray(
        [(2 * item + 1) * int(values.size) // (2 * states) for item in range(states)], dtype=cp.int32
    )]
    for _ in range(30):
        assignment = cp.argmin(cp.abs(values[:, None] - centers[None, :]), axis=1)
        counts = cp.bincount(assignment, minlength=states)
        sums = cp.bincount(assignment, weights=values, minlength=states)
        centers = cp.where(counts > 0, sums / cp.maximum(counts, 1), centers)
    order = cp.argsort(centers)
    centers = centers[order]
    bps = int(np.log2(states))
    labels = cp.asarray(np.array([_integer_bits(_gray(item), bps) for item in range(states)], np.uint8))
    distance = (values[:, None] - centers[None, :]) ** 2
    nearest = cp.argmin(distance, axis=1)
    hard = labels[nearest].reshape(-1)
    variance = np.float32(max(float(noise_variance), 1.0e-8))
    llrs = []
    for bit in range(bps):
        llrs.append((cp.min(distance[:, labels[:, bit] == 0], axis=1) - cp.min(distance[:, labels[:, bit] == 1], axis=1)) / variance)
    return cp.asnumpy(hard).astype(np.uint8, copy=False), cp.asnumpy(cp.stack(llrs, axis=1).reshape(-1)).astype(np.float32, copy=False)


@lru_cache(maxsize=1)
def _viterbi_kernel():
    cp = _cp()
    return cp.RawKernel(r'''
    extern "C" __global__ void viterbi_k7(const float* llrs, const int symbols,
                                             unsigned char* predecessor,
                                             unsigned char* decoded, float* margin) {
        __shared__ float metric[64];
        __shared__ float next_metric[64];
        const int state = threadIdx.x;
        metric[state] = state == 0 ? 0.0f : -1.0e30f;
        __syncthreads();
        for (int time = 0; time < symbols; ++time) {
            const int input = state >> 5;
            const int base = (state & 31) << 1;
            const int previous0 = base;
            const int previous1 = base | 1;
            const unsigned int reg0 = (input << 6) | previous0;
            const unsigned int reg1 = (input << 6) | previous1;
            const int sign00 = (__popc(reg0 & 0171) & 1) ? 1 : -1;
            const int sign01 = (__popc(reg0 & 0133) & 1) ? 1 : -1;
            const int sign10 = (__popc(reg1 & 0171) & 1) ? 1 : -1;
            const int sign11 = (__popc(reg1 & 0133) & 1) ? 1 : -1;
            const float first = llrs[2 * time];
            const float second = llrs[2 * time + 1];
            const float candidate0 = metric[previous0] + first * sign00 + second * sign01;
            const float candidate1 = metric[previous1] + first * sign10 + second * sign11;
            // Native traversal reaches lower predecessor first, so retain it
            // on equality for reproducible CPU/GPU traceback decisions.
            const bool choose0 = candidate0 >= candidate1;
            next_metric[state] = choose0 ? candidate0 : candidate1;
            predecessor[time * 64 + state] = (unsigned char)(choose0 ? previous0 : previous1);
            __syncthreads();
            metric[state] = next_metric[state];
            __syncthreads();
        }
        if (state == 0) {
            float best = metric[0], second = -1.0e30f;
            int best_state = 0;
            for (int item = 1; item < 64; ++item) {
                const float value = metric[item];
                if (value > best) { second = best; best = value; best_state = item; }
                else if (value > second) second = value;
            }
            margin[0] = best - second;
            for (int time = symbols - 1; time >= 0; --time) {
                decoded[time] = (unsigned char)(best_state >> 5);
                best_state = predecessor[time * 64 + best_state];
            }
        }
    }
    ''', "viterbi_k7")


def gpu_viterbi_k7_r12(llrs: np.ndarray) -> tuple[np.ndarray, float]:
    """Decode one K=7 R=1/2 LLR stream with the CUDA ACS/traceback kernel."""
    cp = _cp()
    input_llrs = np.ascontiguousarray(llrs, dtype=np.float32)
    if input_llrs.size % 2:
        raise ValueError("K=7 rate-1/2 Viterbi input requires an even number of LLRs")
    if not np.isfinite(input_llrs).all():
        raise ValueError("Viterbi input LLRs must all be finite")
    symbol_count = input_llrs.size // 2
    if symbol_count == 0:
        return np.zeros(0, np.uint8), 0.0
    device_llrs = cp.asarray(input_llrs)
    predecessor = cp.empty((symbol_count, 64), dtype=cp.uint8)
    decoded = cp.empty(symbol_count, dtype=cp.uint8)
    margin = cp.empty(1, dtype=cp.float32)
    _viterbi_kernel()((1,), (64,), (device_llrs, np.int32(symbol_count), predecessor, decoded, margin))
    return cp.asnumpy(decoded), float(cp.asnumpy(margin)[0])


@lru_cache(maxsize=1)
def _rs_syndrome_kernel():
    cp = _cp()
    return cp.RawKernel(r'''
    __device__ unsigned char gf_multiply(unsigned char left, unsigned char right,
                                         const unsigned char* log_table,
                                         const unsigned char* exp_table) {
        if (left == 0 || right == 0) return 0;
        return exp_table[(int)log_table[left] + (int)log_table[right]];
    }
    extern "C" __global__ void rs_syndrome(const unsigned char* words, const int n,
                                             const int parity, const unsigned char* log_table,
                                             const unsigned char* exp_table, unsigned char* invalid) {
        const int word = blockIdx.x;
        const int root = threadIdx.x;
        if (root >= parity) return;
        const unsigned char point = exp_table[root];
        unsigned char value = 0;
        const unsigned char* input = words + word * n;
        for (int item = 0; item < n; ++item) value = gf_multiply(value, point, log_table, exp_table) ^ input[item];
        if (value != 0) invalid[word] = 1;
    }
    ''', "rs_syndrome")


@lru_cache(maxsize=1)
def _gf_tables() -> tuple[np.ndarray, np.ndarray]:
    exp = np.zeros(510, dtype=np.uint8)
    log = np.zeros(256, dtype=np.uint8)
    value = 1
    for index in range(255):
        exp[index] = value
        log[value] = index
        value <<= 1
        if value & 0x100:
            value ^= 0x11D
    exp[255:] = exp[:255]
    return log, exp


def gpu_rs_zero_syndrome(words: np.ndarray, n: int, k: int) -> np.ndarray:
    """CUDA screen for complete RS codewords with zero syndrome.

    It is safe to accept only a zero-syndrome word here.  Non-zero words are
    deliberately sent to the established native correction implementation;
    this avoids treating a GPU pre-screen as an unverified RS decoder.
    """
    cp = _cp()
    values = np.ascontiguousarray(words, dtype=np.uint8)
    if values.size % n:
        raise ValueError("RS GPU screen requires complete codewords")
    count = values.size // n
    if count == 0:
        return np.zeros(0, dtype=bool)
    parity = n - k
    if not 0 < parity <= 32:
        raise ValueError("GPU RS syndrome screen supports RS parity between 1 and 32")
    log, exp = _gf_tables()
    invalid = cp.zeros(count, dtype=cp.uint8)
    _rs_syndrome_kernel()((count,), (32,), (cp.asarray(values), np.int32(n), np.int32(parity), cp.asarray(log), cp.asarray(exp), invalid))
    return cp.asnumpy(invalid) == 0


def gpu_correlate(bits: np.ndarray, llrs: np.ndarray, patterns: Iterable[tuple[str, np.ndarray]], maximum_fraction: float) -> list[tuple[str, int, int, float]]:
    """Compute sliding Hamming and LLR correlation on CUDA.

    Periodicity classification is intentionally left to the existing host
    evidence logic so GPU and CPU use identical scientific thresholds.
    """
    cp = _cp()
    device_bits = cp.asarray(np.ascontiguousarray(bits, dtype=np.uint8))
    device_llrs = cp.asarray(np.ascontiguousarray(llrs, dtype=np.float32)) if len(llrs) == len(bits) else None
    if device_llrs is not None and device_llrs.size:
        # C++ uses nth_element at N//2 (the upper middle value), not the
        # averaged median used by cupy.median for an even-sized stream.
        magnitude = cp.abs(device_llrs)
        median = float(cp.asnumpy(cp.partition(magnitude, magnitude.size // 2)[magnitude.size // 2]))
    else:
        median = 0.0
    output: list[tuple[str, int, int, float]] = []
    for name, raw_pattern in patterns:
        pattern = np.ascontiguousarray(raw_pattern, dtype=np.uint8)
        if pattern.size == 0 or pattern.size > bits.size:
            continue
        device_pattern = cp.asarray(pattern)
        # for binary values, d(a,b)=a+b-2ab; correlate is valid at every bit offset.
        dot = cp.correlate(device_bits.astype(cp.int32), device_pattern.astype(cp.int32), mode="valid")
        distance = cp.cumsum(device_bits.astype(cp.int32))[pattern.size - 1:]
        distance = distance - cp.concatenate((cp.zeros(1, cp.int32), cp.cumsum(device_bits.astype(cp.int32))[:-pattern.size]))
        distance = distance + int(pattern.sum()) - 2 * dot
        maximum = 0 if pattern.size < 16 else int(np.floor(pattern.size * maximum_fraction))
        offsets = cp.asnumpy(cp.nonzero(distance <= maximum)[0])
        if not len(offsets):
            continue
        distances = cp.asnumpy(distance[offsets])
        if device_llrs is not None:
            local = cp.asnumpy(cp.convolve(cp.abs(device_llrs), cp.ones(pattern.size, dtype=cp.float32), mode="valid")[offsets])
        else:
            local = np.zeros(len(offsets), dtype=np.float32)
        for offset, hamming, local_value in zip(offsets, distances, local):
            base = 1.0 - float(hamming) / float(maximum + 1)
            confidence = base * (min(1.0, float(local_value) / (median * pattern.size)) if median > 1.0e-9 else 0.5)
            output.append((name, int(offset), int(hamming), float(confidence)))
    return output
