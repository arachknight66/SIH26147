"""Repeatable native microbenchmarks; results are measurements, not acceptance claims."""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np

from signal_analysis.native import require_native, runtime_info


def _median_seconds(function, iterations: int) -> float:
    elapsed = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        elapsed.append(time.perf_counter() - start)
    return float(np.median(elapsed))


def run(samples: int, iterations: int, seed: int) -> dict:
    if samples < 1024 or iterations < 1:
        raise ValueError("samples must be >= 1024 and iterations must be positive")
    native = require_native()
    rng = np.random.default_rng(seed)
    iq = np.ascontiguousarray(
        (rng.standard_normal(samples) + 1j * rng.standard_normal(samples)).astype(np.complex64)
    )
    spectral_config = native.SpectralConfig()
    spectral_config.fft_size = 1024
    spectral_config.hop_size = 512
    spectral_config.max_stft_frames = 16

    def spectral_run():
        analyzer = native.SpectralAnalyzer(spectral_config)
        analyzer.update(iq)
        return analyzer.finish()

    # K=7, r=1/2 needs an even number of soft values.
    llrs = np.ascontiguousarray(rng.standard_normal(samples + samples % 2).astype(np.float32))
    viterbi_seconds = _median_seconds(lambda: native.decode_viterbi_k7_r12(llrs), iterations)
    spectral_seconds = _median_seconds(spectral_run, iterations)
    return {
        "benchmark_schema_version": 1,
        "native": runtime_info().__dict__,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "parameters": {"samples": samples, "iterations": iterations, "seed": seed},
        "measurements": {
            "spectral_seconds_median": spectral_seconds,
            "spectral_samples_per_second": samples / spectral_seconds,
            "viterbi_seconds_median": viterbi_seconds,
            "viterbi_llrs_per_second": len(llrs) / viterbi_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure native SIH26147 spectral and Viterbi kernels")
    parser.add_argument("--samples", type=int, default=65_536)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=26147)
    parser.add_argument("--output", type=str, help="Optional JSON output path")
    args = parser.parse_args()
    report = run(args.samples, args.iterations, args.seed)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
