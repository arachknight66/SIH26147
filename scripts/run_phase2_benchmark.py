from __future__ import annotations
import time
import numpy as np
from app.analysis.analyzer import AnalysisConfig, analyze_signal
from app.dsp.detection import detect_signal_regions_spectral
from app.dsp.frequency import estimate_frequency_spectral_peak
from app.dsp.noise import estimate_noise_floor
from app.dsp.psd import compute_psd
from app.dsp.snr import estimate_snr_spectral
from app.dsp.spectrum import compute_spectrum
from app.models.metadata import MetadataSource, MetadataStatus, MetadataValue
from app.models.signal import Endian, IQOrder, SignalRecording, SourceFormat

def run_benchmark() -> None:
    print("=" * 60)
    print("SIH26147 PHASE 2 SCIENTIFIC DSP BENCHMARK")
    print("=" * 60)

    np.random.seed(42)
    n_trials = 100
    n_samples = 16384

    # -------------------------------------------------------------
    # 1. Frequency Estimation Benchmark
    # -------------------------------------------------------------
    print("\n1. FREQUENCY ESTIMATION ACCURACY")
    print("-" * 40)
    freq_errors: list[float] = []
    
    for _ in range(n_trials):
        true_freq = float(np.random.uniform(-0.4, 0.4))
        snr_db = 15.0
        
        t = np.arange(n_samples)
        signal = np.exp(2j * np.pi * true_freq * t).astype(np.complex64)
        noise_p = 1.0 / (10.0 ** (snr_db / 10.0))
        noise = (np.random.normal(0, np.sqrt(noise_p / 2), n_samples) + 1j * np.random.normal(0, np.sqrt(noise_p / 2), n_samples)).astype(np.complex64)
        x = signal + noise
        
        psd_res = compute_psd(x, segment_length=4096, is_complex=True)
        est = estimate_frequency_spectral_peak(psd_res)
        
        if est.normalized_frequency is not None:
            err = abs(est.normalized_frequency - true_freq)
            freq_errors.append(err)

    freq_errors_arr = np.array(freq_errors)
    print(f"Trials:            {len(freq_errors)}")
    print(f"Median error:      {np.median(freq_errors_arr):.6e} cycles/sample")
    print(f"Mean error:        {np.mean(freq_errors_arr):.6e} cycles/sample")
    print(f"95th percentile:   {np.percentile(freq_errors_arr, 95):.6e} cycles/sample")
    print(f"Max error:         {np.max(freq_errors_arr):.6e} cycles/sample")

    # -------------------------------------------------------------
    # 2. SNR Estimation Benchmark
    # -------------------------------------------------------------
    print("\n2. SNR ESTIMATION ACCURACY & BIAS")
    print("-" * 40)
    target_snrs = [0.0, 5.0, 10.0, 20.0]
    
    for target in target_snrs:
        snr_errors: list[float] = []
        for _ in range(50):
            t = np.arange(n_samples)
            signal = np.exp(2j * np.pi * 0.125 * t).astype(np.complex64)
            noise_p = 1.0 / (10.0 ** (target / 10.0))
            noise = (np.random.normal(0, np.sqrt(noise_p / 2), n_samples) + 1j * np.random.normal(0, np.sqrt(noise_p / 2), n_samples)).astype(np.complex64)
            x = signal + noise

            psd_res = compute_psd(x, segment_length=4096, is_complex=True)
            noise_est = estimate_noise_floor(psd_res.psd)
            snr_est = estimate_snr_spectral(psd_res, noise_est)
            
            if snr_est.snr_db is not None:
                snr_errors.append(snr_est.snr_db - target)

        err_arr = np.array(snr_errors)
        print(f"Target SNR: {target:4.1f} dB | Mean Error (Bias): {np.mean(err_arr):+5.2f} dB | Std Dev: {np.std(err_arr):4.2f} dB | Median: {np.median(err_arr):+5.2f} dB")

    # -------------------------------------------------------------
    # 3. Detection Probability & False Alarm Rate
    # -------------------------------------------------------------
    print("\n3. DETECTION PERFORMANCE (Pd and Pfa)")
    print("-" * 40)
    
    # False alarm test on pure noise
    fa_count = 0
    for _ in range(50):
        noise = (np.random.normal(0, 1, n_samples) + 1j * np.random.normal(0, 1, n_samples)).astype(np.complex64)
        psd_res = compute_psd(noise, segment_length=4096, is_complex=True)
        noise_est = estimate_noise_floor(psd_res.psd)
        regions = detect_signal_regions_spectral(psd_res, noise_est, threshold_db_offset=10.0)
        if len(regions) > 0:
            fa_count += 1

    pfa = fa_count / 50.0

    # Detection probability at 5 dB SNR
    det_count = 0
    for _ in range(50):
        t = np.arange(n_samples)
        signal = np.exp(2j * np.pi * 0.2 * t).astype(np.complex64)
        noise_p = 1.0 / (10.0 ** (5.0 / 10.0))
        noise = (np.random.normal(0, np.sqrt(noise_p / 2), n_samples) + 1j * np.random.normal(0, np.sqrt(noise_p / 2), n_samples)).astype(np.complex64)
        x = signal + noise

        psd_res = compute_psd(x, segment_length=4096, is_complex=True)
        noise_est = estimate_noise_floor(psd_res.psd)
        regions = detect_signal_regions_spectral(psd_res, noise_est, threshold_db_offset=6.0)
        if len(regions) > 0:
            det_count += 1

    pd = det_count / 50.0
    print(f"Detection Probability (Pd @ 5 dB SNR): {pd * 100:.1f}%")
    print(f"False Alarm Rate (Pfa on pure AWGN):   {pfa * 100:.1f}%")

    # -------------------------------------------------------------
    # 4. Performance & Execution Speed Benchmarks
    # -------------------------------------------------------------
    print("\n4. EXECUTION TIMINGS (100k samples)")
    print("-" * 40)
    bench_samples = (np.random.normal(0, 1, 100_000) + 1j * np.random.normal(0, 1, 100_000)).astype(np.complex64)

    t0 = time.perf_counter()
    compute_spectrum(bench_samples, fft_size=4096)
    t_fft = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    psd_out = compute_psd(bench_samples, segment_length=4096)
    t_psd = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    noise_out = estimate_noise_floor(psd_out.psd)
    t_noise = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    detect_signal_regions_spectral(psd_out, noise_out)
    t_det = (time.perf_counter() - t0) * 1000

    rec = SignalRecording(
        samples=bench_samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
    )
    t0 = time.perf_counter()
    analyze_signal(rec)
    t_total = (time.perf_counter() - t0) * 1000

    print(f"FFT Spectrum time:     {t_fft:6.2f} ms")
    print(f"Welch PSD time:        {t_psd:6.2f} ms")
    print(f"Noise estimation time: {t_noise:6.2f} ms")
    print(f"Signal detection time: {t_det:6.2f} ms")
    print(f"Full pipeline time:    {t_total:6.2f} ms")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
