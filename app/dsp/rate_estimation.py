from __future__ import annotations
import numpy as np
import scipy.signal as signal
from app.models.analysis import AutocorrelationResult, SymbolRateCandidate
from app.models.metadata import MetadataStatus

def estimate_symbol_rate_candidates(
    samples: np.ndarray,
    autocorr_result: AutocorrelationResult | None = None,
    *,
    sample_rate_hz: float | None = None,
    max_candidates: int = 4,
) -> list[SymbolRateCandidate]:
    """
    Generate preliminary symbol-rate candidates using cyclostationary transition energy
    and autocorrelation periodicity.

    Parameters
    ----------
    samples : np.ndarray
        Signal samples.
    autocorr_result : AutocorrelationResult | None
        Precomputed autocorrelation result (optional).
    sample_rate_hz : float | None
        Sample rate in Hz if known.
    max_candidates : int
        Maximum number of ranked candidates to return.

    Returns
    -------
    list[SymbolRateCandidate]
    """
    candidates: list[SymbolRateCandidate] = []
    n_samples = len(samples)
    if n_samples < 64:
        return candidates

    # Method 1: Cyclostationary Transition Energy Spectral Lines (Oerder-Meyr / Non-linear Derivative)
    if n_samples >= 128:
        # Differential transition energy highlights periodic symbol boundary transitions
        diff_energy = (np.abs(np.diff(samples)) ** 2).astype(np.float64)
        n_fft = min(8192, 1 << int(np.floor(np.log2(len(diff_energy)))))
        
        if n_fft >= 64:
            seg = diff_energy[:n_fft]
            win = np.hanning(n_fft)
            win_seg = (seg - np.mean(seg)) * win
            win_seg -= np.mean(win_seg)
            
            env_fft = np.abs(np.fft.rfft(win_seg))
            env_freqs = np.fft.rfftfreq(n_fft, d=1.0)

            # Valid candidate baud rate interval (e.g. 0.02 to 0.48 cycles/sample)
            valid_mask = (env_freqs >= 0.02) & (env_freqs <= 0.48)
            valid_fft = env_fft[valid_mask]
            valid_freqs = env_freqs[valid_mask]

            if len(valid_fft) > 10:
                med_val = float(np.median(valid_fft))
                mad_val = float(np.median(np.abs(valid_fft - med_val)))
                sigma_val = 1.4826 * mad_val
                min_prom = max(1e-6, 2.5 * sigma_val)
                max_peak = float(np.max(valid_fft))

                peaks, props = signal.find_peaks(valid_fft, prominence=min_prom, distance=max(3, n_fft // 256))
                prominences = props.get("prominences", np.zeros(len(peaks)))

                for i, p_idx in enumerate(peaks):
                    f_cand = float(valid_freqs[p_idx])
                    sps = float(1.0 / f_cand)
                    prom_val = float(prominences[i]) if i < len(prominences) else min_prom
                    # Normalized prominence score relative to maximum spectral component
                    score = float(np.clip(prom_val / max(max_peak, 1e-12), 0.05, 0.99))
                    rate_hz = float(f_cand * sample_rate_hz) if (sample_rate_hz and sample_rate_hz > 0) else None

                    candidates.append(
                        SymbolRateCandidate(
                            normalized_rate=round(f_cand, 6),
                            estimated_samples_per_symbol=round(sps, 3),
                            rate_hz=round(rate_hz, 2) if rate_hz is not None else None,
                            method="cyclostationary_transition_spectrum",
                            score=round(score, 3),
                            assumptions=[
                                "Assumes non-linear transition squaring reveals cyclostationary symbol-rate spectral lines.",
                                "Assumes pulse-shaped modulation with periodic symbol transitions.",
                            ],
                            confidence=round(score * 0.80, 3),
                            status=MetadataStatus.AMBIGUOUS,
                        )
                    )

    # Method 2: Autocorrelation Secondary Peak / Periodicity
    if autocorr_result is not None and len(autocorr_result.normalized_magnitude) > 8:
        mag = autocorr_result.normalized_magnitude
        lags = autocorr_result.lags

        # Search lags >= 2
        min_lag_idx = 2
        search_mag = mag[min_lag_idx:]
        search_lags = lags[min_lag_idx:]

        if len(search_mag) > 6:
            peaks, props = signal.find_peaks(search_mag, prominence=0.03, distance=2)
            prominences = props.get("prominences", np.zeros(len(peaks)))
            for i, p_idx in enumerate(peaks):
                lag_val = int(search_lags[p_idx])
                norm_rate = float(1.0 / lag_val)
                prom = float(prominences[i]) if i < len(prominences) else 0.05
                score = float(np.clip(prom * 5.0, 0.1, 0.90))
                rate_hz = float(norm_rate * sample_rate_hz) if (sample_rate_hz and sample_rate_hz > 0) else None

                candidates.append(
                    SymbolRateCandidate(
                        normalized_rate=round(norm_rate, 6),
                        estimated_samples_per_symbol=round(float(lag_val), 3),
                        rate_hz=round(rate_hz, 2) if rate_hz is not None else None,
                        method="autocorrelation_peak",
                        score=round(score, 3),
                        assumptions=[
                            "Assumes autocorrelation peak corresponds to symbol period rather than carrier/preamble harmonics.",
                        ],
                        confidence=round(score * 0.70, 3),
                        status=MetadataStatus.AMBIGUOUS,
                    )
                )

    # Sort candidates by score descending and deduplicate similar rates (within 3%)
    sorted_candidates = sorted(candidates, key=lambda c: -c.score)
    deduped: list[SymbolRateCandidate] = []
    for cand in sorted_candidates:
        if cand.normalized_rate is None:
            continue
        is_dup = False
        for existing in deduped:
            if existing.normalized_rate is not None:
                rel_diff = abs(cand.normalized_rate - existing.normalized_rate) / existing.normalized_rate
                if rel_diff < 0.03:
                    is_dup = True
                    break
        if not is_dup:
            deduped.append(cand)
            if len(deduped) >= max_candidates:
                break

    return deduped
