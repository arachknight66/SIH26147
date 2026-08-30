from __future__ import annotations
import math
from typing import Sequence
import numpy as np
import scipy.signal as signal
from app.dsp.autocorrelation import compute_autocorrelation
from app.dsp.noise import estimate_noise_floor
from app.models.analysis import AutocorrelationResult, SymbolRateCandidate
from app.models.metadata import MetadataStatus

# Estimator Family Identifiers
METHOD_TRANSITION_ENERGY = "cyclostationary_transition_spectrum"
METHOD_SQUARED_MAGNITUDE = "gardner_squared_magnitude_spectrum"
METHOD_ENVELOPE = "envelope_spectrum"
METHOD_AUTOCORRELATION = "autocorrelation_peak"
METHOD_CONSENSUS = "multi_estimator_consensus"


def _refine_spectral_peak(
    vals: np.ndarray,
    freqs: np.ndarray,
    peak_idx: int,
) -> tuple[float, float]:
    """
    Sub-bin quadratic peak interpolation on spectral array.

    Returns (refined_freq, refined_magnitude).
    """
    n_bins = len(vals)
    if peak_idx <= 0 or peak_idx >= n_bins - 1:
        return float(freqs[peak_idx]), float(vals[peak_idx])

    alpha = float(vals[peak_idx - 1])
    beta = float(vals[peak_idx])
    gamma = float(vals[peak_idx + 1])

    denom = alpha - 2.0 * beta + gamma
    if abs(denom) > 1e-12:
        delta = 0.5 * (alpha - gamma) / denom
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0

    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    refined_freq = float(freqs[peak_idx] + delta * df)
    refined_val = float(beta - 0.25 * (alpha - gamma) * delta)
    return refined_freq, refined_val


def _extract_spectral_lines_from_sequence(
    seq: np.ndarray,
    *,
    min_freq: float = 0.02,
    max_freq: float = 0.48,
    significance_thresh_sigma: float = 5.5,
    significance_thresh_db: float | None = None,
    max_fft: int = 8192,
) -> list[tuple[float, float, dict[str, float]]]:
    """
    Compute real FFT of a non-linear transformed sequence, subtract continuous spectral baseline,
    and detect discrete cyclostationary baud lines using MAD iterative sigma-clipping.

    Returns
    -------
    list[tuple[peak_freq, score, metadata_dict]]
    """
    if significance_thresh_db is not None:
        significance_thresh_sigma = max(5.5, significance_thresh_db)

    n = len(seq)
    if n < 32:
        return []

    n_fft = min(max_fft, 1 << int(np.floor(np.log2(n))))
    if n_fft < 32:
        return []

    seg = seq[:n_fft].astype(np.float64)
    seg_mean = np.mean(seg)
    seg_std = np.std(seg)
    # If sequence is constant (e.g. CW carrier produces constant |x|^2 or |dx|^2), return empty
    if seg_std < 1e-6 * (abs(seg_mean) + 1e-12):
        return []

    win = np.hanning(n_fft)
    win_seg = (seg - seg_mean) * win
    win_seg -= np.mean(win_seg)

    fft_vals = np.abs(np.fft.rfft(win_seg))
    freqs = np.fft.rfftfreq(n_fft, d=1.0)

    # Valid candidate baud rate interval (0.02 to 0.48 cycles/sample)
    valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
    if np.sum(valid_mask) < 8:
        return []

    valid_fft = fft_vals[valid_mask]
    valid_freqs = freqs[valid_mask]
    n_valid = len(valid_fft)

    # Background baseline estimation via moving median filter
    win_len = max(15, int(n_valid // 20))
    if win_len % 2 == 0:
        win_len += 1
    
    pad_width = win_len // 2
    padded = np.pad(valid_fft, pad_width, mode="reflect")
    bg = np.array([np.median(padded[i : i + win_len]) for i in range(n_valid)])
    
    diff = valid_fft - bg
    med_diff = float(np.median(diff))
    mad_diff = float(np.median(np.abs(diff - med_diff)))
    sigma_res = max(1.4826 * mad_diff, 1e-12)

    residual = np.maximum(0.0, diff)
    max_res = float(np.max(residual))

    # Significance test on sharp discrete line (must exceed noise floor by significance_thresh_sigma)
    if max_res < (significance_thresh_sigma * sigma_res):
        return []

    min_prom = max(significance_thresh_sigma * sigma_res * 0.8, 0.20 * max_res)
    min_dist = max(3, n_fft // 256)
    peaks, props = signal.find_peaks(residual, prominence=min_prom, distance=min_dist)
    prominences = props.get("prominences", np.zeros(len(peaks)))

    lines: list[tuple[float, float, dict[str, float]]] = []
    for i, p_idx in enumerate(peaks):
        prom = float(prominences[i]) if i < len(prominences) else 0.0
        refined_f, refined_val = _refine_spectral_peak(residual, valid_freqs, p_idx)
        line_sig = refined_val / sigma_res

        if line_sig >= significance_thresh_sigma and min_freq <= refined_f <= max_freq:
            prom_ratio = prom / max(max_res, 1e-12)
            score = float(np.clip(prom_ratio * 0.85 + min(line_sig / 40.0, 0.15), 0.10, 0.99))
            lines.append((
                refined_f,
                score,
                {"prominence": prom, "sigma": sigma_res, "line_sig": line_sig, "peak_val": refined_val},
            ))

    return lines


# =============================================================================
# 1. INDEPENDENT ESTIMATOR IMPLEMENTATIONS
# =============================================================================

def estimate_rate_transition_energy(
    samples: np.ndarray,
    *,
    min_freq: float = 0.02,
    max_freq: float = 0.48,
    significance_thresh_sigma: float = 5.5,
) -> list[tuple[float, float, dict[str, float]]]:
    """
    Differential transition energy cyclostationary estimator:
    Forms |x[n] - x[n-1]|^2 and identifies spectral lines at symbol transitions.
    """
    if len(samples) < 64:
        return []
    diff_energy = (np.abs(np.diff(samples)) ** 2).astype(np.float64)
    return _extract_spectral_lines_from_sequence(
        diff_energy,
        min_freq=min_freq,
        max_freq=max_freq,
        significance_thresh_sigma=significance_thresh_sigma,
    )


def estimate_rate_squared_magnitude(
    samples: np.ndarray,
    *,
    min_freq: float = 0.02,
    max_freq: float = 0.48,
    significance_thresh_sigma: float = 5.5,
) -> list[tuple[float, float, dict[str, float]]]:
    """
    Squared-magnitude cyclostationary estimator (Gardner spectral-correlation):
    Forms |x[n]|^2, removes mean, and searches power spectrum for envelope periodicity.
    Sensitive to amplitude-modulated formats (QAM/ASK/stochastic envelope variations).
    """
    if len(samples) < 64:
        return []
    sq_mag = (np.abs(samples) ** 2).astype(np.float64)
    return _extract_spectral_lines_from_sequence(
        sq_mag,
        min_freq=min_freq,
        max_freq=max_freq,
        significance_thresh_sigma=significance_thresh_sigma,
    )


def estimate_rate_envelope_spectrum(
    samples: np.ndarray,
    *,
    min_freq: float = 0.02,
    max_freq: float = 0.48,
    significance_thresh_sigma: float = 5.5,
) -> list[tuple[float, float, dict[str, float]]]:
    """
    Delay-and-multiply / envelope spectral estimator:
    Forms |x[n]| (envelope magnitude only, discarding phase) and searches spectrum.
    Strictly invariant to carrier frequency offset (CFO).
    """
    if len(samples) < 64:
        return []
    env_mag = np.abs(samples).astype(np.float64)
    return _extract_spectral_lines_from_sequence(
        env_mag,
        min_freq=min_freq,
        max_freq=max_freq,
        significance_thresh_sigma=significance_thresh_sigma,
    )


def estimate_rate_autocorrelation(
    samples: np.ndarray,
    autocorr_result: AutocorrelationResult | None = None,
    *,
    min_freq: float = 0.02,
    max_freq: float = 0.48,
) -> list[tuple[float, float, dict[str, float]]]:
    """
    Autocorrelation secondary-peak periodicity estimator with parabolic sub-bin lag refinement.
    """
    n_samples = len(samples)
    if autocorr_result is None:
        if n_samples < 32:
            return []
        autocorr_result = compute_autocorrelation(samples, max_lag=min(1024, n_samples - 1))

    mag = autocorr_result.normalized_magnitude
    lags = autocorr_result.lags
    if len(mag) < 8:
        return []

    # Lag limits corresponding to [min_freq, max_freq]
    min_lag = max(2, int(np.floor(1.0 / max_freq)))
    max_lag = min(len(mag) - 1, int(np.ceil(1.0 / min_freq)))

    if min_lag >= max_lag or max_lag >= len(mag):
        return []

    search_mag = mag[min_lag : max_lag + 1]
    search_lags = lags[min_lag : max_lag + 1]

    if len(search_mag) < 4:
        return []

    # Statistically significant autocorrelation prominence above noise variance
    min_prom = max(0.06, 4.0 / np.sqrt(max(n_samples, 64)))
    peaks, props = signal.find_peaks(search_mag, prominence=min_prom, distance=2)
    prominences = props.get("prominences", np.zeros(len(peaks)))

    results = []
    for i, p_idx in enumerate(peaks):
        raw_lag = int(search_lags[p_idx])
        # Quadratic interpolation around autocorrelation peak
        idx_in_mag = min_lag + p_idx
        if 1 <= idx_in_mag < len(mag) - 1:
            y0 = float(mag[idx_in_mag - 1])
            y1 = float(mag[idx_in_mag])
            y2 = float(mag[idx_in_mag + 1])
            denom = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
            refined_lag = raw_lag + delta
        else:
            refined_lag = float(raw_lag)

        if refined_lag <= 0:
            continue
        norm_rate = float(1.0 / refined_lag)
        if not (min_freq <= norm_rate <= max_freq):
            continue

        prom = float(prominences[i]) if i < len(prominences) else 0.05
        peak_val = float(search_mag[p_idx])
        score = float(np.clip(prom * 4.0 + peak_val * 0.5, 0.10, 0.92))
        results.append((norm_rate, score, {"lag": refined_lag, "prominence": prom, "peak_val": peak_val}))

    return results


# =============================================================================
# 2. SEGMENT BOOTSTRAPPING & HARMONIC COLLAPSE
# =============================================================================

def _bootstrap_estimator_variance(
    samples: np.ndarray,
    estimator_func,
    target_rate: float,
    *,
    num_segments: int = 4,
    tolerance: float = 0.05,
) -> tuple[float, int, list[float]]:
    """
    Partition record into temporal segments, evaluate estimator on each, and return
    (uncertainty_std, corroborated_count, segment_rates).
    """
    n_samples = len(samples)
    seg_len = n_samples // num_segments
    if seg_len < 64:
        fft_len = min(8192, 1 << int(np.floor(np.log2(n_samples))))
        unc = max(0.5 / max(fft_len, 64), 0.02 * target_rate)
        return unc, 1, [target_rate]

    corroborated: list[float] = []
    for k in range(num_segments):
        sub_samples = samples[k * seg_len : (k + 1) * seg_len]
        try:
            sub_results = estimator_func(sub_samples)
        except Exception:
            continue

        for r_cand, _, _ in sub_results:
            if abs(r_cand - target_rate) / target_rate <= tolerance:
                corroborated.append(r_cand)
                break

    m = len(corroborated)
    fft_len = min(8192, 1 << int(np.floor(np.log2(seg_len))))
    bin_unc = 0.5 / max(fft_len, 64)

    if m >= 2:
        std_val = float(np.std(corroborated, ddof=1))
        unc = max(std_val, bin_unc)
    else:
        unc = max(bin_unc * 2.0, 0.04 * target_rate)

    return unc, m, corroborated


def _resolve_harmonic_aliasing(
    raw_candidates: list[tuple[float, float, str, float, list[str]]],
    tolerance: float = 0.035,
) -> list[tuple[float, float, str, float, list[str]]]:
    """
    Identify and collapse harmonic (2x, 3x, 4x) and subharmonic (1/2x) relationships into
    the fundamental symbol rate.
    """
    if len(raw_candidates) <= 1:
        return raw_candidates

    # Sort by score descending (strongest periodic components first)
    sorted_by_score = sorted(raw_candidates, key=lambda c: -c[1])
    collapsed: list[tuple[float, float, str, float, list[str]]] = []
    absorbed_indices: set[int] = set()

    for i, (f_i, score_i, meth_i, unc_i, assump_i) in enumerate(sorted_by_score):
        if i in absorbed_indices:
            continue

        current_assumptions = list(assump_i)
        current_score = score_i
        current_unc = unc_i
        fund_f = f_i
        fund_meth = meth_i

        # Check if there is a fundamental subharmonic (e.g. f_i / 2) present among candidates
        for j, (f_j, score_j, meth_j, unc_j, assump_j) in enumerate(sorted_by_score):
            if j == i or j in absorbed_indices:
                continue
            # Check if f_j is half of f_i (f_j is fundamental, f_i is 2nd harmonic)
            if abs(f_i - 2.0 * f_j) / (2.0 * f_j) <= tolerance and f_j >= 0.02:
                fund_f = f_j
                fund_meth = meth_j
                current_score = max(score_i, score_j)
                current_unc = min(unc_i / 2.0, unc_j)
                current_assumptions = list(assump_j)
                current_assumptions.append(
                    f"Collapsed 2x harmonic component at f={f_i:.5f} ({meth_i}) into fundamental baud rate f={fund_f:.5f}."
                )
                absorbed_indices.add(j)
                break

        # Absorb any remaining higher harmonics (2x, 3x, 4x of fund_f)
        for j, (f_j, score_j, meth_j, unc_j, assump_j) in enumerate(sorted_by_score):
            if j == i or j in absorbed_indices:
                continue
            ratio = f_j / max(fund_f, 1e-12)
            is_harmonic = False
            harmonic_k = 0
            for k in (2, 3, 4):
                if abs(ratio - k) / k <= tolerance:
                    is_harmonic = True
                    harmonic_k = k
                    break
            if is_harmonic:
                absorbed_indices.add(j)
                current_assumptions.append(
                    f"Collapsed {harmonic_k}x harmonic component at f={f_j:.5f} ({meth_j}) into fundamental baud rate f={fund_f:.5f}."
                )
                current_unc = min(current_unc, unc_j / harmonic_k)

        collapsed.append((fund_f, current_score, fund_meth, current_unc, current_assumptions))

    return collapsed


# =============================================================================
# 3. MULTI-ESTIMATOR CONSENSUS & SYMBOL RATE ENGINE
# =============================================================================

def estimate_symbol_rate_consensus(
    candidates_by_family: dict[str, list[tuple[float, float, float, list[str]]]],
    *,
    sample_rate_hz: float | None = None,
    sample_rate_confidence: float = 1.0,
    occupied_bandwidth_normalized: float | None = None,
    agreement_tolerance: float = 0.03,
    max_candidates: int = 4,
) -> list[SymbolRateCandidate]:
    """
    Synthesize multi-estimator observations into a cross-validated, uncertainty-quantified
    candidate list.
    """
    # Flatten all raw family candidates
    flat_items: list[tuple[float, float, str, float, list[str]]] = []
    for fam, items in candidates_by_family.items():
        for rate_val, score_val, unc_val, assump_list in items:
            flat_items.append((rate_val, score_val, fam, unc_val, assump_list))

    if not flat_items:
        return []

    # 1. Harmonic aliasing resolution
    collapsed_items = _resolve_harmonic_aliasing(flat_items, tolerance=agreement_tolerance)

    # 2. Inter-method clustering across independent estimator families
    clusters: list[list[tuple[float, float, str, float, list[str]]]] = []
    for item in collapsed_items:
        r_cand = item[0]
        placed = False
        for cl in clusters:
            cl_rates = [x[0] for x in cl]
            cl_mean_r = float(np.mean(cl_rates))
            if abs(r_cand - cl_mean_r) / cl_mean_r <= agreement_tolerance:
                cl.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])

    final_candidates: list[SymbolRateCandidate] = []

    for cl in clusters:
        rates = [x[0] for x in cl]
        scores = [x[1] for x in cl]
        families = list(dict.fromkeys([x[2] for x in cl]))  # unique families
        uncs = [max(x[3], 1e-6) for x in cl]
        all_assumptions = []
        for x in cl:
            for a in x[4]:
                if a not in all_assumptions:
                    all_assumptions.append(a)

        num_families = len(families)

        # Precision-weighted rate average
        weights = [1.0 / (u ** 2) for u in uncs]
        total_w = sum(weights)
        norm_rate = float(sum(r * w for r, w in zip(rates, weights)) / total_w)

        # Combined uncertainty in normalized domain
        combined_unc = float(1.0 / math.sqrt(total_w))
        combined_unc = max(combined_unc, min(uncs) / math.sqrt(max(1, len(cl))), 0.0005)

        sps = float(1.0 / norm_rate)
        max_score = float(max(scores))

        # Epistemic Promotion & Confidence Scoring
        if num_families >= 2:
            status = MetadataStatus.ESTIMATED
            # Multi-family corroboration boosts confidence
            confidence = float(np.clip(max_score * 0.85 + 0.10 * (num_families - 1), 0.70, 0.98))
            method_str = f"consensus_{num_families}_estimators"
            all_assumptions.insert(
                0,
                f"Cross-validated across {num_families} independent estimator families: {', '.join(families)}."
            )
        else:
            status = MetadataStatus.AMBIGUOUS
            confidence = float(np.clip(max_score * 0.65, 0.15, 0.65))
            method_str = families[0] if families else "single_estimator"
            all_assumptions.insert(
                0,
                f"Candidate identified solely by {families[0] if families else 'unknown'}; lacks independent multi-method cross-validation."
            )

        # 3. Physical Plausibility Gating: Low SPS (< 4.0 samples/symbol)
        if sps < 4.0:
            confidence = round(confidence * 0.75, 3)
            all_assumptions.append(
                f"Samples per symbol ({sps:.2f}) < 4.0 implies coarse oversampling for reliable pulse reconstruction; confidence down-weighted."
            )

        # 4. Physical Plausibility Gating: Bandwidth Cross-Check
        if occupied_bandwidth_normalized is not None and occupied_bandwidth_normalized > 0:
            if norm_rate > 1.25 * occupied_bandwidth_normalized:
                confidence = round(confidence * 0.40, 3)
                status = MetadataStatus.AMBIGUOUS
                all_assumptions.append(
                    f"Implied baud rate ({norm_rate:.4f}) exceeds measured 99% occupied bandwidth ({occupied_bandwidth_normalized:.4f}); down-weighted due to physical inconsistency."
                )

        # 5. Propagation to Physical Hz with Sample Rate Uncertainty
        rate_hz: float | None = None
        rate_hz_unc: float | None = None
        if sample_rate_hz is not None and sample_rate_hz > 0:
            rate_hz = float(norm_rate * sample_rate_hz)
            rel_rate_unc = combined_unc / norm_rate
            sr_conf = float(np.clip(sample_rate_confidence, 0.0, 1.0))
            rel_sr_unc = 1.0 - sr_conf
            total_rel_unc = math.sqrt(rel_rate_unc ** 2 + rel_sr_unc ** 2)
            rate_hz_unc = float(rate_hz * total_rel_unc)
            if sr_conf < 1.0:
                all_assumptions.append(
                    f"Sample rate metadata confidence ({sr_conf:.2f}) combined with normalized rate uncertainty in quadrature."
                )

        final_candidates.append(
            SymbolRateCandidate(
                normalized_rate=round(norm_rate, 6),
                estimated_samples_per_symbol=round(sps, 3),
                rate_hz=round(rate_hz, 2) if rate_hz is not None else None,
                method=method_str,
                score=round(max_score, 3),
                assumptions=all_assumptions,
                confidence=round(confidence, 3),
                status=status,
                uncertainty=round(combined_unc, 6),
                rate_hz_uncertainty=round(rate_hz_unc, 2) if rate_hz_unc is not None else None,
            )
        )

    # Sort candidates by status (ESTIMATED before AMBIGUOUS) and then by confidence descending
    final_candidates.sort(key=lambda c: (1 if c.status == MetadataStatus.ESTIMATED else 0, c.confidence, c.score), reverse=True)

    # Deduplicate closely matching final candidates (within 3%)
    deduped: list[SymbolRateCandidate] = []
    for cand in final_candidates:
        if cand.normalized_rate is None:
            continue
        is_dup = False
        for existing in deduped:
            if existing.normalized_rate is not None:
                rel_diff = abs(cand.normalized_rate - existing.normalized_rate) / existing.normalized_rate
                if rel_diff < agreement_tolerance:
                    is_dup = True
                    break
        if not is_dup:
            deduped.append(cand)
            if len(deduped) >= max_candidates:
                break

    return deduped


def estimate_symbol_rate_candidates(
    samples: np.ndarray,
    autocorr_result: AutocorrelationResult | None = None,
    *,
    sample_rate_hz: float | None = None,
    sample_rate_confidence: float = 1.0,
    occupied_bandwidth_normalized: float | None = None,
    max_candidates: int = 4,
    significance_thresh_sigma: float = 5.5,
    significance_thresh_db: float | None = None,
    min_freq: float = 0.02,
    max_freq: float = 0.48,
) -> list[SymbolRateCandidate]:
    """
    Generate multi-estimator, cross-validated symbol-rate candidates with segment bootstrapping
    variance quantification and physical plausibility gating.
    """
    if significance_thresh_db is not None:
        significance_thresh_sigma = max(5.5, significance_thresh_db)

    n_samples = len(samples)
    if n_samples < 64:
        return []

    x = np.asarray(samples)

    # -------------------------------------------------------------
    # 1. Run Independent Estimators on Full Record
    # -------------------------------------------------------------
    raw_trans = estimate_rate_transition_energy(
        x, min_freq=min_freq, max_freq=max_freq, significance_thresh_sigma=significance_thresh_sigma
    )
    raw_sq_mag = estimate_rate_squared_magnitude(
        x, min_freq=min_freq, max_freq=max_freq, significance_thresh_sigma=significance_thresh_sigma
    )
    raw_env = estimate_rate_envelope_spectrum(
        x, min_freq=min_freq, max_freq=max_freq, significance_thresh_sigma=significance_thresh_sigma
    )
    raw_autocorr = estimate_rate_autocorrelation(
        x, autocorr_result=autocorr_result, min_freq=min_freq, max_freq=max_freq
    )

    # -------------------------------------------------------------
    # 2. Segment Bootstrapping Variance Estimation per Method
    # -------------------------------------------------------------
    candidates_by_family: dict[str, list[tuple[float, float, float, list[str]]]] = {
        METHOD_TRANSITION_ENERGY: [],
        METHOD_SQUARED_MAGNITUDE: [],
        METHOD_ENVELOPE: [],
        METHOD_AUTOCORRELATION: [],
    }

    # Method 1: Transition energy
    for r_cand, score, meta in raw_trans:
        unc, corr_cnt, _ = _bootstrap_estimator_variance(
            x,
            lambda s: estimate_rate_transition_energy(s, min_freq=min_freq, max_freq=max_freq, significance_thresh_sigma=significance_thresh_sigma),
            r_cand,
        )
        assump = [
            "Assumes non-linear differential transition squaring reveals cyclostationary symbol-rate spectral lines.",
            "Assumes pulse-shaped modulation with periodic symbol transitions.",
        ]
        if corr_cnt < 2:
            assump.append("Segment-to-segment bootstrapping showed low cross-segment consistency; capped at ambiguous.")
        candidates_by_family[METHOD_TRANSITION_ENERGY].append((r_cand, score, unc, assump))

    # Method 2: Squared magnitude
    for r_cand, score, meta in raw_sq_mag:
        unc, corr_cnt, _ = _bootstrap_estimator_variance(
            x,
            lambda s: estimate_rate_squared_magnitude(s, min_freq=min_freq, max_freq=max_freq, significance_thresh_sigma=significance_thresh_sigma),
            r_cand,
        )
        assump = [
            "Assumes instantaneous power (magnitude-squared) spectral correlation reveals baud line (Gardner approach).",
            "Sensitive to envelope modulation and pulse-shaped amplitude variations.",
        ]
        if corr_cnt < 2:
            assump.append("Segment-to-segment bootstrapping showed low cross-segment consistency; capped at ambiguous.")
        candidates_by_family[METHOD_SQUARED_MAGNITUDE].append((r_cand, score, unc, assump))

    # Method 3: Envelope spectrum
    for r_cand, score, meta in raw_env:
        unc, corr_cnt, _ = _bootstrap_estimator_variance(
            x,
            lambda s: estimate_rate_envelope_spectrum(s, min_freq=min_freq, max_freq=max_freq, significance_thresh_sigma=significance_thresh_sigma),
            r_cand,
        )
        assump = [
            "Assumes signal envelope spectrum contains discrete baud line.",
            "Invariant to carrier frequency offset due to phase discard.",
        ]
        if corr_cnt < 2:
            assump.append("Segment-to-segment bootstrapping showed low cross-segment consistency; capped at ambiguous.")
        candidates_by_family[METHOD_ENVELOPE].append((r_cand, score, unc, assump))

    # Method 4: Autocorrelation
    for r_cand, score, meta in raw_autocorr:
        unc, corr_cnt, _ = _bootstrap_estimator_variance(
            x,
            lambda s: estimate_rate_autocorrelation(s, min_freq=min_freq, max_freq=max_freq),
            r_cand,
        )
        assump = [
            "Assumes autocorrelation secondary peak reflects symbol period rather than carrier/preamble periodicity.",
        ]
        if corr_cnt < 2:
            assump.append("Segment-to-segment bootstrapping showed low cross-segment consistency; capped at ambiguous.")
        candidates_by_family[METHOD_AUTOCORRELATION].append((r_cand, score, unc, assump))

    # -------------------------------------------------------------
    # 3. Consensus Synthesis & Cross-Validation
    # -------------------------------------------------------------
    return estimate_symbol_rate_consensus(
        candidates_by_family,
        sample_rate_hz=sample_rate_hz,
        sample_rate_confidence=sample_rate_confidence,
        occupied_bandwidth_normalized=occupied_bandwidth_normalized,
        max_candidates=max_candidates,
    )

