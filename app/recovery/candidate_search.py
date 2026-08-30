from __future__ import annotations
from typing import Sequence
import numpy as np
from app.modulation.models import ModulationAnalysis, ModulationFamily, ModulationHypothesis
from .fsk_receiver import run_fsk_receiver
from .models import RecoveryCandidate, RecoveryConfig
from .psk_receiver import run_psk_receiver
from .qam_receiver import run_qam_receiver

def build_candidate_configurations(
    modulation_analysis: ModulationAnalysis | None,
    config: RecoveryConfig | None = None,
) -> list[tuple[ModulationFamily, int, float, float]]:
    """
    Extract prioritized receiver configurations (family, order, nominal_sps, phase3_score) from Phase 3 hypotheses.

    Parameters
    ----------
    modulation_analysis : ModulationAnalysis | None
        Analysis results from Phase 3.
    config : RecoveryConfig | None
        Recovery configuration.

    Returns
    -------
    configs : list[tuple[ModulationFamily, int, float, float]]
    """
    cfg = config or RecoveryConfig()
    configs: list[tuple[ModulationFamily, int, float, float]] = []

    if modulation_analysis and modulation_analysis.hypotheses:
        # Take top candidates up to max_candidates
        for h in modulation_analysis.hypotheses[:cfg.max_candidates]:
            sps_val = float(h.candidate_parameters.get("candidate_samples_per_symbol") or 8.0)
            order_val = h.order if h.order is not None else (4 if h.family == ModulationFamily.PSK else (16 if h.family == ModulationFamily.QAM else 2))
            configs.append((h.family, order_val, sps_val, h.score))
    else:
        # Default fallback candidates
        configs = [
            (ModulationFamily.PSK, 4, 8.0, 0.50),   # QPSK
            (ModulationFamily.PSK, 2, 8.0, 0.40),   # BPSK
            (ModulationFamily.QAM, 16, 8.0, 0.35),  # 16-QAM
            (ModulationFamily.FSK, 2, 8.0, 0.30),   # BFSK
        ]

    return configs[:cfg.max_candidates]

def execute_candidate_recovery(
    samples: np.ndarray,
    candidate_id: int,
    family: ModulationFamily,
    order: int,
    nominal_sps: float,
    phase3_score: float,
    config: RecoveryConfig | None = None,
) -> RecoveryCandidate:
    """
    Execute receiver recovery for a single modulation candidate with local SPS grid search.

    Parameters
    ----------
    samples : np.ndarray
        Prepared complex64 samples.
    candidate_id : int
        Candidate identifier.
    family : ModulationFamily
        Target modulation family.
    order : int
        Target modulation order.
    nominal_sps : float
        Nominal samples per symbol.
    phase3_score : float
        Phase 3 prior score.
    config : RecoveryConfig | None
        Recovery configuration.

    Returns
    -------
    RecoveryCandidate
        Best candidate result across the local SPS search grid.
    """
    cfg = config or RecoveryConfig()
    
    # Generate local SPS search grid (e.g. nominal, nominal - 0.25, nominal + 0.25)
    sps_candidates = [nominal_sps]
    if cfg.max_symbol_rate_candidates > 1:
        for offset in (-cfg.sps_search_step, cfg.sps_search_step, -2 * cfg.sps_search_step, 2 * cfg.sps_search_step):
            test_sps = nominal_sps + offset
            if test_sps >= 2.0:
                sps_candidates.append(test_sps)
    
    sps_candidates = sps_candidates[:cfg.max_symbol_rate_candidates]
    best_candidate: RecoveryCandidate | None = None

    for sps_val in sps_candidates:
        # RRC roll-off is a receiver assumption, so test the configured bounded
        # set instead of silently fixing it to one value.
        rolloffs = (None,) if family == ModulationFamily.FSK else cfg.rrc_rolloffs
        for rolloff in rolloffs:
            if family == ModulationFamily.FSK:
                res = run_fsk_receiver(samples, candidate_id=candidate_id, sps=sps_val, phase3_score=phase3_score, config=cfg)
            elif family == ModulationFamily.PSK:
                res = run_psk_receiver(samples, order=order, candidate_id=candidate_id, sps=sps_val, phase3_score=phase3_score, rrc_alpha=float(rolloff), config=cfg)
            elif family == ModulationFamily.QAM:
                res = run_qam_receiver(samples, order=order, candidate_id=candidate_id, sps=sps_val, phase3_score=phase3_score, rrc_alpha=float(rolloff), config=cfg)
            else:
                res = run_psk_receiver(samples, order=order, candidate_id=candidate_id, sps=sps_val, phase3_score=phase3_score, rrc_alpha=float(rolloff), config=cfg)

            if best_candidate is None or res.quality.composite_score > best_candidate.quality.composite_score:
                best_candidate = res

            # Early termination only after a high-quality lock; lower-quality
            # candidates continue through the configured receiver assumptions.
            if best_candidate.quality.composite_score >= 0.85:
                break
        if best_candidate is not None and best_candidate.quality.composite_score >= 0.85:
            break

    return best_candidate or res
