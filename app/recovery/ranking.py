from __future__ import annotations
import numpy as np
from app.models.metadata import Diagnostic, DiagnosticSeverity
from .models import (
    RecoveredSignal,
    RecoveryCandidate,
    RecoveryQualityLevel,
    RecoveryStatus,
)

def rank_and_select_candidates(
    candidates: list[RecoveryCandidate],
) -> tuple[list[RecoveryCandidate], RecoveryCandidate | None, RecoveredSignal | None, bool, list[Diagnostic]]:
    """
    Empirically rank receiver candidates, detect wrong Phase 3 hypotheses, and construct RecoveredSignal.

    Parameters
    ----------
    candidates : list[RecoveryCandidate]
        Executed receiver candidate results.

    Returns
    -------
    ranked_candidates : list[RecoveryCandidate]
    selected_candidate : RecoveryCandidate | None
    recovered_signal : RecoveredSignal | None
    wrong_hypothesis_detected : bool
    diagnostics : list[Diagnostic]
    """
    diagnostics: list[Diagnostic] = []
    if not candidates:
        return [], None, None, False, [
            Diagnostic(
                code="NO_RECOVERY_CANDIDATES",
                message="No receiver candidates were evaluated.",
                severity=DiagnosticSeverity.ERROR,
            )
        ]

    # Score by combining receiver empirical quality (70%) with Phase 3 prior (30%)
    def candidate_rank_key(cand: RecoveryCandidate) -> float:
        # High penalty if recovery failed
        if cand.status in (RecoveryStatus.TIMING_UNLOCKED, RecoveryStatus.CARRIER_UNLOCKED, RecoveryStatus.PREPROCESSING_FAILED):
            return 0.20 * cand.quality.composite_score
        return 0.70 * cand.quality.composite_score + 0.30 * cand.phase3_score

    ranked = sorted(candidates, key=candidate_rank_key, reverse=True)
    winner = ranked[0]
    wrong_hyp_detected = False

    # Check for Wrong Phase 3 Hypothesis Promotion
    # If the candidate with highest Phase 3 score is NOT the winner and winner has substantially better receiver quality
    phase3_first = max(candidates, key=lambda c: c.phase3_score)
    if winner.candidate_id != phase3_first.candidate_id:
        if winner.quality.quality_level in (RecoveryQualityLevel.HIGH, RecoveryQualityLevel.MODERATE) and phase3_first.quality.composite_score < 0.45:
            wrong_hyp_detected = True
            diagnostics.append(
                Diagnostic(
                    code="POSSIBLE_WRONG_MODULATION_HYPOTHESIS",
                    message=(
                        f"Phase 3 candidate {phase3_first.label} (score={phase3_first.phase3_score:.2f}) failed receiver lock "
                        f"(EVM={phase3_first.constellation.evm_percent if phase3_first.constellation else 100:.1f}%), "
                        f"whereas {winner.label} achieved strong receiver lock (EVM={winner.constellation.evm_percent if winner.constellation else 0:.1f}%). "
                        f"Promoting {winner.label} as recovery winner."
                    ),
                    severity=DiagnosticSeverity.INFO,
                )
            )

    # Acceptance threshold
    recovered_sig: RecoveredSignal | None = None
    if winner.status == RecoveryStatus.RECOVERED and winner.quality.quality_level in (RecoveryQualityLevel.HIGH, RecoveryQualityLevel.MODERATE):
        if winner.constellation and winner.demodulation:
            # Generate sample strobe indices
            sps = winner.samples_per_symbol
            n_syms = len(winner.constellation.symbols)
            sample_strobes = np.arange(n_syms, dtype=np.float64) * sps
            
            cfo_val = winner.synchronization.frequency.coarse_cfo_normalized if winner.synchronization else 0.0
            phase_val = winner.synchronization.carrier.phase_estimate_rad if winner.synchronization else 0.0

            recovered_sig = RecoveredSignal(
                symbols=winner.constellation.symbols,
                hard_bits=winner.demodulation.hard_bits,
                soft_bits=winner.demodulation.soft_decisions,
                symbol_indices=winner.demodulation.symbol_indices,
                sample_indices=sample_strobes,
                modulation_family=winner.family,
                modulation_order=winner.order,
                symbol_rate_normalized=winner.symbol_rate_normalized,
                samples_per_symbol=winner.samples_per_symbol,
                cfo_normalized=cfo_val,
                carrier_phase_rad=phase_val,
                evm_percent=winner.constellation.evm_percent,
                decision_margin=winner.constellation.decision_margin,
                rotational_ambiguities_deg=winner.constellation.rotational_ambiguity_deg,
                bit_polarity_status=winner.demodulation.bit_polarity,
                provenance={"winner_label": winner.label, "composite_score": winner.quality.composite_score},
            )
            selected = winner
        else:
            selected = None
    else:
        selected = None
        diagnostics.append(
            Diagnostic(
                code="RECOVERY_INCONCLUSIVE",
                message="No candidate receiver achieved acceptable synchronization lock and EVM criteria.",
                severity=DiagnosticSeverity.WARNING,
            )
        )

    return ranked, selected, recovered_sig, wrong_hyp_detected, diagnostics
