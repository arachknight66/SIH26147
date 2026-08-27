from __future__ import annotations
from typing import Any
import numpy as np
from app.models.metadata import Diagnostic, DiagnosticSeverity

CRITICAL_FAIL_CODES = {
    "PHYSICAL_MEASUREMENT_INCONSISTENCY",
    "MODULATION_CONTRADICTED",
    "SYNC_TEMPORAL_INSTABILITY",
    "FRAME_BOUNDARY_INSTABILITY",
    "FEC_OVER_CORRECTION_VIOLATION",
    "FEC_DEGRADES_RECONSTRUCTION",
    "CRC_OVERFITTING_SUSPECTED",
    "BOUNDARY_PERTURBATION_FAILED",
    "NON_REPRODUCIBLE_EXECUTION",
    "ACCIDENTAL_MATCH_HIGH_PROBABILITY",
}

def validate_physical_invariants(
    samples: np.ndarray | None,
    power: float,
    snr_db: float,
) -> tuple[bool, list[Diagnostic]]:
    """Validate physical invariant constraints."""
    diags: list[Diagnostic] = []
    if samples is not None:
        if not np.all(np.isfinite(samples)):
            diags.append(
                Diagnostic(
                    code="NON_FINITE_SAMPLES",
                    message="Signal contains NaN or Inf non-finite values.",
                    severity=DiagnosticSeverity.ERROR,
                )
            )
            return False, diags

    if power <= 0.0 or not np.isfinite(power):
        diags.append(
            Diagnostic(
                code="DEGENERATE_SIGNAL_POWER",
                message="Signal power is zero or non-finite.",
                severity=DiagnosticSeverity.ERROR,
            )
        )
        return False, diags

    return True, diags

def validate_fec_invariants(
    correction_fraction: float,
    max_allowable: float = 0.10,
    ber_before: float | None = None,
    ber_after: float | None = None,
) -> tuple[bool, list[Diagnostic]]:
    """Validate forward error correction invariants."""
    diags: list[Diagnostic] = []
    if correction_fraction > max_allowable:
        diags.append(
            Diagnostic(
                code="FEC_OVER_CORRECTION_VIOLATION",
                message=f"Correction fraction ({correction_fraction * 100:.2f}%) exceeds maximum allowable budget ({max_allowable * 100:.2f}%).",
                severity=DiagnosticSeverity.ERROR,
            )
        )
        return False, diags

    if ber_before is not None and ber_after is not None:
        if ber_after > ber_before:
            diags.append(
                Diagnostic(
                    code="FEC_DEGRADES_RECONSTRUCTION",
                    message=f"FEC decoding increased bit error rate from {ber_before:.4f} to {ber_after:.4f}.",
                    severity=DiagnosticSeverity.ERROR,
                )
            )
            return False, diags

    return True, diags
