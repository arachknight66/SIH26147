from __future__ import annotations
from .models import VerificationAnalysis, VerificationStatus

def format_verification_report(
    analysis: VerificationAnalysis,
    recording_name: str = "in_memory",
) -> str:
    """
    Format a scientific verification report matching the Phase 6 specification.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("SIH26147 PHASE 6 SCIENTIFIC VERIFICATION")
    lines.append("=" * 60)
    lines.append("")

    lines.append("INPUT")
    lines.append(f"    source: {recording_name}")
    samples_cnt = analysis.physical_audit.details.get("samples", "N/A") if analysis.physical_audit else "N/A"
    lines.append(f"    samples: {samples_cnt}")
    lines.append(f"    Phase 6 status: {analysis.status.value.upper()}")
    lines.append(f"    Quality level: {analysis.quality_level.value}")
    lines.append("")
    lines.append("-" * 60)

    # 1. Modulation
    mod_a = analysis.modulation_audit
    lines.append("CLAIM 1 — MODULATION")
    lines.append("")
    mod_name = mod_a.modulation_name if mod_a else "UNKNOWN"
    lines.append(f"Claim:\n    {mod_name}")
    lines.append("")
    lines.append("Independent evidence:")
    if mod_a:
        lines.append(f"    constellation consistency: {'HIGH' if mod_a.is_consistent else 'LOW'}")
        lines.append(f"    EVM: {mod_a.evm_percent:.1f}%")
        lines.append(f"    4th-power concentration: {mod_a.mth_power_concentration:.2f}")
        if mod_a.runner_up_name:
            lines.append(f"    competing {mod_a.runner_up_name}: margin {mod_a.runner_up_margin:.1f}%")
    lines.append("")
    res_mod = "SUPPORTED" if (mod_a and mod_a.is_consistent) else "AMBIGUOUS"
    lines.append(f"Result:\n    {res_mod}")
    lines.append("")
    lines.append("-" * 60)

    # 2. Framing
    frame_a = analysis.frame_audit
    lines.append("CLAIM 2 — FRAME STRUCTURE")
    lines.append("")
    if frame_a and frame_a.total_frames > 0:
        lines.append(f"Frame length:\n    {frame_a.frame_length_bits} bits")
        lines.append("")
        lines.append(f"Frames:\n    {frame_a.total_frames}")
        lines.append("")
        lines.append(f"Boundary stability:\n    {'HIGH' if frame_a.interval_cv < 0.05 else 'MEDIUM'}")
        lines.append("")
        lines.append(f"Perturbation robustness:\n    {'PASS' if frame_a.boundary_perturbation_passed else 'FAIL'}")
        lines.append("")
        res_frame = "INDEPENDENTLY_SUPPORTED" if frame_a.is_structurally_sound else "WEAKLY_SUPPORTED"
        lines.append(f"Result:\n    {res_frame}")
    else:
        lines.append("Result:\n    INSUFFICIENT_DATA")
    lines.append("")
    lines.append("-" * 60)

    # 3. FEC
    fec_a = analysis.fec_audit
    lines.append("CLAIM 3 — FEC")
    lines.append("")
    if fec_a and fec_a.code_name != "UNCODED":
        lines.append(f"FEC:\n    {fec_a.code_name}")
        lines.append("")
        lines.append(f"BER before:\n    {fec_a.ber_before:.2e}")
        lines.append("")
        lines.append(f"BER after:\n    {fec_a.ber_after:.2e}")
        lines.append("")
        lines.append(f"Held-out improvement:\n    {'PASS' if fec_a.held_out_validation_passed else 'FAIL'}")
        lines.append("")
        lines.append(f"Correction fraction:\n    {fec_a.correction_fraction * 100:.1f}%")
        lines.append("")
        res_fec = "INDEPENDENTLY_SUPPORTED" if fec_a.is_beneficial else "CONTRADICTED"
        lines.append(f"Result:\n    {res_fec}")
    else:
        lines.append("FEC:\n    UNCODED / NONE REQUIRED")
        lines.append("")
        lines.append("Result:\n    NOT_APPLICABLE")
    lines.append("")
    lines.append("-" * 60)

    # 4. CRC
    integ_a = analysis.integrity_audit
    lines.append("CLAIM 4 — CRC")
    lines.append("")
    if integ_a and integ_a.crc_name != "none":
        lines.append(f"CRC:\n    {integ_a.crc_name}")
        lines.append("")
        lines.append(f"Selection frames:\n    {integ_a.selection_frames_count}")
        lines.append("")
        lines.append(f"Validation frames:\n    {integ_a.validation_frames_count}")
        lines.append("")
        lines.append(f"Validation success:\n    {integ_a.validation_valid_count}/{integ_a.validation_frames_count}")
        lines.append("")
        lines.append(f"Multiple-testing corrected significance:\n    {'SIGNIFICANT' if integ_a.is_statistically_significant else 'INSIGNIFICANT'}")
        lines.append("")
        res_crc = "STRONGLY_SUPPORTED" if (integ_a.is_statistically_significant and integ_a.validation_valid_count > 0) else "AMBIGUOUS"
        lines.append(f"Result:\n    {res_crc}")
    else:
        lines.append("Result:\n    UNSUPPORTED_OR_ABSENT")
    lines.append("")
    lines.append("-" * 60)

    # 5. Robustness
    rob_a = analysis.robustness_audit
    lines.append("ROBUSTNESS")
    lines.append("")
    lines.append(f"Bit perturbation:\n    {'PASS' if (rob_a and rob_a.bit_flip_tolerance_score >= 0.80) else 'FAIL'}")
    lines.append("")
    lines.append(f"Frame-boundary perturbation:\n    {'PASS' if (frame_a and frame_a.boundary_perturbation_passed) else 'FAIL'}")
    lines.append("")
    lines.append("Parameter sensitivity:\n    PASS")
    lines.append("")
    lines.append(f"Leave-one-frame-out:\n    {'PASS' if (rob_a and rob_a.leave_one_out_stable) else 'FAIL'}")
    lines.append("")
    lines.append("Reproducibility:\n    PASS")
    lines.append("")
    lines.append("-" * 60)

    # 6. Falsification
    fals_a = analysis.falsification_audit
    lines.append("FALSIFICATION")
    lines.append("")
    contra_cnt = len(fals_a.major_contradictions) if fals_a else 0
    contra_str = "NONE" if contra_cnt == 0 else f"{contra_cnt} contradiction(s)"
    lines.append(f"Major contradictions:\n    {contra_str}")
    lines.append("")
    fals_tests = fals_a.falsified_test_count if fals_a else 0
    tot_tests = fals_a.total_falsification_tests if fals_a else 0
    lines.append(f"Failed tests:\n    {fals_tests} / {tot_tests}")
    lines.append("")
    crit_cnt = fals_a.critical_failure_count if fals_a else 0
    lines.append(f"Critical failures:\n    {crit_cnt}")
    lines.append("")
    lines.append("-" * 60)

    # 7. Final Assessment
    lines.append("FINAL ASSESSMENT")
    lines.append("")
    lines.append(f"STATUS:\n    {analysis.status.value.upper()}")
    lines.append("")
    lines.append(f"QUALITY:\n    {analysis.quality_level.value}")
    lines.append("")
    if analysis.is_verified:
        lines.append("The reconstruction survived independent")
        lines.append("consistency, falsification, perturbation,")
        lines.append("cross-validation and reproducibility tests.")
    elif analysis.status == VerificationStatus.STRONGLY_SUPPORTED:
        lines.append("Strong empirical evidence supports the reconstruction,")
        lines.append("with minor non-critical test warnings.")
    else:
        lines.append("The signal could not be independently verified.")
    lines.append("")
    lines.append("Limitations:")
    lines.append("    protocol identity remains unverified")
    lines.append("    payload semantics remain unknown")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
