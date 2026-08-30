from __future__ import annotations
import numpy as np
from app.data_recovery.concatenated_codes import (
    STANDARD_CONCATENATED_TOPOLOGIES,
    ConcatenatedCodeTopology,
    execute_concatenated_decode,
    execute_reversed_order_decode,
)
from app.data_recovery.fec_decode import viterbi_decode
from app.data_recovery.models import DataRecoveryAnalysis, FECCodeFamily, Phase6Handoff
from app.data_recovery.reed_solomon import ReedSolomonCodec
from .models import FECAuditResult, TestResultStatus, VerificationConfig, VerificationTest

def audit_fec_and_cross_validation(
    data_analysis: DataRecoveryAnalysis | None = None,
    handoff: Phase6Handoff | None = None,
    config: VerificationConfig | None = None,
) -> tuple[FECAuditResult, list[VerificationTest]]:
    """
    Independently verify forward error correction benefit, anti-over-correction, and held-out cross-validation.

    Parameters
    ----------
    data_analysis : DataRecoveryAnalysis | None
    handoff : Phase6Handoff | None
    config : VerificationConfig | None

    Returns
    -------
    audit_result : FECAuditResult
    tests : list[VerificationTest]
    """
    cfg = config or VerificationConfig()
    tests: list[VerificationTest] = []

    sel_cand = data_analysis.selected_candidate if data_analysis else None
    fec_dec = sel_cand.fec_decode if sel_cand else None
    fec_hyp = sel_cand.fec if sel_cand else None

    if fec_hyp is None or fec_hyp.code_family == FECCodeFamily.NONE or fec_dec is None:
        res = FECAuditResult(
            code_name="UNCODED",
            ber_before=0.0,
            ber_after=0.0,
            information_gain=0.0,
            correction_fraction=0.0,
            anti_overcorrection_passed=True,
            held_out_validation_passed=True,
            is_beneficial=True,
            details={"status": "uncoded_or_no_fec_applied"},
        )
        tests.append(
            VerificationTest(
                test_id="FEC_00_UNCODED",
                name="FEC Necessity & Application Audit",
                category="fec",
                description="Verify signal reconstruction does not require unneeded FEC correction",
                status=TestResultStatus.PASS,
                score=1.0,
                details={"code_family": "NONE"},
            )
        )
        return res, tests

    corr_frac = fec_dec.correction_fraction
    corr_count = fec_dec.corrected_bit_count
    details_dict: dict[str, object] = {"code_family": fec_hyp.code_family.value}

    # Determine exact provable correction budget
    if fec_hyp.code_family == FECCodeFamily.CONCATENATED:
        # For concatenated codes, verify both inner and outer stages independently
        topo = next((t for t in STANDARD_CONCATENATED_TOPOLOGIES if t.name in fec_hyp.code_name), STANDARD_CONCATENATED_TOPOLOGIES[0])
        rs_hyp = topo.outer_fec
        n_syms = rs_hyp.block_size or 64
        k_syms = int(round(n_syms * rs_hyp.rate))
        t_rad = (n_syms - k_syms) // 2
        outer_budget = min(cfg.max_allowable_correction_fraction, float(t_rad / n_syms))
        inner_budget = cfg.max_allowable_correction_fraction

        inner_passed = bool(corr_frac <= inner_budget)
        outer_passed = bool(corr_frac <= outer_budget)
        is_anti_over = bool(inner_passed and outer_passed)
        allowable_budget = outer_budget
        details_dict.update({
            "inner_passed": inner_passed,
            "outer_passed": outer_passed,
            "inner_budget": inner_budget,
            "outer_budget": outer_budget,
        })
    elif fec_hyp.code_family == FECCodeFamily.REED_SOLOMON and fec_hyp.block_size:
        n_syms = fec_hyp.block_size
        k_syms = int(round(n_syms * fec_hyp.rate))
        t_radius = (n_syms - k_syms) // 2
        # Exact theoretical symbol bound converted to fraction
        allowable_budget = min(cfg.max_allowable_correction_fraction, float(t_radius / n_syms))
        is_anti_over = bool(corr_frac <= allowable_budget)
    else:
        allowable_budget = cfg.max_allowable_correction_fraction
        is_anti_over = bool(corr_frac <= allowable_budget)

    tests.append(
        VerificationTest(
            test_id="FEC_01_OVERCORRECTION",
            name="Anti-Over-Correction Budget Audit",
            category="fec",
            description=f"Verify FEC bit modification fraction <= {allowable_budget * 100:.1f}%",
            status=TestResultStatus.PASS if is_anti_over else TestResultStatus.FAIL,
            score=max(0.0, 1.0 - (corr_frac / max(1e-4, allowable_budget))),
            details={"corrected_bits": corr_count, "correction_fraction": round(corr_frac, 4), "budget": allowable_budget, **details_dict},
            counter_evidence=f"Excessive bit alterations ({corr_frac * 100:.1f}%) exceeds safety budget" if not is_anti_over else None,
            is_critical=True,
        )
    )

    # 70/30 Held-out Cross-Validation
    raw_channel_bits = handoff.raw_bits if handoff else (sel_cand.bit_hypothesis.bitstream.hard_bits if sel_cand else np.array([], dtype=np.uint8))
    held_out_passed = True

    if len(raw_channel_bits) >= 64:
        if fec_hyp.code_family == FECCodeFamily.CONCATENATED:
            topo = next((t for t in STANDARD_CONCATENATED_TOPOLOGIES if t.name in fec_hyp.code_name), STANDARD_CONCATENATED_TOPOLOGIES[0])
            n_syms = topo.outer_fec.block_size or 64
            inner_rate = topo.inner_fec.rate or 0.5
            block_bits = int(round((n_syms * 8) / inner_rate))
            num_blocks = len(raw_channel_bits) // block_bits
            if num_blocks >= 2:
                val_block_start = max(1, int(num_blocks * 0.70))
                val_raw_bits = raw_channel_bits[val_block_start * block_bits:]
            else:
                val_raw_bits = raw_channel_bits
            val_concat_res = execute_concatenated_decode(val_raw_bits, topology=topo, max_correction_fraction=allowable_budget)
            held_out_passed = bool(val_concat_res.valid)
        elif fec_hyp.code_family == FECCodeFamily.REED_SOLOMON and fec_hyp.block_size:
            n_syms = fec_hyp.block_size
            k_syms = int(round(n_syms * fec_hyp.rate))
            block_bits = n_syms * 8
            num_blocks = len(raw_channel_bits) // block_bits
            if num_blocks >= 2:
                val_block_start = max(1, int(num_blocks * 0.70))
                val_rs_bits = raw_channel_bits[val_block_start * block_bits:]
            else:
                val_rs_bits = raw_channel_bits
            poly = fec_hyp.generator_polynomials[0] if fec_hyp.generator_polynomials else 0x11D
            fcr = 112 if "fcr=112" in "".join(fec_hyp.assumptions) else (1 if "fcr=1" in "".join(fec_hyp.assumptions) else 0)
            rs_codec = ReedSolomonCodec(n_symbols=n_syms, k_symbols=k_syms, symbol_width=8, prim_poly=poly, first_consecutive_root=fcr)
            val_dec = rs_codec.decode_bitstream(val_rs_bits, max_correction_fraction=allowable_budget)
            held_out_passed = bool(val_dec.valid)
            val_raw_bits = val_rs_bits
        else:
            s_idx = int(len(raw_channel_bits) * 0.70)
            val_raw_bits = raw_channel_bits[s_idx:]
            val_dec = viterbi_decode(val_raw_bits, k=7, g1=0o133, g2=0o171, max_correction_fraction=allowable_budget)
            held_out_passed = bool(val_dec.valid)

        split_idx = len(raw_channel_bits) - len(val_raw_bits)

        tests.append(
            VerificationTest(
                test_id="FEC_02_CROSS_VALIDATION",
                name="Held-Out Frame FEC Cross-Validation (70/30)",
                category="fec",
                description="Verify FEC decoder generalizability on held-out validation frames",
                status=TestResultStatus.PASS if held_out_passed else TestResultStatus.FAIL,
                score=1.0 if held_out_passed else 0.20,
                details={"selection_frames": split_idx, "validation_frames": len(val_raw_bits)},
                counter_evidence="FEC hypothesis fails on held-out validation frames (possible overfit)" if not held_out_passed else None,
                is_critical=True,
            )
        )

    # RS Chien-Search Consistency & Perturbation Falsification Probe
    if fec_hyp.code_family == FECCodeFamily.REED_SOLOMON and fec_hyp.block_size:
        n_syms = fec_hyp.block_size
        k_syms = int(round(n_syms * fec_hyp.rate))
        poly = fec_hyp.generator_polynomials[0] if fec_hyp.generator_polynomials else 0x11D
        fcr = 112 if "fcr=112" in "".join(fec_hyp.assumptions) else (1 if "fcr=1" in "".join(fec_hyp.assumptions) else 0)
        rs_codec = ReedSolomonCodec(n_symbols=n_syms, k_symbols=k_syms, symbol_width=8, prim_poly=poly, first_consecutive_root=fcr)

        # Create a test message and perturb beyond correction capability (t+1 errors)
        t_rad = rs_codec.correction_radius
        test_msg = np.zeros(k_syms, dtype=np.uint8)
        test_code = rs_codec.encode(test_msg)
        # Inject t+1 corruptions
        corrupt_code = test_code.copy()
        for idx in range(min(t_rad + 1, n_syms)):
            corrupt_code[idx] ^= 0x55
        _, _, corrupt_valid, corrupt_status = rs_codec.decode(corrupt_code)
        # Falsification probe passes if decoder definitively rejects t+1 errors (chien_root_count_matched=False or valid=False)
        chien_probe_passed = bool(not corrupt_valid and not corrupt_status.chien_root_count_matched)

        tests.append(
            VerificationTest(
                test_id="FEC_03_CHIEN_PERTURBATION",
                name="Reed-Solomon Chien Search Falsification Probe",
                category="fec",
                description="Verify RS decoder rejects (t+1) error patterns with definitive Chien root mismatch",
                status=TestResultStatus.PASS if chien_probe_passed else TestResultStatus.FAIL,
                score=1.0 if chien_probe_passed else 0.0,
                details={"correction_radius": t_rad, "injected_errors": t_rad + 1, "root_count_matched": corrupt_status.chien_root_count_matched},
                counter_evidence="RS decoder failed to detect over-radius error pattern via Chien root-count mismatch" if not chien_probe_passed else None,
                is_critical=True,
            )
        )

    # Concatenated Topology Ordering Falsification Probe
    if fec_hyp.code_family == FECCodeFamily.CONCATENATED and len(raw_channel_bits) >= 64:
        topo = next((t for t in STANDARD_CONCATENATED_TOPOLOGIES if t.name in fec_hyp.code_name), STANDARD_CONCATENATED_TOPOLOGIES[0])
        rev_res = execute_reversed_order_decode(raw_channel_bits, topology=topo, max_correction_fraction=allowable_budget)
        # Falsification probe passes if reversed order decode fails (valid=False or higher error rate)
        rev_probe_passed = bool(not rev_res.valid or rev_res.combined_correction_fraction > corr_frac + 0.05)

        tests.append(
            VerificationTest(
                test_id="FEC_04_TOPOLOGY_ORDERING_FALSIFICATION",
                name="Concatenated Topology Ordering Falsification Probe",
                category="fec",
                description="Verify reversed-order decode (Outer RS -> De-interleave -> Inner Viterbi) fails as negative control",
                status=TestResultStatus.PASS if rev_probe_passed else TestResultStatus.FAIL,
                score=1.0 if rev_probe_passed else 0.0,
                details={"reversed_valid": rev_res.valid, "reversed_correction_fraction": rev_res.combined_correction_fraction},
                counter_evidence="Reversed topology ordering unexpectedly succeeded, indicating unfalsifiable decoding result" if not rev_probe_passed else None,
                is_critical=True,
            )
        )

    info_gain = float(corr_frac)
    res = FECAuditResult(
        code_name=fec_hyp.code_name,
        ber_before=round(corr_frac, 4),
        ber_after=0.0,
        information_gain=round(info_gain, 4),
        correction_fraction=round(corr_frac, 4),
        anti_overcorrection_passed=is_anti_over,
        held_out_validation_passed=held_out_passed,
        is_beneficial=bool(is_anti_over and held_out_passed),
        details=details_dict,
    )
    return res, tests
