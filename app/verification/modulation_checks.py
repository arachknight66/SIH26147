from __future__ import annotations
import numpy as np
from app.modulation.models import ModulationAnalysis, ModulationFamily
from app.recovery.constellation import get_ideal_constellation
from app.recovery.models import RecoveredSignal, RecoveryAnalysis
from .models import ModulationAuditResult, TestResultStatus, VerificationTest

def audit_modulation_and_constellation(
    recovery: RecoveryAnalysis | RecoveredSignal | None = None,
    mod_analysis: ModulationAnalysis | None = None,
) -> tuple[ModulationAuditResult, list[VerificationTest]]:
    """
    Independently verify constellation geometry, EVM, 4th-power collapse, and competing modulation fit.

    Parameters
    ----------
    recovery : RecoveryAnalysis | RecoveredSignal | None
    mod_analysis : ModulationAnalysis | None

    Returns
    -------
    audit_result : ModulationAuditResult
    tests : list[VerificationTest]
    """
    tests: list[VerificationTest] = []

    rec_sig: RecoveredSignal | None = None
    if isinstance(recovery, RecoveryAnalysis):
        rec_sig = recovery.recovered_signal
    elif isinstance(recovery, RecoveredSignal):
        rec_sig = recovery

    if rec_sig is None or len(rec_sig.symbols) < 16:
        res = ModulationAuditResult(
            modulation_name="unknown",
            evm_percent=100.0,
            cluster_variance=1.0,
            mth_power_concentration=0.0,
            decision_margin=0.0,
            runner_up_name=None,
            runner_up_margin=0.0,
            is_consistent=False,
            details={"status": "no_recovered_symbols_available"},
        )
        tests.append(
            VerificationTest(
                test_id="MOD_00_INPUT",
                name="Modulation Input Check",
                category="modulation",
                description="Check availability of 1-SPS recovered symbols",
                status=TestResultStatus.FAIL,
                score=0.0,
                counter_evidence="No 1-SPS recovered constellation symbols available for verification",
                is_critical=True,
            )
        )
        return res, tests

    symbols = rec_sig.symbols.astype(np.complex64)
    n_syms = len(symbols)
    mod_fam = rec_sig.modulation_family
    mod_order = rec_sig.modulation_order or 4
    mod_label = f"{mod_order}-PSK" if mod_fam == ModulationFamily.PSK else f"{mod_order}-QAM"

    # Normalize symbols to unit average power
    p_avg = float(np.mean(np.abs(symbols) ** 2))
    norm_syms = symbols / np.sqrt(max(1e-9, p_avg))

    # 1. Independent EVM calculation against ideal constellation
    ideal_const = get_ideal_constellation(mod_fam, mod_order)
    # Nearest neighbor assignment
    dists = np.abs(norm_syms[:, None] - ideal_const[None, :])
    nearest_idx = np.argmin(dists, axis=1)
    nearest_ideal = ideal_const[nearest_idx]
    
    evm_linear = float(np.sqrt(np.mean(np.abs(norm_syms - nearest_ideal) ** 2)))
    evm_pct = evm_linear * 100.0

    cluster_var = float(np.var(np.abs(norm_syms - nearest_ideal)))
    dec_margin = float(np.min(dists))

    # 2. 4th-power phase concentration (for 4-ary PSK/QAM)
    z4 = norm_syms ** 4
    mth_conc = float(np.abs(np.mean(z4)))

    # 3. Competing hypothesis check (e.g. compare QPSK vs 8-PSK)
    runner_up = "8PSK" if mod_order == 4 else "QPSK"
    ideal_alt = get_ideal_constellation(ModulationFamily.PSK, 8 if mod_order == 4 else 4)
    dists_alt = np.abs(norm_syms[:, None] - ideal_alt[None, :])
    evm_alt_pct = float(np.sqrt(np.mean(np.min(dists_alt, axis=1) ** 2))) * 100.0
    runner_up_margin = float(evm_alt_pct - evm_pct)

    # Verification Tests
    is_evm_pass = bool(evm_pct <= 25.0)
    tests.append(
        VerificationTest(
            test_id="MOD_01_EVM",
            name="Constellation EVM Verification",
            category="modulation",
            description="Verify recovered 1-SPS EVM <= 25.0%",
            status=TestResultStatus.PASS if is_evm_pass else (TestResultStatus.WEAK_PASS if evm_pct <= 35.0 else TestResultStatus.FAIL),
            score=float(np.clip(1.0 - (evm_pct / 35.0), 0.0, 1.0)),
            details={"evm_percent": round(evm_pct, 2), "cluster_variance": round(cluster_var, 4)},
            counter_evidence=f"EVM ({evm_pct:.1f}%) exceeds acceptable demodulation threshold (25.0%)" if not is_evm_pass else None,
            is_critical=True,
        )
    )

    is_sym_pass = bool(mth_conc >= 0.20 or mod_fam != ModulationFamily.PSK)
    tests.append(
        VerificationTest(
            test_id="MOD_02_PHASE_CONCENTRATION",
            name="M-th Power Phase Symmetry",
            category="modulation",
            description="Verify rotational phase state concentration",
            status=TestResultStatus.PASS if is_sym_pass else TestResultStatus.FAIL,
            score=float(np.clip(mth_conc, 0.0, 1.0)),
            details={"mth_power_concentration": round(mth_conc, 3)},
            counter_evidence="Weak 4th-power phase concentration indicates phase jitter or wrong modulation order" if not is_sym_pass else None,
        )
    )

    tests.append(
        VerificationTest(
            test_id="MOD_03_COMPETING_HYPOTHESIS",
            name="Competing Hypothesis Dominance",
            category="modulation",
            description="Verify statistical separation against competing modulation alternative",
            status=TestResultStatus.PASS if runner_up_margin >= 0.0 else TestResultStatus.WEAK_PASS,
            score=float(np.clip(0.5 + (runner_up_margin / 20.0), 0.0, 1.0)),
            details={"selected": mod_label, "runner_up": runner_up, "margin_percent": round(runner_up_margin, 2)},
        )
    )

    is_consistent = bool(is_evm_pass and (mth_conc >= 0.15 or mod_fam != ModulationFamily.PSK))

    res = ModulationAuditResult(
        modulation_name=mod_label,
        evm_percent=round(evm_pct, 2),
        cluster_variance=round(cluster_var, 4),
        mth_power_concentration=round(mth_conc, 3),
        decision_margin=round(dec_margin, 3),
        runner_up_name=runner_up,
        runner_up_margin=round(runner_up_margin, 2),
        is_consistent=is_consistent,
        details={"num_symbols": n_syms},
    )
    return res, tests
