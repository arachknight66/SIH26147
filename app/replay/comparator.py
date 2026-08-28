from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from app.orchestration.pipeline_runner import PipelineResult

class DifferentialStatus(str, Enum):
    IDENTICAL = "IDENTICAL"
    NUMERICALLY_EQUIVALENT = "NUMERICALLY_EQUIVALENT"
    MATERIALLY_DIFFERENT = "MATERIALLY_DIFFERENT"
    INCOMPARABLE = "INCOMPARABLE"

@dataclass(frozen=True)
class StageComparison:
    stage_name: str
    is_identical: bool
    status: DifferentialStatus
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RunComparisonResult:
    overall_status: DifferentialStatus
    first_divergent_stage: str | None
    stage_comparisons: dict[str, StageComparison]
    reproducibility_hash_a: str
    reproducibility_hash_b: str

def compare_runs(run_a: PipelineResult, run_b: PipelineResult) -> RunComparisonResult:
    """
    Compare two execution runs stage-by-stage and classify divergence.
    """
    comps: dict[str, StageComparison] = {}
    first_div: str | None = None

    stages = ["phase1", "phase2", "phase3", "phase4", "phase5", "phase6"]
    for st in stages:
        res_a = getattr(run_a, f"{st}_result")
        res_b = getattr(run_b, f"{st}_result")

        if res_a is None or res_b is None:
            status = DifferentialStatus.IDENTICAL if (res_a is None and res_b is None) else DifferentialStatus.MATERIALLY_DIFFERENT
            comps[st] = StageComparison(stage_name=st, is_identical=(status == DifferentialStatus.IDENTICAL), status=status)
            if status != DifferentialStatus.IDENTICAL and first_div is None:
                first_div = st
            continue

        if not res_a.success or not res_b.success:
            ident = bool(res_a.success == res_b.success)
            comps[st] = StageComparison(stage_name=st, is_identical=ident, status=DifferentialStatus.IDENTICAL if ident else DifferentialStatus.MATERIALLY_DIFFERENT)
            if not ident and first_div is None:
                first_div = st
            continue

        # If both stages succeeded
        ident = True
        if st == "phase3":
            # Compare selected hypothesis labels
            w_a = res_a.output.selected_hypothesis.label if res_a.output.selected_hypothesis else None
            w_b = res_b.output.selected_hypothesis.label if res_b.output.selected_hypothesis else None
            ident = bool(w_a == w_b)
        elif st == "phase4":
            # Compare demodulated hard bits
            bits_a = res_a.output.recovered_signal.hard_bits if res_a.output.recovered_signal else None
            bits_b = res_b.output.recovered_signal.hard_bits if res_b.output.recovered_signal else None
            if bits_a is not None and bits_b is not None:
                ident = bool(len(bits_a) == len(bits_b) and (bits_a == bits_b).all())
            else:
                ident = bool(bits_a is None and bits_b is None)
        elif st == "phase6":
            # Compare verification status
            s_a = res_a.output.status if res_a.output else None
            s_b = res_b.output.status if res_b.output else None
            ident = bool(s_a == s_b)

        status = DifferentialStatus.IDENTICAL if ident else DifferentialStatus.MATERIALLY_DIFFERENT
        comps[st] = StageComparison(stage_name=st, is_identical=ident, status=status)
        if not ident and first_div is None:
            first_div = st

    hash_a = run_a.provenance.reproducibility_hash if run_a.provenance else ""
    hash_b = run_b.provenance.reproducibility_hash if run_b.provenance else ""

    overall = DifferentialStatus.IDENTICAL if hash_a == hash_b else (
        DifferentialStatus.MATERIALLY_DIFFERENT if first_div is not None else DifferentialStatus.NUMERICALLY_EQUIVALENT
    )

    return RunComparisonResult(
        overall_status=overall,
        first_divergent_stage=first_div,
        stage_comparisons=comps,
        reproducibility_hash_a=hash_a,
        reproducibility_hash_b=hash_b,
    )
