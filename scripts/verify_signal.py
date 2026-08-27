from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from app.data_recovery.analyzer import recover_data
from app.io.loader import load_signal
from app.modulation.analyzer import analyze_modulation
from app.recovery.analyzer import recover_signal
from app.analysis.analyzer import analyze_signal
from app.verification.analyzer import verify_result
from app.verification.report import format_verification_report

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SIH26147 Phase 6: Independent Scientific Verification & Falsification Engine",
    )
    parser.add_argument("input_path", type=str, help="Path to input signal recording (.iq, .wav, .sigmf-meta)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--report", action="store_true", help="Output scientific verification report (default)")
    parser.add_argument("--dump-evidence", action="store_true", help="Include detailed claim evidence")
    parser.add_argument("--dump-tests", action="store_true", help="Include all executed verification tests")
    parser.add_argument("--dump-falsification", action="store_true", help="Include falsification details")
    parser.add_argument("--dump-robustness", action="store_true", help="Include robustness and leave-one-out metrics")
    parser.add_argument("--dump-uncertainty", action="store_true", help="Include itemized uncertainty budget")

    args = parser.parse_args()
    input_file = Path(args.input_path)

    if not input_file.exists():
        print(f"Error: Input file '{input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Execute full pipeline Phases 1 through 6
    recording = load_signal(input_file)
    phase2 = analyze_signal(recording)
    phase3 = analyze_modulation(recording, phase2)
    phase4 = recover_signal(recording, phase3, phase2)
    phase5 = recover_data(phase4)
    phase6 = verify_result(
        phase5_result=phase5,
        phase4_result=phase4,
        phase3_result=phase3,
        phase2_result=phase2,
        phase1_result=recording,
    )

    if args.json:
        out_dict = {
            "status": phase6.status.value,
            "quality_level": phase6.quality_level.value,
            "is_verified": phase6.is_verified,
            "is_falsified": phase6.is_falsified,
            "is_ambiguous": phase6.is_ambiguous,
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "claim_text": c.claim_text,
                    "status": c.status.value,
                    "confidence": c.confidence,
                    "independence_level": c.independence_level.value,
                }
                for c in phase6.claims
            ],
            "falsification": {
                "total_tests": phase6.falsification_audit.total_falsification_tests if phase6.falsification_audit else 0,
                "falsified_tests": phase6.falsification_audit.falsified_test_count if phase6.falsification_audit else 0,
                "critical_failures": phase6.falsification_audit.critical_failure_count if phase6.falsification_audit else 0,
                "outcome": phase6.falsification_audit.outcome.value if phase6.falsification_audit else "none",
            },
            "error_budget": {
                "total_uncertainty": phase6.error_budget.total_composite_uncertainty if phase6.error_budget else 0.0,
                "summary": phase6.error_budget.summary if phase6.error_budget else "",
            },
        }
        print(json.dumps(out_dict, indent=2))
    else:
        report_text = format_verification_report(phase6, recording_name=str(input_file.name))
        print(report_text)

if __name__ == "__main__":
    main()
