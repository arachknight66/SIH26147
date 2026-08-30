from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from app.orchestration.pipeline_config import PresetName, get_preset_config
from app.orchestration.pipeline_runner import run_pipeline
from app.reporting.artifact_manifest import export_all_artifacts
from app.replay.runner import replay_experiment
from app.replay.comparator import compare_runs
from app.ui.main_window import launch_gui


def _add_input_options(parser: argparse.ArgumentParser) -> None:
    """Options required to interpret metadata-free raw IQ recordings."""
    parser.add_argument("--raw-dtype", choices=["complex64", "float32", "int8", "int16", "uint8"], default="complex64", help="Scalar storage type for .iq/.raw/.bin inputs")
    parser.add_argument("--iq-order", choices=["IQ", "QI"], default="IQ", help="Order of scalar pairs in raw IQ input")
    parser.add_argument("--endian", choices=["little", "big"], default="little", help="Byte order for raw IQ input")
    parser.add_argument("--sample-rate", type=float, default=None, help="Known sample rate in Hz for metadata-free raw IQ")
    parser.add_argument("--center-frequency", type=float, default=None, help="Known RF center frequency in Hz for metadata-free raw IQ")


def _with_input_overrides(config, args):
    overrides = dict(config.user_overrides)
    overrides.update({
        "raw_dtype": args.raw_dtype,
        "iq_order": args.iq_order,
        "endianness": args.endian,
        "sample_rate_hz": args.sample_rate,
        "center_frequency_hz": args.center_frequency,
    })
    return replace(config, user_overrides=overrides)

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sih26147",
        description="SIH26147 Scientific Signal Recovery & Verification Engine v0.7.0",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: analyze
    p_analyze = subparsers.add_parser("analyze", help="Execute complete 6-phase analysis pipeline on input recording")
    p_analyze.add_argument("input_path", type=str, help="Path to input signal recording (.iq, .wav, .sigmf-meta)")
    p_analyze.add_argument("--preset", choices=["fast", "standard", "deep", "forensic"], default="standard", help="Pipeline execution preset")
    p_analyze.add_argument("--reproducible", action="store_true", help="Enforce deterministic random seed and configuration freezing")
    p_analyze.add_argument("--export", type=str, default=None, help="Directory to export HTML/JSON/CSV artifacts")
    p_analyze.add_argument("--json", action="store_true", help="Print result as JSON to stdout")
    _add_input_options(p_analyze)

    # Command: verify
    p_verify = subparsers.add_parser("verify", help="Execute Phase 6 independent scientific verification")
    p_verify.add_argument("input_path", type=str, help="Path to input signal recording")
    p_verify.add_argument("--strict", action="store_true", help="Enforce strict falsification criteria")
    _add_input_options(p_verify)

    # Command: replay
    p_replay = subparsers.add_parser("replay", help="Replay a previously executed experiment bundle")
    p_replay.add_argument("experiment_file", type=str, help="Path to experiment.json bundle")

    # Command: compare
    p_compare = subparsers.add_parser("compare", help="Differentially compare two experiment runs")
    p_compare.add_argument("run_a", type=str, help="Path to first experiment recording or JSON")
    p_compare.add_argument("run_b", type=str, help="Path to second experiment recording or JSON")

    # Command: gui
    p_gui = subparsers.add_parser("gui", help="Launch interactive desktop GUI")

    # Command: web
    p_web = subparsers.add_parser("web", help="Launch interactive local web application (opens in browser)")
    p_web.add_argument("--port", type=int, default=8050, help="Port to run web server on")

    # Command: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run comprehensive system benchmark suite")

    args = parser.parse_args()

    if args.command == "gui" or len(sys.argv) == 1:
        sys.exit(launch_gui())

    elif args.command == "web":
        from app.ui.web_app import launch_web_server
        launch_web_server(port=args.port, open_browser=True)

    elif args.command == "analyze":
        preset_map = {
            "fast": PresetName.FAST_SCREENING,
            "standard": PresetName.STANDARD_ANALYSIS,
            "deep": PresetName.DEEP_ANALYSIS,
            "forensic": PresetName.FORENSIC_ANALYSIS,
        }
        cfg = _with_input_overrides(get_preset_config(preset_map[args.preset], seed=42), args)
        print(f"Executing SIH26147 Pipeline on: {args.input_path} (Preset: {args.preset.upper()})...")
        res = run_pipeline(args.input_path, config=cfg)

        if args.export:
            artifacts = export_all_artifacts(res, args.export)
            print(f"Artifacts exported to: {args.export}")

        if args.json:
            from app.reporting.json_report import build_json_report
            print(json.dumps(build_json_report(res), indent=2))
        else:
            print("\n" + "=" * 65)
            print("SIH26147 FINAL SCIENTIFIC ASSESSMENT")
            print("=" * 65)
            print(f"Assessment:   {res.final_assessment_text}")
            print(f"Verified:     {res.is_verified}")
            print(f"Total Time:   {res.total_duration_seconds:.2f}s")
            print(f"SHA-256 Repro:{res.provenance.reproducibility_hash if res.provenance else 'N/A'}")
            print("=" * 65)

    elif args.command == "verify":
        cfg = _with_input_overrides(get_preset_config(PresetName.FORENSIC_ANALYSIS if args.strict else PresetName.STANDARD_ANALYSIS), args)
        res = run_pipeline(args.input_path, config=cfg)
        from app.verification.report import format_verification_report
        if res.phase6_result and res.phase6_result.output:
            print(format_verification_report(res.phase6_result.output, args.input_path))
        else:
            print("Verification could not be performed due to upstream pipeline failure.")

    elif args.command == "replay":
        res, matches = replay_experiment(args.experiment_file)
        print(f"Replay Status: {'SUCCESS (Reproducibility Hash Matches)' if matches else 'DIVERGED'}")
        print(f"Verified: {res.is_verified}")

    elif args.command == "compare":
        cfg = get_preset_config(PresetName.STANDARD_ANALYSIS)
        run_a = run_pipeline(args.run_a, config=cfg)
        run_b = run_pipeline(args.run_b, config=cfg)
        comp = compare_runs(run_a, run_b)
        print("\n" + "=" * 65)
        print("DIFFERENTIAL RUN COMPARISON")
        print("=" * 65)
        print(f"Overall Divergence Status: {comp.overall_status.value}")
        print(f"First Divergent Stage:     {comp.first_divergent_stage or 'NONE'}")
        for st, st_comp in comp.stage_comparisons.items():
            print(f"  • {st:15s}: {st_comp.status.value}")
        print("=" * 65)

    elif args.command == "benchmark":
        from scripts.run_full_benchmark import run_comprehensive_benchmark
        run_comprehensive_benchmark()

if __name__ == "__main__":
    main()
