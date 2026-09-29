import sys
import json
import argparse
from pathlib import Path
from dataclasses import asdict
from typing import Any

def _enum_to_str(obj: Any) -> Any:
    if hasattr(obj, 'value') and hasattr(obj, 'name'):
        return obj.value
    if isinstance(obj, dict):
        return {k: _enum_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_enum_to_str(v) for v in obj]
    if hasattr(obj, '__dataclass_fields__'):
        return _enum_to_str(asdict(obj))
    if isinstance(obj, float):
        return round(obj, 6)
    if hasattr(obj, 'tolist'): # NumPy arrays
        arr = obj.tolist()
        if len(arr) > 100:
            return arr[:100] + ["..."]
        return arr
    return obj

def run_cli():
    """
        Entry point for the CLI.

        Parses arguments, loads the signal, and runs the pipeline.
        """
    # Defensive check against GUI imports in headless mode
    for mod in sys.modules:
        if 'PySide6' in mod or 'pyqtgraph' in mod:
            print("ERROR: GUI modules imported in CLI mode!", file=sys.stderr)
            sys.exit(1)
            
    parser = argparse.ArgumentParser(description="Signal Analysis MVP Headless CLI")
    parser.add_argument("input", help="Path to input file or directory")
    parser.add_argument("--output", choices=["json", "text"], default="json", help="Output format")
    parser.add_argument("--wav-stereo-mode", choices=["unresolved", "stereo_real", "stereo_iq"], default="unresolved", help="Stereo interpretation for WAV files")
    parser.add_argument("--raw-dtype", help="Raw IQ dtype (requires --sample-rate-hz)")
    parser.add_argument("--raw-iq-order", choices=["iq", "qi"], default="iq", help="Raw real-pair order")
    parser.add_argument("--raw-endian", choices=["little", "big"], default="little", help="Raw IQ byte order")
    parser.add_argument("--sample-rate-hz", type=float, help="Raw IQ sample rate in Hz")
    parser.add_argument("--center-frequency-hz", type=float, default=0.0, help="Raw IQ center frequency in Hz")
    parser.add_argument("--fec-profile", default="UNCODED", help="Explicit FEC profile for the production pipeline")
    parser.add_argument(
        "--frame-profiles",
        type=Path,
        help="JSON file containing explicit frame profiles for fixed-boundary CRC verification",
    )
    args = parser.parse_args()
    
    # We defer these imports so we don't accidentally import GUI stuff at module load
    from .loaders import RawIQConfig
    from .release import build_run_metadata
    from .workflow import AnalysisRequest, run_production_analysis
    
    input_path = Path(args.input)
    frame_profiles = None
    if args.frame_profiles is not None:
        try:
            frame_profiles = json.loads(args.frame_profiles.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --frame-profiles: {exc}")
        if not isinstance(frame_profiles, list) or not all(isinstance(item, dict) for item in frame_profiles):
            parser.error("--frame-profiles must contain a JSON list of profile objects")
    files_to_process = []
    
    if input_path.is_dir():
        files_to_process = list(input_path.glob("*"))
    else:
        files_to_process = [input_path]
        
    results = {}
    
    for fpath in files_to_process:
        if not fpath.is_file():
            continue
            
        try:
            raw_config = None
            if fpath.suffix.lower() not in {".wav"} and not str(fpath).lower().endswith(".sigmf-meta"):
                if args.raw_dtype is None or args.sample_rate_hz is None:
                    raise ValueError("raw IQ input requires --raw-dtype and --sample-rate-hz")
                raw_config = RawIQConfig(
                    dtype=args.raw_dtype,
                    sample_rate_hz=args.sample_rate_hz,
                    center_frequency_hz=args.center_frequency_hz,
                    iq_order=args.raw_iq_order.upper(),
                    endian=args.raw_endian,
                )
            outcome = run_production_analysis(AnalysisRequest(
                path=fpath,
                wav_stereo_mode=args.wav_stereo_mode,
                raw_iq_config=raw_config,
                pipeline_config={
                    "fec_profile": args.fec_profile,
                    **({"frame_profiles": frame_profiles} if frame_profiles is not None else {}),
                },
                origin="cli",
            ))
            pipe_res = outcome.pipeline_result
            if pipe_res is None:
                raise RuntimeError(f"analysis did not complete: {outcome.execution_status.value}")
            
            # Serialize
            res_dict = _enum_to_str(pipe_res)
            res_dict["run_metadata"] = build_run_metadata(outcome)
            
            # Remove giant arrays from the output explicitly
            if 'recording' in res_dict:
                res_dict['recording'].pop('samples', None)
            if 'demod_result' in res_dict and res_dict['demod_result']:
                res_dict['demod_result'].pop('hard_bits', None)
                res_dict['demod_result'].pop('soft_llrs', None)
                res_dict['demod_result'].pop('symbol_decisions', None)
            if 'deint_result' in res_dict and res_dict['deint_result']:
                res_dict['deint_result'].pop('bits', None)
                res_dict['deint_result'].pop('llrs_reordered', None)
            if 'fec_result' in res_dict and res_dict['fec_result']:
                res_dict['fec_result'].pop('decoded_bits', None)
                
            results[str(fpath)] = res_dict
        except Exception as e:
            results[str(fpath)] = {"error": str(e)}
            
    if args.output == "json":
        print(json.dumps(results, indent=2))
    else:
        for f, res in results.items():
            print(f"--- {f} ---")
            if "error" in res:
                print(f"Error: {res['error']}")
            else:
                print(f"Hypothesis: {res.get('hypothesis_status')}")
                if res.get('top_hypothesis'):
                    print(f"  Top: {res['top_hypothesis'].get('label')} - {res['top_hypothesis'].get('status')}")
                print(f"Sync: {res.get('sync_status')}")
                print(f"FEC: {res.get('fec_status')}")
                print(f"Framing: {res.get('framing_status')}")
                if res.get('frame_structure'):
                    fs = res['frame_structure']
                    print(f"  Frame Status: {fs.get('status')}")
                    
if __name__ == "__main__":
    run_cli()
