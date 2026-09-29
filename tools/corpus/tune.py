"""Calibration-only threshold proposal writer; deliberately never edits native constants."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .evaluate import ROOT, evaluate, FirewallError

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,default=ROOT/"corpus/manifest.json"); p.add_argument("--output",type=Path,default=ROOT/"corpus/proposed_thresholds.json"); p.add_argument("--tier",choices=["T0","T1","T2"],default="T1"); p.add_argument("--high-confidence-threshold",type=float,default=.8); args=p.parse_args()
    try: report=evaluate(args.manifest,tier=args.tier,split="calibration",config={"high_confidence_threshold":args.high_confidence_threshold})
    except FirewallError as exc: print(f"FIREWALL REFUSAL: {exc}"); return 3
    args.output.write_text(json.dumps({"status":"PROPOSED_HUMAN_REVIEW_REQUIRED", "scope":"calibration_only", "proposed":{"high_confidence_threshold":args.high_confidence_threshold}, "calibration_metrics":report["metrics"], "native_constants_modified":False},indent=2),encoding="utf-8"); print(args.output); return 0
if __name__ == "__main__": raise SystemExit(main())
