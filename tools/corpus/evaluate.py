"""Corpus evaluator. Evaluation truth is loaded only after production analysis completes."""
from __future__ import annotations
import argparse, hashlib, json, math, time
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
from scipy.stats import beta
from signal_analysis.native import runtime_info
from signal_analysis.workflow import run_production_analysis
from .ingest import production_request
from .validate_manifest import validate

ROOT = Path(__file__).resolve().parents[2]
REPORT_VERSION = 1
class FirewallError(RuntimeError): pass

def config_hash(config: dict) -> str: return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
def clopper_pearson(successes: int, total: int, alpha: float = .05) -> list[float] | None:
    if not total: return None
    return [0.0 if not successes else float(beta.ppf(alpha / 2, successes, total - successes + 1)), 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))]
def bootstrap_interval(values: list[float], seed: int = 0, rounds: int = 2000) -> list[float] | None:
    if not values: return None
    samples = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(samples, (rounds, samples.size), replace=True), axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]
def minimum_n_zero_failure(rate: float, confidence: float = .95) -> int:
    """n making the two-sided 95% Clopper--Pearson upper bound fall below rate."""
    return math.ceil(math.log((1 - confidence) / 2) / math.log(1 - rate))

def _truth(record: dict, root: Path) -> dict:
    ref = (root / record["truth_ref"]).resolve()
    truth = json.loads(ref.read_text(encoding="utf-8"))["captures"].get(record["capture_id"])
    if not truth: raise ValueError(f"missing evaluation truth for {record['capture_id']}")
    for field in ("symbol_rate", "snr"):
        if truth.get(field, {}).get("status") not in {"KNOWN", "MEASURED_INDEPENDENTLY", "UNKNOWN"}: raise ValueError(f"truth {record['capture_id']} missing {field} status")
    return truth
def _prediction(outcome: Any) -> dict:
    result = outcome.pipeline_result
    top = result.top_hypothesis if result else None
    label = top.label if top else "UNKNOWN"
    family = "UNSUPPORTED" if label == "UNSUPPORTED" else ("QAM" if "QAM" in label else "PSK" if "PSK" in label else "FSK" if "FSK" in label else "UNKNOWN")
    score = float(top.score) if top else 0.0
    params = top.candidate_parameters if top else None
    carrier = params.carrier_offset if params else None
    uncertainty = params.carrier_offset_uncertainty if params else None
    return {"family": family, "exact_label": label, "status": "CANDIDATE" if top and top.status.name == "HYPOTHESIS_UNVERIFIED" else (top.status.name if top else "UNKNOWN"), "score": score, "symbol_rate": None if not params else params.symbol_rate, "carrier_offset": carrier, "carrier_uncertainty": uncertainty, "snr": None if not params else params.snr_db, "receiver_locked": bool(result and result.demod_result and result.demod_result.sync_result.acquisition_status == "LOCKED")}

def _audit(root: Path, ids: list[str], split: str, cfg_hash: str, tuning: bool = False) -> int:
    path = root / "corpus/evaluation_invocations.jsonl"; path.parent.mkdir(exist_ok=True)
    prior = [] if not path.exists() else [json.loads(x) for x in path.read_text().splitlines() if x]
    reuse = sum(1 for x in prior if x.get("split") == "heldout")
    entry = {"time": time.time(), "ids": ids, "split": split, "config_hash": cfg_hash, "tuning": tuning}
    with path.open("a", encoding="utf-8") as stream: stream.write(json.dumps(entry, sort_keys=True) + "\n")
    if tuning and any(x.get("split") == "heldout" for x in prior + [entry]): raise FirewallError("threshold tuning touched held-out IDs; held-out report refused")
    return reuse

def evaluate(manifest_path: Path, *, tier: str | None, split: str, config: dict | None = None, output_dir: Path | None = None, allow_missing: bool = False) -> dict:
    manifest = validate(manifest_path); root = manifest_path.parent.parent; config = config or {}; cfg_hash = config_hash(config)
    if tier is None:
        raise ValueError("a single provenance tier is required; cross-tier metrics are prohibited")
    records = [r for r in manifest["captures"] if (tier is None or r["tier"] == tier) and r["split"] == split]
    heldout_reuse = _audit(root, [r["capture_id"] for r in records], split, cfg_hash)
    rows, skipped = [], []
    for record in records:
        try:
            outcome = run_production_analysis(production_request(record, root)) # truth intentionally not passed
            rows.append({"capture_id": record["capture_id"], "tier": record["tier"], "truth": _truth(record, root), "prediction": _prediction(outcome)})
        except Exception as exc:
            if not allow_missing: raise
            skipped.append({"capture_id": record["capture_id"], "error": str(exc)})
    matrix = Counter((r["truth"]["family"], r["prediction"]["family"]) for r in rows)
    supported = [r for r in rows if r["truth"]["supported_by_engine"]]
    negatives = [r for r in rows if not r["truth"]["supported_by_engine"]]
    correct = sum(r["truth"]["family"] == r["prediction"]["family"] for r in supported)
    exact = sum(r["truth"]["exact_label"] == r["prediction"]["exact_label"] for r in supported)
    high_wrong = sum(r["prediction"]["status"] == "CANDIDATE" and r["prediction"]["score"] >= config.get("high_confidence_threshold", .8) and r["prediction"]["family"] not in {"UNSUPPORTED", "UNKNOWN"} for r in negatives)
    rate_errors = [abs(r["prediction"]["symbol_rate"] - r["truth"]["symbol_rate"]["value"]) / r["truth"]["symbol_rate"]["value"] for r in rows if r["truth"]["symbol_rate"]["status"] in {"KNOWN", "MEASURED_INDEPENDENTLY"} and r["prediction"]["symbol_rate"] is not None and r["truth"]["symbol_rate"]["value"]]
    snr_errors = [r["prediction"]["snr"] - r["truth"]["snr"]["value"] for r in rows if r["truth"]["snr"]["status"] == "KNOWN" and r["prediction"]["snr"] is not None]
    carrier = [(r["prediction"], r["truth"]["carrier_offset"]) for r in rows if r["truth"].get("carrier_offset", {}).get("status") in {"KNOWN", "MEASURED_INDEPENDENTLY"} and r["prediction"]["carrier_offset"] is not None and r["prediction"]["carrier_uncertainty"] is not None]
    inside = sum(abs(p["carrier_offset"] - t["value"]) <= p["carrier_uncertainty"] for p,t in carrier)
    abstained = sum(r["prediction"]["family"] == "UNKNOWN" or r["prediction"]["status"] == "AMBIGUOUS" for r in supported)
    min_neg = minimum_n_zero_failure(.01); runtime = runtime_info()
    gates = {"family_accuracy": {"verdict": "INSUFFICIENT_POWER" if len(supported) < 30 else ("PASS" if correct / len(supported) >= .95 else "FAIL"), "n": len(supported), "interval_95": clopper_pearson(correct, len(supported))}, "unsupported_false_confidence": {"verdict": "INSUFFICIENT_POWER" if len(negatives) < min_neg else ("PASS" if high_wrong == 0 and clopper_pearson(high_wrong, len(negatives))[1] < .01 else "FAIL"), "n": len(negatives), "interval_95": clopper_pearson(high_wrong, len(negatives)), "minimum_n_zero_failure": min_neg}, "symbol_rate": {"verdict": "NOT_MEASURABLE" if not rate_errors else ("PASS" if np.mean(np.asarray(rate_errors) <= .02) >= .95 else "FAIL"), "n": len(rate_errors), "within_2_percent": None if not rate_errors else float(np.mean(np.asarray(rate_errors) <= .02))}}
    report = {"schema_version": REPORT_VERSION, "representativeness": "NOT_REPRESENTATIVE" if tier == "T1" else ("REPRESENTATIVE_EVIDENCE_PENDING" if not any(r["tier"] == "T2" for r in rows) else "T2_ONLY"), "engine": {"version": runtime.version, "native_api_version": runtime.api_version}, "config_hash": cfg_hash, "manifest_hash": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "tier": tier, "split": split, "split_counts": dict(Counter(r["tier"] for r in rows)), "t2_n": sum(r["tier"] == "T2" for r in rows), "heldout_reuse_count_before_invocation": heldout_reuse, "high_confidence_definition": f"status==CANDIDATE and top score>={config.get('high_confidence_threshold', .8)}", "rows": rows, "skipped": skipped, "confusion_matrix": {f"{a}->{b}": n for (a,b),n in matrix.items()}, "metrics": {"family_accuracy": {"n":len(supported), "value":None if not supported else correct/len(supported), "interval_95":clopper_pearson(correct,len(supported))}, "exact_label_accuracy": {"n":len(supported), "value":None if not supported else exact/len(supported), "interval_95":clopper_pearson(exact,len(supported))}, "high_confidence_wrong_unsupported": {"n":len(negatives), "value":None if not negatives else high_wrong/len(negatives), "interval_95":clopper_pearson(high_wrong,len(negatives))}, "symbol_rate_relative_error": {"n":len(rate_errors), "mean":None if not rate_errors else float(np.mean(rate_errors)), "mean_interval_95":bootstrap_interval(rate_errors), "fraction_within_2_percent":None if not rate_errors else float(np.mean(np.asarray(rate_errors)<=.02))}, "carrier_uncertainty_coverage": {"n":len(carrier), "value":None if not carrier else inside/len(carrier), "interval_95":clopper_pearson(inside,len(carrier)), "calibration": "NOT_MEASURABLE" if not carrier else ("UNDERCONFIDENT" if inside>.98 else "OVERCONFIDENT" if inside<.90 else "CONSISTENT")}, "snr_error_db": {"n":len(snr_errors), "mean":None if not snr_errors else float(np.mean(snr_errors)), "mean_interval_95":bootstrap_interval(snr_errors), "note":"unreliable unless truth SNR is KNOWN"}, "receiver": {"locked_rate":None if not supported else sum(r["prediction"]["receiver_locked"] for r in supported)/len(supported), "bits":"bits unverified unless independently supplied transmitted_bits_ref"}, "abstention": {"n":len(supported), "rate":None if not supported else abstained/len(supported), "selective_accuracy":None if correct == 0 else correct/max(1,len(supported)-abstained)}}, "gates": gates}
    output_dir = output_dir or root / "corpus/reports"; output_dir.mkdir(parents=True, exist_ok=True); stem = f"{tier or 'all'}_{split}"
    (output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / f"{stem}.md").write_text(f"# Corpus evaluation ({tier or 'all'} / {split})\n\nTier T2 captures: **{report['t2_n']}**. Representativeness: **{report['representativeness']}**.\n\n| Gate | Verdict | n | 95% interval |\n|---|---|---:|---|\n" + "\n".join(f"| {name} | {value['verdict']} | {value['n']} | {value.get('interval_95')} |" for name,value in gates.items()) + "\n", encoding="utf-8")
    return report

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,default=ROOT/"corpus/manifest.json"); p.add_argument("--tier",choices=["T0","T1","T2"]); p.add_argument("--split",choices=["calibration","heldout"],default="calibration"); p.add_argument("--allow-missing",action="store_true"); p.add_argument("--output-dir",type=Path); args=p.parse_args()
    try: report=evaluate(args.manifest,tier=args.tier,split=args.split,output_dir=args.output_dir,allow_missing=args.allow_missing)
    except FirewallError as exc: print(f"FIREWALL REFUSAL: {exc}"); return 3
    print(json.dumps({"report": report["gates"], "t2_n": report["t2_n"]}, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
