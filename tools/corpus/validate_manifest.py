"""Validate corpus metadata without treating labels as analysis input."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALID_CAPTURE_STATUSES = {"KNOWN", "ASSUMED", "MISSING"}
VALID_TRUTH_STATUSES = {"KNOWN", "MEASURED_INDEPENDENTLY", "UNKNOWN"}

class ManifestError(ValueError): pass

def split_for(capture_id: str, salt: str) -> tuple[str, str]:
    digest = hashlib.sha256((salt + ":" + capture_id).encode()).hexdigest()
    return ("calibration" if int(digest[0], 16) < 8 else "heldout"), digest

def _load(path: Path) -> dict:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise ManifestError(f"cannot read {path}: {exc}") from exc

def validate(manifest_path: Path, *, verify_files: bool = False) -> dict:
    manifest = _load(manifest_path)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("split_salt"), str) or len(manifest["split_salt"]) < 16: raise ManifestError("schema_version=1 and a frozen split_salt (>=16 chars) are required")
    captures = manifest.get("captures")
    if not isinstance(captures, list): raise ManifestError("captures must be a list")
    seen, tiers = set(), set()
    for record in captures:
        required = {"capture_id", "path", "sha256", "tier", "source_description", "license", "acquisition", "center_frequency_hz", "sample_rate_hz", "format", "import_config", "duration_seconds", "split", "split_assignment", "truth_ref", "notes"}
        missing = required - record.keys()
        if missing: raise ManifestError(f"{record.get('capture_id', '<unknown>')}: missing {sorted(missing)}")
        cid = record["capture_id"]
        if cid in seen: raise ManifestError(f"duplicate capture_id: {cid}")
        seen.add(cid); tiers.add(record["tier"])
        if record["tier"] not in {"T0", "T1", "T2"}: raise ManifestError(f"{cid}: invalid tier")
        if len(record["sha256"]) != 64 or any(ch not in "0123456789abcdef" for ch in record["sha256"]): raise ManifestError(f"{cid}: missing or invalid sha256")
        for field in ("center_frequency_hz", "sample_rate_hz"):
            value = record[field]
            if not isinstance(value, dict) or value.get("status") not in VALID_CAPTURE_STATUSES or "value" not in value: raise ManifestError(f"{cid}: {field} requires value and KNOWN/ASSUMED/MISSING status")
            if value["status"] == "MISSING" and value["value"] is not None: raise ManifestError(f"{cid}: MISSING {field} must have null value")
        expected, digest = split_for(cid, manifest["split_salt"])
        assignment = record["split_assignment"]
        if record["split"] != expected or assignment.get("algorithm") != "sha256" or assignment.get("rule") != "first_hex_nibble_lt_8_calibration" or assignment.get("digest") != digest: raise ManifestError(f"{cid}: split is not the required frozen hash-derived assignment")
        if record["tier"] == "T1" and not isinstance(record.get("impairment_recipe"), dict): raise ManifestError(f"{cid}: T1 requires an impairment_recipe")
        if record["tier"] != "T1" and record.get("impairment_recipe") is not None: raise ManifestError(f"{cid}: impairment recipes are T1-only")
        truth_path = (manifest_path.parent.parent / record["truth_ref"]).resolve()
        truth_document = _load(truth_path)
        truth = truth_document.get("captures", {}).get(cid)
        if truth_document.get("schema_version") != 1 or not isinstance(truth, dict): raise ManifestError(f"{cid}: missing separate evaluation truth")
        if truth.get("family") not in {"PSK", "QAM", "FSK", "UNSUPPORTED", "UNKNOWN"} or not isinstance(truth.get("supported_by_engine"), bool): raise ManifestError(f"{cid}: truth requires family and supported_by_engine")
        for truth_field in ("symbol_rate", "snr"):
            field = truth.get(truth_field)
            if not isinstance(field, dict) or field.get("status") not in VALID_TRUTH_STATUSES or "value" not in field or "unit" not in field: raise ManifestError(f"{cid}: truth {truth_field} requires value, unit, and explicit status")
        if verify_files:
            data = (manifest_path.parent.parent / record["path"]).resolve()
            if not data.is_file(): raise ManifestError(f"{cid}: data file unavailable: {data}")
            if hashlib.sha256(data.read_bytes()).hexdigest() != record["sha256"]: raise ManifestError(f"{cid}: sha256 mismatch")
    return manifest

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("manifest", nargs="?", type=Path, default=ROOT / "corpus/manifest.json"); parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()
    try: manifest = validate(args.manifest, verify_files=args.verify_files)
    except ManifestError as exc: print(f"MANIFEST INVALID: {exc}"); return 2
    counts = {tier: sum(c["tier"] == tier for c in manifest["captures"]) for tier in ("T0", "T1", "T2")}
    print(json.dumps({"manifest": str(args.manifest), "valid": True, "capture_counts": counts, "t2_status": "EMPTY" if not counts["T2"] else "PARTIAL_OR_POPULATED"}, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
