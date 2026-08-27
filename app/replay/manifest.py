from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def read_manifest(manifest_path: str | Path) -> dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

def verify_manifest_integrity(manifest_dict: dict[str, Any]) -> bool:
    required_keys = ["input_sha256", "configuration_hash", "reproducibility_hash", "software_version"]
    return all(k in manifest_dict for k in required_keys)
