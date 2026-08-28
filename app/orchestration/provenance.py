from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import os
import platform
import sys
from typing import Any
from .pipeline_config import PipelineConfig

@dataclass(frozen=True)
class ProvenanceManifest:
    input_sha256: str
    software_name: str
    software_version: str
    python_version: str
    platform_system: str
    platform_machine: str
    configuration_hash: str
    configuration_dict: dict[str, Any]
    random_seed: int
    reproducibility_hash: str
    timestamp_utc: str
    stage_durations: dict[str, float] = field(default_factory=dict)
    user_overrides: dict[str, Any] = field(default_factory=dict)

def compute_reproducibility_hash(
    input_sha256: str,
    config_hash: str,
    stage_hashes: dict[str, str],
) -> str:
    """
    Compute a deterministic reproducibility hash that is independent of wall-clock time
    and file system paths.
    """
    elements = {
        "input_sha256": input_sha256,
        "config_hash": config_hash,
        "stages": stage_hashes,
    }
    s = json.dumps(elements, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def build_provenance_manifest(
    input_sha256: str,
    config: PipelineConfig,
    stage_hashes: dict[str, str],
    stage_durations: dict[str, float],
    timestamp_utc: str,
    software_version: str = "0.7.0",
) -> ProvenanceManifest:
    config_hash = config.compute_hash()
    repro_hash = compute_reproducibility_hash(input_sha256, config_hash, stage_hashes)

    return ProvenanceManifest(
        input_sha256=input_sha256,
        software_name="SIH26147 Signal Recovery & Scientific Verification Engine",
        software_version=software_version,
        python_version=sys.version.split()[0],
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        configuration_hash=config_hash,
        configuration_dict=config.to_dict(),
        random_seed=config.random_seed,
        reproducibility_hash=repro_hash,
        timestamp_utc=timestamp_utc,
        stage_durations=stage_durations,
        user_overrides=config.user_overrides,
    )
