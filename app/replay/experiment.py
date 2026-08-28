from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from app.orchestration.pipeline_config import PipelineConfig, get_preset_config
from app.orchestration.pipeline_runner import PipelineResult, run_pipeline

@dataclass
class ExperimentBundle:
    experiment_id: str
    source_path: str
    source_sha256: str
    config_dict: dict[str, Any]
    reproducibility_hash: str
    report_json: dict[str, Any]

def save_experiment(result: PipelineResult, file_path: str | Path) -> None:
    """Save an experiment bundle for deterministic replay."""
    prov = result.provenance
    data = {
        "experiment_id": prov.reproducibility_hash[:16] if prov else "unknown",
        "source_path": result.input_path or "in_memory",
        "source_sha256": result.input_sha256,
        "config_dict": prov.configuration_dict if prov else {},
        "reproducibility_hash": prov.reproducibility_hash if prov else "unknown",
        "report_json": result.phase6_result.output.__dict__ if (result.phase6_result and result.phase6_result.output) else {},
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

def load_experiment(file_path: str | Path) -> ExperimentBundle:
    """Load an experiment bundle from disk."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ExperimentBundle(
        experiment_id=data.get("experiment_id", "unknown"),
        source_path=data.get("source_path", ""),
        source_sha256=data.get("source_sha256", ""),
        config_dict=data.get("config_dict", {}),
        reproducibility_hash=data.get("reproducibility_hash", ""),
        report_json=data.get("report_json", {}),
    )
