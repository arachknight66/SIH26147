from __future__ import annotations
from pathlib import Path
from typing import Any
from app.orchestration.pipeline_config import PipelineConfig, get_preset_config
from app.orchestration.pipeline_runner import PipelineResult, run_pipeline
from .experiment import ExperimentBundle, load_experiment

def replay_experiment(bundle_or_path: str | Path | ExperimentBundle) -> tuple[PipelineResult, bool]:
    """
    Rerun an experiment from its saved bundle and check whether the reproducibility hash matches.
    Returns (PipelineResult, hash_matches).
    """
    if isinstance(bundle_or_path, ExperimentBundle):
        bundle = bundle_or_path
    else:
        bundle = load_experiment(bundle_or_path)

    # Reconstruct configuration
    cfg = get_preset_config(bundle.config_dict.get("preset", "standard_analysis"), seed=bundle.config_dict.get("random_seed", 42))

    res = run_pipeline(bundle.source_path, config=cfg)
    new_hash = res.provenance.reproducibility_hash if res.provenance else ""
    matches = bool(new_hash == bundle.reproducibility_hash)
    return res, matches
