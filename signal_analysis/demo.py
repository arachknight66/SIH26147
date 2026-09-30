"""Manifest-backed deterministic Demo Mode without analysis-time ground truth access."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loaders import RawIQConfig
from .models import MetadataStatus
from .workflow import AnalysisRequest


_ROOT = Path(__file__).resolve().parents[1]
_DEMO_DIR = _ROOT / "fixtures" / "data_"
_CATALOG_PATH = _DEMO_DIR / "catalog.json"
_TRUTH_PATH = _DEMO_DIR / "truth.json"


@dataclass(frozen=True)
class DemoFixture:
    identifier: str
    title: str
    filename: str
    narration: str
    wav_stereo_mode: str = "unresolved"
    raw_iq: dict[str, Any] | None = None

    @property
    def path(self) -> Path:
        return _DEMO_DIR / self.filename

    def analysis_request(self, pipeline_config: dict[str, Any] | None = None) -> AnalysisRequest:
        raw_iq_config = None
        if self.raw_iq is not None:
            settings = dict(self.raw_iq)
            status = settings.get("sample_rate_status")
            if isinstance(status, str):
                settings["sample_rate_status"] = MetadataStatus[status]
            raw_iq_config = RawIQConfig(**settings)
        return AnalysisRequest(
            path=self.path,
            wav_stereo_mode=self.wav_stereo_mode,
            raw_iq_config=raw_iq_config,
            pipeline_config=dict(pipeline_config or {}),
            origin="demo",
        )


def demo_catalog() -> list[DemoFixture]:
    entries = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    return [DemoFixture(**entry) for entry in entries]


def get_demo(identifier: str) -> DemoFixture:
    for fixture in demo_catalog():
        if fixture.identifier == identifier:
            return fixture
    raise KeyError(f"unknown demo fixture: {identifier}")


def reveal_ground_truth(identifier: str) -> dict[str, Any]:
    """Read separate evaluation metadata only after an explicit reveal action."""
    truth = json.loads(_TRUTH_PATH.read_text(encoding="utf-8"))
    if identifier not in truth:
        raise KeyError(f"no ground truth for demo fixture: {identifier}")
    return dict(truth[identifier])
