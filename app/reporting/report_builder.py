from __future__ import annotations
from pathlib import Path
from typing import Any
from app.orchestration.pipeline_runner import PipelineResult
from .artifact_manifest import export_all_artifacts
from .csv_export import export_frames_csv, export_parameters_csv
from .html_report import build_html_report, export_html_report
from .json_report import build_json_report, export_json_report

class ReportBuilder:
    @staticmethod
    def to_json(result: PipelineResult) -> dict[str, Any]:
        return build_json_report(result)

    @staticmethod
    def to_html(result: PipelineResult) -> str:
        return build_html_report(result)

    @staticmethod
    def export(result: PipelineResult, output_dir: str | Path) -> dict[str, str]:
        return export_all_artifacts(result, output_dir)
