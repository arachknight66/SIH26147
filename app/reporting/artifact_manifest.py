from __future__ import annotations
import json
from pathlib import Path
from app.orchestration.pipeline_runner import PipelineResult
from .csv_export import export_frames_csv, export_parameters_csv
from .html_report import export_html_report
from .json_report import export_json_report

def export_all_artifacts(result: PipelineResult, output_dir: str | Path) -> dict[str, str]:
    """
    Export all human-readable and machine-readable artifacts into a target directory.
    Returns mapping of artifact type to absolute path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "report.json"
    html_path = out / "report.html"
    frames_csv_path = out / "frames.csv"
    params_csv_path = out / "parameters.csv"
    manifest_path = out / "manifest.json"

    export_json_report(result, str(json_path))
    export_html_report(result, str(html_path))
    export_frames_csv(result, str(frames_csv_path))
    export_parameters_csv(result, str(params_csv_path))

    if result.provenance:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(result.provenance.__dict__, f, indent=2)

    return {
        "json_report": str(json_path.resolve()),
        "html_report": str(html_path.resolve()),
        "frames_csv": str(frames_csv_path.resolve()),
        "parameters_csv": str(params_csv_path.resolve()),
        "manifest": str(manifest_path.resolve()) if result.provenance else "",
    }
