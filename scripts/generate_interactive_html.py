from __future__ import annotations
import json
from pathlib import Path
from app.reporting.json_report import build_json_report
from app.orchestration.pipeline_runner import run_pipeline
from app.orchestration.pipeline_config import get_preset_config, PresetName
from scripts.generate_digital_dataset import generate_digital_stream
from tests.test_phase6_cases import _make_rec_sig
from app.ui.web_app import HTML_TEMPLATE

def generate_interactive_dashboard(file_path: str = "dashboard.html") -> None:
    # 1. Run complete pipeline
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    res = run_pipeline(rec, config=get_preset_config(PresetName.FAST_SCREENING))
    data = build_json_report(res)
    json_str = json.dumps(data)

    # 2. Inject initial payload into the HTML template
    injected_html = HTML_TEMPLATE.replace(
        "let currentData = null;",
        f"let currentData = {json_str};\nwindow.addEventListener('DOMContentLoaded', () => {{ updateUI(currentData); }});"
    )

    # 3. Write standalone file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(injected_html)
    print(f"Interactive standalone dashboard written to: {Path(file_path).resolve()}")

if __name__ == "__main__":
    generate_interactive_dashboard("dashboard.html")
