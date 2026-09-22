"""Run the reproducible local validation sequence and retain machine-readable evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], environment: dict[str, str]) -> dict:
    started = time.time()
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "duration_seconds": round(time.time() - started, 3),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SIH26147 validation gates and write a JSON transcript")
    parser.add_argument("--output", type=Path, default=Path("build/validation-report.json"))
    parser.add_argument("--skip-native", action="store_true", help="Skip CTest; useful when only Python changed")
    parser.add_argument("--quick", action="store_true", help="Run focused binding, integration, and release tests")
    args = parser.parse_args()
    environment = dict(os.environ)
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    commands: list[list[str]] = []
    if not args.skip_native:
        commands.extend([
            ["cmake", "--preset", "native-release"],
            ["cmake", "--build", "--preset", "native-release"],
            ["ctest", "--preset", "native-release"],
        ])
    tests = [
        "tests/test_native_bindings.py", "tests/test_phase2_native.py", "tests/test_phase3_native.py",
        "tests/test_phase4_native.py", "tests/test_phase5_native.py", "tests/test_phase6_workflow.py",
        "tests/test_phase7_release.py", "tests/test_cli.py", "tests/test_gui.py",
        "tests/test_gui_pipeline_integration.py",
    ] if args.quick else ["tests"]
    commands.append([sys.executable, "-m", "pytest", "-q", *tests])
    report = {"validation_schema_version": 1, "quick": args.quick, "checks": []}
    for command in commands:
        check = _run(command, environment)
        report["checks"].append(check)
        if check["returncode"] != 0:
            break
    report["passed"] = all(check["returncode"] == 0 for check in report["checks"])
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validation report: {output}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
