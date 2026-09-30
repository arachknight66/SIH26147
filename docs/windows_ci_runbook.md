# Windows native CI runbook

## Evidence status

As of 2026-09-29, Windows execution is **NOT-YET-RUN**. This document is a handoff for producing a real transcript; it is not evidence that the project builds on Windows.

The repository workflow [`.github/workflows/native.yml`](../.github/workflows/native.yml) is reachable on `windows-latest`: `build-and-test` has `os: [ubuntu-latest, windows-latest]`, no `exclude` entries or job/step `if:` guards, and `fail-fast: false`. Therefore a Windows failure remains visible even if the Ubuntu matrix entry fails.

This workflow validates the native build, selected Python tests, and a GUI
import. It does **not** install or exercise the separate GNU Radio runtime
required by ordinary GUI/CLI/Demo analysis jobs, and it does not validate the
optional CUDA or ML extras.

## GitHub Actions path

The workflow triggers on `push` and `pull_request` (it has no `workflow_dispatch`). Push the reviewed commit or open/update a pull request, then save the complete **build-and-test (windows-latest, 3.12)** log as `windows-native-YYYY-MM-DD.log`. Include its Actions URL and commit SHA with the transcript.

## Equivalent local Windows runner path

Use a clean Windows Server 2022 or Windows 11 x64 machine with the Visual Studio 2022 Build Tools C++ workload, CMake >= 3.20, Git, and PowerShell. `windows-latest` normally exposes the Visual Studio/MSVC environment; because the workflow does not select a generator, record the first CMake configure banner (generator and `MSVC` compiler version) rather than assuming it.

Open an **x64 Native Tools Command Prompt for VS 2022** at the repository root. Install uv using the current official Windows installer if it is not already on PATH, then execute these commands in order, without adding configuration flags:

```powershell
uv --version
cmake --version
cl

cmake -S . -B build/native-tests -DSIH_BUILD_PYTHON=OFF -DSIH_BUILD_TESTS=ON
cmake --build build/native-tests --config Release
ctest --test-dir build/native-tests -C Release --output-on-failure

uv sync --extra test
uv run pytest tests/test_native_bindings.py tests/test_phase2_native.py tests/test_measurements.py tests/test_loaders.py
uv run --extra gui python -c "from signal_analysis.gui import HAS_QT; assert HAS_QT"
```

Those commands mirror every Windows step in `native.yml` after checkout and `astral-sh/setup-uv@v6`; do not replace them with presets or a Linux shell translation. Redirect all output before running if a file is preferred:

```powershell
& { <commands above> } *>&1 | Tee-Object -FilePath windows-native-YYYY-MM-DD.log
```

## Risk register: observations, not pre-emptive fixes

| Area | Look for in the transcript | Finding if observed | Likely follow-up after evidence |
|---|---|---|---|
| A — compiler flags | `MSVC` / Visual Studio generator banner and successful `/W4 /permissive-` compilation | MSVC branch is exercised. If GCC-style `-Wall` fails under MSVC, generator/toolchain selection differs from expectation. | Pin/configure a compatible generator or correct the conditional only after preserving the failing log. |
| B — dependency probes | Three `Native dependency probe:` lines, whether `found` or `not found` | Runner package-discovery state is unobserved. The current probes only emit status messages and do not switch the build's linked/compiled path. | Record any found package/config location; investigate a code-path difference only if a build rule consumes it. |
| C — paths/encoding | Configure/build errors containing backslash paths, missing generated files, or Unicode/path failures | Windows filesystem semantics exposed a path/encoding defect. | Apply the smallest `pathlib`/CMake install-path correction against the captured failure. |
| D — wheel extension | Whether the wheel contains `signal_analysis/_native*.pyd`; import outcome | **Install-destination hypothesis resolved statically:** `_native` is a `MODULE` target and CMake installs modules via `LIBRARY DESTINATION` on Windows, which the project already supplies. A missing `.pyd` or DLL-load error would be a new, unobserved packaging/runtime issue. | Inspect the actual wheel and loader error; fix only an observed issue. See [static risk review](windows_known_risks.md). |
| E — GUI extra | The final import command fails, especially Qt platform-plugin errors | Windows GUI wheel/platform assumptions differ from the runner. | Record exact plugin error and add the minimal Windows-safe CI environment/setup change after evidence. |

## Progress status template (do not paste as a claim before a transcript exists)

```text
- Windows native evidence (YYYY-MM-DD): [PASS | FAIL | NOT-YET-RUN].
  Source: [GitHub Actions URL or attached local transcript], commit [SHA].
  Toolchain: [generator], [compiler/version], Python [version].
  Probes: FFTW3 [found/not found], libsndfile [found/not found], AFF3CT [found/not found].
  Native CTest: [result]. Python binding/Phase 2 tests: [result]. GUI import: [result].
  Findings/fixes: [none, or link to captured before/after evidence].
```
