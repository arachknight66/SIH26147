# SIH26147 implementation guidance

## Read first

- [prd.md](prd.md): product requirements and beta scope.
- [plan.md](plan.md): the seven-phase migration plan, interfaces, dependencies, and acceptance gates.
- [progress.md](progress.md): completed work, verification evidence, and next steps.

The repository currently contains the Python prototype. The native architecture in the plan is a target, not an implemented capability. Check executable paths before trusting existing README, architecture, demo narration, or historical test claims.

## Working approach

- Follow the user's current task and implement the requested scope. The presence of the full plan does not authorize implementing unrelated phases in the same task.
- Migrate incrementally. Preserve the existing Qt GUI, pyqtgraph views, Demo Mode, and useful Python-facing entry points. Avoid an unnecessary GUI or application rewrite.
- Put meaningful DSP, estimation, synchronization, demodulation, deinterleaving, FEC, and bitstream computation in modular C++20 components. Keep GUI, scheduling, configuration, plotting, and reporting in Python.
- Keep the C++ library independently buildable and testable without Python or Qt. Isolate third-party libraries behind narrow adapters.
- Maintain a runnable application at each phase boundary. During migration, distinguish an explicitly selected legacy/reference path from the native path; never silently fall back and present the result as native.
- Keep Linux and Windows build compatibility in scope from the foundation phase. Pin dependencies and preserve license notices.

## Scientific correctness

- Distinguish known metadata, user assumptions, estimates, and missing values. Do not infer an absolute sample rate from bare IQ bytes or mark an untouched UI default as known metadata.
- Distinguish stage execution from hypothesis confidence and decode validity. A decoder finishing, low EVM, or one short sync/CRC match is not sufficient evidence of the correct processing chain.
- Preserve unknown, ambiguous, unsupported, uncoded, and search-budget-exhausted outcomes. Do not force a FEC chain or silently discard failed alternatives.
- Preserve bit order, constellation mapping, code parameters, interleaving units, frame boundaries, and sample/bit provenance. Make acquisition transients, padding, residual data, and decoder termination explicit.
- Keep positive LLR = bit 1 at the public boundary. Adapt library sign conventions explicitly. Do not invent soft information after hard decoding.
- Normalize appropriately before nonlinear operations and test numerical range; existing audit runs emitted FFT overflow warnings.

## Python/C++ boundary and jobs

- Use coarse operations on chunks, windows, and frames. Avoid per-sample Python callbacks or conversions through Python lists.
- Validate dtype, shape, strides, alignment requirements, and endianness. Borrow compatible arrays with retained ownership; expose any required conversion copy.
- Keep input buffers alive and unmodified while native work uses them. Returned NumPy views must retain the owning native allocation.
- Release the GIL for compute. Do not access Python objects without the GIL. Deliver progress/results through the orchestration layer; update Qt widgets only on the main thread.
- Preserve native state across chunks and support cooperative cancellation. Report actual processed coverage and use bounded memory.

## Demo integrity

- Demo Mode must submit the same production job and source adapter as ordinary files.
- Fixtures must be deterministic and genuinely contain their advertised modulation, interleaving, and FEC.
- Keep ground truth separate from capture/import metadata and production configuration. Only the reveal/evaluation path may read it.
- Do not select analysis results or decoder profiles by fixture filename, demo label, or hidden truth. Test that renaming a capture preserves its analysis result.
- User-selected profiles are allowed, but they must be explicit and available through the ordinary workflow as well.

## Verification and documentation

- Use independent golden vectors and reference implementations for algorithm acceptance. Old Python output is a regression reference only where its correctness has been established.
- Check recovered bits/payloads, not merely nonempty output, a green stage label, or a successful function return. Include wrong-profile and noise/unsupported cases.
- Run relevant native tests, binding tests, and Python integration tests. Exercise the real decoder chain for end-to-end tests; mocks are appropriate only for isolated orchestration/UI behavior.
- Do not run the old suite blindly in the working tree: `tests/test_gui_pipeline_integration.py` overwrites `dab_test.wav`, and root `test_qpsk_cfo.py` writes `test_16qam_cfo.wav` at import time. Isolate or repair these tests within the authorized task before running them.
- Do not execute root `append*`, `fix_gui*`, or `patch*` scripts as setup; they mutate source files.
- Update [progress.md](progress.md) with the actual changes, checks, outcomes, and remaining blockers after implementation tasks. Record skipped or unavailable checks explicitly.
- Mark phases complete only when their acceptance criteria have evidence. Performance thresholds in the plan are targets until measured on recorded reference hardware.
- Keep [prd.md](prd.md) and [plan.md](plan.md) aligned with explicit scope decisions. Report material deviations rather than silently reducing promised coverage.
