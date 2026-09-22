"""Configured native LDPC decoding; blind matrix/profile inference is not supported."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .models import DeinterleavingResult, Diagnostic, FECDecodeResult, Severity
from .native import decode_ldpc as native_decode_ldpc, require_native


def decode_ldpc(deint: DeinterleavingResult, profile: Dict[str, Any]) -> FECDecodeResult:
    """Run normalized min-sum against an explicit sparse matrix or `.alist` file."""
    native = require_native()
    matrix = native.LdpcMatrix()
    if "alist_path" in profile:
        matrix = native.LdpcMatrix.from_alist(str(profile["alist_path"]))
    elif "checks" in profile and "variable_count" in profile:
        matrix.variable_count = int(profile["variable_count"])
        matrix.checks = [[int(index) for index in row] for row in profile["checks"]]
    else:
        return FECDecodeResult(
            decoded_bits=np.zeros(0, dtype=np.uint8), corrected_bit_count=0,
            corrected_bit_fraction=0.0, decode_success=False, codec_name="LDPC",
            pre_correction_metric=0.0,
            diagnostics=[Diagnostic(Severity.WARNING, "LDPC_MATRIX_REQUIRED", "LDPC decoding requires an explicit sparse matrix or alist_path.", "")],
        )
    llrs = np.ascontiguousarray(deint.llrs_reordered, dtype=np.float32)
    if llrs.size != matrix.variable_count:
        return FECDecodeResult(
            decoded_bits=np.zeros(0, dtype=np.uint8), corrected_bit_count=0,
            corrected_bit_fraction=0.0, decode_success=False, codec_name="LDPC",
            pre_correction_metric=0.0,
            diagnostics=[Diagnostic(Severity.WARNING, "LDPC_LENGTH_MISMATCH", "LLR length does not match the configured LDPC matrix.", f"llrs={llrs.size}, variables={matrix.variable_count}")],
        )
    config = native.LdpcConfig()
    config.maximum_iterations = int(profile.get("maximum_iterations", 50))
    config.normalization = float(profile.get("normalization", 0.8))
    result = native_decode_ldpc(llrs, matrix=matrix, config=config)
    diagnostics = [
        Diagnostic(getattr(Severity, item.severity.name), item.code, item.message, item.evidence)
        for item in result.diagnostics
    ]
    return FECDecodeResult(
        decoded_bits=np.asarray(result.decoded_bits), corrected_bit_count=0,
        corrected_bit_fraction=0.0, decode_success=bool(result.converged),
        codec_name="LDPC normalized min-sum", pre_correction_metric=float(result.syndrome_weight),
        diagnostics=diagnostics,
    )
