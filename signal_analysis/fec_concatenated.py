import numpy as np
from typing import Tuple, List
from .models import DemodulationResult, DeinterleavingResult, FECDecodeResult, DeinterleaverFamily, Diagnostic, Severity
from .deinterleaving import attempt_deinterleaving, _deinterleave_block
from .fec_convolutional import viterbi_decode_soft
from .fec_reed_solomon import decode_reed_solomon
from .fec_ldpc import decode_ldpc

from typing import Optional, Dict, Any
def decode_concatenated(demod_result: DemodulationResult, config: Optional[Dict[str, Any]] = None) -> Tuple[FECDecodeResult, FECDecodeResult, DeinterleavingResult]:
    """
    Standard RS(outer) + interleaver + convolutional(inner) composition.
    Enforces pipeline ordering: Viterbi (inner) -> Deinterleave -> RS (outer).
    LDPC is explicitly OUT OF SCOPE.
    
    Since Viterbi runs first, it uses the soft LLRs straight from demod_result.
    Then we run deinterleaving on Viterbi's hard bit output.
    Then RS runs on the de-interleaved bits.
    """
    config = config or {}
    profile = str(config.get("fec_profile", "UNCODED")).upper()
    # Uncoded is a first-class candidate.  Do not force a speculative FEC chain
    # onto an unfamiliar recording merely because the pipeline reached this stage.
    if profile == "UNCODED":
        deint_res, _ = attempt_deinterleaving(demod_result, config)
        uncoded = FECDecodeResult(
            decoded_bits=deint_res.bits, corrected_bit_count=0, corrected_bit_fraction=0.0,
            decode_success=True, codec_name="Uncoded", pre_correction_metric=0.0,
            diagnostics=[Diagnostic(Severity.INFO, "FEC_UNCODED_CANDIDATE", "No FEC profile was selected; preserved the deinterleaved bitstream.", "")],
        )
        return uncoded, uncoded, deint_res

    # The inner decoder consumes raw demodulator LLRs.  Its hard output carries
    # no fabricated post-decoder soft information.
    from .models import DeinterleaverHypothesis, HypothesisStatus
    none_hyp = DeinterleaverHypothesis(DeinterleaverFamily.NONE, {}, 0.0, [], HypothesisStatus.HYPOTHESIS_UNVERIFIED)
    initial_deint = DeinterleavingResult(demod_result.hard_bits, demod_result.soft_llrs, none_hyp, 0.0)
    if profile == "RS_255_223":
        deint_res, _ = attempt_deinterleaving(demod_result, config)
        rs_res = decode_reed_solomon(deint_res)
        return rs_res, rs_res, deint_res
    if profile == "LDPC":
        deint_res, _ = attempt_deinterleaving(demod_result, config)
        ldpc_res = decode_ldpc(deint_res, config.get("ldpc_profile", {}))
        return ldpc_res, ldpc_res, deint_res
    if profile not in {"CONVOLUTIONAL_K7_R12", "CONCATENATED_K7_RS_255_223"}:
        unsupported = FECDecodeResult(
            decoded_bits=np.zeros(0, dtype=np.uint8), corrected_bit_count=0, corrected_bit_fraction=0.0,
            decode_success=False, codec_name=profile, pre_correction_metric=0.0,
            diagnostics=[Diagnostic(Severity.WARNING, "FEC_PROFILE_UNSUPPORTED", "The selected FEC profile is not implemented.", profile)],
        )
        return unsupported, unsupported, initial_deint

    viterbi_res = viterbi_decode_soft(initial_deint)
    fake_demod = DemodulationResult(
        hard_bits=viterbi_res.decoded_bits,
        soft_llrs=np.zeros(0, dtype=np.float32),
        bits_per_symbol=1,
        symbol_decisions=np.array([]),
        sync_result=demod_result.sync_result,
        source_hypothesis_label="VITERBI_OUT",
        hypothesis_confirmed=True
    )
    
    deint_res, _ = attempt_deinterleaving(fake_demod, config)
    if profile == "CONVOLUTIONAL_K7_R12":
        return viterbi_res, viterbi_res, deint_res
    rs_res = decode_reed_solomon(deint_res)
    return viterbi_res, rs_res, deint_res
