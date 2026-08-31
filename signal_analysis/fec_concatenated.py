import numpy as np
from typing import Tuple, List
from .models import DemodulationResult, DeinterleavingResult, FECDecodeResult, DeinterleaverFamily, Diagnostic, Severity
from .deinterleaving import attempt_deinterleaving, _deinterleave_block
from .fec_convolutional import viterbi_decode_soft
from .fec_reed_solomon import decode_reed_solomon

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
    # 1. Inner Viterbi
    
    
    # 2. De-interleave inner output
    # To do this, we need to create a dummy DemodulationResult to feed the deinterleaver search
    # But wait, the prompt says: 
    # "The FEC decoder's input type should only be constructible from a DeinterleavingResult"
    # For Viterbi to run first, it must be running on a DeinterleavingResult IF it was interleaved?
    # Actually, standard concatenated is: Viterbi -> Deinterleave -> RS.
    # The interleaver sits BETWEEN RS and Viterbi to spread burst errors that Viterbi emits!
    # Yes, Viterbi burst errors need to be spread before RS.
    # So the pipeline is:
    # Channel -> Viterbi (Inner) -> Deinterleaver -> RS (Outer).
    # Since Viterbi input isn't interleaved, it just takes DemodulationResult or a NONE DeinterleavingResult.
    
    # Let's wrap DemodulationResult into a NONE DeinterleavingResult for Viterbi
    # to strictly satisfy "FEC decoder's input type should only be constructible from a DeinterleavingResult"
    from .models import DeinterleaverHypothesis, HypothesisStatus
    none_hyp = DeinterleaverHypothesis(DeinterleaverFamily.NONE, {}, 0.0, [], HypothesisStatus.HYPOTHESIS_UNVERIFIED)
    initial_deint = DeinterleavingResult(demod_result.hard_bits, demod_result.soft_llrs, none_hyp, 0.0)
    
    viterbi_res = viterbi_decode_soft(initial_deint)
    
    # Now we need to deinterleave the Viterbi output. We build a fake DemodulationResult to search interleavers on it.
    fake_demod = DemodulationResult(
        hard_bits=viterbi_res.decoded_bits,
        soft_llrs=np.zeros_like(viterbi_res.decoded_bits, dtype=np.float32), # No soft LLRs after Viterbi hard output
        bits_per_symbol=1,
        symbol_decisions=np.array([]),
        sync_result=demod_result.sync_result,
        source_hypothesis_label="VITERBI_OUT",
        hypothesis_confirmed=True
    )
    
    deint_res, _ = attempt_deinterleaving(fake_demod, config)
    
    # 3. Outer RS
    rs_res = decode_reed_solomon(deint_res)
    
    return viterbi_res, rs_res, deint_res
