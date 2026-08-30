from __future__ import annotations
from .models import FECCodeFamily, FECHypothesis

STANDARD_FEC_CONFIGURATIONS: list[FECHypothesis] = [
    FECHypothesis(
        code_family=FECCodeFamily.NONE,
        code_name="UNCODED",
        rate=1.0,
        constraint_length=None,
        generator_polynomials=(),
        block_size=None,
        confidence=0.80,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.CONVOLUTIONAL,
        code_name="CONV_K7_R12_NASA",
        rate=0.5,
        constraint_length=7,
        generator_polynomials=(0o133, 0o171),  # (1011011_2, 1111001_2) = (91, 121)
        block_size=None,
        assumptions=("Non-punctured", "Zero-tail termination"),
        confidence=0.50,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.CONVOLUTIONAL,
        code_name="CONV_K3_R12",
        rate=0.5,
        constraint_length=3,
        generator_polynomials=(0o7, 0o5),      # (111_2, 101_2) = (7, 5)
        block_size=None,
        assumptions=("Non-punctured", "Zero-tail termination"),
        confidence=0.40,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.HAMMING,
        code_name="HAMMING_7_4",
        rate=4.0 / 7.0,
        constraint_length=None,
        generator_polynomials=(),
        block_size=7,
        confidence=0.30,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.HAMMING,
        code_name="HAMMING_8_4",
        rate=4.0 / 8.0,
        constraint_length=None,
        generator_polynomials=(),
        block_size=8,
        confidence=0.30,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.REPETITION,
        code_name="REPETITION_3",
        rate=1.0 / 3.0,
        constraint_length=None,
        generator_polynomials=(),
        block_size=3,
        confidence=0.20,
        valid=True,
    ),
    # -------------------------------------------------------------------------
    # Reed-Solomon Standard Configurations
    # -------------------------------------------------------------------------
    FECHypothesis(
        code_family=FECCodeFamily.REED_SOLOMON,
        code_name="RS_255_223_CCSDS",
        rate=223.0 / 255.0,
        constraint_length=None,
        generator_polynomials=(0x187,),  # Dual basis / CCSDS Blue Book 131.0-B-3
        block_size=255,
        assumptions=("N=255, K=223, 2t=32 (t=16), fcr=112, poly=0x187", "CCSDS telemetry outer code"),
        confidence=0.50,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.REED_SOLOMON,
        code_name="RS_255_239_DVB",
        rate=239.0 / 255.0,
        constraint_length=None,
        generator_polynomials=(0x11D,),  # DVB / ITU-T J.83 standard
        block_size=255,
        assumptions=("N=255, K=239, 2t=16 (t=8), fcr=0, poly=0x11D", "DVB/ITU digital broadcast outer code"),
        confidence=0.45,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.REED_SOLOMON,
        code_name="RS_204_188_DVB_SHORTENED",
        rate=188.0 / 204.0,
        constraint_length=None,
        generator_polynomials=(0x11D,),  # DVB-T / DVB-S MPEG-TS outer code
        block_size=204,
        assumptions=("N=204, K=188, 2t=16 (t=8), fcr=0, poly=0x11D", "DVB MPEG transport stream packet code"),
        confidence=0.50,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.REED_SOLOMON,
        code_name="RS_128_112_TELEMETRY",
        rate=112.0 / 128.0,
        constraint_length=None,
        generator_polynomials=(0x11D,),  # Medium-length telemetry frame code
        block_size=128,
        assumptions=("N=128, K=112, 2t=16 (t=8), fcr=0, poly=0x11D", "Standard telemetry framing code"),
        confidence=0.40,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.REED_SOLOMON,
        code_name="RS_64_48_COMPACT",
        rate=48.0 / 64.0,
        constraint_length=None,
        generator_polynomials=(0x11D,),  # Compact packet telemetry
        block_size=64,
        assumptions=("N=64, K=48, 2t=16 (t=8), fcr=1, poly=0x11D", "Compact packet telemetry code"),
        confidence=0.35,
        valid=True,
    ),
    # -------------------------------------------------------------------------
    # Concatenated Standard Configurations (Lower priors reflecting model complexity)
    # -------------------------------------------------------------------------
    FECHypothesis(
        code_family=FECCodeFamily.CONCATENATED,
        code_name="CCSDS_CONCATENATED_TELEMETRY",
        rate=(223.0 / 255.0) * 0.5,
        constraint_length=7,
        generator_polynomials=(0x187, 0o133, 0o171),
        block_size=255,
        assumptions=("Outer: RS(255,223)", "Interleaver: Block", "Inner: Conv(7,[133,171])", "CCSDS 131.0-B-3 telemetry"),
        confidence=0.35,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.CONCATENATED,
        code_name="DVB_S_CONCATENATED_BROADCAST",
        rate=(188.0 / 204.0) * 0.5,
        constraint_length=7,
        generator_polynomials=(0x11D, 0o133, 0o171),
        block_size=204,
        assumptions=("Outer: RS(204,188)", "Interleaver: Conv (M=4,D=2)", "Inner: Conv(7,[133,171])", "DVB-S broadcast"),
        confidence=0.35,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.CONCATENATED,
        code_name="VOYAGER_CONCATENATED_CLASSIC",
        rate=(223.0 / 255.0) * 0.5,
        constraint_length=7,
        generator_polynomials=(0x187, 0o133, 0o171),
        block_size=255,
        assumptions=("Outer: RS(255,223)", "Interleaver: Block (16x8)", "Inner: Conv(7,[133,171])", "Voyager telemetry"),
        confidence=0.30,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.CONCATENATED,
        code_name="COMPACT_CONCATENATED_PACKET",
        rate=(48.0 / 64.0) * 0.5,
        constraint_length=7,
        generator_polynomials=(0x11D, 0o133, 0o171),
        block_size=64,
        assumptions=("Outer: RS(64,48)", "Interleaver: Block (8x8)", "Inner: Conv(7,[133,171])", "Micro-satellite telemetry"),
        confidence=0.35,
        valid=True,
    ),
    # -------------------------------------------------------------------------
    # LDPC Standard Configurations (Quasi-Cyclic and Gallager Regular)
    # Permanent Epistemic Note: Blind unconstrained LDPC matrix discovery from raw
    # data is an ill-posed inverse problem; hypothesis search is strictly restricted
    # to standard registered codes with known structural priors.
    # -------------------------------------------------------------------------
    FECHypothesis(
        code_family=FECCodeFamily.LDPC,
        code_name="QC_LDPC_N128_R12",
        rate=0.5,
        constraint_length=None,
        generator_polynomials=(),
        block_size=128,
        assumptions=("N=128, M=64, K=64, R=0.5, Z=16", "Quasi-cyclic block LDPC, girth >= 6"),
        confidence=0.35,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.LDPC,
        code_name="QC_LDPC_N256_R12",
        rate=0.5,
        constraint_length=None,
        generator_polynomials=(),
        block_size=256,
        assumptions=("N=256, M=128, K=128, R=0.5, Z=32", "Quasi-cyclic block LDPC, girth >= 6"),
        confidence=0.35,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.LDPC,
        code_name="GALLAGER_N96_R12",
        rate=0.52,
        constraint_length=None,
        generator_polynomials=(),
        block_size=96,
        assumptions=("N=96, M=48, K=50, R=0.52", "Gallager (3,6) regular LDPC, girth >= 6"),
        confidence=0.30,
        valid=True,
    ),
    FECHypothesis(
        code_family=FECCodeFamily.LDPC,
        code_name="GALLAGER_N192_R12",
        rate=0.51,
        constraint_length=None,
        generator_polynomials=(),
        block_size=192,
        assumptions=("N=192, M=96, K=98, R=0.51", "Gallager (3,6) regular LDPC, girth >= 6"),
        confidence=0.30,
        valid=True,
    ),
]


