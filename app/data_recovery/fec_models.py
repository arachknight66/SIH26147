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
]
