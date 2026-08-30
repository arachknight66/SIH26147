from __future__ import annotations
import pytest
import numpy as np
from app.data_recovery.galois_field import GaloisField, STANDARD_PRIMITIVE_POLYNOMIALS


@pytest.mark.parametrize("m,poly_dict", STANDARD_PRIMITIVE_POLYNOMIALS.items())
def test_galois_field_construction_and_closure(m: int, poly_dict: dict[str, int]):
    """
    Verify finite field construction, cyclic group generation, and closure across
    all standard primitive polynomials in dimensions m=3..8.
    """
    for name, poly in poly_dict.items():
        gf = GaloisField(symbol_width=m, prim_poly=poly)
        assert gf.field_order == (1 << m)
        assert gf.cardinality == (1 << m) - 1

        # Assert alpha^(2^m - 1) == 1
        assert gf.alpha(gf.cardinality) == 1
        assert gf.alpha(0) == 1

        # Assert cyclic order: all powers alpha^0 .. alpha^(2^m - 2) are unique
        powers = [gf.alpha(i) for i in range(gf.cardinality)]
        assert len(set(powers)) == gf.cardinality
        assert 0 not in powers


def test_galois_field_arithmetic_axioms():
    """
    Exhaustively verify algebraic field axioms in GF(2^8):
    1. Additive identity: a + 0 = a
    2. Additive inverse: a + a = 0 (characteristic 2)
    3. Multiplicative identity: a * 1 = a
    4. Multiplicative inverse: a * inv(a) = 1 for all a != 0
    5. Commutativity of addition and multiplication
    6. Distributivity: a * (b + c) = (a * b) + (a * c)
    7. Division: (a / b) * b = a
    """
    gf = GaloisField(symbol_width=8, prim_poly=0x11D)
    rng = np.random.default_rng(42)

    # Exhaustive check of multiplicative inverses
    for a in range(1, 256):
        inv_a = gf.inv(a)
        assert gf.mul(a, inv_a) == 1
        assert gf.div(a, a) == 1
        assert gf.power(a, 0) == 1
        assert gf.power(a, 255) == 1

    # Random triad algebraic property tests
    for _ in range(500):
        a = int(rng.integers(0, 256))
        b = int(rng.integers(0, 256))
        c = int(rng.integers(0, 256))

        # Additive identity & inverse
        assert gf.add(a, 0) == a
        assert gf.sub(a, 0) == a
        assert gf.add(a, a) == 0
        assert gf.sub(a, a) == 0
        assert gf.add(a, b) == gf.sub(a, b)

        # Multiplicative identity
        assert gf.mul(a, 1) == a
        assert gf.mul(a, 0) == 0

        # Commutativity
        assert gf.add(a, b) == gf.add(b, a)
        assert gf.mul(a, b) == gf.mul(b, a)

        # Associativity
        assert gf.add(gf.add(a, b), c) == gf.add(a, gf.add(b, c))
        assert gf.mul(gf.mul(a, b), c) == gf.mul(a, gf.mul(b, c))

        # Distributivity
        lhs = gf.mul(a, gf.add(b, c))
        rhs = gf.add(gf.mul(a, b), gf.mul(a, c))
        assert lhs == rhs

        # Division
        if b != 0:
            quot = gf.div(a, b)
            assert gf.mul(quot, b) == a


def test_division_by_zero_rejection():
    """Verify division by zero raises ZeroDivisionError."""
    gf = GaloisField(symbol_width=8, prim_poly=0x11D)
    with pytest.raises(ZeroDivisionError):
        gf.div(42, 0)
    with pytest.raises(ZeroDivisionError):
        gf.inv(0)


def test_non_primitive_polynomial_rejection():
    """Verify that reducible or non-primitive polynomials are rejected at initialization."""
    # x^8 + 1 is reducible: (x + 1)^8
    reducible_poly = 0x101
    with pytest.raises(ValueError, match="is not primitive"):
        GaloisField(symbol_width=8, prim_poly=reducible_poly)


def test_polynomial_arithmetic_operations():
    """
    Verify polynomial addition, scaling, multiplication, division with remainder,
    Horner evaluation, and formal differentiation over GF(2^8).
    """
    gf = GaloisField(symbol_width=8, prim_poly=0x11D)

    # 1. Polynomial Addition & Subtraction
    p1 = [1, 2, 3]       # x^2 + 2x + 3
    p2 = [4, 3]          # 4x + 3
    p_add = gf.poly_add(p1, p2)
    # (x^2 + 2x + 3) + (4x + 3) = x^2 + (2^4)x + (3^3) = x^2 + 6x + 0 = [1, 6, 0]
    assert np.array_equal(p_add, [1, 6, 0])
    assert np.array_equal(gf.poly_sub(p1, p2), p_add)

    # 2. Polynomial Scaling
    p_scaled = gf.poly_scale(p1, 2)
    # 2*(x^2 + 2x + 3) = 2x^2 + (2*2)x + (2*3) = 2x^2 + 4x + 6
    assert np.array_equal(p_scaled, [2, 4, 6])
    assert np.array_equal(gf.poly_scale(p1, 0), [0])

    # 3. Polynomial Multiplication
    # (x + 2) * (x + 3) = x^2 + (2^3)x + (2*3) = x^2 + 1x + 6
    f1 = [1, 2]
    f2 = [1, 3]
    prod = gf.poly_mul(f1, f2)
    assert np.array_equal(prod, [1, 1, 6])

    # 4. Polynomial Division with Remainder: A = Q * B + R
    dividend = [1, 5, 6, 7]  # x^3 + 5x^2 + 6x + 7
    divisor = [1, 2]         # x + 2
    q, r = gf.poly_divmod(dividend, divisor)
    reconstructed = gf.poly_add(gf.poly_mul(q, divisor), r)
    assert np.array_equal(reconstructed, dividend)
    assert gf.poly_degree(r) < gf.poly_degree(divisor)

    # 5. Horner Evaluation
    # P(x) = x^2 + 1x + 6 evaluated at root x=2:
    # 2^2 + 1*2 + 6 = 4 ^ 2 ^ 6 = 0
    assert gf.poly_eval(prod, 2) == 0
    # evaluated at root x=3:
    # 3^2 + 1*3 + 6 = 5 ^ 3 ^ 6 = 0
    assert gf.poly_eval(prod, 3) == 0

    # 6. Formal Derivative in Characteristic 2
    # P(x) = c3 x^3 + c2 x^2 + c1 x + c0
    # P'(x) = 3*c3 x^2 + 2*c2 x + c1 = c3 x^2 + 0 x + c1 = [c3, 0, c1]
    poly_test = [7, 8, 9, 10]
    deriv = gf.poly_deriv(poly_test)
    assert np.array_equal(deriv, [7, 0, 9])
