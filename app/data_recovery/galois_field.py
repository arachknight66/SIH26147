from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

# Standard Primitive Polynomials for Common Galois Fields GF(2^m)
# Represented as integers where bit i corresponds to coefficient of x^i
STANDARD_PRIMITIVE_POLYNOMIALS: dict[int, dict[str, int]] = {
    8: {
        # x^8 + x^4 + x^3 + x^2 + 1 (CCSDS, DVB-T, ITU-T J.83, QR Code)
        "CCSDS_DVB_QR": 0x11D,      # 285 decimal, octal 0o435
        # x^8 + x^5 + x^3 + x^2 + 1
        "STANDARD_0x12D": 0x12D,    # 301 decimal
        # x^8 + x^6 + x^3 + x^2 + 1
        "STANDARD_0x14D": 0x14D,    # 333 decimal
        # x^8 + x^7 + x^6 + x + 1
        "STANDARD_0x1C3": 0x1C3,    # 451 decimal
        # x^8 + x^7 + x^2 + x + 1 (Mil-Std 188-165 / Intelsat)
        "INTELSAT": 0x187,          # 391 decimal
    },
    7: {
        # x^7 + x^3 + 1
        "STANDARD_7": 0x89,         # 137 decimal
    },
    6: {
        # x^6 + x + 1
        "STANDARD_6": 0x43,         # 67 decimal
    },
    5: {
        # x^5 + x^2 + 1
        "STANDARD_5": 0x25,         # 37 decimal
    },
    4: {
        # x^4 + x + 1
        "STANDARD_4": 0x13,         # 19 decimal
    },
    3: {
        # x^3 + x + 1
        "STANDARD_3": 0x0B,         # 11 decimal
    },
}


class GaloisField:
    """
    Galois Field GF(2^m) arithmetic engine over binary extension fields.

    Operates on symbol width m (field order 2^m) using exponential (antilog) and logarithm
    lookup tables for high-performance exact multiplication, division, and inversion.

    Epistemic & Mathematical Invariants:
    1. Addition and Subtraction in GF(2^m) are strictly bitwise XOR (characteristic 2).
    2. The field-generator polynomial must be strictly primitive; construction asserts
       cyclic group closure and reciprocal existence across all non-zero elements.
    """

    def __init__(
        self,
        symbol_width: int = 8,
        prim_poly: int = 0x11D,
        generator: int = 2,
    ) -> None:
        """
        Initialize and validate Galois Field GF(2^m).

        Parameters
        ----------
        symbol_width : int
            Symbol bit-width m (e.g. 8 for GF(256)).
        prim_poly : int
            Primitive polynomial represented as integer (e.g. 0x11D for x^8 + x^4 + x^3 + x^2 + 1).
        generator : int
            Field primitive root / generator alpha (default 2, corresponding to polynomial 'x').
        """
        if symbol_width < 2 or symbol_width > 16:
            raise ValueError(f"Symbol width m={symbol_width} out of supported range [2, 16]")

        self.symbol_width = symbol_width
        self.field_order = 1 << symbol_width
        self.cardinality = self.field_order - 1  # 2^m - 1
        self.prim_poly = prim_poly
        self.generator = generator

        # Precompute Exponential (Antilog) and Logarithm Tables
        self.exp_table = np.zeros(2 * self.cardinality, dtype=np.int32)
        self.log_table = np.zeros(self.field_order, dtype=np.int32)

        self._build_and_validate_tables()

    def _build_and_validate_tables(self) -> None:
        """Construct exponential/logarithm lookup tables and assert field closure and invertibility."""
        x = 1
        seen = set()

        for i in range(self.cardinality):
            self.exp_table[i] = x
            self.log_table[x] = i
            seen.add(x)

            # Multiply by alpha (shift left, reduce modulo primitive polynomial if overflow)
            x <<= 1
            if x & self.field_order:
                x ^= self.prim_poly

        # Group closure & primitivity assertion:
        # A valid primitive polynomial must generate all 2^m - 1 non-zero elements in a single cycle
        if len(seen) != self.cardinality or x != 1:
            raise ValueError(
                f"Polynomial 0x{self.prim_poly:X} ({self.prim_poly}) is not primitive for GF(2^{self.symbol_width}): "
                f"generated {len(seen)} unique elements out of required {self.cardinality}."
            )

        # Duplicate upper half of exp_table for fast modulo-free lookup
        for i in range(self.cardinality, 2 * self.cardinality):
            self.exp_table[i] = self.exp_table[i - self.cardinality]

        # Log of 0 is mathematically undefined (-infinity / None proxy)
        self.log_table[0] = -1

        # Multiplicative inverse sanity check across all non-zero elements
        for a in range(1, self.field_order):
            inv_a = self.inv(a)
            prod = self.mul(a, inv_a)
            if prod != 1:
                raise ValueError(
                    f"GF(2^{self.symbol_width}) self-consistency failure: "
                    f"element {a} * inv({a})={inv_a} yielded {prod} != 1"
                )

    def add(self, a: int, b: int) -> int:
        """Finite field addition: in GF(2^m), addition is bitwise XOR."""
        return int(a ^ b)

    def sub(self, a: int, b: int) -> int:
        """Finite field subtraction: in GF(2^m), subtraction is identical to addition (bitwise XOR)."""
        return int(a ^ b)

    def mul(self, a: int, b: int) -> int:
        """Finite field multiplication using precomputed log/exp tables."""
        if a == 0 or b == 0:
            return 0
        return int(self.exp_table[self.log_table[a] + self.log_table[b]])

    def div(self, a: int, b: int) -> int:
        """Finite field division a / b."""
        if b == 0:
            raise ZeroDivisionError("Division by zero in Galois field GF(2^m)")
        if a == 0:
            return 0
        diff = self.log_table[a] - self.log_table[b] + self.cardinality
        return int(self.exp_table[diff])

    def inv(self, a: int) -> int:
        """Multiplicative inverse of a in GF(2^m), such that a * inv(a) == 1."""
        if a == 0:
            raise ZeroDivisionError("Zero has no multiplicative inverse in Galois field GF(2^m)")
        return int(self.exp_table[self.cardinality - self.log_table[a]])

    def power(self, a: int, p: int) -> int:
        """Compute a^p in GF(2^m)."""
        if a == 0:
            if p == 0:
                return 1
            if p < 0:
                raise ZeroDivisionError("0 cannot be raised to negative power in GF(2^m)")
            return 0
        if p == 0:
            return 1
        p_mod = p % self.cardinality
        if p_mod < 0:
            p_mod += self.cardinality
        return int(self.exp_table[(self.log_table[a] * p_mod) % self.cardinality])

    def alpha(self, i: int) -> int:
        """Return the i-th power of primitive element alpha: alpha^i."""
        i_mod = i % self.cardinality
        if i_mod < 0:
            i_mod += self.cardinality
        return int(self.exp_table[i_mod])

    # =========================================================================
    # POLYNOMIAL ARITHMETIC UTILITY LAYER OVER GF(2^m)
    # Coefficients ordered from highest degree to lowest degree: [c_d, ..., c_0]
    # =========================================================================

    def poly_strip(self, poly: Sequence[int]) -> np.ndarray:
        """Strip leading zero coefficients from polynomial."""
        arr = np.asarray(poly, dtype=np.int32)
        nonzero = np.where(arr != 0)[0]
        if len(nonzero) == 0:
            return np.array([0], dtype=np.int32)
        return arr[nonzero[0]:]

    def poly_degree(self, poly: Sequence[int]) -> int:
        """Return degree of polynomial (deg([0]) == 0)."""
        stripped = self.poly_strip(poly)
        if len(stripped) == 1 and stripped[0] == 0:
            return 0
        return len(stripped) - 1

    def poly_add(self, p1: Sequence[int], p2: Sequence[int]) -> np.ndarray:
        """Add two polynomials over GF(2^m)."""
        a1 = np.asarray(p1, dtype=np.int32)
        a2 = np.asarray(p2, dtype=np.int32)
        max_len = max(len(a1), len(a2))
        res = np.zeros(max_len, dtype=np.int32)
        if len(a1) > 0:
            res[-len(a1):] ^= a1
        if len(a2) > 0:
            res[-len(a2):] ^= a2
        return self.poly_strip(res)

    def poly_sub(self, p1: Sequence[int], p2: Sequence[int]) -> np.ndarray:
        """Subtract two polynomials over GF(2^m) (identical to addition)."""
        return self.poly_add(p1, p2)

    def poly_scale(self, poly: Sequence[int], scalar: int) -> np.ndarray:
        """Multiply polynomial by a field scalar."""
        if scalar == 0:
            return np.array([0], dtype=np.int32)
        if scalar == 1:
            return self.poly_strip(poly)
        res = np.array([self.mul(c, scalar) for c in poly], dtype=np.int32)
        return self.poly_strip(res)

    def poly_mul(self, p1: Sequence[int], p2: Sequence[int]) -> np.ndarray:
        """Multiply two polynomials over GF(2^m)."""
        a1 = self.poly_strip(p1)
        a2 = self.poly_strip(p2)
        if (len(a1) == 1 and a1[0] == 0) or (len(a2) == 1 and a2[0] == 0):
            return np.array([0], dtype=np.int32)

        res_len = len(a1) + len(a2) - 1
        res = np.zeros(res_len, dtype=np.int32)

        for i, c1 in enumerate(a1):
            if c1 == 0:
                continue
            for j, c2 in enumerate(a2):
                if c2 == 0:
                    continue
                prod = self.mul(c1, c2)
                res[i + j] ^= prod

        return self.poly_strip(res)

    def poly_divmod(
        self, dividend: Sequence[int], divisor: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Polynomial long division with remainder over GF(2^m): dividend = quotient * divisor + remainder.

        Parameters
        ----------
        dividend : Sequence[int]
        divisor : Sequence[int]

        Returns
        -------
        quotient : np.ndarray
        remainder : np.ndarray
        """
        num = list(self.poly_strip(dividend))
        den = list(self.poly_strip(divisor))

        if len(den) == 1 and den[0] == 0:
            raise ZeroDivisionError("Polynomial division by zero polynomial")

        if len(num) < len(den):
            return np.array([0], dtype=np.int32), np.array(num, dtype=np.int32)

        quotient_len = len(num) - len(den) + 1
        quotient = [0] * quotient_len
        inv_leading_den = self.inv(den[0])

        for i in range(quotient_len):
            coeff = self.mul(num[i], inv_leading_den)
            quotient[i] = coeff
            if coeff != 0:
                for j in range(len(den)):
                    num[i + j] ^= self.mul(coeff, den[j])

        rem = self.poly_strip(num[quotient_len:])
        return self.poly_strip(quotient), rem

    def poly_eval(self, poly: Sequence[int], x: int) -> int:
        """
        Evaluate polynomial at field element x using Horner's method.

        P(x) = c_d * x^d + ... + c_1 * x + c_0
        """
        stripped = self.poly_strip(poly)
        val = 0
        for coeff in stripped:
            val = self.add(self.mul(val, x), int(coeff))
        return val

    def poly_deriv(self, poly: Sequence[int]) -> np.ndarray:
        """
        Formal derivative of polynomial in GF(2^m) (characteristic 2).

        In GF(2^m), for term c_k * x^k:
        k * c_k = c_k if k is odd, and 0 if k is even.
        """
        p = self.poly_strip(poly)
        deg = len(p) - 1
        if deg <= 0:
            return np.array([0], dtype=np.int32)

        deriv_coeffs: list[int] = []
        for i, coeff in enumerate(p[:-1]):
            power = deg - i
            # If power is odd, term survives: (power mod 2) == 1
            if power % 2 == 1:
                deriv_coeffs.append(int(coeff))
            else:
                deriv_coeffs.append(0)

        return self.poly_strip(deriv_coeffs)
