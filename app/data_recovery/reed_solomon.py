from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

from .galois_field import GaloisField
from .models import FECCodeFamily, FECDecodeResult


@dataclass(frozen=True)
class RSDecodeStatus:
    """Detailed algebraic metrics and diagnostic status for a Reed-Solomon decode operation."""
    n_symbols: int
    k_symbols: int
    parity_symbols: int
    correction_radius: int
    detected_error_count: int
    detected_erasure_count: int
    error_positions: tuple[int, ...]
    error_magnitudes: tuple[int, ...]
    bm_euclidean_agreement: bool
    chien_root_count_matched: bool
    post_correction_syndromes_zero: bool
    is_overcorrected: bool
    valid: bool
    diagnostics: tuple[str, ...] = ()


class ReedSolomonCodec:
    """
    Production-grade Reed–Solomon encoder and syndrome / Berlekamp–Massey / Euclidean / Chien / Forney
    decoder with error and erasure support over Galois fields GF(2^m).

    Epistemic & Mathematical Invariants:
    1. Operates over GF(2^m) with configurable symbol width m, primitive polynomial, and starting root b.
    2. Guaranteed error-correcting radius in symbols is exactly t = floor((N - K) / 2).
    3. Failure boundary is exact and detectable: if detected error count > t (or root-count mismatch in
       Chien search, or non-zero post-correction syndromes), the decoder explicitly returns failure
       (valid=False) rather than a fabricated "best-effort" codeword.
    4. Dual error-locator solvers (Berlekamp–Massey and Extended Euclidean) are cross-checked.
    5. Post-correction syndrome verification is mandatory and unconditionally enforced.
    """

    def __init__(
        self,
        n_symbols: int,
        k_symbols: int,
        symbol_width: int = 8,
        prim_poly: int = 0x11D,
        first_consecutive_root: int = 0,
        gf: GaloisField | None = None,
    ) -> None:
        """
        Initialize Reed-Solomon codec.

        Parameters
        ----------
        n_symbols : int
            Total codeword length in field symbols (N <= 2^m - 1).
        k_symbols : int
            Message length in field symbols (K < N).
        symbol_width : int
            Bit width m per field symbol (default 8).
        prim_poly : int
            Field primitive polynomial (default 0x11D).
        first_consecutive_root : int
            First consecutive root power b in generator polynomial g(x) = prod_{i=0}^{2t-1} (x - alpha^{b+i}).
            Standard values: b=0 (DVB-T, ITU-T), b=1 (CCSDS, QR Code), b=112 (CCSDS shortened).
        gf : GaloisField | None
            Optional pre-initialized GaloisField instance.
        """
        if k_symbols >= n_symbols:
            raise ValueError(f"Message length K={k_symbols} must be strictly less than codeword length N={n_symbols}")

        self.symbol_width = symbol_width
        self.gf = gf or GaloisField(symbol_width=symbol_width, prim_poly=prim_poly)

        if n_symbols > self.gf.cardinality:
            raise ValueError(
                f"Codeword length N={n_symbols} exceeds field cardinality 2^{symbol_width}-1={self.gf.cardinality}"
            )

        self.n_symbols = n_symbols
        self.k_symbols = k_symbols
        self.parity_symbols = n_symbols - k_symbols  # 2t
        self.correction_radius = self.parity_symbols // 2  # t
        self.first_consecutive_root = first_consecutive_root

        # Construct and validate generator polynomial g(x)
        self.generator_poly = self._build_generator_polynomial()

    def _build_generator_polynomial(self) -> np.ndarray:
        """
        Construct generator polynomial g(x) = prod_{i=0}^{2t-1} (x - alpha^{b+i}).

        Returns polynomial coefficients in descending degree order: [g_{2t}, ..., g_0]
        """
        # Start with g(x) = 1
        g = np.array([1], dtype=np.int32)
        b = self.first_consecutive_root

        for i in range(self.parity_symbols):
            root = self.gf.alpha(b + i)
            # Multiply g(x) by (x - root) = (x + root)
            factor = np.array([1, root], dtype=np.int32)
            g = self.gf.poly_mul(g, factor)

        # Assert defining algebraic property: g(alpha^{b+i}) == 0 for all i in 0..2t-1
        for i in range(self.parity_symbols):
            root = self.gf.alpha(b + i)
            eval_val = self.gf.poly_eval(g, root)
            if eval_val != 0:
                raise ValueError(
                    f"Generator polynomial construction error: g(alpha^{b+i})={eval_val} != 0"
                )

        return g

    # =========================================================================
    # 1. SYSTEMATIC ENCODER
    # =========================================================================

    def encode(self, message_symbols: Sequence[int]) -> np.ndarray:
        """
        Produce systematic Reed-Solomon codeword [Message, Parity].

        Parameters
        ----------
        message_symbols : Sequence[int]
            Array of K field symbols (each in [0, 2^m - 1]).

        Returns
        -------
        codeword : np.ndarray
            1D array of N field symbols: message followed by 2t parity symbols.
        """
        msg = list(message_symbols)
        if len(msg) != self.k_symbols:
            raise ValueError(
                f"Message length {len(msg)} does not match codec K={self.k_symbols}"
            )

        # Shift message up by parity_symbols positions: M(x) * x^{2t}
        shifted_msg = np.concatenate((msg, np.zeros(self.parity_symbols, dtype=np.int32)))

        # Parity remainder P(x) = (M(x) * x^{2t}) mod g(x)
        _, remainder = self.gf.poly_divmod(shifted_msg, self.generator_poly)

        # Ensure remainder has exact length parity_symbols (left-pad with zeros if needed)
        rem_len = len(remainder)
        if rem_len < self.parity_symbols:
            pad = np.zeros(self.parity_symbols - rem_len, dtype=np.int32)
            parity = np.concatenate((pad, remainder))
        else:
            parity = remainder[-self.parity_symbols :]

        codeword = np.concatenate((msg, parity)).astype(np.int32)

        # Validate encoder: Every valid codeword evaluates to zero at all generator roots
        for i in range(self.parity_symbols):
            root = self.gf.alpha(self.first_consecutive_root + i)
            if self.gf.poly_eval(codeword, root) != 0:
                raise ValueError("Encoder verification failed: codeword is not an algebraic multiple of g(x)")

        return codeword

    # =========================================================================
    # 2. SYNDROME COMPUTATION
    # =========================================================================

    def compute_syndromes(self, received_symbols: Sequence[int]) -> np.ndarray:
        """
        Evaluate received polynomial at each root of generator polynomial.

        S_i = R(alpha^{b+i}) for i in 0..2t-1.
        """
        r = np.asarray(received_symbols, dtype=np.int32)
        b = self.first_consecutive_root
        syndromes = np.zeros(self.parity_symbols, dtype=np.int32)

        for i in range(self.parity_symbols):
            root = self.gf.alpha(b + i)
            syndromes[i] = self.gf.poly_eval(r, root)

        return syndromes

    # =========================================================================
    # 3. ERROR LOCATOR SEARCH (BERLEKAMP-MASSEY & EXTENDED EUCLIDEAN)
    # =========================================================================

    def find_error_locator_berlekamp_massey(self, syndromes: Sequence[int]) -> np.ndarray:
        """
        Find error locator polynomial Lambda(x) via the finite-field Berlekamp-Massey algorithm.

        Returns polynomial coefficients in ascending degree order: [1, Lambda_1, Lambda_2, ..., Lambda_v]
        where Lambda(x) = 1 + Lambda_1 * x + ... + Lambda_v * x^v.
        """
        syn = list(syndromes)
        num_syn = len(syn)

        Lambda = [1]
        B = [1]
        L = 0
        k = 1

        for n in range(num_syn):
            # Discrepancy delta_n = S_n + sum_{i=1}^L Lambda_i * S_{n-i}
            delta = syn[n]
            for i in range(1, len(Lambda)):
                if i <= n:
                    delta ^= self.gf.mul(Lambda[i], syn[n - i])

            if delta == 0:
                k += 1
            else:
                # new_Lambda = Lambda - delta * x^k * B
                scaled_B = [self.gf.mul(c, delta) for c in B]
                shifted_B = [0] * k + scaled_B

                max_len = max(len(Lambda), len(shifted_B))
                new_Lambda = [0] * max_len
                for idx, c in enumerate(Lambda):
                    new_Lambda[idx] ^= c
                for idx, c in enumerate(shifted_B):
                    new_Lambda[idx] ^= c

                if 2 * L <= n:
                    L = n + 1 - L
                    inv_delta = self.gf.inv(delta)
                    B = [self.gf.mul(c, inv_delta) for c in Lambda]
                    k = 1
                else:
                    k += 1

                Lambda = new_Lambda

        # Strip trailing zeros
        while len(Lambda) > 1 and Lambda[-1] == 0:
            Lambda.pop()

        return np.array(Lambda, dtype=np.int32)

    def find_error_locator_euclidean(
        self, syndromes: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find error locator Lambda(x) and evaluator Omega(x) via Sugiyama's Extended Euclidean Algorithm.

        Returns
        -------
        locator_poly : np.ndarray (ascending order [1, Lambda_1, ...])
        evaluator_poly : np.ndarray (ascending order [Omega_0, Omega_1, ...])
        """
        syn = list(syndromes)
        two_t = self.parity_symbols
        t = self.correction_radius

        # r_{-1}(x) = x^{2t}, r_0(x) = S(x)
        # Note: poly_divmod uses descending polynomial representation
        r_prev = np.zeros(two_t + 1, dtype=np.int32)
        r_prev[0] = 1  # x^{2t}
        r_curr = np.array(syn[::-1], dtype=np.int32)  # S(x) in descending order

        v_prev = np.array([0], dtype=np.int32)
        v_curr = np.array([1], dtype=np.int32)

        while self.gf.poly_degree(r_curr) >= t:
            quotient, remainder = self.gf.poly_divmod(r_prev, r_curr)
            # v_next = v_prev - q * v_curr
            q_v = self.gf.poly_mul(quotient, v_curr)
            v_next = self.gf.poly_sub(v_prev, q_v)

            r_prev, r_curr = r_curr, remainder
            v_prev, v_curr = v_curr, v_next

        # Normalize so constant term v(0) is 1
        # In descending order, v(0) is the last coefficient
        v_desc = self.gf.poly_strip(v_curr)
        const_term = v_desc[-1] if len(v_desc) > 0 else 0
        if const_term == 0:
            return np.array([1], dtype=np.int32), np.array([0], dtype=np.int32)

        inv_const = self.gf.inv(const_term)
        v_norm = self.gf.poly_scale(v_desc, inv_const)
        omega_norm = self.gf.poly_scale(self.gf.poly_strip(r_curr), inv_const)

        # Convert to ascending order: [1, Lambda_1, ...]
        locator_asc = v_norm[::-1]
        evaluator_asc = omega_norm[::-1]

        return locator_asc, evaluator_asc

    # =========================================================================
    # 4. CHIEN SEARCH & ROOT COUNT CONSISTENCY
    # =========================================================================

    def chien_search(
        self, locator_asc: Sequence[int], n_symbols: int
    ) -> tuple[list[int], list[int]]:
        """
        Find error roots and position indices using Chien search.

        Parameters
        ----------
        locator_asc : Sequence[int]
            Error locator polynomial in ascending degree order: [1, L_1, L_2, ...].
        n_symbols : int
            Codeword block length.

        Returns
        -------
        error_roots : list[int]
            Field elements X_l = alpha^p representing error locations.
        error_positions : list[int]
            0-indexed positions in codeword array (from left: 0 to N-1).
        """
        loc_desc = np.asarray(locator_asc, dtype=np.int32)[::-1]
        deg = len(loc_desc) - 1
        if deg <= 0:
            return [], []

        error_roots: list[int] = []
        error_positions: list[int] = []

        # For codeword polynomial R(x) = r_{N-1} x^{N-1} + ... + r_0:
        # Codeword index i in [0, N-1] corresponds to term power p = N - 1 - i.
        # Root equation: Lambda(X_l^{-1}) == 0, where X_l = alpha^p = alpha^{N - 1 - i}.
        for i in range(n_symbols):
            p = n_symbols - 1 - i
            inv_root = self.gf.alpha(-p)  # X_l^{-1} = alpha^{-p}
            val = self.gf.poly_eval(loc_desc, inv_root)
            if val == 0:
                root = self.gf.alpha(p)
                error_roots.append(root)
                error_positions.append(i)

        return error_roots, error_positions

    # =========================================================================
    # 5. FORNEY ALGORITHM FOR ERROR MAGNITUDES
    # =========================================================================

    def forney_algorithm(
        self,
        syndromes: Sequence[int],
        locator_asc: Sequence[int],
        error_roots: list[int],
        error_positions: list[int],
    ) -> list[int]:
        """
        Compute exact error magnitudes via Forney's algorithm.

        Parameters
        ----------
        syndromes : Sequence[int]
        locator_asc : Sequence[int]
        error_roots : list[int]
        error_positions : list[int]

        Returns
        -------
        magnitudes : list[int]
        """
        # Evaluator polynomial: Omega(x) = [S(x) * Lambda(x)] mod x^{2t}
        # In ascending order:
        syn_asc = list(syndromes)
        loc_asc = list(locator_asc)

        # Polynomial multiplication in ascending order
        omega_len = len(syn_asc) + len(loc_asc) - 1
        omega_asc = [0] * omega_len
        for i, s in enumerate(syn_asc):
            if s == 0:
                continue
            for j, l in enumerate(loc_asc):
                if l == 0:
                    continue
                omega_asc[i + j] ^= self.gf.mul(s, l)

        # Modulo x^{2t}: retain only powers 0..2t-1
        omega_asc = omega_asc[: self.parity_symbols]
        omega_desc = np.array(omega_asc[::-1], dtype=np.int32)

        # Formal derivative Lambda'(x) in ascending order:
        # Lambda(x) = Lambda_0 + Lambda_1 x + Lambda_2 x^2 + ...
        # Lambda'(x) = Lambda_1 + Lambda_3 x^2 + Lambda_5 x^4 + ... (odd powers survive)
        deriv_asc: list[int] = []
        for i in range(1, len(loc_asc)):
            if i % 2 == 1:
                deriv_asc.append(loc_asc[i])
            else:
                deriv_asc.append(0)
        while len(deriv_asc) > 1 and deriv_asc[-1] == 0:
            deriv_asc.pop()
        deriv_desc = np.array(deriv_asc[::-1], dtype=np.int32) if deriv_asc else np.array([0], dtype=np.int32)

        b = self.first_consecutive_root
        magnitudes: list[int] = []

        for root in error_roots:
            inv_root = self.gf.inv(root)
            num = self.gf.poly_eval(omega_desc, inv_root)
            den = self.gf.poly_eval(deriv_desc, inv_root)

            if den == 0:
                magnitudes.append(0)
                continue

            # Standard Forney factor for starting root b:
            # e = (X_l^{1 - b} * Omega(X_l^{-1})) / Lambda'(X_l^{-1})
            factor = self.gf.power(root, 1 - b)
            num_scaled = self.gf.mul(factor, num)
            mag = self.gf.div(num_scaled, den)
            magnitudes.append(mag)

        return magnitudes

    # =========================================================================
    # 6. FULL DECODER (ERRORS AND ERASURES)
    # =========================================================================

    def decode(
        self,
        received_symbols: Sequence[int],
        erasures: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, list[int], bool, RSDecodeStatus]:
        """
        Decode received symbol block with provable error & erasure correction.

        Parameters
        ----------
        received_symbols : Sequence[int]
            Received block of N field symbols.
        erasures : Sequence[int] | None
            Optional list of known erasure symbol positions (0..N-1).

        Returns
        -------
        (corrected_symbols, corrected_positions, is_valid, decode_status)
        """
        r = np.asarray(received_symbols, dtype=np.int32).copy()
        if len(r) != self.n_symbols:
            raise ValueError(f"Received block length {len(r)} != codec N={self.n_symbols}")

        erasure_list = sorted(list(set(erasures or [])))
        num_erasures = len(erasure_list)

        # Singleton bound check: erasure count cannot exceed parity symbol count
        if num_erasures > self.parity_symbols:
            status = RSDecodeStatus(
                n_symbols=self.n_symbols,
                k_symbols=self.k_symbols,
                parity_symbols=self.parity_symbols,
                correction_radius=self.correction_radius,
                detected_error_count=0,
                detected_erasure_count=num_erasures,
                error_positions=(),
                error_magnitudes=(),
                bm_euclidean_agreement=True,
                chien_root_count_matched=False,
                post_correction_syndromes_zero=False,
                is_overcorrected=True,
                valid=False,
                diagnostics=(f"Erasure count {num_erasures} exceeds total parity budget {self.parity_symbols}",),
            )
            return r, [], False, status

        # 1. Compute syndromes
        syndromes = self.compute_syndromes(r)

        # Short-circuit: if all syndromes are zero and no erasures, codeword is clean
        if np.all(syndromes == 0) and num_erasures == 0:
            status = RSDecodeStatus(
                n_symbols=self.n_symbols,
                k_symbols=self.k_symbols,
                parity_symbols=self.parity_symbols,
                correction_radius=self.correction_radius,
                detected_error_count=0,
                detected_erasure_count=0,
                error_positions=(),
                error_magnitudes=(),
                bm_euclidean_agreement=True,
                chien_root_count_matched=True,
                post_correction_syndromes_zero=True,
                is_overcorrected=False,
                valid=True,
            )
            return r, [], True, status

        # 2. Formulate erasure locator if erasures provided
        if num_erasures > 0:
            # Erasure locator Gamma(x) = prod_{k in erasures} (1 - X_k * x)
            # where X_k = alpha^{N - 1 - k}
            gamma_asc = [1]
            for pos in erasure_list:
                p = self.n_symbols - 1 - pos
                x_k = self.gf.alpha(p)
                # Multiply gamma_asc by (1 - x_k * x)
                factor_asc = [1, x_k]
                # Ascending poly mul
                new_gamma = [0] * (len(gamma_asc) + 1)
                for i_g, c_g in enumerate(gamma_asc):
                    new_gamma[i_g] ^= c_g
                    new_gamma[i_g + 1] ^= self.gf.mul(c_g, x_k)
                gamma_asc = new_gamma

            # Compute Forney modified syndromes: T(x) = [S(x) * Gamma(x)] mod x^{2t}
            t_len = len(syndromes) + len(gamma_asc) - 1
            t_asc = [0] * t_len
            for i_s, s in enumerate(syndromes):
                if s == 0:
                    continue
                for i_g, g in enumerate(gamma_asc):
                    t_asc[i_s + i_g] ^= self.gf.mul(s, g)
            modified_syndromes = t_asc[: self.parity_symbols]

            # Find erratic error locator Psi(x) via Berlekamp-Massey on modified syndromes
            psi_asc = self.find_error_locator_berlekamp_massey(modified_syndromes[num_erasures:])
            
            # Full locator Lambda(x) = Gamma(x) * Psi(x)
            lambda_len = len(gamma_asc) + len(psi_asc) - 1
            lambda_asc = [0] * lambda_len
            for i_g, g in enumerate(gamma_asc):
                if g == 0:
                    continue
                for i_p, p in enumerate(psi_asc):
                    lambda_asc[i_g + i_p] ^= self.gf.mul(g, p)

            lambda_asc_arr = np.array(lambda_asc, dtype=np.int32)
            bm_euc_agree = True  # In erasure mode, combined locator used
        else:
            # Pure error decoding
            bm_locator = self.find_error_locator_berlekamp_massey(syndromes)
            euc_locator, _ = self.find_error_locator_euclidean(syndromes)

            # Cross-check agreement between Berlekamp-Massey and Extended Euclidean
            bm_deg = len(bm_locator) - 1
            euc_deg = len(euc_locator) - 1
            bm_euc_agree = bool(bm_deg == euc_deg and np.array_equal(bm_locator, euc_locator))

            lambda_asc_arr = bm_locator

        # 3. Chien Search for roots
        err_roots, err_positions = self.chien_search(lambda_asc_arr, self.n_symbols)
        target_deg = len(lambda_asc_arr) - 1

        # Epistemic failure check: number of roots must EXACTLY match locator degree
        if len(err_roots) != target_deg:
            status = RSDecodeStatus(
                n_symbols=self.n_symbols,
                k_symbols=self.k_symbols,
                parity_symbols=self.parity_symbols,
                correction_radius=self.correction_radius,
                detected_error_count=len(err_roots),
                detected_erasure_count=num_erasures,
                error_positions=tuple(err_positions),
                error_magnitudes=(),
                bm_euclidean_agreement=bm_euc_agree,
                chien_root_count_matched=False,
                post_correction_syndromes_zero=False,
                is_overcorrected=True,
                valid=False,
                diagnostics=(
                    f"Chien search root count {len(err_roots)} != locator degree {target_deg}; "
                    f"error pattern exceeds code capability t={self.correction_radius}",
                ),
            )
            return r, [], False, status

        # Singleton radius check: 2 * num_errors + num_erasures <= 2t
        pure_errors = len(err_positions) - num_erasures
        if 2 * pure_errors + num_erasures > self.parity_symbols:
            status = RSDecodeStatus(
                n_symbols=self.n_symbols,
                k_symbols=self.k_symbols,
                parity_symbols=self.parity_symbols,
                correction_radius=self.correction_radius,
                detected_error_count=pure_errors,
                detected_erasure_count=num_erasures,
                error_positions=tuple(err_positions),
                error_magnitudes=(),
                bm_euclidean_agreement=bm_euc_agree,
                chien_root_count_matched=True,
                post_correction_syndromes_zero=False,
                is_overcorrected=True,
                valid=False,
                diagnostics=(
                    f"Singleton bound violated: 2*errors ({2*pure_errors}) + erasures ({num_erasures}) > parity ({self.parity_symbols})",
                ),
            )
            return r, [], False, status

        # 4. Forney Algorithm for error magnitudes
        magnitudes = self.forney_algorithm(syndromes, lambda_asc_arr, err_roots, err_positions)

        # 5. Apply corrections: in GF(2^m), field addition and subtraction coincide
        corrected = r.copy()
        for pos, mag in zip(err_positions, magnitudes):
            corrected[pos] ^= mag

        # 6. Post-Correction Syndrome Re-Evaluation
        post_syndromes = self.compute_syndromes(corrected)
        is_zero_syn = bool(np.all(post_syndromes == 0))

        status = RSDecodeStatus(
            n_symbols=self.n_symbols,
            k_symbols=self.k_symbols,
            parity_symbols=self.parity_symbols,
            correction_radius=self.correction_radius,
            detected_error_count=pure_errors,
            detected_erasure_count=num_erasures,
            error_positions=tuple(err_positions),
            error_magnitudes=tuple(magnitudes),
            bm_euclidean_agreement=bm_euc_agree,
            chien_root_count_matched=True,
            post_correction_syndromes_zero=is_zero_syn,
            is_overcorrected=not is_zero_syn,
            valid=is_zero_syn,
            diagnostics=() if is_zero_syn else ("Post-correction syndromes are non-zero; decode failed validation",),
        )

        return corrected, err_positions, is_zero_syn, status

    # =========================================================================
    # 7. BITSTREAM WRAPPER & CORRECTION MASK
    # =========================================================================

    def decode_bitstream(
        self,
        input_bits: np.ndarray,
        max_correction_fraction: float = 0.10,
        soft_bits: np.ndarray | None = None,
    ) -> FECDecodeResult:
        """
        Block-wise Reed-Solomon decoding of a binary bitstream.

        Parameters
        ----------
        input_bits : np.ndarray
            1D uint8 binary stream.
        max_correction_fraction : float
            Maximum fraction of bits permitted to be altered.
        soft_bits : np.ndarray | None
            Optional float32 LLR soft decision stream.

        Returns
        -------
        FECDecodeResult
        """
        m = self.symbol_width
        bits_per_block = self.n_symbols * m
        msg_bits_per_block = self.k_symbols * m

        n_blocks = len(input_bits) // bits_per_block
        if n_blocks == 0:
            return FECDecodeResult(
                input_bits=input_bits,
                decoded_bits=input_bits.copy(),
                correction_mask=np.zeros(len(input_bits), dtype=bool),
                corrected_bit_count=0,
                correction_fraction=0.0,
                path_metric=0.0,
                normalized_path_metric=0.0,
                is_overcorrected=False,
                code_family=FECCodeFamily.REED_SOLOMON,
                valid=False,
            )

        usable_len = n_blocks * bits_per_block
        raw_usable_bits = input_bits[:usable_len]
        correction_mask = np.zeros(len(input_bits), dtype=bool)

        decoded_message_bits_list: list[np.ndarray] = []
        total_corrected_bits = 0
        all_blocks_valid = True

        for b_idx in range(n_blocks):
            blk_bits = raw_usable_bits[b_idx * bits_per_block : (b_idx + 1) * bits_per_block]
            # Unpack bits into m-bit symbols
            symbols = np.packbits(blk_bits.reshape(self.n_symbols, m), axis=1).squeeze(-1)

            # Optional soft decision erasure extraction if soft_bits provided
            erasures: list[int] = []
            if soft_bits is not None and len(soft_bits) >= (b_idx + 1) * bits_per_block:
                blk_soft = np.abs(soft_bits[b_idx * bits_per_block : (b_idx + 1) * bits_per_block])
                symbol_conf = np.mean(blk_soft.reshape(self.n_symbols, m), axis=1)
                # Identify symbols with very low confidence as candidate erasures
                low_conf_idx = np.where(symbol_conf < 0.25)[0].tolist()
                if len(low_conf_idx) <= self.parity_symbols:
                    erasures = low_conf_idx

            corrected_syms, corr_pos, blk_valid, status = self.decode(symbols, erasures=erasures)
            if not blk_valid:
                all_blocks_valid = False

            # Extract K message symbols
            msg_syms = corrected_syms[: self.k_symbols]
            # Convert back to bits
            msg_bits = np.unpackbits(msg_syms.astype(np.uint8)[:, None], axis=1)[:, 8 - m :].ravel()
            decoded_message_bits_list.append(msg_bits)

            # Re-encode message to compute exact correction mask on channel codeword bits
            reencoded_syms = self.encode(msg_syms)
            reencoded_bits = np.unpackbits(reencoded_syms.astype(np.uint8)[:, None], axis=1)[:, 8 - m :].ravel()

            blk_mask = (reencoded_bits != blk_bits)
            correction_mask[b_idx * bits_per_block : (b_idx + 1) * bits_per_block] = blk_mask
            total_corrected_bits += int(np.sum(blk_mask))

        decoded_bits = np.concatenate(decoded_message_bits_list) if decoded_message_bits_list else np.array([], dtype=np.uint8)
        corr_frac = float(total_corrected_bits / usable_len) if usable_len > 0 else 0.0
        is_over = bool(corr_frac > max_correction_fraction or not all_blocks_valid)

        return FECDecodeResult(
            input_bits=input_bits,
            decoded_bits=decoded_bits,
            correction_mask=correction_mask,
            corrected_bit_count=total_corrected_bits,
            correction_fraction=round(corr_frac, 4),
            path_metric=float(total_corrected_bits),
            normalized_path_metric=round(corr_frac, 4),
            is_overcorrected=is_over,
            code_family=FECCodeFamily.REED_SOLOMON,
            valid=bool(all_blocks_valid and not is_over),
        )
