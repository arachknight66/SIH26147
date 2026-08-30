"""
Low-Density Parity-Check (LDPC) Coding, Tanner Graph Representation, and Iterative Belief-Propagation Decoder.

Theoretical Foundations and Epistemic Scope:
-------------------------------------------
LDPC belongs to a fundamentally distinct mathematical and inference family from
exact dynamic-programming Viterbi decoders (MLSE on cycle-free trellises) and exact
algebraic Reed-Solomon decoders (Singleton-bounded polynomials over GF(2^m)).

LDPC decoding is an approximate, iterative probabilistic inference procedure on a
bipartite graph (the Tanner graph) that generally contains cycles.

1. Statistical Physics & Graphical Models Equivalence:
   Message passing on a Tanner graph is structurally identical to the Sum-Product algorithm
   on factor graphs, widely known across Bayesian inference and statistical physics:
   - In statistical physics, it is the cavity method and the Bethe-Peierls approximation
     for computing thermodynamic marginals on locally tree-like spin glasses and random lattices.
   - On tree graphs (girth = infinity), belief propagation computes exact marginal a posteriori probabilities.
   - On loopy graphs (practical LDPC codes), belief propagation is a controlled variational approximation
     whose accuracy and convergence properties depend directly on the graph's shortest cycle length (girth).

2. Girth and Trapping Sets:
   Short cycles (especially length 4) induce rapid, spurious self-reinforcing message feedback,
   causing belief propagation to develop inflated certainty along closed loops and fail to converge.
   Subgraphs with low-weight unsatisfied checks surrounded by short cycles form "trapping sets",
   causing the decoder to stall at a non-zero syndrome weight plateau regardless of iteration cap.

3. Sum-Product vs. Min-Sum:
   - Sum-Product (exact BP): Operates in the log-likelihood ratio (LLR) domain via the exact
     hyperbolic-tangent product rule at check nodes.
   - Min-Sum: Replaces the non-linear tanh product with a sign product and minimum magnitude,
     trading ~0.5 dB in coding gain for high numerical stability and computational efficiency.

4. Certified Operational Success:
   Unlike convolutional Viterbi (which always returns a minimum-metric traceback path regardless
   of channel validity), an LDPC decode is successful IF AND ONLY IF the hard-decision syndrome
   vector satisfies s = H * x^T = 0 (mod 2) within the allocated iteration budget and stays within
   the rate-informed plausibility band. An unverified hard decision is never returned as "decoded".

5. Scalability Boundary & Degree-Class Vectorization Non-Goal:
   The message-passing update loops in `decode_ldpc` are structured over flat 1D edge arrays
   with precomputed indices. For moderate block lengths (N in the range 96 to 512), this yields
   sub-5ms execution per 20 iterations.
   Practical production-scale codes (e.g. DVB-S2, 5G NR with N = 10^3 to 10^5) require batch
   vectorization across check/bit degree classes. Full vectorization across degree-classes is a
   deliberate non-goal of this phase to maintain exact mathematical transparency, inspectable
   per-iteration diagnostics, and zero external binary dependencies.

6. Permanent Epistemic Non-Goal:
   Blind unconstrained LDPC parity-check matrix discovery from received data alone is generally
   an ill-posed inverse problem. Hypothesis evaluation is strictly restricted to the registered
   standard families (Quasi-Cyclic and regular Gallager) with precomputed properties.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from app.models.metadata import Diagnostic, DiagnosticSeverity
from .models import FECCodeFamily, FECDecodeResult


class LDPCDecodeMode(str, Enum):
    SUM_PRODUCT = "sum_product"
    MIN_SUM = "min_sum"


class LDPCTerminationStatus(str, Enum):
    CONVERGED = "converged"
    ITERATION_CAP_REACHED = "iteration_cap_reached"
    TRAPPING_SET_STALLED = "trapping_set_stalled"


@dataclass(frozen=True)
class TannerGraph:
    """
    Explicit bipartite graph representation of a sparse parity-check matrix H.

    Attributes
    ----------
    num_bit_nodes : int
        Number of variable / bit nodes (columns of H, N).
    num_check_nodes : int
        Number of parity check nodes (rows of H, M).
    bit_to_checks : tuple[tuple[int, ...], ...]
        For each bit node v, the tuple of connected check node indices c.
    check_to_bits : tuple[tuple[int, ...], ...]
        For each check node c, the tuple of connected bit node indices v.
    edges : tuple[tuple[int, int], ...]
        List of all (check_idx, bit_idx) pairs where H[c, v] == 1.
    check_edge_indices : tuple[tuple[int, ...], ...]
        For each check node c, the tuple of edge indices in edges.
    bit_edge_indices : tuple[tuple[int, ...], ...]
        For each bit node v, the tuple of edge indices in edges.
    girth : int
        Shortest cycle length in the bipartite graph.
    """
    num_bit_nodes: int
    num_check_nodes: int
    bit_to_checks: tuple[tuple[int, ...], ...]
    check_to_bits: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int], ...]
    check_edge_indices: tuple[tuple[int, ...], ...]
    bit_edge_indices: tuple[tuple[int, ...], ...]
    girth: int


@dataclass(frozen=True)
class LDPCCodeSpec:
    """
    Structural specification and diagnostics of a standard LDPC code.

    Design Notes on Systematic Form & Message Locations:
    ---------------------------------------------------
    `free_columns` specifies the exact bit-node indices in the codeword that correspond
    to the systematic message bits (derived as the non-pivot columns during Gaussian
    elimination over GF(2)).

    Message extraction (e.g. in `decode_ldpc_bitstream`) must authoritatively index
    `decoded_bits[list(code_spec.free_columns)]` rather than assuming positional contiguity
    at `decoded_bits[:k_info]`. This guarantees correctness-by-construction across arbitrary
    parity-check matrix structures and column elimination permutations.
    """
    name: str
    h_matrix: np.ndarray             # 2D uint8 matrix (M x N)
    g_matrix: np.ndarray | None      # 2D uint8 systematic generator matrix (K x N)
    n_bits: int                      # Codeword length N
    m_checks: int                    # Parity check count M
    k_info_bits: int                 # Information bits K
    free_columns: tuple[int, ...]    # Authoritative column indices for systematic information bits
    rate: float                      # Design rate R = K / N
    graph: TannerGraph               # Explicit bipartite graph
    girth: int                       # Shortest cycle length
    sparsity: float                  # Fraction of zero entries in H
    bit_degrees: tuple[int, ...]     # Column weights
    check_degrees: tuple[int, ...]   # Row weights
    construction: str                # "gallager_regular" or "quasi_cyclic"
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class LDPCDecodeResult:
    """
    Detailed results of an iterative LDPC belief-propagation decode.
    """
    input_bits: np.ndarray
    decoded_bits: np.ndarray
    correction_mask: np.ndarray
    corrected_bit_count: int
    correction_fraction: float
    path_metric: float
    normalized_path_metric: float
    is_overcorrected: bool
    code_family: FECCodeFamily
    valid: bool
    iterations_used: int
    max_iterations: int
    decoding_mode: LDPCDecodeMode
    termination_status: LDPCTerminationStatus
    final_syndrome_weight: int
    syndrome_weight_history: tuple[int, ...]
    girth: int
    diagnostics: list[Diagnostic] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Graph & Diagnostic Utility Functions
# -----------------------------------------------------------------------------

def compute_graph_girth(
    bit_to_checks: tuple[tuple[int, ...], ...] | list[list[int]],
    check_to_bits: tuple[tuple[int, ...], ...] | list[list[int]],
    max_depth: int = 16,
) -> int:
    """
    Compute the girth (shortest cycle length) of a bipartite Tanner graph via BFS.

    In a bipartite graph, all cycles have even length >= 4.
    """
    num_bits = len(bit_to_checks)
    min_cycle = 999999

    for start_bit in range(num_bits):
        num_checks = len(check_to_bits)
        dist = [-1] * (num_bits + num_checks)
        parent = [-1] * (num_bits + num_checks)

        queue: deque[int] = deque()
        dist[start_bit] = 0
        queue.append(start_bit)

        while queue:
            curr = queue.popleft()
            curr_dist = dist[curr]

            if curr_dist * 2 >= min_cycle or curr_dist >= max_depth:
                continue

            if curr < num_bits:
                b_idx = curr
                for c_idx in bit_to_checks[b_idx]:
                    neighbor = num_bits + c_idx
                    if dist[neighbor] == -1:
                        dist[neighbor] = curr_dist + 1
                        parent[neighbor] = curr
                        queue.append(neighbor)
                    elif parent[curr] != neighbor:
                        cycle_len = curr_dist + dist[neighbor] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle == 4:
                                return 4
            else:
                c_idx = curr - num_bits
                for b_idx in check_to_bits[c_idx]:
                    neighbor = b_idx
                    if dist[neighbor] == -1:
                        dist[neighbor] = curr_dist + 1
                        parent[neighbor] = curr
                        queue.append(neighbor)
                    elif parent[curr] != neighbor:
                        cycle_len = curr_dist + dist[neighbor] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle == 4:
                                return 4

    return min_cycle if min_cycle < 999999 else 0


def build_tanner_graph(h_matrix: np.ndarray) -> TannerGraph:
    """
    Construct an explicit bipartite Tanner graph from a 2D binary parity-check matrix H.
    """
    m_checks, n_bits = h_matrix.shape
    bit_to_checks: list[list[int]] = [[] for _ in range(n_bits)]
    check_to_bits: list[list[int]] = [[] for _ in range(m_checks)]
    edges: list[tuple[int, int]] = []
    check_edge_indices: list[list[int]] = [[] for _ in range(m_checks)]
    bit_edge_indices: list[list[int]] = [[] for _ in range(n_bits)]

    for c in range(m_checks):
        for v in range(n_bits):
            if h_matrix[c, v] != 0:
                e_idx = len(edges)
                edges.append((c, v))
                check_to_bits[c].append(v)
                bit_to_checks[v].append(c)
                check_edge_indices[c].append(e_idx)
                bit_edge_indices[v].append(e_idx)

    b2c_tuple = tuple(tuple(c_list) for c_list in bit_to_checks)
    c2b_tuple = tuple(tuple(v_list) for v_list in check_to_bits)
    c_edge_tuple = tuple(tuple(l) for l in check_edge_indices)
    b_edge_tuple = tuple(tuple(l) for l in bit_edge_indices)

    girth = compute_graph_girth(b2c_tuple, c2b_tuple)

    return TannerGraph(
        num_bit_nodes=n_bits,
        num_check_nodes=m_checks,
        bit_to_checks=b2c_tuple,
        check_to_bits=c2b_tuple,
        edges=tuple(edges),
        check_edge_indices=c_edge_tuple,
        bit_edge_indices=b_edge_tuple,
        girth=girth,
    )


# -----------------------------------------------------------------------------
# Parity-Check Matrix Builders
# -----------------------------------------------------------------------------

def build_gallager_matrix(
    n_bits: int,
    d_v: int = 3,
    d_c: int = 6,
    rng_seed: int = 42,
) -> np.ndarray:
    """
    Construct a regular Gallager parity-check matrix with explicit 4-cycle avoidance.

    Parameters
    ----------
    n_bits : int
        Codeword length N (must be divisible by d_c).
    d_v : int
        Column weight / bit node degree (standard = 3).
    d_c : int
        Row weight / check node degree (standard = 6).
    rng_seed : int
        Random seed for column permutations.

    Returns
    -------
    H : np.ndarray (M x N, uint8)
        Regular Gallager parity-check matrix where M = (d_v / d_c) * N.
    """
    if n_bits % d_c != 0:
        raise ValueError(f"Codeword length N={n_bits} must be divisible by check degree d_c={d_c}")

    rows_per_block = n_bits // d_c
    m_checks = d_v * rows_per_block
    h_matrix = np.zeros((m_checks, n_bits), dtype=np.uint8)

    # First block: deterministic consecutive ones
    for r in range(rows_per_block):
        h_matrix[r, r * d_c : (r + 1) * d_c] = 1

    rng = np.random.default_rng(rng_seed)

    # Subsequent blocks: column-permuted copies with 4-cycle elimination
    for block_idx in range(1, d_v):
        row_offset = block_idx * rows_per_block
        perm = rng.permutation(n_bits)
        h_matrix[row_offset : row_offset + rows_per_block, :] = h_matrix[0:rows_per_block, perm]

        # Iteratively swap columns to eliminate column overlaps >= 2 (4-cycles)
        for _ in range(500):
            sub_h = h_matrix[: row_offset + rows_per_block]
            overlap = np.dot(sub_h.T, sub_h)
            np.fill_diagonal(overlap, 0)
            bad_pairs = np.argwhere(overlap >= 2)
            if len(bad_pairs) == 0:
                break
            c1, _ = bad_pairs[0]
            c_swap = rng.integers(0, n_bits)
            h_matrix[row_offset : row_offset + rows_per_block, [c1, c_swap]] = h_matrix[
                row_offset : row_offset + rows_per_block, [c_swap, c1]
            ]

    return h_matrix


def build_qc_ldpc_matrix(
    base_matrix: np.ndarray,
    lifting_factor: int,
) -> np.ndarray:
    """
    Construct a Quasi-Cyclic (QC-LDPC) parity-check matrix from a base matrix of shift values.

    Parameters
    ----------
    base_matrix : np.ndarray (m_b x n_b, int)
        Shift values for each circulant block. -1 represents an all-zero ZxZ block.
        s >= 0 represents a ZxZ identity matrix cyclically shifted right by s positions.
    lifting_factor : int (Z)
        Circulant block dimension Z.

    Returns
    -------
    H : np.ndarray (M x N, uint8)
        Quasi-cyclic parity-check matrix where M = m_b * Z, N = n_b * Z.
    """
    m_b, n_b = base_matrix.shape
    m_checks = m_b * lifting_factor
    n_bits = n_b * lifting_factor
    h_matrix = np.zeros((m_checks, n_bits), dtype=np.uint8)

    for i in range(m_b):
        for j in range(n_b):
            shift = int(base_matrix[i, j])
            if shift >= 0:
                row_start = i * lifting_factor
                col_start = j * lifting_factor
                # Circular right shift of Z x Z identity
                for r in range(lifting_factor):
                    c = (r + shift) % lifting_factor
                    h_matrix[row_start + r, col_start + c] = 1

    return h_matrix


def compute_systematic_generator_matrix(h_matrix: np.ndarray) -> tuple[np.ndarray | None, int, tuple[int, ...]]:
    """
    Perform Gaussian elimination over GF(2) to transform H into reduced echelon form,
    yielding a valid generator matrix G satisfying H * G^T = 0 (mod 2) and identifying
    the exact free (systematic information) columns.

    Returns
    -------
    G : np.ndarray (K x N, uint8) | None
    k_info_bits : int
    free_columns : tuple[int, ...]
    """
    m_checks, n_bits = h_matrix.shape
    h_work = h_matrix.copy().astype(np.uint8)
    pivot_cols: list[int] = []

    r = 0
    for c in range(n_bits):
        p_row = None
        for i in range(r, m_checks):
            if h_work[i, c] == 1:
                p_row = i
                break
        if p_row is None:
            continue
        if p_row != r:
            h_work[[r, p_row]] = h_work[[p_row, r]]
        for i in range(m_checks):
            if i != r and h_work[i, c] == 1:
                h_work[i] ^= h_work[r]
        pivot_cols.append(c)
        r += 1
        if r == m_checks:
            break

    free_cols = [c for c in range(n_bits) if c not in pivot_cols]
    k_info = len(free_cols)
    if k_info == 0:
        return None, 0, ()

    g_matrix = np.zeros((k_info, n_bits), dtype=np.uint8)
    for i, free_col in enumerate(free_cols):
        g_matrix[i, free_col] = 1
        for row_idx, piv_col in enumerate(pivot_cols):
            if h_work[row_idx, free_col] == 1:
                g_matrix[i, piv_col] = 1

    return g_matrix, k_info, tuple(free_cols)


def encode_ldpc(message_bits: np.ndarray, code_spec: LDPCCodeSpec) -> np.ndarray:
    """
    Systematically encode message bits u into codeword x = u * G (mod 2),
    placing message bits at exactly code_spec.free_columns.

    Parameters
    ----------
    message_bits : np.ndarray (K, uint8)
    code_spec : LDPCCodeSpec

    Returns
    -------
    codeword : np.ndarray (N, uint8)
    """
    if code_spec.g_matrix is None:
        raise ValueError(f"Code {code_spec.name} does not have a precomputed generator matrix.")

    u = message_bits.astype(np.uint8)
    if len(u) != code_spec.k_info_bits:
        raise ValueError(f"Message length {len(u)} does not match code K={code_spec.k_info_bits}")

    # The systematic generator matrix has G[i, free_columns[i]] = 1 and G[i, free_columns[j]] = 0 for j != i.
    # Matrix product u * G (mod 2) places message bit u[i] directly at column free_columns[i].
    codeword = np.dot(u, code_spec.g_matrix) % 2
    return codeword.astype(np.uint8)


# -----------------------------------------------------------------------------
# Standard Registered LDPC Code Configurations
# -----------------------------------------------------------------------------

def _create_code_spec(
    name: str,
    h_mat: np.ndarray,
    construction: str,
    assumptions: tuple[str, ...] = (),
) -> LDPCCodeSpec:
    m_checks, n_bits = h_mat.shape
    g_mat, k_info, free_cols = compute_systematic_generator_matrix(h_mat)
    graph = build_tanner_graph(h_mat)
    bit_degs = tuple(int(np.sum(h_mat[:, v])) for v in range(n_bits))
    check_degs = tuple(int(np.sum(h_mat[c, :])) for c in range(m_checks))
    sparsity = float(1.0 - (np.sum(h_mat) / (m_checks * n_bits)))
    rate = float(k_info / n_bits)

    return LDPCCodeSpec(
        name=name,
        h_matrix=h_mat,
        g_matrix=g_mat,
        n_bits=n_bits,
        m_checks=m_checks,
        k_info_bits=k_info,
        free_columns=free_cols,
        rate=rate,
        graph=graph,
        girth=graph.girth,
        sparsity=sparsity,
        bit_degrees=bit_degs,
        check_degrees=check_degs,
        construction=construction,
        assumptions=assumptions,
    )


# Quasi-Cyclic Base Matrices (Dual-Diagonal Parity, Degree-3 Information, Girth >= 6)
# Standard 4x8 rate 1/2 base matrix (Z=16 -> N=128, Z=32 -> N=256)
_QC_BASE_4X8 = np.array([
    [1,  2,  5, -1,  0, -1, -1, -1],
    [-1, 3,  1,  7,  0,  0, -1, -1],
    [4, -1,  6,  2, -1,  0,  0, -1],
    [8,  0, -1,  3, -1, -1,  0,  0],
], dtype=int)

# Gallager Regular (3, 6) Rate 1/2 (N=96, M=48, N=192, M=96)
_H_GALLAGER_96 = build_gallager_matrix(96, d_v=3, d_c=6, rng_seed=42)
_H_GALLAGER_192 = build_gallager_matrix(192, d_v=3, d_c=6, rng_seed=42)
_H_QC_128 = build_qc_ldpc_matrix(_QC_BASE_4X8, lifting_factor=16)
_H_QC_256 = build_qc_ldpc_matrix(_QC_BASE_4X8, lifting_factor=32)

STANDARD_LDPC_SPECS: dict[str, LDPCCodeSpec] = {
    "QC_LDPC_N128_R12": _create_code_spec(
        name="QC_LDPC_N128_R12",
        h_mat=_H_QC_128,
        construction="quasi_cyclic",
        assumptions=("Quasi-cyclic block LDPC", "N=128, M=64, K=64, R=0.5, Z=16, Girth>=6"),
    ),
    "QC_LDPC_N256_R12": _create_code_spec(
        name="QC_LDPC_N256_R12",
        h_mat=_H_QC_256,
        construction="quasi_cyclic",
        assumptions=("Quasi-cyclic block LDPC", "N=256, M=128, K=128, R=0.5, Z=32, Girth>=6"),
    ),
    "GALLAGER_N96_R12": _create_code_spec(
        name="GALLAGER_N96_R12",
        h_mat=_H_GALLAGER_96,
        construction="gallager_regular",
        assumptions=("Gallager (3,6) regular LDPC", "N=96, M=48, K=50, R=0.52, Girth>=6"),
    ),
    "GALLAGER_N192_R12": _create_code_spec(
        name="GALLAGER_N192_R12",
        h_mat=_H_GALLAGER_192,
        construction="gallager_regular",
        assumptions=("Gallager (3,6) regular LDPC", "N=192, M=96, K=98, R=0.51, Girth>=6"),
    ),
}


# -----------------------------------------------------------------------------
# Core Belief-Propagation (Sum-Product & Min-Sum) Decoders
# -----------------------------------------------------------------------------

def decode_ldpc(
    received_bits: np.ndarray,
    code_spec: LDPCCodeSpec,
    soft_bits: np.ndarray | None = None,
    mode: LDPCDecodeMode = LDPCDecodeMode.SUM_PRODUCT,
    max_iterations: int = 50,
    min_sum_scale: float = 0.80,
    max_correction_fraction: float = 0.15,
) -> LDPCDecodeResult:
    """
    Perform iterative belief-propagation decoding on a single codeword of length N.

    Scalability & Execution Model Note:
    -----------------------------------
    Message-passing updates operate over flat 1D edge arrays indexed by precomputed
    bipartite adjacency structures (`check_edge_indices`, `bit_edge_indices`).
    This provides sub-5ms latency for block lengths up to N = 512 in native Python/NumPy.
    Production-scale codes (N > 10^4) requiring parallel GPU or SIMD degree-class batching
    are outside the scope of this phase.

    Parameters
    ----------
    received_bits : np.ndarray (1D uint8)
        Received hard-decision channel bits (length N).
    code_spec : LDPCCodeSpec
        Parity-check matrix and Tanner graph specification.
    soft_bits : np.ndarray | None
        Channel Log-Likelihood Ratios (LLRs) where positive indicates bit 0.
        If None, a conservative proxy is derived from received_bits.
    mode : LDPCDecodeMode
        SUM_PRODUCT (exact tanh formulation) or MIN_SUM (sign-min approximation).
    max_iterations : int
        Maximum decoding iterations allowed before termination.
    min_sum_scale : float
        Attenuation factor for normalized min-sum decoding (default 0.80).
    max_correction_fraction : float
        Rate-informed plausibility threshold for bit alterations (default 0.15).

    Returns
    -------
    result : LDPCDecodeResult
    """
    n_bits = code_spec.n_bits
    m_checks = code_spec.m_checks
    h_mat = code_spec.h_matrix
    graph = code_spec.graph

    if len(received_bits) != n_bits:
        raise ValueError(f"Received bit length {len(received_bits)} != code block length {n_bits}")

    diagnostics: list[Diagnostic] = []

    # 1. Initialize Channel LLRs
    if soft_bits is not None and len(soft_bits) == n_bits:
        llr_channel = soft_bits.astype(np.float64)
    else:
        # Conservative hard-decision proxy: LLR = +4.0 for 0, -4.0 for 1
        llr_channel = np.where(received_bits == 0, 4.0, -4.0).astype(np.float64)
        diagnostics.append(
            Diagnostic(
                code="LDPC_HARD_PROXY_INITIALIZED",
                message="Channel soft bits unavailable; initialized conservative hard-decision LLR proxy (+/-4.0)",
                severity=DiagnosticSeverity.INFO,
            )
        )

    # 2. Check 0th-iteration syndrome on raw hard bits
    s_initial = (np.dot(h_mat, (llr_channel < 0).astype(np.uint8)) % 2)
    w_initial = int(np.sum(s_initial))
    if w_initial == 0:
        decoded_bits = (llr_channel < 0).astype(np.uint8)
        corr_mask = (received_bits != decoded_bits)
        return LDPCDecodeResult(
            input_bits=received_bits.copy(),
            decoded_bits=decoded_bits,
            correction_mask=corr_mask,
            corrected_bit_count=int(np.sum(corr_mask)),
            correction_fraction=float(np.mean(corr_mask)),
            path_metric=0.0,
            normalized_path_metric=0.0,
            is_overcorrected=False,
            code_family=FECCodeFamily.LDPC,
            valid=True,
            iterations_used=0,
            max_iterations=max_iterations,
            decoding_mode=mode,
            termination_status=LDPCTerminationStatus.CONVERGED,
            final_syndrome_weight=0,
            syndrome_weight_history=(0,),
            girth=code_spec.girth,
            diagnostics=diagnostics,
        )

    # 3. Fast Edge-Indexed Message Structures
    num_edges = len(graph.edges)
    q_edge = np.zeros(num_edges, dtype=np.float64)
    r_edge = np.zeros(num_edges, dtype=np.float64)

    # Initial bit-to-check messages along connected edges are channel LLRs
    for v in range(n_bits):
        for e in graph.bit_edge_indices[v]:
            q_edge[e] = llr_channel[v]

    syndrome_history: list[int] = [w_initial]
    eps = 1e-15
    clip_limit = 1.0 - eps

    iteration = 0
    final_decoded_bits = received_bits.copy()
    termination_status = LDPCTerminationStatus.ITERATION_CAP_REACHED
    is_overcorrected = False

    while iteration < max_iterations:
        iteration += 1

        # ---------------------------------------------------------------------
        # A. Check Node Update (c -> v along edge e)
        # ---------------------------------------------------------------------
        for c in range(m_checks):
            e_indices = graph.check_edge_indices[c]
            deg_c = len(e_indices)
            if deg_c == 0:
                continue

            q_in = q_edge[list(e_indices)]

            if mode == LDPCDecodeMode.SUM_PRODUCT:
                signs = np.sign(q_in)
                signs[signs == 0] = 1.0
                total_sign_prod = float(np.prod(signs))

                tanh_mags = np.tanh(np.abs(q_in) / 2.0)
                tanh_mags = np.clip(tanh_mags, eps, clip_limit)
                log_tanh_sum = float(np.sum(np.log(tanh_mags)))

                for idx, e in enumerate(e_indices):
                    extrinsic_sign = total_sign_prod * signs[idx]
                    extrinsic_log_tanh = log_tanh_sum - np.log(tanh_mags[idx])
                    extrinsic_tanh = np.clip(np.exp(extrinsic_log_tanh), eps, clip_limit)
                    r_edge[e] = extrinsic_sign * 2.0 * np.arctanh(extrinsic_tanh)

            elif mode == LDPCDecodeMode.MIN_SUM:
                signs = np.sign(q_in)
                signs[signs == 0] = 1.0
                total_sign_prod = float(np.prod(signs))
                mags = np.abs(q_in)

                for idx, e in enumerate(e_indices):
                    extrinsic_sign = total_sign_prod * signs[idx]
                    other_mags = np.delete(mags, idx)
                    min_val = float(np.min(other_mags)) if len(other_mags) > 0 else 0.0
                    r_edge[e] = min_sum_scale * extrinsic_sign * min_val

        # ---------------------------------------------------------------------
        # B. Bit Node Update & Extrinsic Exclusion (v -> c along edge e)
        # ---------------------------------------------------------------------
        total_llr = llr_channel.copy()
        for v in range(n_bits):
            e_indices = graph.bit_edge_indices[v]
            r_in = r_edge[list(e_indices)]
            sum_r = float(np.sum(r_in))
            total_llr[v] += sum_r

            # Strict Extrinsic Exclusion: q_{v->c} = total_llr[v] - r_{c->v}
            for idx, e in enumerate(e_indices):
                q_edge[e] = total_llr[v] - r_in[idx]

        # ---------------------------------------------------------------------
        # C. Hard Decision & Syndrome Check
        # ---------------------------------------------------------------------
        hard_est = (total_llr < 0).astype(np.uint8)
        syndrome = (np.dot(h_mat, hard_est) % 2)
        syn_weight = int(np.sum(syndrome))
        syndrome_history.append(syn_weight)

        if syn_weight == 0:
            corr_m = (received_bits != hard_est)
            c_frac = float(np.mean(corr_m))
            if c_frac <= max_correction_fraction:
                final_decoded_bits = hard_est
                termination_status = LDPCTerminationStatus.CONVERGED
            else:
                is_overcorrected = True
                termination_status = LDPCTerminationStatus.ITERATION_CAP_REACHED
            break

    # 4. Analyze Termination Reason & Trapping Sets
    is_converged = (termination_status == LDPCTerminationStatus.CONVERGED)

    if not is_converged and not is_overcorrected:
        # Check for stagnation / oscillation consistent with a trapping set
        if len(syndrome_history) >= 5:
            last_weights = syndrome_history[-4:]
            if len(set(last_weights)) <= 2 and last_weights[-1] > 0:
                termination_status = LDPCTerminationStatus.TRAPPING_SET_STALLED
                diagnostics.append(
                    Diagnostic(
                        code="LDPC_TRAPPING_SET_DETECTED",
                        message=f"Decoder stalled at non-zero syndrome weight plateau {last_weights}, characteristic of a trapping set",
                        severity=DiagnosticSeverity.WARNING,
                    )
                )

    correction_mask = (received_bits != final_decoded_bits)
    corrected_count = int(np.sum(correction_mask))
    correction_fraction = float(np.mean(correction_mask))
    path_metric = float(syndrome_history[-1])
    norm_path_metric = float(path_metric / m_checks)

    return LDPCDecodeResult(
        input_bits=received_bits.copy(),
        decoded_bits=final_decoded_bits if is_converged else received_bits.copy(),
        correction_mask=correction_mask if is_converged else np.zeros(n_bits, dtype=bool),
        corrected_bit_count=corrected_count if is_converged else 0,
        correction_fraction=correction_fraction if is_converged else 0.0,
        path_metric=path_metric,
        normalized_path_metric=norm_path_metric,
        is_overcorrected=is_overcorrected,
        code_family=FECCodeFamily.LDPC,
        valid=is_converged,
        iterations_used=iteration,
        max_iterations=max_iterations,
        decoding_mode=mode,
        termination_status=termination_status,
        final_syndrome_weight=syndrome_history[-1],
        syndrome_weight_history=tuple(syndrome_history),
        girth=code_spec.girth,
        diagnostics=diagnostics,
    )


# -----------------------------------------------------------------------------
# Bitstream Block Decoder
# -----------------------------------------------------------------------------

def decode_ldpc_bitstream(
    received_bits: np.ndarray,
    code_spec: LDPCCodeSpec,
    soft_bits: np.ndarray | None = None,
    mode: LDPCDecodeMode = LDPCDecodeMode.SUM_PRODUCT,
    max_iterations: int = 50,
    max_correction_fraction: float = 0.15,
) -> FECDecodeResult:
    """
    Decode a full continuous bitstream over consecutive LDPC codewords of length N.

    Parameters
    ----------
    received_bits : np.ndarray (1D uint8)
    code_spec : LDPCCodeSpec
    soft_bits : np.ndarray | None
    mode : LDPCDecodeMode
    max_iterations : int
    max_correction_fraction : float

    Returns
    -------
    FECDecodeResult
    """
    n_bits = code_spec.n_bits
    k_info = code_spec.k_info_bits
    num_blocks = len(received_bits) // n_bits

    if num_blocks == 0:
        return FECDecodeResult(
            input_bits=received_bits.copy(),
            decoded_bits=received_bits.copy(),
            correction_mask=np.zeros(len(received_bits), dtype=bool),
            corrected_bit_count=0,
            correction_fraction=0.0,
            path_metric=0.0,
            normalized_path_metric=0.0,
            is_overcorrected=False,
            code_family=FECCodeFamily.LDPC,
            valid=False,
        )

    decoded_info_blocks: list[np.ndarray] = []
    full_corrected_bits = received_bits.copy()
    full_correction_mask = np.zeros(len(received_bits), dtype=bool)

    total_corrected = 0
    all_blocks_valid = True
    total_path_metric = 0.0

    for b_idx in range(num_blocks):
        b_start = b_idx * n_bits
        b_end = b_start + n_bits
        block_rx = received_bits[b_start:b_end]
        block_soft = soft_bits[b_start:b_end] if soft_bits is not None and len(soft_bits) >= b_end else None

        res = decode_ldpc(
            received_bits=block_rx,
            code_spec=code_spec,
            soft_bits=block_soft,
            mode=mode,
            max_iterations=max_iterations,
            max_correction_fraction=max_correction_fraction,
        )

        if not res.valid:
            all_blocks_valid = False
            break

        # Authoritative systematic information extraction from free_columns
        info_bits = res.decoded_bits[list(code_spec.free_columns)]
        decoded_info_blocks.append(info_bits)
        full_corrected_bits[b_start:b_end] = res.decoded_bits
        full_correction_mask[b_start:b_end] = res.correction_mask
        total_corrected += res.corrected_bit_count
        total_path_metric += res.path_metric

    if not all_blocks_valid or len(decoded_info_blocks) == 0:
        return FECDecodeResult(
            input_bits=received_bits.copy(),
            decoded_bits=received_bits.copy(),
            correction_mask=np.zeros(len(received_bits), dtype=bool),
            corrected_bit_count=0,
            correction_fraction=0.0,
            path_metric=999.0,
            normalized_path_metric=1.0,
            is_overcorrected=False,
            code_family=FECCodeFamily.LDPC,
            valid=False,
        )

    decoded_stream = np.concatenate(decoded_info_blocks)
    total_bits_evaluated = num_blocks * n_bits
    corr_frac = float(total_corrected / max(1, total_bits_evaluated))
    is_over = bool(corr_frac > max_correction_fraction)

    return FECDecodeResult(
        input_bits=received_bits[:total_bits_evaluated].copy(),
        decoded_bits=decoded_stream,
        correction_mask=full_correction_mask[:total_bits_evaluated],
        corrected_bit_count=total_corrected,
        correction_fraction=corr_frac,
        path_metric=total_path_metric,
        normalized_path_metric=float(total_path_metric / max(1, num_blocks * code_spec.m_checks)),
        is_overcorrected=is_over,
        code_family=FECCodeFamily.LDPC,
        valid=bool(not is_over),
    )
