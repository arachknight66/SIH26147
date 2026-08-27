from __future__ import annotations
import numpy as np
from .models import ConstellationResult, ModulationFamily

def get_ideal_constellation(family: ModulationFamily, order: int) -> np.ndarray:
    """Return ideal unit-average-energy constellation points."""
    if family == ModulationFamily.PSK:
        if order == 2:
            return np.array([-1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64)
        elif order == 4:
            return np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)
        elif order == 8:
            angles = np.arange(8) * (2.0 * np.pi / 8.0)
            return np.exp(1j * angles).astype(np.complex64)
        else:
            angles = np.arange(order) * (2.0 * np.pi / order)
            return np.exp(1j * angles).astype(np.complex64)

    elif family == ModulationFamily.QAM:
        if order == 16:
            grid_1d = np.array([-3.0, -1.0, 1.0, 3.0])
            i_grid, q_grid = np.meshgrid(grid_1d, grid_1d)
            const = (i_grid.ravel() + 1j * q_grid.ravel()) / np.sqrt(10.0)
            return const.astype(np.complex64)
        elif order == 64:
            grid_1d = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
            i_grid, q_grid = np.meshgrid(grid_1d, grid_1d)
            const = (i_grid.ravel() + 1j * q_grid.ravel()) / np.sqrt(42.0)
            return const.astype(np.complex64)
        else:
            return np.array([-1.0+0j, 1.0+0j], dtype=np.complex64)

    return np.array([-1.0 + 0j, 1.0 + 0j], dtype=np.complex64)

def analyze_constellation(
    symbols: np.ndarray,
    family: ModulationFamily,
    order: int = 4,
) -> ConstellationResult:
    """
    Normalize 1-SPS symbol constellation, compute EVM, cluster centroids, and decision margins.

    Parameters
    ----------
    symbols : np.ndarray
        Extracted 1-SPS symbol samples.
    family : ModulationFamily
        Modulation family.
    order : int
        Modulation order.

    Returns
    -------
    ConstellationResult
    """
    n_syms = len(symbols)
    if n_syms < 8:
        return ConstellationResult(
            symbols=symbols.copy().astype(np.complex64),
            cluster_centroids=np.array([], dtype=np.complex64),
            cluster_variances=np.array([], dtype=np.float64),
            rms_radius=1.0,
            evm_linear=1.0,
            evm_percent=100.0,
            evm_db=0.0,
            decision_margin=0.0,
            phase_error_rms_rad=1.0,
            amplitude_error_rms=1.0,
            rotational_ambiguity_deg=(0.0,),
            valid=False,
        )

    # 1. Unit Average Energy Normalization: E[|z|^2] = 1.0
    rms_rad = float(np.sqrt(np.mean(np.abs(symbols) ** 2)))
    norm_symbols = (symbols / (rms_rad + 1e-12)).astype(np.complex64)

    # 2. Reference Constellation & Slicing
    ideal_pts = get_ideal_constellation(family, order)
    
    # Nearest neighbor slicing: for each symbol z_k, find closest ideal point s_hat_k
    dists = np.abs(norm_symbols[:, None] - ideal_pts[None, :])
    nearest_idx = np.argmin(dists, axis=1)
    ideal_assigned = ideal_pts[nearest_idx]
    
    # 3. Error Vectors and EVM
    err_vectors = norm_symbols - ideal_assigned
    err_pwr = np.sum(np.abs(err_vectors) ** 2)
    ref_pwr = np.sum(np.abs(ideal_assigned) ** 2) + 1e-12
    
    evm_lin = float(np.sqrt(err_pwr / ref_pwr))
    evm_pct = float(evm_lin * 100.0)
    evm_db = float(20.0 * np.log10(evm_lin + 1e-12))

    # Phase and Amplitude Error RMS
    amp_actual = np.abs(norm_symbols)
    amp_ideal = np.abs(ideal_assigned)
    amp_err_rms = float(np.sqrt(np.mean((amp_actual - amp_ideal) ** 2)))

    phase_diffs = np.angle(norm_symbols * np.conj(ideal_assigned))
    phase_err_rms = float(np.sqrt(np.mean(phase_diffs ** 2)))

    # 4. Cluster Centroids and Variances
    centroids_list = []
    variances_list = []
    for m in range(len(ideal_pts)):
        pts_in_cluster = norm_symbols[nearest_idx == m]
        if len(pts_in_cluster) > 0:
            c_mean = complex(np.mean(pts_in_cluster))
            c_var = float(np.var(pts_in_cluster))
        else:
            c_mean = ideal_pts[m]
            c_var = 0.0
        centroids_list.append(c_mean)
        variances_list.append(c_var)

    cluster_centroids = np.array(centroids_list, dtype=np.complex64)
    cluster_variances = np.array(variances_list, dtype=np.float64)

    # 5. Decision Margin: 1.0 - (mean error distance / half nearest neighbor distance)
    if len(ideal_pts) >= 2:
        all_pair_dists = np.abs(ideal_pts[:, None] - ideal_pts[None, :])
        np.fill_diagonal(all_pair_dists, np.inf)
        min_inter_dist = float(np.min(all_pair_dists))
        margin = float(np.clip(1.0 - (np.mean(np.abs(err_vectors)) / (0.5 * min_inter_dist + 1e-6)), 0.0, 1.0))
    else:
        margin = 1.0

    # 6. Rotational Ambiguity Set
    if family == ModulationFamily.PSK:
        if order == 2:
            rot_amb = (0.0, 180.0)
        elif order == 4:
            rot_amb = (0.0, 90.0, 180.0, 270.0)
        elif order == 8:
            rot_amb = tuple(sorted([round(float(k * 45.0), 1) for k in range(8)]))
        else:
            rot_amb = (0.0,)
    elif family == ModulationFamily.QAM:
        rot_amb = (0.0, 90.0, 180.0, 270.0)
    else:
        rot_amb = (0.0,)

    return ConstellationResult(
        symbols=norm_symbols,
        cluster_centroids=cluster_centroids,
        cluster_variances=cluster_variances,
        rms_radius=round(rms_rad, 4),
        evm_linear=round(evm_lin, 4),
        evm_percent=round(evm_pct, 2),
        evm_db=round(evm_db, 2),
        decision_margin=round(margin, 4),
        phase_error_rms_rad=round(phase_err_rms, 4),
        amplitude_error_rms=round(amp_err_rms, 4),
        rotational_ambiguity_deg=rot_amb,
        valid=True,
    )
