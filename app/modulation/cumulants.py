from __future__ import annotations
import numpy as np
from .models import CumulantFeatures, FeatureValidity

def extract_cumulant_features(samples: np.ndarray) -> CumulantFeatures:
    """
    Extract 2nd and 4th order complex cumulants and normalized discriminant ratios.

    Definitions:
    C20 = E[x^2]
    C21 = E[|x|^2]
    C40 = E[x^4] - 3*(C20^2)
    C41 = E[x^3 * conj(x)] - 3*C20*C21
    C42 = E[|x|^4] - |C20|^2 - 2*(C21^2)

    Normalized Ratios:
    f20 = |C20| / C21
    f40 = |C40| / (C21^2)
    f41 = |C41| / (C21^2)
    f42 = |C42| / (C21^2)

    Parameters
    ----------
    samples : np.ndarray
        Complex signal samples (zero-mean, RMS-normalized).

    Returns
    -------
    CumulantFeatures
    """
    n_samples = len(samples)
    if n_samples < 32:
        return CumulantFeatures(
            c20=0.0j,
            c21=0.0,
            c40=0.0j,
            c41=0.0j,
            c42=0.0,
            f20=0.0,
            f40=0.0,
            f41=0.0,
            f42=0.0,
            validity=FeatureValidity.UNAVAILABLE,
        )

    # Remove complex mean for unbiased central cumulants
    x = samples.astype(np.complex128) - np.mean(samples.astype(np.complex128))
    
    # 2nd order moments
    x_sq = x ** 2
    x_mag_sq = np.abs(x) ** 2

    m20 = complex(np.mean(x_sq))
    m21 = float(np.mean(x_mag_sq))

    if m21 <= 1e-12:
        return CumulantFeatures(
            c20=0.0j,
            c21=0.0,
            c40=0.0j,
            c41=0.0j,
            c42=0.0,
            f20=0.0,
            f40=0.0,
            f41=0.0,
            f42=0.0,
            validity=FeatureValidity.UNRELIABLE,
        )

    # 4th order moments
    m40 = complex(np.mean(x ** 4))
    m41 = complex(np.mean((x ** 3) * np.conj(x)))
    m42 = float(np.mean(x_mag_sq ** 2))

    # Cumulants
    c20 = m20
    c21 = m21
    c40 = m40 - 3.0 * (c20 ** 2)
    c41 = m41 - 3.0 * c20 * c21
    c42 = m42 - (abs(c20) ** 2) - 2.0 * (c21 ** 2)

    # Normalized ratios
    f20 = float(abs(c20) / c21)
    f40 = float(abs(c40) / (c21 ** 2))
    f41 = float(abs(c41) / (c21 ** 2))
    f42 = float(abs(c42) / (c21 ** 2))

    validity = FeatureValidity.VALID if n_samples >= 128 else FeatureValidity.PARTIALLY_VALID

    return CumulantFeatures(
        c20=complex(round(c20.real, 6), round(c20.imag, 6)),
        c21=round(c21, 6),
        c40=complex(round(c40.real, 6), round(c40.imag, 6)),
        c41=complex(round(c41.real, 6), round(c41.imag, 6)),
        c42=round(c42, 6),
        f20=round(f20, 6),
        f40=round(f40, 6),
        f41=round(f41, 6),
        f42=round(f42, 6),
        validity=validity,
    )
