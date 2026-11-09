"""Realized semicorrelation matrices (Section 5.1, Eqs. (16)-(19)).

From the realized covariance decomposition RCOV = P + N + M+ + M- the paper
builds three semicorrelation matrices used as DCC regressors:

    R^P_t = diag(P_t)^{-1/2} P_t diag(P_t)^{-1/2}                    (Eq. 16)
    R^N_t = diag(N_t)^{-1/2} N_t diag(N_t)^{-1/2}                    (Eq. 17)
    R^M_t = diag(RCOV_t)^{-1/2} M_t diag(RCOV_t)^{-1/2} + I_N        (Eq. 19)

where M_t = M+_t + M-_t. R^M is normalized by the *total* realized variances
(not the semivariances), and has the identity added so that it has a unit
diagonal like a correlation matrix (M_t has a zero diagonal). Appendix A
shows, via Cauchy-Schwarz, that each off-diagonal entry of R^M - I lies in
[-1, 0]; this bound is checked in the test suite.
"""
from __future__ import annotations

import numpy as np


def _inv_sqrt_diag(mat: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    """diag(mat)^{-1/2} for a (T, N, N) stack, with a small floor against
    division by zero on days with no intraday variation for some asset."""
    d = np.diagonal(mat, axis1=-2, axis2=-1).copy()
    d = np.clip(d, eps, None)
    return 1.0 / np.sqrt(d)


def realized_semicorrelations(rcov: np.ndarray, p: np.ndarray, n_mat: np.ndarray,
                               mp: np.ndarray, mn: np.ndarray) -> dict[str, np.ndarray]:
    """Compute R^P, R^N, R^M for a (T, N, N) stack of daily matrices."""
    T, N, _ = rcov.shape
    eye = np.eye(N)

    inv_sqrt_p = _inv_sqrt_diag(p)
    r_p = inv_sqrt_p[:, :, None] * p * inv_sqrt_p[:, None, :]

    inv_sqrt_n = _inv_sqrt_diag(n_mat)
    r_n = inv_sqrt_n[:, :, None] * n_mat * inv_sqrt_n[:, None, :]

    inv_sqrt_rcov = _inv_sqrt_diag(rcov)
    m = mp + mn
    r_m = inv_sqrt_rcov[:, :, None] * m * inv_sqrt_rcov[:, None, :] + eye

    return {"R_P": r_p, "R_N": r_n, "R_M": r_m}


def check_semicorrelation_bounds(r_p: np.ndarray, r_n: np.ndarray, r_m: np.ndarray) -> dict[str, float]:
    """Diagnostics: R^P, R^N are correlation-like (diagonal 1, |off-diag|<=1);
    R^M - I is bounded in [-1, 0] off-diagonal (Appendix A)."""
    N = r_p.shape[-1]
    off = ~np.eye(N, dtype=bool)
    return {
        "max_abs_offdiag_RP": float(np.abs(r_p[:, off]).max()),
        "max_abs_offdiag_RN": float(np.abs(r_n[:, off]).max()),
        "min_offdiag_RM_minus_I": float((r_m[:, off] - 0.0).min()),
        "max_offdiag_RM_minus_I": float((r_m[:, off] - 1.0 + 1.0).max()),
    }
