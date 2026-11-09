"""Unit tests for realized semicorrelation matrices (Eqs. (16)-(19))."""
from __future__ import annotations

import numpy as np

from src.semicorrelation import check_semicorrelation_bounds, realized_semicorrelations


def test_semicorrelations_diagonal_and_bounds(synthetic_bivariate_data):
    data = synthetic_bivariate_data
    semicorr = realized_semicorrelations(data["RCOV"], data["P"], data["N"], data["Mp"], data["Mn"])
    N = data["RCOV"].shape[-1]
    diag_p = np.diagonal(semicorr["R_P"], axis1=-2, axis2=-1)
    diag_n = np.diagonal(semicorr["R_N"], axis1=-2, axis2=-1)
    diag_m = np.diagonal(semicorr["R_M"], axis1=-2, axis2=-1)
    np.testing.assert_allclose(diag_p, 1.0, atol=1e-8)
    np.testing.assert_allclose(diag_n, 1.0, atol=1e-8)
    np.testing.assert_allclose(diag_m, 1.0, atol=1e-8)

    bounds = check_semicorrelation_bounds(semicorr["R_P"], semicorr["R_N"], semicorr["R_M"])
    assert bounds["max_abs_offdiag_RP"] <= 1.0 + 1e-6
    assert bounds["max_abs_offdiag_RN"] <= 1.0 + 1e-6
    # Appendix A: off-diagonal entries of R^M (= normalized M, since the added
    # identity only affects the diagonal) are bounded in [-1, 0] by Cauchy-Schwarz.
    off = ~np.eye(N, dtype=bool)
    assert (semicorr["R_M"][:, off] <= 1e-6).all()
    assert (semicorr["R_M"][:, off] >= -1.0 - 1e-6).all()


def test_r_m_equals_identity_when_no_discordant_mass():
    """If M+ = M- = 0 (returns never have opposite signs within the day),
    R^M collapses to the identity matrix exactly (Eq. (19))."""
    T, N = 5, 2
    rcov = np.stack([np.eye(N) for _ in range(T)])
    p = rcov.copy()
    n_mat = np.zeros((T, N, N))
    mp = np.zeros((T, N, N))
    mn = np.zeros((T, N, N))
    semicorr = realized_semicorrelations(rcov, p, n_mat, mp, mn)
    np.testing.assert_allclose(semicorr["R_M"], np.stack([np.eye(N)] * T), atol=1e-10)
