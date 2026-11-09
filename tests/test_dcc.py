"""Unit tests for the high-dimensional DCC recursions (Eqs. (22)-(27))."""
from __future__ import annotations

import numpy as np

from src.dcc import (
    DCC_MODEL_NAMES, average_correlation, build_dcc_regressors, fit_all_dcc_models,
    realized_correlation, standardized_returns, std_correlation,
)


def _make_system(T: int = 150, N: int = 4, seed: int = 1):
    rng = np.random.default_rng(seed)
    base = 0.3 * np.ones((N, N)) + 0.7 * np.eye(N)
    L = np.linalg.cholesky(base)
    z = rng.standard_normal((T, N))
    daily_returns = (z @ L.T) * 0.01
    h = np.full((T, N), 1e-4)

    n_intraday = 20
    rcov = np.empty((T, N, N))
    p = np.empty((T, N, N))
    n_mat = np.empty((T, N, N))
    mp = np.empty((T, N, N))
    mn = np.empty((T, N, N))
    for t in range(T):
        intraday = rng.multivariate_normal(np.zeros(N), base / n_intraday, size=n_intraday) * 0.01
        r_pos = np.clip(intraday, 0, None)
        r_neg = np.clip(intraday, None, 0)
        p[t] = r_pos.T @ r_pos
        n_mat[t] = r_neg.T @ r_neg
        mp[t] = r_pos.T @ r_neg
        mn[t] = mp[t].T
        rcov[t] = p[t] + n_mat[t] + mp[t] + mn[t]
    return daily_returns, h, rcov, p, n_mat, mp, mn


def test_realized_correlation_has_unit_diagonal():
    _, _, rcov, _, _, _, _ = _make_system()
    rcor = realized_correlation(rcov)
    diag = np.diagonal(rcor, axis1=-2, axis2=-1)
    np.testing.assert_allclose(diag, 1.0, atol=1e-8)


def test_standardized_returns_scale_by_inverse_volatility():
    r = np.array([[0.02, -0.01]])
    h = np.array([[0.0004, 0.0001]])
    z = standardized_returns(r, h)
    np.testing.assert_allclose(z, [[1.0, -1.0]])


def test_fit_all_dcc_models_returns_valid_correlation_matrices():
    daily_returns, h, rcov, p, n_mat, mp, mn = _make_system()
    z = standardized_returns(daily_returns, h)
    fits = fit_all_dcc_models(z, rcov, p, n_mat, mp, mn)
    assert set(fits.keys()) == set(DCC_MODEL_NAMES)
    for name, fit in fits.items():
        R = fit["R"]
        diag = np.diagonal(R, axis1=-2, axis2=-1)
        np.testing.assert_allclose(diag, 1.0, atol=1e-6)
        assert np.isfinite(fit["loglik"])


def test_average_and_std_correlation_bounds():
    daily_returns, h, rcov, p, n_mat, mp, mn = _make_system()
    z = standardized_returns(daily_returns, h)
    fits = fit_all_dcc_models(z, rcov, p, n_mat, mp, mn)
    R = fits["DCC"]["R"]
    avg = average_correlation(R)
    std = std_correlation(R)
    assert (avg >= -1.0 - 1e-6).all() and (avg <= 1.0 + 1e-6).all()
    assert (std >= -1e-8).all()
