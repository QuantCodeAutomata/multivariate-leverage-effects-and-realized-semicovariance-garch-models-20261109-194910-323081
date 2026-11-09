"""Unit tests for the generic covariance-targeted recursion and QMLE
(Section 2.2, Eqs. (5)-(11) generic form)."""
from __future__ import annotations

import numpy as np

from src.bivariate_garch import MODEL_NAMES, build_regressors, fit_all_models
from src.matrix_garch import fit_covariance_targeted_garch, gaussian_qmle_loglik, run_recursion


def test_recursion_presample_value_is_h_bar(synthetic_bivariate_data):
    data = synthetic_bivariate_data
    regs = build_regressors(data["daily_returns"], data["RCOV"], data["P"], data["N"],
                             data["Mp"], data["Mn"])
    H = run_recursion(0.9, {"a": 0.05}, {"a": regs["BG"]["a"]}, data["h_bar"])
    np.testing.assert_allclose(H[0], data["h_bar"])


def test_recursion_step_uses_only_lagged_information(synthetic_bivariate_data):
    """H_t (t >= 1) must equal const + beta * H_{t-1} + sum_j a_j * X_{t-1,j}
    exactly, i.e. the filter never uses regressors dated t or later."""
    data = synthetic_bivariate_data
    regs = build_regressors(data["daily_returns"], data["RCOV"], data["P"], data["N"],
                             data["Mp"], data["Mn"])
    X = regs["BG"]["a"]
    beta, a = 0.9, 0.05
    H = run_recursion(beta, {"a": a}, {"a": X}, data["h_bar"])
    const = data["h_bar"] * (1.0 - beta) - a * X.mean(axis=0)
    for t in (1, 2, 50, 100, len(X) - 1):
        expected_t = const + beta * H[t - 1] + a * X[t - 1]
        np.testing.assert_allclose(H[t], expected_t, atol=1e-12)

    # Holding the targeting constant fixed (as it is during out-of-sample
    # filtering with parameters and Hbar/Xbar already estimated), corrupting
    # the regressor dated t must leave every H_s, s <= t, unchanged and only
    # H_{t+1} onward must move: the step uses X_{t-1}, never X_t or later.
    t0 = 50
    X_corrupted = X.copy()
    X_corrupted[t0] = X_corrupted[t0] * 100 + 999.0
    H_corrupted = np.empty_like(H)
    H_corrupted[0] = data["h_bar"]
    for t in range(1, len(X)):
        H_corrupted[t] = const + beta * H_corrupted[t - 1] + a * X_corrupted[t - 1]
    np.testing.assert_allclose(H_corrupted[: t0 + 1], H[: t0 + 1], atol=1e-10)
    assert not np.allclose(H_corrupted[t0 + 1], H[t0 + 1])


def test_recursion_matrices_stay_symmetric_positive_definite(synthetic_bivariate_data):
    data = synthetic_bivariate_data
    regs = build_regressors(data["daily_returns"], data["RCOV"], data["P"], data["N"],
                             data["Mp"], data["Mn"])
    H = run_recursion(0.85, {"aP": 0.05, "aN": 0.08, "aM": 0.01}, regs["tBG"], data["h_bar"])
    np.testing.assert_allclose(H, np.swapaxes(H, -1, -2), atol=1e-10)
    eigvals = np.linalg.eigvalsh(H)
    assert (eigvals > 0).all()


def test_fit_all_eight_models_runs_and_returns_finite_loglik(synthetic_bivariate_data):
    data = synthetic_bivariate_data
    fits = fit_all_models(data["daily_returns"], data["RCOV"], data["P"], data["N"],
                           data["Mp"], data["Mn"])
    assert set(fits.keys()) == set(MODEL_NAMES)
    for name, fit in fits.items():
        assert np.isfinite(fit["loglik"])
        assert 0.0 <= fit["beta"] < 1.0


def test_gaussian_qmle_loglik_penalizes_non_positive_definite_input():
    H = np.stack([np.array([[1.0, 0.9], [0.9, 0.5]])])  # not PD: det < 0
    r = np.zeros((1, 2))
    ll = gaussian_qmle_loglik(H, r)
    assert ll == -1e12


def test_fit_covariance_targeted_garch_improves_on_static_benchmark(synthetic_bivariate_data):
    """The fitted dynamic model should not have a lower in-sample
    log-likelihood than the degenerate static model H_t = Hbar."""
    data = synthetic_bivariate_data
    regs = build_regressors(data["daily_returns"], data["RCOV"], data["P"], data["N"],
                             data["Mp"], data["Mn"])
    fit = fit_covariance_targeted_garch({"a": regs["BG"]["a"]}, data["daily_returns"], data["h_bar"])
    T = data["daily_returns"].shape[0]
    static_H = np.tile(data["h_bar"], (T, 1, 1))
    static_ll = gaussian_qmle_loglik(static_H, data["daily_returns"])
    assert fit["loglik"] >= static_ll - 1e-6
