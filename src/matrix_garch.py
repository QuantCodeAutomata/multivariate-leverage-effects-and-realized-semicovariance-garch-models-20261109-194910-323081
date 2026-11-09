"""Generic covariance-targeted GARCH(1,1)-type recursion and QMLE.

All eight bivariate semicovariance GARCH specifications of Section 2.2
(Eqs. (5)-(11)), the five univariate realized-GARCH specifications of
Section 5 (Appendix D), and the five DCC-type correlation recursions of
Section 5.2-5.3 (Eqs. (23)-(27)) share one algebraic structure:

    H_t = (1 - sum_j a_j - b) * Hbar  +  b * H_{t-1}  +  sum_j a_j * X_{t-1,j}

where Hbar is fixed by covariance targeting (Section 2.2: "we concentrate
Hbar out of the likelihood... replacing it with the sample analogue") and
the X_j are lagged regressor matrices specific to each model (outer
products of returns, realized (semi)covariances, or their threshold-masked
/ split ("T(.)") versions). Substituting the targeting constraint
Hbar = (1 - sum_j a_j - b) Hbar + sum_j a_j * mean(X_j) + b * Hbar gives the
equivalent, numerically convenient form used below:

    H_t = Hbar * (1 - b) - sum_j a_j * (mean(X_j) - X_{t-1,j})  +  b * H_{t-1}

which is algebraically identical to the paper's alpha* adjustment (e.g.
Eq. (7): a* = (aP Pbar + aN Nbar + aM Mbar) Hbar^{-1}) without requiring the
matrix inverse Hbar^{-1} to be formed explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


@dataclass
class ModelSpec:
    """One covariance-targeted recursion: regressors X_j and their names."""

    name: str
    regressors: dict[str, np.ndarray]  # each (T, n, n), aligned with `returns`
    n_assets: int


def run_recursion(beta: float, coefs: dict[str, float], regressors: dict[str, np.ndarray],
                   h_bar: np.ndarray, h_init: np.ndarray | None = None) -> np.ndarray:
    """Filter the covariance path H_1..H_T (Eqs. (5)-(11) generic form).

    ``h_bar`` fixes the covariance-targeting constant (the long-run mean the
    recursion reverts to). The presample value ``H_1`` defaults to ``h_bar``
    (the in-sample estimation assumption), but can be overridden with
    ``h_init`` -- the last covariance filtered on the estimation window --
    when this recursion continues a path out-of-sample instead of starting
    one: re-seeding an out-of-sample forecast at the unconditional mean
    rather than at the model's own last filtered value discards the
    persistence the recursion was estimated to have (``beta`` close to the
    stationary bound), and no future information enters either way since
    ``H_t`` for t >= 2 is built from regressors dated t-1.
    """
    any_reg = next(iter(regressors.values()))
    T, n, _ = any_reg.shape
    x_bar = {k: v.mean(axis=0) for k, v in regressors.items()}
    const = h_bar * (1.0 - beta) - sum(coefs[k] * x_bar[k] for k in coefs)
    H = np.empty((T, n, n))
    H[0] = h_bar if h_init is None else h_init
    for t in range(1, T):
        H[t] = const + beta * H[t - 1] + sum(coefs[k] * regressors[k][t - 1] for k in coefs)
    return H


def gaussian_qmle_loglik(H: np.ndarray, returns: np.ndarray) -> float:
    """Multivariate Gaussian QMLE criterion, L = sum_t (-log|H_t| - r_t' H_t^-1 r_t).

    Matches the paper's quasi-likelihood objective (Section 2.2): the
    constant -N log(2 pi) and the conventional 1/2 scaling are omitted since
    they do not affect the maximizer; reported log-likelihoods therefore
    differ from the paper's Table 2/3 values by that fixed affine scaling.
    """
    T = H.shape[0]
    total = 0.0
    for t in range(T):
        Ht = H[t]
        sign, logdet = np.linalg.slogdet(Ht)
        if sign <= 0:
            return -1e12
        r = returns[t]
        quad = r @ np.linalg.solve(Ht, r)
        total += -logdet - quad
    return total


def _unpack(theta: np.ndarray, names: list[str]) -> tuple[float, dict[str, float]]:
    beta = theta[0]
    coefs = {name: theta[i + 1] for i, name in enumerate(names)}
    return beta, coefs


def fit_covariance_targeted_garch(regressors: dict[str, np.ndarray], returns: np.ndarray,
                                   h_bar: np.ndarray, x0: dict[str, float] | None = None
                                   ) -> dict:
    """QMLE estimation of beta and the a_j coefficients by constrained
    numerical optimization (scipy.optimize.minimize, SLSQP), subject to
    stationarity b + sum_j a_j < 1 and non-negativity a_j, b >= 0, as used
    throughout Section 2.2 for the covariance-targeted recursions.
    """
    names = list(regressors.keys())
    k = len(names)
    if x0 is None:
        x0 = {name: 0.05 / k for name in names}
    theta0 = np.array([0.90] + [x0[name] for name in names])

    def neg_ll(theta: np.ndarray) -> float:
        beta, coefs = _unpack(theta, names)
        if beta < 0 or beta > 0.999 or any(c < -0.5 for c in coefs.values()):
            return 1e12
        if beta + sum(coefs.values()) >= 0.999:
            return 1e12
        H = run_recursion(beta, coefs, regressors, h_bar)
        ll = gaussian_qmle_loglik(H, returns)
        if not np.isfinite(ll):
            return 1e12
        return -ll

    bounds = [(0.0, 0.999)] + [(-0.5, 0.999) for _ in names]
    cons = {"type": "ineq", "fun": lambda th: 0.999 - th[0] - sum(th[1:])}
    res = minimize(neg_ll, theta0, method="SLSQP", bounds=bounds, constraints=[cons],
                    options={"maxiter": 300, "ftol": 1e-8})
    beta, coefs = _unpack(res.x, names)
    H = run_recursion(beta, coefs, regressors, h_bar)
    loglik = gaussian_qmle_loglik(H, returns)
    return {
        "beta": beta,
        "coefs": coefs,
        "loglik": loglik,
        "H": H,
        "converged": bool(res.success),
        "n_params": k + 1,
        "names": names,
    }
