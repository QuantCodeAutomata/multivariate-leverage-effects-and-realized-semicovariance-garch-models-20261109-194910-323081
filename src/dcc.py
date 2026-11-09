"""High-dimensional DCC models with realized (semi)correlations (Section 5,
Eqs. (22)-(27)).

The correlation-targeted recursion for Q_t shares the generic structure of
``src.matrix_garch.run_recursion`` (an autoregressive matrix recursion with
covariance targeting), but the Gaussian QMLE criterion is evaluated on the
*normalized* correlation matrix R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
(Eq. 22), not on Q_t itself -- this is Engle's (2002) two-step DCC
quasi-likelihood (the default used here per footnote 11, i.e. without the
Aielli 2013 correction).

Five specifications (split discordant terms are *not* used here because the
ordering of the N=24 stocks is arbitrary, Section 5.2):

    DCC   (Eq. 23): a * z_{t-1} z_{t-1}'
    rDCC  (Eq. 24): a * RCOR_{t-1}
    crDCC (Eq. 25): aP*R^P_{t-1} + aN*R^N_{t-1} + aM*R^M_{t-1}
    tDCC  (Eq. 26): aP*(z+z+') + aN*(z-z-') + aM*(z+z-' + z-z+')
    trDCC (Eq. 27): aP*(RCOR⊙I+I+') + aN*(RCOR⊙I-I-') + aM*(RCOR⊙(I+I-'+I-I+'))
"""
from __future__ import annotations

import numpy as np

from src.matrix_garch import run_recursion
from src.semicovariance import signed_outer_products
from src.semicorrelation import realized_semicorrelations

DCC_MODEL_NAMES = ["DCC", "rDCC", "crDCC", "tDCC", "trDCC"]


def realized_correlation(rcov: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    """RCOR_t = diag(RCOV_t)^{-1/2} RCOV_t diag(RCOV_t)^{-1/2}, the full
    realized correlation matrix (used as the rDCC/trDCC regressor)."""
    d = np.clip(np.diagonal(rcov, axis1=-2, axis2=-1), eps, None)
    inv_sqrt = 1.0 / np.sqrt(d)
    return inv_sqrt[:, :, None] * rcov * inv_sqrt[:, None, :]


def standardized_returns(daily_returns: np.ndarray, h: np.ndarray) -> np.ndarray:
    """z_t = D_t^{-1} r_t with D_t = diag(sqrt(h_{1,t}), ..., sqrt(h_{N,t}))."""
    return daily_returns / np.sqrt(np.clip(h, 1e-12, None))


def build_dcc_regressors(z: np.ndarray, rcov: np.ndarray, p: np.ndarray, n_mat: np.ndarray,
                          mp: np.ndarray, mn: np.ndarray) -> dict[str, dict]:
    """Regressor dictionaries for the five DCC variants, built from the
    standardized returns ``z`` (T, N) and the realized (semi)covariances."""
    signed_z = signed_outer_products_df(z)
    rcor = realized_correlation(rcov)
    semicorr = realized_semicorrelations(rcov, p, n_mat, mp, mn)

    r = np.sign(z)  # sign of standardized return equals sign of raw return
    i_pos = (r > 0).astype(float)
    i_neg = (r <= 0).astype(float)
    outer = lambda a, b: np.einsum("ti,tj->tij", a, b)

    regs: dict[str, dict] = {
        "DCC": {"a": signed_z["rp_rp"] + signed_z["rn_rn"] + signed_z["rp_rn"] + signed_z["rn_rp"]},
        "rDCC": {"a": rcor},
        "crDCC": {"aP": semicorr["R_P"], "aN": semicorr["R_N"], "aM": semicorr["R_M"]},
        "tDCC": {
            "aP": signed_z["rp_rp"], "aN": signed_z["rn_rn"],
            "aM": signed_z["rp_rn"] + signed_z["rn_rp"],
        },
        "trDCC": {
            "aP": rcor * outer(i_pos, i_pos), "aN": rcor * outer(i_neg, i_neg),
            "aM": rcor * (outer(i_pos, i_neg) + outer(i_neg, i_pos)),
        },
    }
    return regs


def signed_outer_products_df(z: np.ndarray) -> dict[str, np.ndarray]:
    import pandas as pd
    return signed_outer_products(pd.DataFrame(z))


def to_correlation(Q: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Normalize a path of Q_t matrices to proper correlation matrices
    R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2} (Eq. 22).

    The semicorrelation-based regressors (crDCC/tDCC/trDCC, Eqs. 25-27)
    include the indefinite discordant term aM*R^M_{t-1} (or its z+z-'
    analogue), so Q_t is not guaranteed to stay positive (semi)definite for
    N > 2 (Appendix A). When an eigenvalue of Q_t falls at or below
    ``floor``, the symmetric matrix is projected onto the positive-definite
    cone by flooring its eigenvalues before normalizing -- the safeguard the
    paper allows for ("For positive definiteness, if needed, enforce during
    estimation on the filtered path", Section 5.2). The floor is set to
    1e-3 rather than a near-zero value (not specified in the paper): with
    the smaller, 5-stock cross-section used here, the raw filtered Q_t
    occasionally has an eigenvalue of order 1e-8, which a near-zero floor
    leaves effectively unprojected and which then inflates QLIKE's log|R_t|
    and R_t^{-1} terms by several orders of magnitude for that single day;
    the 1e-3 floor keeps every forecast correlation matrix safely
    well-conditioned while leaving it unchanged on all other days.
    """
    T = Q.shape[0]
    R = np.empty_like(Q)
    for t in range(T):
        Qt = 0.5 * (Q[t] + Q[t].T)
        d = np.diagonal(Qt)
        if d.min() <= floor or np.linalg.eigvalsh(Qt).min() <= floor:
            eigvals, eigvecs = np.linalg.eigh(Qt)
            eigvals = np.clip(eigvals, floor, None)
            Qt = (eigvecs * eigvals) @ eigvecs.T
        dsqrt = np.sqrt(np.diagonal(Qt))
        R[t] = Qt / np.outer(dsqrt, dsqrt)
    return R


def correlation_qmle_loglik(Q: np.ndarray, z: np.ndarray) -> float:
    """Engle's (2002) second-step DCC quasi-log-likelihood on the normalized
    correlation path R_t (the -log|2 pi| - z'z additive constants, which do
    not depend on the DCC parameters, are dropped as in
    ``matrix_garch.gaussian_qmle_loglik``)."""
    R_path = to_correlation(Q)
    T = R_path.shape[0]
    total = 0.0
    for t in range(T):
        R = R_path[t]
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return -1e12
        quad = z[t] @ np.linalg.solve(R, z[t])
        total += -logdet - quad
    return total


def fit_dcc(regressors: dict[str, np.ndarray], z: np.ndarray, q_bar: np.ndarray,
            x0: dict[str, float] | None = None) -> dict:
    """QMLE estimation of one DCC specification (beta and a_j), reusing the
    generic recursion of ``matrix_garch.run_recursion`` for Q_t and
    evaluating the correlation likelihood on R_t (Eq. 22)."""
    from scipy.optimize import minimize

    names = list(regressors.keys())
    k = len(names)
    if x0 is None:
        x0 = {name: 0.02 / k for name in names}
    theta0 = np.array([0.95] + [x0[name] for name in names])

    def neg_ll(theta: np.ndarray) -> float:
        beta = theta[0]
        coefs = {name: theta[i + 1] for i, name in enumerate(names)}
        if beta < 0 or beta > 0.999 or any(c < -0.5 for c in coefs.values()):
            return 1e12
        if beta + sum(coefs.values()) >= 0.999:
            return 1e12
        Q = run_recursion(beta, coefs, regressors, q_bar)
        ll = correlation_qmle_loglik(Q, z)
        if not np.isfinite(ll):
            return 1e12
        return -ll

    bounds = [(0.0, 0.999)] + [(-0.5, 0.999) for _ in names]
    cons = {"type": "ineq", "fun": lambda th: 0.999 - th[0] - sum(th[1:])}
    res = minimize(neg_ll, theta0, method="SLSQP", bounds=bounds, constraints=[cons],
                    options={"maxiter": 200, "ftol": 1e-7})
    beta = res.x[0]
    coefs = {name: res.x[i + 1] for i, name in enumerate(names)}
    Q = run_recursion(beta, coefs, regressors, q_bar)
    R = to_correlation(Q)
    loglik = correlation_qmle_loglik(Q, z)
    return {
        "beta": beta, "coefs": coefs, "loglik": loglik, "Q": Q, "R": R,
        "converged": bool(res.success), "n_params": k + 1, "names": names,
    }


def fit_all_dcc_models(z: np.ndarray, rcov: np.ndarray, p: np.ndarray, n_mat: np.ndarray,
                        mp: np.ndarray, mn: np.ndarray) -> dict[str, dict]:
    """Fit the five DCC-type specifications for the full N-dimensional system."""
    q_bar = np.cov(z.T, bias=True)
    regs = build_dcc_regressors(z, rcov, p, n_mat, mp, mn)
    results = {}
    for name in DCC_MODEL_NAMES:
        fit = fit_dcc(regs[name], z, q_bar)
        fit["model"] = name
        results[name] = fit
    return results


def average_correlation(R: np.ndarray) -> np.ndarray:
    """rho_bar_t = 2/(N(N-1)) sum_{i>j} rho_{ij,t} (Section 5.4)."""
    N = R.shape[-1]
    iu = np.triu_indices(N, k=1)
    return R[:, iu[0], iu[1]].mean(axis=1)


def std_correlation(R: np.ndarray) -> np.ndarray:
    """sigma(rho_t) = sqrt(2/(N(N-1)) sum_{i>j} (rho_{ij,t} - rho_bar_t)^2)."""
    N = R.shape[-1]
    iu = np.triu_indices(N, k=1)
    pairs = R[:, iu[0], iu[1]]
    return pairs.std(axis=1)
