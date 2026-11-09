"""Univariate GARCH model variants used in the first step of the DCC models
(Section 5, Appendix D). Five specifications per stock:

    GARCH   : h_t = omega + beta h_{t-1} + alpha r_{t-1}^2
    tGARCH  : h_t = omega + beta h_{t-1} + alpha r_{t-1}^2 + gamma r_{t-1}^2 I{r_{t-1}<0}  (GJR)
    rGARCH  : h_t = omega + beta h_{t-1} + alpha RV_{t-1}
    trGARCH : h_t = omega + beta h_{t-1} + alpha RV_{t-1} + gamma RV_{t-1} I{r_{t-1}<0}
    crGARCH : h_t = omega + beta h_{t-1} + alpha^+ V+_{t-1} + alpha^- V-_{t-1}

where RV_t is the realized variance (diagonal of RCOV_t) and V+_t, V-_t are
the realized semivariances (diagonals of P_t, N_t). GARCH and tGARCH are
fit with the ``arch`` package (ZeroMean, GARCH/GJR volatility processes);
rGARCH, trGARCH and crGARCH are realized-measure extensions not covered by
``arch`` and are fit with the covariance-targeted recursion engine in
``src.matrix_garch`` (reduced to the 1x1 / scalar case).
"""
from __future__ import annotations

import numpy as np
from arch import arch_model

from src.matrix_garch import fit_covariance_targeted_garch, gaussian_qmle_loglik

UNIVARIATE_MODELS = ["GARCH", "tGARCH", "rGARCH", "trGARCH", "crGARCH"]


def _scalar_regressor(x: np.ndarray) -> np.ndarray:
    """Reshape a (T,) series of realized measures into (T, 1, 1) matrices so
    that the shared ``matrix_garch`` recursion engine can be reused for the
    scalar univariate case."""
    return x.reshape(-1, 1, 1)


def fit_garch_arch_pkg(returns_pct: np.ndarray, asymmetric: bool) -> dict:
    """GARCH(1,1) or GJR-GARCH(1,1) via the ``arch`` package (Bollerslev
    1986 / Glosten-Jagannathan-Runkle 1993), used for the GARCH and tGARCH
    variants (daily returns only, no realized measures)."""
    o = 1 if asymmetric else 0
    am = arch_model(returns_pct, mean="Zero", vol="GARCH", p=1, o=o, q=1, dist="normal")
    res = am.fit(disp="off", show_warning=False)
    h_pct = np.asarray(res.conditional_volatility) ** 2
    h_dec = h_pct / 100.0 ** 2
    returns_dec = returns_pct / 100.0
    # arch's res.loglikelihood is on the percent-return scale and includes the
    # 0.5 and -N log(2 pi) terms that src.matrix_garch.gaussian_qmle_loglik
    # omits; recompute the same QMLE criterion on decimal returns here so that
    # GARCH/tGARCH are comparable on an identical scale and formula to
    # rGARCH/trGARCH/crGARCH (Section 5: "the model with highest
    # log-likelihood is selected" presumes a common likelihood definition).
    loglik = gaussian_qmle_loglik(h_dec.reshape(-1, 1, 1), returns_dec.reshape(-1, 1))
    return {
        "loglik": loglik,
        "params": res.params.to_dict(),
        "h": h_dec,
        "converged": res.convergence_flag == 0,
    }


def fit_univariate_models(returns: np.ndarray, rv: np.ndarray, pos_var: np.ndarray,
                           neg_var: np.ndarray) -> dict[str, dict]:
    """Fit all five univariate specifications for one stock.

    ``returns`` (T,) are daily simple returns (decimal, not percent); ``rv``
    is the realized variance RCOV_{ii,t}; ``pos_var``/``neg_var`` are the
    realized semivariances P_{ii,t}/N_{ii,t}. All series share the same T
    trading days, aligned so that day t uses information up to and including
    day t for the *regressor* (dated t-1 relative to h_t in the recursion).
    """
    results: dict[str, dict] = {}
    returns_pct = returns * 100.0  # arch package convention; scaled back after

    results["GARCH"] = fit_garch_arch_pkg(returns_pct, asymmetric=False)
    results["tGARCH"] = fit_garch_arch_pkg(returns_pct, asymmetric=True)

    h_bar = np.array([[np.var(returns)]])
    rv3 = _scalar_regressor(rv)
    pv3 = _scalar_regressor(pos_var)
    nv3 = _scalar_regressor(neg_var)
    r3 = returns.reshape(-1, 1)

    fit_r = fit_covariance_targeted_garch({"a": rv3}, r3, h_bar)
    results["rGARCH"] = {"loglik": fit_r["loglik"], "params": fit_r["coefs"] | {"beta": fit_r["beta"]},
                          "h": fit_r["H"][:, 0, 0], "converged": fit_r["converged"]}

    i_neg = (returns < 0).astype(float).reshape(-1, 1, 1)
    trv3 = rv3 * i_neg
    fit_tr = fit_covariance_targeted_garch({"a": rv3, "g": trv3}, r3, h_bar)
    results["trGARCH"] = {"loglik": fit_tr["loglik"], "params": fit_tr["coefs"] | {"beta": fit_tr["beta"]},
                           "h": fit_tr["H"][:, 0, 0], "converged": fit_tr["converged"]}

    fit_cr = fit_covariance_targeted_garch({"ap": pv3, "an": nv3}, r3, h_bar)
    results["crGARCH"] = {"loglik": fit_cr["loglik"], "params": fit_cr["coefs"] | {"beta": fit_cr["beta"]},
                           "h": fit_cr["H"][:, 0, 0], "converged": fit_cr["converged"]}

    return results


def select_best_model(results: dict[str, dict]) -> str:
    """Pick the univariate specification with the highest log-likelihood
    (Section 5: "the model with highest log-likelihood is selected")."""
    return max(results, key=lambda name: results[name]["loglik"])
