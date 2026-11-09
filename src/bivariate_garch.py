"""The eight bivariate semicovariance GARCH specifications (Section 2.2).

Each specification is a ``matrix_garch.ModelSpec`` built from the daily and
realized (semi)covariance series of ``src.semicovariance``. The regressor
dictionaries follow the paper's equations exactly:

* BG   (Eq. 5):  a * r_{t-1} r_{t-1}'
* tBG  (Eq. 9):  aP*(r+r+') + aN*(r-r-') + aM*(r+r-' + r-r+')
* rBG  (Eq. 6):  a * RCOV_{t-1}
* trBG (Eq. 11): aP*(RCOV ⊙ I+I+') + aN*(RCOV ⊙ I-I-') + aM*(RCOV ⊙ (I+I-'+I-I+'))
* crBG (Eq. 7):  aP*P_{t-1} + aN*N_{t-1} + aM*(M+_{t-1}+M-_{t-1})
* tBG-S  (Eq. 10): aP*(r+r+') + aN*(r-r-') + aM+*T(r+r-') + aM-*T(r-r+')
* trBG-S (footnote 9, split analogue of Eq. 11):
      aP*(RCOV⊙I+I+') + aN*(RCOV⊙I-I-') + aM+*T(RCOV⊙I+I-') + aM-*T(RCOV⊙I-I+')
* crBG-S (Eq. 8): aP*P_{t-1} + aN*N_{t-1} + aM+*T(M+_{t-1}) + aM-*T(M-_{t-1})

where T(.) is the symmetrizing operator of footnote 7 (copies the upper
triangle onto the lower triangle), implemented in
``src.semicovariance.t_operator``.
"""
from __future__ import annotations

import numpy as np

from src.matrix_garch import fit_covariance_targeted_garch
from src.semicovariance import signed_outer_products, t_operator

MODEL_NAMES = ["BG", "tBG", "rBG", "trBG", "crBG", "tBG-S", "trBG-S", "crBG-S"]


def build_regressors(daily_returns: np.ndarray, rcov: np.ndarray, p: np.ndarray,
                      n_mat: np.ndarray, mp: np.ndarray, mn: np.ndarray) -> dict[str, dict]:
    """All regressor matrices needed by every bivariate specification.

    ``daily_returns``, ``rcov``, ``p``, ``n_mat``, ``mp``, ``mn`` must share
    the same length T and day alignment (trading days with both intraday
    and daily data available).
    """
    signed = signed_outer_products_df(daily_returns)
    regs: dict[str, dict] = {}
    regs["BG"] = {"a": np.einsum("ti,tj->tij", daily_returns, daily_returns)}
    regs["tBG"] = {
        "aP": signed["rp_rp"], "aN": signed["rn_rn"],
        "aM": signed["rp_rn"] + signed["rn_rp"],
    }
    regs["rBG"] = {"a": rcov}
    regs["trBG"] = {
        "aP": rcov * signed["ip_ip"], "aN": rcov * signed["in_in"],
        "aM": rcov * (signed["ip_in"] + signed["in_ip"]),
    }
    regs["crBG"] = {"aP": p, "aN": n_mat, "aM": mp + mn}
    regs["tBG-S"] = {
        "aP": signed["rp_rp"], "aN": signed["rn_rn"],
        "aMp": t_operator(signed["rp_rn"]), "aMn": t_operator(signed["rn_rp"]),
    }
    regs["trBG-S"] = {
        "aP": rcov * signed["ip_ip"], "aN": rcov * signed["in_in"],
        "aMp": t_operator(rcov * signed["ip_in"]), "aMn": t_operator(rcov * signed["in_ip"]),
    }
    regs["crBG-S"] = {"aP": p, "aN": n_mat, "aMp": t_operator(mp), "aMn": t_operator(mn)}
    return regs


def signed_outer_products_df(daily_returns: np.ndarray) -> dict[str, np.ndarray]:
    """Thin numpy wrapper around ``semicovariance.signed_outer_products``."""
    import pandas as pd

    df = pd.DataFrame(daily_returns)
    return signed_outer_products(df)


def fit_all_models(daily_returns: np.ndarray, rcov: np.ndarray, p: np.ndarray,
                    n_mat: np.ndarray, mp: np.ndarray, mn: np.ndarray) -> dict[str, dict]:
    """QMLE-fit the eight bivariate models on one (stock, market) pair.

    ``daily_returns`` (T, 2) must be aligned day-by-day with the realized
    measures ``rcov``, ``p``, ``n_mat``, ``mp``, ``mn`` (T, 2, 2), using
    returns dated t-1 to build H_t (Section 2.2, no look-ahead).
    """
    h_bar = np.cov(daily_returns.T, bias=True)
    regs = build_regressors(daily_returns, rcov, p, n_mat, mp, mn)
    results = {}
    for name in MODEL_NAMES:
        fit = fit_covariance_targeted_garch(regs[name], daily_returns, h_bar)
        fit["model"] = name
        results[name] = fit
    return results
