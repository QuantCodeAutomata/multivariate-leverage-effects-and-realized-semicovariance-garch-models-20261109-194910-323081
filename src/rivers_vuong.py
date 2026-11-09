"""Rivers and Vuong (2002) non-nested model comparison test.

For two non-nested models A and B with per-observation log-likelihood
contributions ell_A,t and ell_B,t, the test statistic is the studentized
mean log-likelihood-ratio

    RV = sqrt(T) * mean(ell_A,t - ell_B,t) / std(ell_A,t - ell_B,t)

which is asymptotically standard normal under the null of equal fit
(Rivers and Vuong, 2002, "Model selection tests for nonlinear dynamic
models", Econometrics Journal). A positive and significant RV favors model A.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def rivers_vuong_test(loglik_a: np.ndarray, loglik_b: np.ndarray) -> dict[str, float]:
    """Two-sided Rivers-Vuong statistic and p-value for per-day
    log-likelihood contributions of two non-nested models."""
    d = np.asarray(loglik_a) - np.asarray(loglik_b)
    T = d.shape[0]
    mean_d = d.mean()
    std_d = d.std(ddof=1)
    if std_d <= 1e-14:
        stat = 0.0
    else:
        stat = np.sqrt(T) * mean_d / std_d
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(stat)))
    return {"statistic": float(stat), "p_value": float(p_value), "favors_a": bool(stat > 0)}


def pairwise_rivers_vuong(per_day_ll: dict[str, np.ndarray], reference: str,
                          challengers: list[str]) -> dict[str, dict[str, float]]:
    """Run the Rivers-Vuong test of ``reference`` against each model in
    ``challengers`` (used to test whether crBG/crBG-S are favored against the
    other six specifications, Section 4.1)."""
    out = {}
    for name in challengers:
        out[name] = rivers_vuong_test(per_day_ll[reference], per_day_ll[name])
    return out
