"""News Impact Surfaces for the bivariate semicovariance GARCH models
(Appendix B.1).

Under the assumption that H_{t-1} = Hbar and every realized (semi)covariance
regressor is at its unconditional (time-series mean) level, the next-period
conditional covariance element H_{t,ij} is a function of the standardized
shock pair (z_tilde_m, z_tilde_i) alone (z_tilde = r / sqrt(Hbar) the
diagonal-scaled shock). The main-diagonal cross-section sets
z_tilde_m = z_tilde_i; the secondary-diagonal cross-section sets
z_tilde_m = -z_tilde_i.

Under the maintained assumption that the sum of intraday returns within the
day equals the realized daily return exactly and that all the intraday mass
is concentrated in the shock's own sign bucket (Appendix B.1), the realized
semicovariance regressors reduce to the same signed outer products as the
daily-threshold model for a single shock pair, so the NIS is evaluated with
the *daily* regressor map of ``src.bivariate_garch.build_regressors`` applied
to a one-observation path (T=1) built from the shock pair itself.
"""
from __future__ import annotations

import numpy as np

from src.semicovariance import t_operator


def _signed(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.clip(z, 0, None), np.clip(z, None, 0)


def news_impact_value(model: str, coefs: dict[str, float], beta: float, h_bar: np.ndarray,
                       z_m: float, z_i: float) -> float:
    """H_{t,12} (stock-market covariance element) implied by one shock pair
    (z_m, z_i), given average parameter estimates ``coefs``/``beta`` and the
    unconditional covariance ``h_bar`` (Table 3 "average parameter
    estimates"; Appendix B.1).
    """
    # Appendix B.1 defines the NIS cross-sections directly in standardized
    # shock space (z_tilde); the shock pair is used as the regressor input
    # exactly as the daily sign-based regressors of the paper's Eqs. (9)-(11).
    r = np.array([z_i, z_m])  # order: [stock, market], element (0,1) is H_{12}
    r_pos, r_neg = _signed(r)
    i_pos = (r > 0).astype(float)
    i_neg = (r <= 0).astype(float)
    outer = lambda a, b: np.outer(a, b)

    const = h_bar * (1.0 - beta) - sum(coefs[k] * MEANS[model][k] for k in coefs)
    if model == "BG":
        X = coefs["a"] * outer(r, r)
    elif model == "tBG":
        X = (coefs["aP"] * outer(r_pos, r_pos) + coefs["aN"] * outer(r_neg, r_neg)
             + coefs["aM"] * (outer(r_pos, r_neg) + outer(r_neg, r_pos)))
    elif model == "rBG":
        X = coefs["a"] * outer(r, r)
    elif model == "trBG":
        X = (coefs["aP"] * outer(r, r) * outer(i_pos, i_pos)
             + coefs["aN"] * outer(r, r) * outer(i_neg, i_neg)
             + coefs["aM"] * outer(r, r) * (outer(i_pos, i_neg) + outer(i_neg, i_pos)))
    elif model == "crBG":
        X = coefs["aP"] * outer(r_pos, r_pos) + coefs["aN"] * outer(r_neg, r_neg) \
            + coefs["aM"] * (outer(r_pos, r_neg) + outer(r_neg, r_pos))
    elif model in ("tBG-S", "trBG-S", "crBG-S"):
        base = outer(r, r) if "r" in model else None
        if model == "tBG-S":
            X = (coefs["aP"] * outer(r_pos, r_pos) + coefs["aN"] * outer(r_neg, r_neg)
                 + coefs["aMp"] * t_operator(outer(r_pos, r_neg))
                 + coefs["aMn"] * t_operator(outer(r_neg, r_pos)))
        elif model == "trBG-S":
            X = (coefs["aP"] * outer(r, r) * outer(i_pos, i_pos)
                 + coefs["aN"] * outer(r, r) * outer(i_neg, i_neg)
                 + coefs["aMp"] * t_operator(outer(r, r) * outer(i_pos, i_neg))
                 + coefs["aMn"] * t_operator(outer(r, r) * outer(i_neg, i_pos)))
        else:  # crBG-S
            X = (coefs["aP"] * outer(r_pos, r_pos) + coefs["aN"] * outer(r_neg, r_neg)
                 + coefs["aMp"] * t_operator(outer(r_pos, r_neg))
                 + coefs["aMn"] * t_operator(outer(r_neg, r_pos)))
    else:
        raise ValueError(f"unknown model {model}")
    H = const + beta * h_bar + X
    return float(H[0, 1])


MEANS: dict[str, dict[str, np.ndarray]] = {}  # populated by `set_unconditional_means`


def set_unconditional_means(model: str, means: dict[str, np.ndarray]) -> None:
    """Register the time-series means of each regressor (e.g. Pbar, Nbar,
    Mbar) used for the covariance-targeting constant in ``news_impact_value``."""
    MEANS[model] = means


def news_impact_surface(model: str, coefs: dict[str, float], beta: float, h_bar: np.ndarray,
                         z_grid: np.ndarray, diagonal: str = "main") -> np.ndarray:
    """Evaluate the NIS cross-section H_{t,12}(z) along the main diagonal
    (z_tilde_m = z_tilde_i) or secondary diagonal (z_tilde_m = -z_tilde_i),
    Appendix B.1.
    """
    sign = 1.0 if diagonal == "main" else -1.0
    out = np.empty_like(z_grid)
    for idx, z in enumerate(z_grid):
        out[idx] = news_impact_value(model, coefs, beta, h_bar, z_m=sign * z, z_i=z)
    return out
