"""Non-linear impulse response functions via Monte Carlo simulation
(Appendix B.2).

For a given bivariate model, an initial shock pair, and average parameter
estimates, the IRF at horizon h is

    IRF(h) = E[H_{t+h} | F_t, z_t] - E[H_{t+h} | F_t, z_t = 0]

estimated by simulating many future paths of 15-minute returns (drawn from a
constant-volatility Gaussian model within each day, consistent with the
"filtered H_t paths" approach of Appendix B.2), aggregating them into daily
realized semicovariances, and filtering H_{t+h} forward with the estimated
recursion.
"""
from __future__ import annotations

import numpy as np

from src.bivariate_garch import build_regressors
from src.matrix_garch import run_recursion


def simulate_intraday_day(h_day: np.ndarray, n_intraday: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n_intraday`` 15-minute return vectors for one day from
    N(0, h_day / n_intraday) (constant intraday volatility within the day,
    consistent with realized-measure consistency E[sum r r'] = h_day)."""
    cov = h_day / n_intraday
    return rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=n_intraday)


def semicovariances_from_intraday(r: np.ndarray) -> dict[str, np.ndarray]:
    """RCOV, P, N, M+, M- from one simulated day of intraday returns."""
    r_pos = np.clip(r, 0, None)
    r_neg = np.clip(r, None, 0)
    P = r_pos.T @ r_pos
    N = r_neg.T @ r_neg
    Mp = r_pos.T @ r_neg
    Mn = Mp.T
    return {"RCOV": P + N + Mp + Mn, "P": P, "N": N, "Mp": Mp, "Mn": Mn}


def simulate_one_path(model: str, coefs: dict[str, float], beta: float, h_bar: np.ndarray,
                       shock_r: np.ndarray, horizon: int, n_intraday: int,
                       rng: np.random.Generator) -> np.ndarray:
    """One Monte-Carlo path of H_{t+1}..H_{t+horizon} starting from an
    initial shock ``shock_r`` applied at day t (Appendix B.2: shocks of size
    3*sqrt(Hbar))."""
    H = np.empty((horizon + 1, 2, 2))
    H[0] = h_bar
    r_hist = [shock_r]
    for h in range(1, horizon + 1):
        r_prev = r_hist[-1]
        day_r = simulate_intraday_day(H[h - 1], n_intraday, rng) if h > 1 else None
        if day_r is None:
            # Day t (the shock day) uses the given shock as the realized daily
            # return; its semicovariances use the shock itself as the
            # one-interval realization (consistent with the NIS convention).
            semis = semicovariances_from_intraday(r_prev.reshape(1, 2))
        else:
            semis = semicovariances_from_intraday(day_r)
            r_prev = day_r.sum(axis=0)
        regs_t = build_regressors(
            np.zeros((1, 2)), semis["RCOV"][None], semis["P"][None],
            semis["N"][None], semis["Mp"][None], semis["Mn"][None],
        )
        reg_map = regs_t[model]
        x_bar = {k: v for k, v in coefs.items()}
        const = h_bar * (1.0 - beta) - sum(coefs[k] * reg_map_mean(model, k) for k in coefs)
        X = sum(coefs[k] * reg_map[k][0] for k in coefs)
        H[h] = const + beta * H[h - 1] + X
        r_hist.append(r_prev)
    return H


def reg_map_mean(model: str, key: str) -> np.ndarray:
    """Time-series mean of a given regressor, registered via
    ``src.nis.set_unconditional_means`` prior to IRF simulation."""
    from src.nis import MEANS
    return MEANS[model][key]


def impulse_response(model: str, coefs: dict[str, float], beta: float, h_bar: np.ndarray,
                      shock_signs: tuple[int, int], horizon: int, n_paths: int,
                      n_intraday: int, shock_size_sd: float, seed: int) -> np.ndarray:
    """Average, over ``n_paths`` Monte-Carlo paths, of H_{t+h} given an
    initial shock of size ``shock_size_sd * sqrt(diag(Hbar))`` with the given
    signs, minus the benchmark path with a zero initial shock (Appendix B.2).
    """
    rng = np.random.default_rng(seed)
    sd = np.sqrt(np.diagonal(h_bar))
    shock = shock_size_sd * sd * np.array(shock_signs)

    acc_shock = np.zeros((horizon + 1, 2, 2))
    acc_zero = np.zeros((horizon + 1, 2, 2))
    for _ in range(n_paths):
        acc_shock += simulate_one_path(model, coefs, beta, h_bar, shock, horizon, n_intraday, rng)
        acc_zero += simulate_one_path(model, coefs, beta, h_bar, np.zeros(2), horizon, n_intraday, rng)
    return (acc_shock - acc_zero) / n_paths
