"""Non-linear impulse response functions via Monte Carlo simulation
(Appendix B.2).

For a given bivariate model, an initial shock pair, and average parameter
estimates, the IRF at horizon h is

    IRF(h) = E[H_{t+h} | F_t, z_t] - E[H_{t+h} | F_t, z_t = 0]

estimated by simulating many future paths of 15-minute returns (drawn from a
constant-volatility Gaussian model within each day, consistent with the
"filtered H_t paths" approach of Appendix B.2), aggregating them into daily
realized semicovariances, and filtering H_{t+h} forward with the estimated
recursion. All ``n_paths`` Monte-Carlo paths are simulated simultaneously with
batched numpy array operations (one (n_paths, 2, 2) state array per horizon
step) rather than a per-path Python loop, which is what makes the paper's
Monte-Carlo path count tractable.
"""
from __future__ import annotations

import numpy as np

from src.semicovariance import t_operator


def _outer_batched(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched outer product a_p b_p' for a, b of shape (n_paths, 2)."""
    return np.einsum("pi,pj->pij", a, b)


def single_step_regressors(model: str, r: np.ndarray, rcov: np.ndarray, p: np.ndarray,
                            n_mat: np.ndarray, mp: np.ndarray, mn: np.ndarray) -> dict[str, np.ndarray]:
    """Batched one-step-ahead regressors for ``model`` (shape (n_paths, 2, 2)
    each), matching ``src.bivariate_garch.build_regressors`` exactly but
    evaluated at a single day's realization per Monte-Carlo path, and using
    the path's own simulated (or shocked) daily return ``r`` for the
    sign-indicator regressors of the threshold models -- unlike a constant
    zero placeholder, which would always classify the threshold indicator
    into the "non-positive" branch regardless of the simulated shock.
    """
    r_pos, r_neg = np.clip(r, 0, None), np.clip(r, None, 0)
    i_pos, i_neg = (r > 0).astype(float), (r <= 0).astype(float)
    if model == "BG":
        return {"a": _outer_batched(r, r)}
    if model == "tBG":
        return {"aP": _outer_batched(r_pos, r_pos), "aN": _outer_batched(r_neg, r_neg),
                "aM": _outer_batched(r_pos, r_neg) + _outer_batched(r_neg, r_pos)}
    if model == "rBG":
        return {"a": rcov}
    if model == "trBG":
        return {"aP": rcov * _outer_batched(i_pos, i_pos), "aN": rcov * _outer_batched(i_neg, i_neg),
                "aM": rcov * (_outer_batched(i_pos, i_neg) + _outer_batched(i_neg, i_pos))}
    if model == "crBG":
        return {"aP": p, "aN": n_mat, "aM": mp + mn}
    if model == "tBG-S":
        return {"aP": _outer_batched(r_pos, r_pos), "aN": _outer_batched(r_neg, r_neg),
                "aMp": t_operator(_outer_batched(r_pos, r_neg)), "aMn": t_operator(_outer_batched(r_neg, r_pos))}
    if model == "trBG-S":
        return {"aP": rcov * _outer_batched(i_pos, i_pos), "aN": rcov * _outer_batched(i_neg, i_neg),
                "aMp": t_operator(rcov * _outer_batched(i_pos, i_neg)),
                "aMn": t_operator(rcov * _outer_batched(i_neg, i_pos))}
    if model == "crBG-S":
        return {"aP": p, "aN": n_mat, "aMp": t_operator(mp), "aMn": t_operator(mn)}
    raise ValueError(f"unknown model {model!r}")


def _semicovariances_batched(r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RCOV, P, N, M+, M- from intraday returns ``r`` of shape
    (n_paths, n_intraday, 2), or from a single realization of shape
    (n_paths, 2) treated as one interval."""
    if r.ndim == 2:
        r = r[:, None, :]
    r_pos, r_neg = np.clip(r, 0, None), np.clip(r, None, 0)
    P = np.einsum("pki,pkj->pij", r_pos, r_pos)
    N = np.einsum("pki,pkj->pij", r_neg, r_neg)
    Mp = np.einsum("pki,pkj->pij", r_pos, r_neg)
    Mn = np.einsum("pki,pkj->pij", r_neg, r_pos)
    return P + N + Mp + Mn, P, N, Mp, Mn


def _project_psd(H: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Symmetrize a batch of 2x2 matrices and floor their eigenvalues at
    ``floor`` times the matrix trace. The recursion [Eq. 5-8] does not
    guarantee an exactly symmetric, positive-definite H_t along every
    simulated Monte-Carlo path (round-off breaks symmetry, and a strongly
    negative discordant coefficient such as alpha^{M-} in crBG-S can push a
    path's covariance matrix briefly off the positive-definite cone); this
    numerical safeguard is applied before every Cholesky factorization so the
    simulation can continue instead of raising a linear-algebra error."""
    H_sym = 0.5 * (H + np.swapaxes(H, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(H_sym)
    trace = np.clip(eigvals.sum(axis=-1, keepdims=True), 1e-12, None)
    eigvals = np.clip(eigvals, floor * trace, None)
    return np.einsum("pij,pj,pkj->pik", eigvecs, eigvals, eigvecs)


def _simulate_batch(model: str, coefs: dict[str, float], beta: float, h_bar: np.ndarray,
                     means: dict[str, np.ndarray], initial_r: np.ndarray, horizon: int,
                     n_paths: int, n_intraday: int, rng: np.random.Generator) -> np.ndarray:
    """All ``n_paths`` Monte-Carlo paths of H_{t+1}..H_{t+horizon}, starting
    from the common initial daily return ``initial_r`` applied at day t."""
    const = h_bar * (1.0 - beta) - sum(coefs[k] * means[k] for k in coefs)
    H_path = np.empty((horizon + 1, n_paths, 2, 2))
    H_path[0] = np.broadcast_to(h_bar, (n_paths, 2, 2))
    r_day = np.broadcast_to(initial_r, (n_paths, 2))
    rcov, p, n_mat, mp, mn = _semicovariances_batched(r_day)
    for h in range(1, horizon + 1):
        regs = single_step_regressors(model, r_day, rcov, p, n_mat, mp, mn)
        X = sum(coefs[k] * regs[k] for k in coefs)
        H_path[h] = _project_psd(const + beta * H_path[h - 1] + X)
        if h < horizon:
            L = np.linalg.cholesky(H_path[h])
            z = rng.standard_normal((n_paths, n_intraday, 2))
            intraday = np.einsum("pij,pkj->pki", L / np.sqrt(n_intraday), z)
            r_day = intraday.sum(axis=1)
            rcov, p, n_mat, mp, mn = _semicovariances_batched(intraday)
    return H_path


def impulse_response(model: str, coefs: dict[str, float], beta: float, h_bar: np.ndarray,
                      shock_signs: tuple[int, int], horizon: int, n_paths: int,
                      n_intraday: int, shock_size_sd: float, seed: int) -> np.ndarray:
    """Average, over ``n_paths`` Monte-Carlo paths, of H_{t+h} given an
    initial shock of size ``shock_size_sd * sqrt(diag(Hbar))`` with the given
    signs, minus the benchmark path with a zero initial shock (Appendix B.2).
    """
    from src.nis import MEANS

    rng = np.random.default_rng(seed)
    sd = np.sqrt(np.diagonal(h_bar))
    shock = shock_size_sd * sd * np.array(shock_signs)
    means = MEANS[model]

    H_shock = _simulate_batch(model, coefs, beta, h_bar, means, shock, horizon, n_paths, n_intraday, rng)
    H_zero = _simulate_batch(model, coefs, beta, h_bar, means, np.zeros(2), horizon, n_paths, n_intraday, rng)
    return H_shock.mean(axis=1) - H_zero.mean(axis=1)
