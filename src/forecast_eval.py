"""Out-of-sample loss functions and the Model Confidence Set (Section 4.3 /
Section 5.5; Hansen, Lunde and Nason, 2011).
"""
from __future__ import annotations

import itertools

import numpy as np


def frobenius_loss(H: np.ndarray, RCOV: np.ndarray) -> np.ndarray:
    """Frobenius distance sqrt(Tr((H_t - RCOV_t)(H_t - RCOV_t)')) for each t."""
    diff = H - RCOV
    return np.sqrt(np.einsum("tij,tij->t", diff, diff))


def qlike_loss(H: np.ndarray, RCOV: np.ndarray) -> np.ndarray:
    """QLIKE loss log|H_t| + Tr(H_t^{-1} RCOV_t) for each t."""
    T = H.shape[0]
    out = np.empty(T)
    for t in range(T):
        sign, logdet = np.linalg.slogdet(H[t])
        out[t] = logdet + np.trace(np.linalg.solve(H[t], RCOV[t]))
    return out


def _bootstrap_resample_mean(losses: np.ndarray, block_length: int, n_boot: int,
                              rng: np.random.Generator) -> np.ndarray:
    """Stationary bootstrap (Politis and Romano, 1994) of the column means of
    a (T, M) loss-differential matrix, returning an (n_boot, M) array of
    resampled means, used to approximate the MCS test statistics'
    distribution (Hansen, Lunde and Nason, 2011, Section 3)."""
    T, M = losses.shape
    p = 1.0 / block_length
    out = np.empty((n_boot, M))
    for b in range(n_boot):
        idx = np.empty(T, dtype=int)
        idx[0] = rng.integers(T)
        for t in range(1, T):
            if rng.random() < p:
                idx[t] = rng.integers(T)
            else:
                idx[t] = (idx[t - 1] + 1) % T
        out[b] = losses[idx].mean(axis=0)
    return out


def model_confidence_set(loss_matrix: dict[str, np.ndarray], alpha: float = 0.20,
                          n_boot: int = 300, block_length: int = 10,
                          seed: int = 0) -> dict[str, float]:
    """Hansen, Lunde and Nason's (2011) Model Confidence Set via the
    range statistic T_R and a stationary bootstrap, applied sequentially
    (elimination rule: drop the worst-performing model, 1 - alpha MCS level
    reported in the paper as "80% MCS" with alpha=0.20).

    Returns, for every model initially in ``loss_matrix``, its MCS p-value
    (1.0 for models that are never eliminated down to the final set).
    """
    rng = np.random.default_rng(seed)
    names = list(loss_matrix.keys())
    L = np.column_stack([loss_matrix[name] for name in names])
    remaining = list(range(len(names)))
    p_values = {name: 1.0 for name in names}

    while len(remaining) > 1:
        sub = L[:, remaining]
        # Loss differential of each model against the cross-sectional average
        # loss at each t (Hansen, Lunde and Nason, 2011, Sec. 2-3): d_i,t =
        # L_i,t - mean_j(L_j,t); its time average d_bar_i measures relative
        # performance (d_bar_i > 0 means model i is worse than the set average).
        d_ij = sub - sub.mean(axis=1, keepdims=True)
        d_bar = d_ij.mean(axis=0)
        boot_means = _bootstrap_resample_mean(d_ij, block_length, n_boot, rng)
        # Recenter the bootstrap draws at the observed d_bar to approximate
        # the null distribution "no model is worse than the average".
        boot_centered = boot_means - d_bar
        denom = boot_centered.std(axis=0) + 1e-12
        t_stat = d_bar / denom
        t_range_obs = t_stat.max() - t_stat.min()
        t_range_boot = (boot_centered / denom).max(axis=1) - (boot_centered / denom).min(axis=1)
        p_val = float((t_range_boot >= t_range_obs).mean())

        worst_local = int(np.argmax(t_stat))
        worst_global = remaining[worst_local]
        p_values[names[worst_global]] = min(p_values[names[worst_global]], p_val)
        if p_val > alpha:
            # Current set already satisfies the MCS confidence level: every
            # remaining model is assigned this p-value and the loop stops.
            for idx in remaining:
                p_values[names[idx]] = max(p_values[names[idx]], p_val)
            break
        remaining.remove(worst_global)

    if len(remaining) == 1:
        p_values[names[remaining[0]]] = 1.0
    return p_values


def mcs_included(p_values: dict[str, float], level: float = 0.80) -> list[str]:
    """Models included in the (1 - alpha) MCS, alpha = 1 - level (Section 4.3:
    "80% Model Confidence Set")."""
    alpha = 1.0 - level
    return [name for name, p in p_values.items() if p >= alpha]
