"""Unit tests for the realized (semi)covariance construction (Section 2.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.semicovariance import (
    daily_realized_semicovariances, signed_outer_products, t_operator,
)


def _make_intraday(n_days: int, n_intraday: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=d)
        for k in range(n_intraday):
            ts = date + pd.Timedelta(minutes=15 * k)
            rows.append((date, ts))
    index = pd.MultiIndex.from_tuples(rows, names=["date", "timestamp"])
    data = rng.standard_normal((len(index), 2)) * 0.001
    return pd.DataFrame(data, index=index, columns=["AAPL", "SPY"])


def test_rcov_equals_sum_of_semicovariance_parts():
    intraday = _make_intraday(10, 26)
    semis = daily_realized_semicovariances(intraday)
    recomposed = semis["P"] + semis["N"] + semis["Mp"] + semis["Mn"]
    np.testing.assert_allclose(semis["RCOV"], recomposed, atol=1e-12)


def test_semicovariance_matrices_are_symmetric_and_psd_pieces():
    intraday = _make_intraday(5, 26)
    semis = daily_realized_semicovariances(intraday)
    for key in ("RCOV", "P", "N"):
        mats = semis[key]
        # P, N, RCOV are each sums of outer products: symmetric and PSD.
        np.testing.assert_allclose(mats, np.swapaxes(mats, -1, -2), atol=1e-12)
        eigvals = np.linalg.eigvalsh(mats)
        assert (eigvals >= -1e-10).all()


def test_mn_is_transpose_of_mp():
    intraday = _make_intraday(5, 26)
    semis = daily_realized_semicovariances(intraday)
    np.testing.assert_allclose(semis["Mn"], np.swapaxes(semis["Mp"], -1, -2), atol=1e-12)


def test_t_operator_symmetrizes_by_copying_upper_triangle():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = t_operator(a)
    assert out[0, 1] == 2.0
    assert out[1, 0] == 2.0
    assert out[0, 0] == 1.0 and out[1, 1] == 4.0


def test_t_operator_on_stacked_series():
    a = np.stack([np.array([[1.0, 5.0], [0.0, 2.0]]) for _ in range(3)])
    out = t_operator(a)
    assert out.shape == a.shape
    np.testing.assert_allclose(out[:, 1, 0], out[:, 0, 1])


def test_signed_outer_products_sign_conventions():
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.0], "B": [-0.01, 0.03, 0.02]})
    out = signed_outer_products(returns)
    # Day 0: A>0 (pos), B<0 (neg, since <=0 convention) -> rp_rp[0] has A*A entry, 0 for B.
    np.testing.assert_allclose(out["rp_rp"][0, 0, 0], 0.01 * 0.01)
    np.testing.assert_allclose(out["rn_rn"][0, 1, 1], (-0.01) * (-0.01))
    # concordant positive-positive is zero when the pair has opposite signs.
    assert out["rp_rp"][0, 1, 1] == 0.0
