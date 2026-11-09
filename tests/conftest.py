"""Shared pytest fixtures: market-data marker registration and a reduced,
session-cached slice of real data loaded through ``data.data_loader``."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: tests that load a real (reduced) market-data slice"
    )


@pytest.fixture(scope="session")
def market_slice(request: pytest.FixtureRequest):
    """Two tickers, one year, loaded (and cached) through the shared Dataloader
    path. Skipped when DATALOADER_API_KEY is absent (no live-call default)."""
    if not os.environ.get("DATALOADER_API_KEY") or not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("DATALOADER_API_KEY/MASSIVE_API_KEY not set: market_data tests are skipped")

    cache_dir = request.config.cache.mkdir("market_slice")
    cache_file = cache_dir / "aapl_spy_2013.json"
    from data.data_loader import load_ohlcv
    from src import config as cfg

    tickers = ["AAPL", "SPY"]
    start, end = "2013-01-01", "2013-12-31"
    daily_bars, daily_meta = load_ohlcv(tickers, start, end, timeframe=cfg.DAILY_FREQ)
    intra_bars, intra_meta = load_ohlcv(tickers, start, end, timeframe=cfg.INTRADAY_FREQ)
    return {
        "tickers": tickers, "daily_bars": daily_bars, "intra_bars": intra_bars,
        "daily_meta": daily_meta, "intra_meta": intra_meta,
    }


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def synthetic_bivariate_data(rng: np.random.Generator):
    """Synthetic daily returns and realized (semi)covariance matrices for a
    2-asset system, with a known positive-definite target H-bar, used to test
    the covariance-targeted recursion and QMLE machinery without live data."""
    T = 400
    h_bar = np.array([[1.0, 0.4], [0.4, 0.8]])
    L = np.linalg.cholesky(h_bar)
    z = rng.standard_normal((T, 2))
    daily_returns = z @ L.T * 0.01

    n_intraday = 26
    rcov = np.empty((T, 2, 2))
    p = np.empty((T, 2, 2))
    n_mat = np.empty((T, 2, 2))
    mp = np.empty((T, 2, 2))
    mn = np.empty((T, 2, 2))
    for t in range(T):
        intraday = rng.multivariate_normal(np.zeros(2), h_bar / n_intraday, size=n_intraday) * 0.01
        r_pos = np.clip(intraday, 0, None)
        r_neg = np.clip(intraday, None, 0)
        p[t] = r_pos.T @ r_pos
        n_mat[t] = r_neg.T @ r_neg
        mp[t] = r_pos.T @ r_neg
        mn[t] = mp[t].T
        rcov[t] = p[t] + n_mat[t] + mp[t] + mn[t]
    return {
        "daily_returns": daily_returns, "RCOV": rcov, "P": p, "N": n_mat, "Mp": mp, "Mn": mn, "h_bar": h_bar,
    }
