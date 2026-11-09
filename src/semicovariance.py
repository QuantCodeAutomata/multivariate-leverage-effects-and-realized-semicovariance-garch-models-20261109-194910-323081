"""Realized (semi)covariance construction from high-frequency returns.

Implements Section 2.1 of Bollerslev, Patton and Quaedvlieg (2020): the
decomposition of the daily realized covariance matrix into concordant
positive, concordant negative, and discordant semicovariance components, and
the companion daily-return sign decomposition used by the threshold models
(Eqs. (9)-(11)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def intraday_log_returns(bars: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """15-minute log returns for ``tickers``, indexed by (date, timestamp).

    ``bars`` holds tidy rows (timestamp, date, ticker, open, high, low,
    close, volume) as returned by ``data.data_loader.load_ohlcv``. Returns
    are computed within each trading day from consecutive close prices
    (Section 3: "returns are computed... at the 15-min frequency"); the
    first intraday observation of a day (the open-to-first-bar return) is
    dropped, consistent with using only within-day price changes to avoid
    contaminating the realized measures with the overnight gap used
    separately for daily close-to-close returns (footnote 6).
    """
    wide = bars.pivot_table(index=["date", "timestamp"], columns="ticker", values="close")
    wide = wide.reindex(columns=tickers).sort_index(level=["date", "timestamp"])
    log_px = np.log(wide)
    rets = log_px.groupby(level="date").diff()
    rets = rets.dropna(how="any")
    return rets


def daily_realized_semicovariances(intraday_returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily RCOV, P, N, M+, M- matrices (Section 2.1, Eqs. preceding Eq. (5)).

    For 15-minute return vectors r_{t,k} (k = 1..m within day t), with
    r^+ = max(r, 0) and r^- = min(r, 0) applied element-wise:
        RCOV_t = sum_k r_{t,k} r_{t,k}'
        P_t = sum_k r_{t,k}^+ r_{t,k}^{+'}      (concordant positive)
        N_t = sum_k r_{t,k}^- r_{t,k}^{-'}      (concordant negative)
        M_t^+ = sum_k r_{t,k}^+ r_{t,k}^{-'}     (discordant, asset "+"/other "-")
        M_t^- = sum_k r_{t,k}^- r_{t,k}^{+'} = (M_t^+)'
    and RCOV_t = P_t + N_t + M_t^+ + M_t^-.

    Returns a dict of {date -> matrix} for each of the five components,
    with tickers in the column order of ``intraday_returns``.
    """
    tickers = list(intraday_returns.columns)
    n = len(tickers)
    out = {key: {} for key in ("RCOV", "P", "N", "Mp", "Mn")}
    for date, group in intraday_returns.groupby(level="date"):
        r = group.to_numpy()
        if np.isnan(r).any() or len(r) == 0:
            continue
        r_pos = np.clip(r, 0, None)
        r_neg = np.clip(r, None, 0)
        P = r_pos.T @ r_pos
        N = r_neg.T @ r_neg
        Mp = r_pos.T @ r_neg
        Mn = Mp.T
        RCOV = P + N + Mp + Mn
        out["RCOV"][date] = RCOV
        out["P"][date] = P
        out["N"][date] = N
        out["Mp"][date] = Mp
        out["Mn"][date] = Mn
    index = pd.DatetimeIndex(sorted(out["RCOV"].keys()))
    frames = {}
    for key, mats in out.items():
        stacked = np.stack([mats[d] for d in index])
        frames[key] = stacked
    frames["index"] = index
    frames["tickers"] = tickers
    return frames


def daily_simple_returns(daily_bars: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Close-to-close daily simple returns, used for covariance targeting of
    ``\\bar H`` and for the daily-threshold models (Eqs. (5), (9), (10),
    footnote 6: "close-to-close" daily returns differ from the open-to-close
    realized measures built from intraday data).
    """
    wide = daily_bars.pivot_table(index="date", columns="ticker", values="close")
    wide = wide.reindex(columns=tickers).sort_index()
    rets = wide.pct_change().dropna(how="any")
    return rets


def signed_outer_products(daily_returns: pd.DataFrame) -> dict[str, np.ndarray]:
    """Daily analogues of the signed outer products used in Eqs. (9)-(11):
    r_t^+ r_t^{+'}, r_t^- r_t^{-'}, r_t^+ r_t^{-'}, r_t^- r_t^{+'}, and the
    sign-indicator outer products I_t^+ I_t^{+'} etc. used to mask RCOV_t in
    Eq. (11). ``I_t^+ = 1{r_t > 0}``, ``I_t^- = 1{r_t <= 0}`` (footnote 8).
    """
    r = daily_returns.to_numpy()
    T, n = r.shape
    r_pos = np.clip(r, 0, None)
    r_neg = np.clip(r, None, 0)
    i_pos = (r > 0).astype(float)
    i_neg = (r <= 0).astype(float)

    def outer_series(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.einsum("ti,tj->tij", a, b)

    return {
        "rp_rp": outer_series(r_pos, r_pos),
        "rn_rn": outer_series(r_neg, r_neg),
        "rp_rn": outer_series(r_pos, r_neg),
        "rn_rp": outer_series(r_neg, r_pos),
        "ip_ip": outer_series(i_pos, i_pos),
        "in_in": outer_series(i_neg, i_neg),
        "ip_in": outer_series(i_pos, i_neg),
        "in_ip": outer_series(i_neg, i_pos),
    }


def t_operator(a: np.ndarray) -> np.ndarray:
    """T(.) operator (paper's footnote 7): sets the lower triangular part of
    a square matrix equal to its upper triangular part, "which ensures that
    T(M_{t-1}^+) and T(M_{t-1}^-) each contain unique information, and that
    the resulting H_t matrix [is] symmetric". Accepts either a single (n, n)
    matrix or a stacked series of matrices (T, n, n).
    """
    upper = np.triu(a)
    lower_strict = np.swapaxes(np.triu(a, k=1), -2, -1)
    return upper + lower_strict
