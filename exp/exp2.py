"""Experiment 2: high-dimensional DCC models with realized semicorrelations.

Reproduces, for the N-stock DJIA system:
  - univariate GARCH model selection per stock (Appendix D),
  - realized semicorrelation matrices R^P, R^N, R^M (Eqs. 16-19),
  - the 5 DCC-type specifications (Eqs. 23-27),
  - the average pairwise correlation and its cross-sectional dispersion
    (Section 5.4), including crisis-period comparisons,
  - out-of-sample Frobenius/QLIKE loss and an 80% Model Confidence Set.

Run as ``python -m exp.exp2`` (reduced configuration) or
``python -m exp.exp2 --full`` (paper configuration, Section 3).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.data_loader import load_ohlcv
from src import config
from src.dcc import (
    DCC_MODEL_NAMES, average_correlation, build_dcc_regressors, fit_dcc,
    realized_correlation, standardized_returns, std_correlation, to_correlation,
)
from src.forecast_eval import frobenius_loss, mcs_included, model_confidence_set, qlike_loss
from src.matrix_garch import run_recursion
from src.semicorrelation import realized_semicorrelations
from src.semicovariance import daily_realized_semicovariances, daily_simple_returns, intraday_log_returns
from src.univariate_garch import fit_univariate_models, select_best_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_system_data(tickers: list[str], start: str, end: str) -> dict:
    """Daily and 15-minute bars for the whole stock system, aligned on the
    common set of trading days with both daily and intraday coverage."""
    daily_bars, daily_meta = load_ohlcv(tickers, start, end, timeframe=config.DAILY_FREQ)
    intra_bars, intra_meta = load_ohlcv(tickers, start, end, timeframe=config.INTRADAY_FREQ)
    intraday_returns = intraday_log_returns(intra_bars, tickers)
    semis = daily_realized_semicovariances(intraday_returns)
    daily_returns = daily_simple_returns(daily_bars, tickers)
    common = semis["index"].intersection(daily_returns.index)
    daily_returns = daily_returns.loc[common]
    keep = [i for i, d in enumerate(semis["index"]) if d in common]
    for key in ("RCOV", "P", "N", "Mp", "Mn"):
        semis[key] = semis[key][keep]
    semis["index"] = semis["index"][keep]
    return {
        "tickers": tickers, "daily_returns": daily_returns.to_numpy(), "dates": daily_returns.index,
        "RCOV": semis["RCOV"], "P": semis["P"], "N": semis["N"], "Mp": semis["Mp"], "Mn": semis["Mn"],
        "daily_meta": daily_meta, "intra_meta": intra_meta,
    }


def fit_univariate_step(data: dict, log: list[str]) -> tuple[np.ndarray, dict]:
    """First-step univariate GARCH models and best-fit selection per stock
    (Section 5, Appendix D)."""
    T, N = data["daily_returns"].shape
    h = np.empty((T, N))
    selection = {}
    for i, ticker in enumerate(data["tickers"]):
        r = data["daily_returns"][:, i]
        rv = data["RCOV"][:, i, i]
        pv = data["P"][:, i, i]
        nv = data["N"][:, i, i]
        fits = fit_univariate_models(r, rv, pv, nv)
        best = select_best_model(fits)
        selection[ticker] = {"best_model": best,
                              "loglik": {m: f["loglik"] for m, f in fits.items()}}
        h[:, i] = fits[best]["h"]
        log.append(f"{ticker}: best univariate model {best}")
    return h, selection


def oos_forecast_system(data: dict, h_uni: np.ndarray, window: int, step: int) -> dict[str, dict]:
    """Rolling re-estimation of the univariate step and the DCC step every
    ``step`` days over a ``window``-day window (Section 5.5)."""
    T, N = data["daily_returns"].shape
    r, RCOV, P, N_, Mp, Mn = data["daily_returns"], data["RCOV"], data["P"], data["N"], data["Mp"], data["Mn"]
    forecasts: dict[str, list] = {name: [] for name in DCC_MODEL_NAMES}
    realized_oos = []
    t = window
    while t < T:
        est_slice = slice(t - window, t)
        z_est = standardized_returns(r[est_slice], h_uni[est_slice])
        q_bar = np.cov(z_est.T, bias=True)
        regs_est = build_dcc_regressors(z_est, RCOV[est_slice], P[est_slice], N_[est_slice],
                                         Mp[est_slice], Mn[est_slice])
        horizon = min(step, T - t)
        z_fore = standardized_returns(r[t - 1:t - 1 + horizon + 1], h_uni[t - 1:t - 1 + horizon + 1])
        regs_fore = build_dcc_regressors(z_fore, RCOV[t - 1:t - 1 + horizon + 1], P[t - 1:t - 1 + horizon + 1],
                                          N_[t - 1:t - 1 + horizon + 1], Mp[t - 1:t - 1 + horizon + 1],
                                          Mn[t - 1:t - 1 + horizon + 1])
        for name in DCC_MODEL_NAMES:
            fit = fit_dcc(regs_est[name], z_est, q_bar)
            Q_path = run_recursion(fit["beta"], fit["coefs"], regs_fore[name], q_bar)
            R_path = to_correlation(Q_path)
            Dmat = np.sqrt(h_uni[t - 1:t - 1 + horizon + 1])
            H_path = R_path * Dmat[:, :, None] * Dmat[:, None, :]
            forecasts[name].extend(list(H_path[1:1 + horizon]))
        realized_oos.extend(list(RCOV[t:t + horizon]))
        t += step
    return {name: {"H": np.stack(forecasts[name]), "RCOV": np.stack(realized_oos)} for name in DCC_MODEL_NAMES}


def run_experiment(tickers: list[str], start: str, end: str, window: int, step: int, log: list[str]) -> dict:
    data = load_system_data(tickers, start, end)
    log.append(f"system: {data['daily_returns'].shape[0]} trading days, {len(tickers)} stocks, "
                f"{data['dates'].min().date()} to {data['dates'].max().date()}")

    h_uni, uni_selection = fit_univariate_step(data, log)
    z = standardized_returns(data["daily_returns"], h_uni)

    semicorr = realized_semicorrelations(data["RCOV"], data["P"], data["N"], data["Mp"], data["Mn"])
    from src.semicorrelation import check_semicorrelation_bounds
    bounds = check_semicorrelation_bounds(semicorr["R_P"], semicorr["R_N"], semicorr["R_M"])

    dcc_fits = {}
    q_bar = np.cov(z.T, bias=True)
    regs = build_dcc_regressors(z, data["RCOV"], data["P"], data["N"], data["Mp"], data["Mn"])
    correlation_paths = {}
    for name in DCC_MODEL_NAMES:
        fit = fit_dcc(regs[name], z, q_bar)
        dcc_fits[name] = {"loglik": fit["loglik"], "beta": fit["beta"], "coefs": fit["coefs"],
                           "converged": fit["converged"]}
        correlation_paths[name] = fit["R"]
        log.append(f"DCC {name}: loglik={fit['loglik']:.2f}, beta={fit['beta']:.4f}, coefs={fit['coefs']}")

    dynamics = {}
    for name in DCC_MODEL_NAMES:
        R = correlation_paths[name]
        avg = average_correlation(R)
        std = std_correlation(R)
        years = pd.Series(data["dates"]).dt.year.to_numpy()
        yearly_avg = pd.Series(avg, index=years).groupby(level=0).mean().to_dict()
        dynamics[name] = {"yearly_avg_correlation": {int(y): float(v) for y, v in yearly_avg.items()},
                           "mean_avg_correlation": float(avg.mean()), "mean_std_correlation": float(std.mean())}

    crisis = {}
    pairs_available = [p for p in config.CORR_DYNAMICS_PAIRS if all(t in tickers for t in p)]
    for (a, b) in pairs_available:
        i, j = tickers.index(a), tickers.index(b)
        for name in DCC_MODEL_NAMES:
            key = f"{a}-{b}_{name}"
            crisis[key] = correlation_paths[name][:, i, j].tolist()

    oos = oos_forecast_system(data, h_uni, window, step)
    mcs_results = {}
    for loss_name, loss_fn in (("Frobenius", frobenius_loss), ("QLIKE", qlike_loss)):
        stacked = {name: loss_fn(oos[name]["H"], oos[name]["RCOV"]) for name in DCC_MODEL_NAMES}
        p_values = model_confidence_set(stacked, alpha=1 - config.MCS_LEVEL,
                                         n_boot=config.MCS_BOOT_REPLICATIONS_RUN,
                                         block_length=config.MCS_BLOCK_LENGTH_RUN, seed=config.RANDOM_SEED)
        mcs_results[loss_name] = {
            "mean_loss": {m: float(np.mean(stacked[m])) for m in DCC_MODEL_NAMES},
            "p_values": p_values,
            "included": mcs_included(p_values, config.MCS_LEVEL),
        }

    return {
        "universe": tickers, "univariate_selection": uni_selection, "semicorrelation_bounds": bounds,
        "dcc_fits": dcc_fits, "correlation_dynamics": dynamics, "crisis_correlations": crisis,
        "oos_mcs": mcs_results, "dates": [str(d.date()) for d in data["dates"]],
        "correlation_paths": {name: correlation_paths[name].tolist() for name in DCC_MODEL_NAMES},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run the paper's full configuration")
    args = parser.parse_args()

    if args.full:
        tickers, start, end = config.DJIA_24_PAPER, config.SAMPLE_START_PAPER, config.SAMPLE_END_PAPER
        window, step = config.OOS_WINDOW_PAPER, config.OOS_STEP_PAPER
    else:
        tickers, start, end = config.DJIA_RUN, config.SAMPLE_START_RUN, config.SAMPLE_END_RUN
        window, step = config.OOS_WINDOW_RUN, config.OOS_STEP_RUN

    log: list[str] = [f"exp2: tickers={tickers}, {start}..{end}, window={window}, step={step}, full={args.full}"]
    t0 = time.time()
    results = run_experiment(tickers, start, end, window, step, log)
    log.append(f"total runtime: {time.time() - t0:.1f}s")

    RESULTS_DIR.mkdir(exist_ok=True)
    corr_paths = results.pop("correlation_paths")
    with open(RESULTS_DIR / "exp2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    np.savez_compressed(RESULTS_DIR / "exp2_correlation_paths.npz",
                        **{name: np.array(v) for name, v in corr_paths.items()})
    with open(RESULTS_DIR / "exp2_log.txt", "w") as f:
        f.write("\n".join(log))

    make_plots(results, corr_paths, tickers)
    print("\n".join(log))


def make_plots(results: dict, corr_paths: dict, tickers: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in DCC_MODEL_NAMES:
        R = np.array(corr_paths[name])
        avg = average_correlation(R)
        ax.plot(avg, label=name, alpha=0.8)
    ax.set_xlabel("trading day index")
    ax.set_ylabel("average pairwise correlation")
    ax.set_title("Filtered average correlation paths (Section 5.4)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_avg_correlation.png", dpi=120)
    plt.close(fig)

    if results["crisis_correlations"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for key, path in results["crisis_correlations"].items():
            ax.plot(path, label=key, alpha=0.7)
        ax.set_xlabel("trading day index")
        ax.set_ylabel("pairwise correlation")
        ax.set_title("Crisis-period pairwise correlations (Section 5.4)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "exp2_crisis_correlations.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
