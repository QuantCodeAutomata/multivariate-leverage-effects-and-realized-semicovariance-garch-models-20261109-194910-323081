"""Experiment 1: bivariate realized semicovariance GARCH models.

Reproduces, for each DJIA stock paired with the S&P 500 proxy:
  - in-sample QMLE estimates of the 8 bivariate specifications (Table 3),
  - Rivers and Vuong (2002) non-nested comparisons against crBG/crBG-S,
  - News Impact Surfaces on the main and secondary diagonal (Appendix B.1),
  - 30-day non-linear impulse response functions (Appendix B.2),
  - out-of-sample Frobenius/QLIKE loss and an 80% Model Confidence Set
    (Section 4.3).

Run as ``python -m exp.exp1`` (reduced configuration) or
``python -m exp.exp1 --full`` (paper configuration, Section 3).
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
from src.bivariate_garch import MODEL_NAMES, build_regressors, fit_all_models
from src.forecast_eval import frobenius_loss, mcs_included, model_confidence_set, qlike_loss
from src.irf import impulse_response
from src.matrix_garch import run_recursion
from src.nis import news_impact_surface, set_unconditional_means
from src.rivers_vuong import pairwise_rivers_vuong
from src.semicovariance import daily_realized_semicovariances, daily_simple_returns, intraday_log_returns

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_pair_data(ticker: str, market: str, start: str, end: str) -> dict:
    """Daily and 15-minute bars for one stock plus the market proxy."""
    daily_bars, daily_meta = load_ohlcv([ticker, market], start, end, timeframe=config.DAILY_FREQ)
    intra_bars, intra_meta = load_ohlcv([ticker, market], start, end, timeframe=config.INTRADAY_FREQ)
    intraday_returns = intraday_log_returns(intra_bars, [ticker, market])
    semis = daily_realized_semicovariances(intraday_returns)
    daily_returns = daily_simple_returns(daily_bars, [ticker, market])
    common = semis["index"].intersection(daily_returns.index)
    daily_returns = daily_returns.loc[common]
    keep = [i for i, d in enumerate(semis["index"]) if d in common]
    for key in ("RCOV", "P", "N", "Mp", "Mn"):
        semis[key] = semis[key][keep]
    semis["index"] = semis["index"][keep]
    return {
        "ticker": ticker, "market": market,
        "daily_returns": daily_returns.to_numpy(),
        "dates": daily_returns.index,
        "RCOV": semis["RCOV"], "P": semis["P"], "N": semis["N"], "Mp": semis["Mp"], "Mn": semis["Mn"],
        "daily_meta": daily_meta, "intra_meta": intra_meta,
    }


def oos_forecast_one_stock(data: dict, window: int, step: int) -> dict[str, dict]:
    """Rolling re-estimation every ``step`` days with a ``window``-day
    estimation window, producing out-of-sample H_t forecasts for every
    bivariate model (Section 4.3)."""
    T = data["daily_returns"].shape[0]
    r, RCOV, P, N, Mp, Mn = data["daily_returns"], data["RCOV"], data["P"], data["N"], data["Mp"], data["Mn"]
    forecasts: dict[str, list] = {name: [] for name in MODEL_NAMES}
    realized_oos = []
    t = window
    while t < T:
        est_slice = slice(t - window, t)
        h_bar = np.cov(r[est_slice].T, bias=True)
        regs = build_regressors(r[est_slice], RCOV[est_slice], P[est_slice], N[est_slice],
                                 Mp[est_slice], Mn[est_slice])
        fits = {}
        for name in MODEL_NAMES:
            from src.matrix_garch import fit_covariance_targeted_garch
            fits[name] = fit_covariance_targeted_garch(regs[name], r[est_slice], h_bar)
        horizon = min(step, T - t)
        full_regs_all = build_regressors(r[t - 1:t - 1 + horizon + 1], RCOV[t - 1:t - 1 + horizon + 1],
                                          P[t - 1:t - 1 + horizon + 1], N[t - 1:t - 1 + horizon + 1],
                                          Mp[t - 1:t - 1 + horizon + 1], Mn[t - 1:t - 1 + horizon + 1])
        for name in MODEL_NAMES:
            fit = fits[name]
            # Seed the forecast path at the model's own last in-sample filtered
            # H_{t-1}, not at h_bar, so the estimated persistence (beta) carries
            # over into the out-of-sample window instead of being reset to the
            # unconditional mean at every re-estimation (see matrix_garch.run_recursion).
            H_path = run_recursion(fit["beta"], fit["coefs"], full_regs_all[name], h_bar,
                                    h_init=fit["H"][-1])
            forecasts[name].extend(list(H_path[1:1 + horizon]))
        realized_oos.extend(list(RCOV[t:t + horizon]))
        t += step
    out = {name: {"H": np.stack(forecasts[name]), "RCOV": np.stack(realized_oos)} for name in MODEL_NAMES}
    return out


def run_experiment(tickers: list[str], market: str, start: str, end: str, window: int, step: int,
                    irf_paths: int, log: list[str]) -> dict:
    all_fits, all_rv, all_oos_losses = {}, {}, {"Frobenius": {m: [] for m in MODEL_NAMES},
                                                 "QLIKE": {m: [] for m in MODEL_NAMES}}
    nis_records, irf_records = [], []

    for ticker in tickers:
        t0 = time.time()
        data = load_pair_data(ticker, market, start, end)
        log.append(f"{ticker}: {data['daily_returns'].shape[0]} trading days, "
                    f"{data['dates'].min().date()} to {data['dates'].max().date()}")
        fits = fit_all_models(data["daily_returns"], data["RCOV"], data["P"], data["N"], data["Mp"], data["Mn"])
        all_fits[ticker] = {name: {"loglik": f["loglik"], "beta": f["beta"], "coefs": f["coefs"],
                                    "converged": f["converged"]} for name, f in fits.items()}

        per_day_ll = {}
        for name, f in fits.items():
            from src.matrix_garch import gaussian_qmle_loglik
            T = f["H"].shape[0]
            per_day_ll[name] = np.array([
                -np.linalg.slogdet(f["H"][t])[1] - data["daily_returns"][t] @ np.linalg.solve(f["H"][t], data["daily_returns"][t])
                for t in range(T)
            ])
        rv = pairwise_rivers_vuong(per_day_ll, reference="crBG-S",
                                    challengers=[m for m in MODEL_NAMES if m != "crBG-S"])
        all_rv[ticker] = {name: {"statistic": r["statistic"], "p_value": r["p_value"], "favors_crBG_S": r["favors_a"]}
                           for name, r in rv.items()}

        for name, f in fits.items():
            means = {k: v.mean(axis=0) if v.ndim == 3 else v for k, v in
                     build_regressors(data["daily_returns"], data["RCOV"], data["P"], data["N"],
                                       data["Mp"], data["Mn"])[name].items()}
            set_unconditional_means(name, means)
        h_bar = np.cov(data["daily_returns"].T, bias=True)
        for name in ("BG", "crBG", "crBG-S"):
            f = fits[name]
            z_grid = np.linspace(-3, 3, config.NIS_Z_GRID)
            for diag in ("main", "secondary"):
                surf = news_impact_surface(name, f["coefs"], f["beta"], h_bar, z_grid, diag)
                nis_records.append({"ticker": ticker, "model": name, "diagonal": diag,
                                     "z": z_grid.tolist(), "H12": surf.tolist()})

        for name in ("crBG", "trBG"):
            f = fits[name]
            for signs in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
                irf = impulse_response(name, f["coefs"], f["beta"], h_bar, signs,
                                        horizon=config.IRF_HORIZON, n_paths=irf_paths,
                                        n_intraday=26, shock_size_sd=config.IRF_SHOCK_SIZE_SD,
                                        seed=config.RANDOM_SEED)
                irf_records.append({"ticker": ticker, "model": name, "signs": signs,
                                     "H12_path": irf[:, 0, 1].tolist()})

        oos = oos_forecast_one_stock(data, window, step)
        for name in MODEL_NAMES:
            all_oos_losses["Frobenius"][name].append(frobenius_loss(oos[name]["H"], oos[name]["RCOV"]))
            all_oos_losses["QLIKE"][name].append(qlike_loss(oos[name]["H"], oos[name]["RCOV"]))
        log.append(f"{ticker}: fitted 8 models + NIS + IRF + OOS in {time.time() - t0:.1f}s")

    mcs_results = {}
    for loss_name, per_model in all_oos_losses.items():
        stacked = {m: np.concatenate(v) for m, v in per_model.items()}
        p_values = model_confidence_set(stacked, alpha=1 - config.MCS_LEVEL,
                                         n_boot=config.MCS_BOOT_REPLICATIONS_RUN,
                                         block_length=config.MCS_BLOCK_LENGTH_RUN, seed=config.RANDOM_SEED)
        mcs_results[loss_name] = {
            "mean_loss": {m: float(np.mean(stacked[m])) for m in MODEL_NAMES},
            "p_values": p_values,
            "included": mcs_included(p_values, config.MCS_LEVEL),
        }

    return {"fits": all_fits, "rivers_vuong": all_rv, "nis": nis_records, "irf": irf_records,
            "oos_mcs": mcs_results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run the paper's full configuration")
    args = parser.parse_args()

    if args.full:
        tickers, start, end = config.DJIA_24_PAPER, config.SAMPLE_START_PAPER, config.SAMPLE_END_PAPER
        window, step, irf_paths = config.OOS_WINDOW_PAPER, config.OOS_STEP_PAPER, config.IRF_PATHS_PAPER
    else:
        tickers, start, end = config.DJIA_RUN, config.SAMPLE_START_RUN, config.SAMPLE_END_RUN
        window, step, irf_paths = config.OOS_WINDOW_RUN, config.OOS_STEP_RUN, config.IRF_PATHS_RUN

    log: list[str] = [f"exp1: tickers={tickers}, market={config.MARKET_TICKER}, {start}..{end}, "
                      f"window={window}, step={step}, irf_paths={irf_paths}, full={args.full}"]
    results = run_experiment(tickers, config.MARKET_TICKER, start, end, window, step, irf_paths, log)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp1_fits.json", "w") as f:
        json.dump(results["fits"], f, indent=2, default=float)
    with open(RESULTS_DIR / "exp1_rivers_vuong.json", "w") as f:
        json.dump(results["rivers_vuong"], f, indent=2, default=float)
    with open(RESULTS_DIR / "exp1_nis.json", "w") as f:
        json.dump(results["nis"], f, indent=2, default=float)
    with open(RESULTS_DIR / "exp1_irf.json", "w") as f:
        json.dump(results["irf"], f, indent=2, default=float)
    with open(RESULTS_DIR / "exp1_oos_mcs.json", "w") as f:
        json.dump(results["oos_mcs"], f, indent=2, default=float)
    with open(RESULTS_DIR / "exp1_log.txt", "w") as f:
        f.write("\n".join(log))

    make_plots(results, tickers)
    print("\n".join(log))


def make_plots(results: dict, tickers: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for rec in results["nis"]:
        if rec["model"] == "crBG" and rec["diagonal"] == "main" and rec["ticker"] == tickers[0]:
            ax.plot(rec["z"], rec["H12"], label=f"{rec['ticker']} crBG main diagonal")
        if rec["model"] == "BG" and rec["diagonal"] == "main" and rec["ticker"] == tickers[0]:
            ax.plot(rec["z"], rec["H12"], label=f"{rec['ticker']} BG main diagonal", linestyle="--")
    ax.set_xlabel("standardized shock z_tilde")
    ax.set_ylabel("H_{t,12} (stock-market covariance)")
    ax.set_title("News Impact Surface, main diagonal (Appendix B.1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_nis_main_diagonal.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for rec in results["irf"]:
        if rec["ticker"] == tickers[0] and rec["model"] == "crBG":
            ax.plot(rec["H12_path"], label=f"signs={rec['signs']}")
    ax.set_xlabel("horizon (days)")
    ax.set_ylabel("IRF of H_{t+h,12}")
    ax.set_title(f"crBG impulse response functions, {tickers[0]} (Appendix B.2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_irf_crBG.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
