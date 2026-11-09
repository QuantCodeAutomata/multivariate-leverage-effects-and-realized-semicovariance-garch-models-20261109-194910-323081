# Multivariate leverage effects and realized semicovariance GARCH models

Replication of: Tim Bollerslev, Andrew J. Patton and Rogier Quaedvlieg (2020),
"Multivariate leverage effects and realized semicovariance GARCH models",
*Journal of Econometrics*, 217(2), 411-430.

The paper decomposes the realized covariance matrix of high-frequency returns
into concordant (joint positive, joint negative) and discordant semicovariance
components and shows that these realized semicovariances drive "continuous
leverage effects" in multivariate GARCH models, beyond the threshold
(sign-based) asymmetry of GJR-type models. Two experiments are implemented:

- **exp_1** -- bivariate semicovariance GARCH models for each DJIA stock
  paired with the S&P 500 index (Sections 2.2 and 4; Appendix B).
- **exp_2** -- 24-dimensional DCC models with realized semicorrelations
  (Section 5).

## Repository layout

```
src/    model and estimation code (flat modules)
exp/    experiment entry points (python -m exp.exp1 / exp.exp2)
data/   data_loader.py: OHLCV bars via the QuantX Dataloader REST API
tests/  pytest unit tests and a market-data integration test
results/ saved metrics, tables and plots (written by the experiments)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

## Tests

```bash
python -m pytest -m "not market_data"      # unit tests, no API calls
python -m pytest                            # also the market-data test (needs keys)
```

## Running the experiments

```bash
python -m exp.exp1            # executed configuration (see Deviations below)
python -m exp.exp2            # executed configuration
python -m exp.exp1 --full     # the paper's full configuration (Section 3)
python -m exp.exp2 --full
```

`reproduce.sh` runs the test suite and both experiments in the executed
configuration.

## Data

High-frequency 15-minute bars and daily bars for the DJIA constituents and the
S&P 500 proxy are loaded through the QuantX Dataloader REST API
(`https://quantx-api.limex.pro/apis/dataloader`), as described in
`data/data_loader.py`. Massive has no entitlement for the raw S&P 500 index
symbol (`I:SPX` returns `NOT_AUTHORIZED`), so the SPY ETF is used as the market
proxy throughout, as the market data guidance allows. The stock universe and
sample period executed here are reduced from the paper's configuration because
15-minute Massive coverage for these tickers does not reach back to 2000; see
`results/RESULTS.md` for the full account of this and every other deviation.

## Results

See `results/RESULTS.md` for the tables, figures, and the paper comparison,
together with the "Deviations from the paper" and "Not reproduced" sections.
