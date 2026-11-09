# AGENTS.md

Guide for contributors and coding agents working in this repository.

## What this repository implements

A replication of Tim Bollerslev, Andrew J. Patton and Rogier Quaedvlieg
(2020), "Multivariate leverage effects and realized semicovariance GARCH
models", *Journal of Econometrics*, 217(2), 411-430. Two experiments are
implemented:

- `exp/exp1.py` -- bivariate realized semicovariance GARCH models for each
  DJIA stock paired with an S&P 500 proxy (paper Section 2.2, Section 4,
  Appendix B): in-sample QMLE of 8 specifications, Rivers-Vuong (2002)
  non-nested tests, News Impact Surfaces, 30-day impulse response functions,
  and out-of-sample Frobenius/QLIKE loss with an 80% Model Confidence Set.
- `exp/exp2.py` -- 24-dimensional DCC models with realized semicorrelations
  (paper Section 5): univariate GARCH model selection per stock, realized
  semicorrelation construction, 5 DCC variants, correlation-dynamics
  diagnostics, and out-of-sample forecast evaluation.

## Layout

```
src/    flat modules: semicovariance/semicorrelation construction, the
        bivariate and DCC GARCH recursions and QMLE, News Impact Surfaces,
        impulse response simulation, Rivers-Vuong and MCS/forecast
        evaluation, and config.py (all parameters, as module-level constants)
exp/    exp1.py and exp2.py, run as `python -m exp.exp1` / `python -m exp.exp2`
data/   data_loader.py: OHLCV bars via the QuantX Dataloader REST API; no
        other loader exists, and no raw market data is ever committed
tests/  pytest unit tests (synthetic fixtures) and a market-data integration
        test marked `market_data`
results/ metrics (JSON), figures (PNG) and RESULTS.md, all overwritten by
        the experiment scripts
```

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env                        # set DATALOADER_API_KEY, MASSIVE_API_KEY

python -m pytest -m "not market_data"        # unit tests, no live API calls
python -m pytest                             # also the market-data test (needs keys)

python -m exp.exp1                           # executed configuration
python -m exp.exp2                           # executed configuration
python -m exp.exp1 --full                    # the paper's full configuration (Section 3)
python -m exp.exp2 --full

./reproduce.sh                               # tests + both experiments, executed configuration
```

## Conventions

- Every experiment parameter lives in `src/config.py` as an upper-case
  module-level constant with a comment stating its meaning, unit and source
  (a paper section/equation, or "not specified in the paper"). No classes or
  parameter dictionaries. Parameters with both a paper value and an executed
  value are named `*_PAPER` / `*_RUN`; `--full` on either experiment script
  selects the `*_PAPER` values.
- Market data is loaded only through `data/data_loader.py`
  (`load_ohlcv`), which calls the QuantX Dataloader REST API
  (`https://quantx-api.limex.pro/apis/dataloader`, `POST /fetch-ohlcvs`).
  `src/` and `exp/` never call a market-data API directly.
- `DATALOADER_API_KEY` and `MASSIVE_API_KEY` are read from the environment
  through `python-dotenv`; they are set only in an uncommitted `.env`
  (`.env.example` holds empty placeholders) and must never appear in code,
  logs, or results.
- Tests on real market data are marked `@pytest.mark.market_data`, use a
  reduced slice of the executed universe cached by `tests/conftest.py`, and
  are skipped automatically when `DATALOADER_API_KEY` is not set.
- Rerunning `python -m exp.exp1` / `exp.exp2` overwrites every file in
  `results/`. The prose, tables and verdicts in `results/RESULTS.md` are
  written by hand from those files after a run and must be checked against
  them again after any rerun; they are not regenerated automatically.
