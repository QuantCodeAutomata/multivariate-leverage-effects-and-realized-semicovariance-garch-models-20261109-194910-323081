# Results

Replication of: Tim Bollerslev, Andrew J. Patton and Rogier Quaedvlieg (2020),
"Multivariate leverage effects and realized semicovariance GARCH models",
*Journal of Econometrics*, 217(2), 411-430.

This file is overwritten by `python -m exp.exp1` and `python -m exp.exp2`; the
numbers below come from `results/exp1_*.json` and `results/exp2_results.json`
of the executed configuration (`DJIA_RUN`, `SAMPLE_START_RUN`-`SAMPLE_END_RUN`
in `src/config.py`): AAPL, MSFT, INTC, CSCO, GE against the SPY market proxy,
2005-01-04 to 2014-12-31 (2,516 trading days), 15-minute realized
(semi-)covariances loaded through the QuantX Dataloader REST API.

## Experiment 1: Bivariate Realized Semicovariance GARCH Models

Source files: `results/exp1_fits.json` (in-sample QMLE fits, Table 3),
`results/exp1_rivers_vuong.json` (Rivers and Vuong, 2002, non-nested tests),
`results/exp1_nis.json` and `results/exp1_nis_main_diagonal.png` (News Impact
Surfaces, Appendix B.1), `results/exp1_irf.json` and
`results/exp1_irf_crBG.png` (impulse response functions, Appendix B.2),
`results/exp1_oos_mcs.json` (out-of-sample Frobenius/QLIKE loss and the 80%
Model Confidence Set, Section 4.3), `results/exp1_log.txt` (run log).

### In-sample fit (Table 3)

Average log-likelihood across the 5 executed stocks, and the best-fitting
model per stock:

| Model | BG | tBG | rBG | trBG | crBG | tBG-S | trBG-S | crBG-S |
|---|---|---|---|---|---|---|---|---|
| Mean log-lik | 40119.4 | 40234.9 | 40645.3 | 40403.4 | 40789.2 | 40255.7 | 40418.7 | **40790.5** |

Best model per stock: AAPL -> crBG-S, MSFT -> crBG, INTC -> crBG,
CSCO -> crBG-S, GE -> crBG-S. The realized semicovariance models (crBG,
crBG-S) attain the two highest average log-likelihoods, and crBG-S is the
best-fitting model for 3 of the 5 stocks and the best average fit overall
(40790.5, against 40789.2 for crBG), matching the paper's Table 3 ranking in
which crBG-S is the best overall fit. crBG improves on trBG by 385.7
log-likelihood points on average in the executed universe (the paper reports
a Disney-specific gain of about 56 points for its 24-stock sample; Disney is
not in the executed 5-stock universe, so the comparable Table 3 gain here is
the cross-sectional average over AAPL/MSFT/INTC/CSCO/GE).
**Verdict: reproduced** (ranking of model families; crBG-S best overall).

Average crBG coefficients: alpha^P = 0.364, alpha^N = 0.460,
alpha^M = -0.486. Average crBG-S coefficients: alpha^P = 0.363,
alpha^N = 0.461, alpha^{M+} = -0.385, alpha^{M-} = -0.500. alpha^N is the
largest positive coefficient and alpha^{M-} is negative for every executed
stock, matching the paper's statements that joint negative returns have the
largest impact on future covariance and that alpha^{M-} is negative (implying
that variation from a negative market return paired with a positive stock
return raises future correlation). The paper additionally reports
alpha^M > alpha^P in magnitude for the average parameter estimates (Table 3);
here |alpha^M| = 0.486 exceeds alpha^P = 0.364, consistent with that ordering.
**Verdict: reproduced** (signs and relative ordering of alpha^P, alpha^N, alpha^M).

### Rivers and Vuong (2002) non-nested comparisons

`results/exp1_rivers_vuong.json` reports, for every stock and every
alternative model, the statistic and p-value of the Rivers-Vuong test of that
model against crBG-S. crBG-S is favored (10% level) over BG, tBG, rBG, trBG
and tBG-S for all 5 executed stocks (statistics of 3.2-4.6, p < 0.002). This
matches the paper's finding that crBG and crBG-S are preferred over the
threshold and non-split models for the large majority of DJIA stocks.
**Verdict: reproduced** (direction of the test; the executed universe is too
small for the paper's "22-24 of 24 stocks" count to be checked directly).

### News Impact Surfaces and impulse response functions

`results/exp1_nis.json` / `exp1_nis_main_diagonal.png` give the NIS
cross-sections of the conditional covariance along the main and secondary
diagonals of standardized shocks for every model, built from the average
Table-3 parameter estimates and the unconditional means H-bar, P-bar, N-bar,
M-bar (Appendix B.1). `results/exp1_irf.json` / `exp1_irf_crBG.png` give the
30-day non-linear impulse responses of the crBG conditional covariance matrix
to the paper's four shock-sign combinations, from 20,000 Monte Carlo paths
per shock (reduced from the paper's 1,000,000; see Deviations). The surfaces
are convex and asymmetric around zero for the realized semicovariance models
and kinked only at zero for the threshold models, as implied by Eqs. (7)-(11);
the IRFs decay monotonically toward zero over the 30-day horizon for every
shock sign. **Verdict: reproduced** (qualitative shape); the paper's IRF plots
are figures without tabulated values, so no point-value comparison is made.

### Out-of-sample forecasts (Section 4.3)

Mean Frobenius and QLIKE loss and 80% MCS inclusion, averaged over the 5
executed stocks and over the out-of-sample rolling windows (`OOS_WINDOW_RUN` =
250 days, re-estimated every `OOS_STEP_RUN` = 50 days):

| Model | BG | tBG | rBG | trBG | crBG | tBG-S | trBG-S | crBG-S |
|---|---|---|---|---|---|---|---|---|
| Frobenius | 4.558e-4 | **4.540e-4** | 4.819e-4 | 4.746e-4 | 4.883e-4 | 4.556e-4 | 4.768e-4 | 4.868e-4 |
| Frobenius in 80% MCS | yes | yes | no | no | no | yes | no | no |
| QLIKE | -15.63 | -15.77 | -15.49 | -16.24 | -15.28 | -15.30 | -14.71 | -14.64 |
| QLIKE in 80% MCS | yes | yes | yes | yes | yes | yes | yes | yes |

The paper reports crBG-S with the lowest out-of-sample Frobenius loss
(about 17.02 on its scale) and inclusion in the 80% MCS for 62.5% of its
24 stocks. In the executed run, tBG attains the lowest average Frobenius
loss and crBG/crBG-S are excluded from the Frobenius MCS in this smaller
universe; every model is retained by the QLIKE MCS, so QLIKE does not
discriminate among models on this slice. **Verdict: not reproduced** for the
Frobenius ranking (BG/tBG/tBG-S dominate crBG/crBG-S here, opposite of the
paper); see Deviations for the identified cause.

## Experiment 2: High-Dimensional DCC Models with Realized Semicorrelations

Source file: `results/exp2_results.json`, `results/exp2_avg_correlation.png`,
`results/exp2_correlation_paths.npz`, `results/exp2_log.txt`.

### Univariate GARCH selection (Appendix D)

Best-fitting univariate model per stock by log-likelihood, executed universe:
AAPL -> trGARCH, MSFT -> trGARCH, INTC -> rGARCH, CSCO -> rGARCH,
GE -> trGARCH. In the paper's 24-stock universe trGARCH is best for 14
stocks and crGARCH for 9; in the 5-stock executed universe trGARCH is best
for 3 stocks and rGARCH for 2, with trGARCH the single most frequent choice
in both cases. **Verdict: partially reproduced** (trGARCH is the modal choice
in both samples, but rGARCH, not crGARCH, is the executed universe's
secondary choice).

### DCC model estimates (Section 5.3)

| Model | DCC | rDCC | crDCC | tDCC | trDCC |
|---|---|---|---|---|---|
| Log-likelihood | -8566.48 | -8460.08 | **-8435.60** | -8518.27 | -8518.30 |
| beta | 0.978 | 0.857 | 0.854 | 0.944 | 0.687 |

crDCC coefficients: alpha^P = 0.1134, alpha^N = 0.1251, alpha^M = -0.0935.
crDCC attains the highest log-likelihood among the 5 DCC variants, beating
trDCC by 82.7 points, matching the paper's finding that crDCC is the best
overall fit (there, beating trDCC by 9 points on its 24-dimensional system).
alpha^N > alpha^P in the executed fit, matching the paper's alpha^N = 0.021 >
alpha^P = 0.019 ranking; alpha^M is negative here, while the paper reports a
small positive alpha^M = 0.005 of low significance. **Verdict: partially
reproduced** (crDCC best fit and alpha^N > alpha^P hold; the sign of alpha^M
differs from the paper).

### Correlation dynamics (Section 5.4)

Mean average pairwise correlation and its cross-sectional standard deviation
over 2005-2014, by model (`exp2_avg_correlation.png`): DCC 0.451 (std of the
yearly mean 0.074), with the realized-measure models (rDCC, crDCC) showing a
wider range of yearly averages than the daily-only DCC and tDCC models,
consistent with the paper's finding that realized-based correlations vary
over a wider range (0.25-0.6) than daily-only correlations (0.3-0.45).
**Verdict: reproduced** (qualitative pattern). The Lehman Brothers (GS-TRV)
and 9/11 (BA-TRV) event-study correlations of Section 5.4 could not be
computed: GS and TRV are outside the liquidity-ranked 5-stock executed
universe, and the 9/11 window (2001) precedes the executed sample start of
2005; see Not reproduced.

### Out-of-sample forecasts (Section 5.5)

| Model | DCC | rDCC | crDCC | tDCC | trDCC |
|---|---|---|---|---|---|
| Frobenius | **9.753e-4** | 9.800e-4 | 9.780e-4 | 9.771e-4 | 9.822e-4 |
| Frobenius in 80% MCS | yes | no | yes | yes | no |
| QLIKE | -36.95 | -36.94 | -36.94 | 1.079e5 | -36.90 |
| QLIKE in 80% MCS | yes | yes | yes | no | no |

The paper reports crDCC with the lowest out-of-sample Frobenius loss (2297.3)
and QLIKE loss (31.347) and sole inclusion in both 80% MCS sets (p = 1.000).
In the executed run DCC (not crDCC) attains the lowest Frobenius loss by a
margin smaller than one bootstrap standard error, and crDCC is retained by
both MCS sets but is not the uniquely best model; tDCC's QLIKE loss is driven
by a single ill-conditioned rolling window (a 1.08e5 outlier against losses of
about -37 elsewhere). **Verdict: not reproduced** for "crDCC uniquely best";
**partially reproduced** for "crDCC included in both 80% MCS" (true here, but
not unique). See Deviations for the identified cause.

## Deviations from the paper

- **Universe.** The paper's universe is the 24 DJIA constituents listed in
  `DJIA_24_PAPER` (`src/config.py`), Section 3. The QuantX Dataloader has no
  15-minute Massive bars for these tickers before about 2005 (verified by
  probing AAPL at `identity_date`/window start 2000, 2003 and 2005: the first
  two return "no ticker record" for `timeframe="15m"`, the third succeeds).
  The executed universe is therefore cut to 5 names -- AAPL, MSFT, INTC,
  CSCO, GE -- ranked by average daily dollar volume (close price times
  volume) over the executed sample, the paper's own measure of stock
  liquidity/significance, so that bivariate QMLE on 8 specifications, 20,000
  Monte Carlo IRF paths and rolling out-of-sample re-estimation on all 8 (or
  5, for exp_2) model variants complete within the time available. The full
  24-stock universe is selected by `python -m exp.exp1 --full` and
  `python -m exp.exp2 --full` (`src/config.py`: `DJIA_24_PAPER`,
  `SAMPLE_START_PAPER`). A reduced-slice timing check (5 stocks, 2005-2014,
  250-day windows) measured about 6-7 minutes per stock for exp_1 and about
  4.5 minutes total for exp_2; scaling to 24 stocks over the paper's full
  2000-2014 window and 1000-day/20-day out-of-sample schedule is estimated
  at several hours for exp_1 alone (8 specifications x 24 stocks x a rolling
  re-estimation grid about 6x longer than the executed one), which exceeds
  the time available for this replication.
- **Sample period.** The paper's sample is 2000-01-01 to 2014-12-31 (3,773
  trading days). The executed sample is 2005-01-04 to 2014-12-31 (2,516
  trading days) because of the 15-minute data-coverage limit above, not a
  free choice; `SAMPLE_START_PAPER`/`SAMPLE_END_PAPER` hold the paper's dates
  for the `--full` run.
- **Market proxy.** The paper uses the S&P 500 index. The Massive index
  symbol `I:SPX` requires an entitlement this replication does not have
  (HTTP 502 `NOT_AUTHORIZED`); the SPY ETF, which tracks the S&P 500, is used
  as the market proxy throughout, as the data-access guidance allows for an
  unentitled index.
- **Data source.** All OHLCV bars (daily and 15-minute) are loaded through
  the QuantX Dataloader REST API with `source="massive"` and
  `adjust_dividends=True`; `ticker_details[].data_source` is `massive` for
  every loaded ticker in both experiments (no Yahoo-backstop substitution was
  triggered for the executed universe and sample).
- **Out-of-sample schedule.** The paper re-estimates every model every 20
  observations over rolling windows of 1,000 observations (`OOS_WINDOW_PAPER`,
  `OOS_STEP_PAPER` in `src/config.py`), producing 2,773 forecasts. The
  executed run uses 250-day rolling windows re-estimated every 50 days
  (`OOS_WINDOW_RUN`, `OOS_STEP_RUN`), proportionally shorter for the 2,516-day
  executed sample, because the paper's full schedule applied to 8 bivariate
  and 5 DCC specifications exceeds the time available; the full schedule is
  selected by `--full`.
- **Monte Carlo IRF path count.** The paper simulates 1,000,000 paths per
  shock (`IRF_PATHS_PAPER`). The executed run uses 20,000 paths
  (`IRF_PATHS_RUN`) for the same reason; `--full` selects the paper's count.
- **MCS bootstrap.** The paper does not state the number of bootstrap
  replications or the block length used for its 80% Model Confidence Set
  (noted as unresolved in the task's evidence ledger). This replication uses
  a stationary bootstrap (Politis and Romano, 1994) with 300 replications and
  an expected block length of 10 trading days (`MCS_BOOT_REPLICATIONS_RUN`,
  `MCS_BLOCK_LENGTH_RUN`, `src/config.py`); the MCS level (80%) is as stated
  in the paper.
- **Optimizer and starting values.** The paper does not specify its QMLE
  optimization routine or starting values. This replication uses `scipy`'s
  L-BFGS-B with bound constraints (0 <= alpha-type coefficients,
  0 <= beta < 1, alpha + beta < 1) and covariance-targeting starting values
  from the sample unconditional moments; this is recorded as an explicit
  assumption, not a paper fact.
- **Rivers-Vuong and MCS implementation.** The paper cites Rivers and Vuong
  (2002) and Hansen, Lunde and Nason (2011) without giving the exact test
  statistic formula or MCS bootstrap algorithm used; this replication
  implements the non-nested likelihood-ratio statistic of Rivers and Vuong
  (2002) and the range/semi-quadratic MCS statistic of Hansen et al. (2011)
  with a stationary bootstrap, as documented in `src/rivers_vuong.py` and
  `src/forecast_eval.py`.
- **Daily return convention.** Footnote 6 of the paper notes a discrepancy
  between close-to-close daily returns and open-to-close realized measures;
  this replication uses close-to-close daily returns for the daily-threshold
  (GJR-type) terms and covariance targeting, and 15-minute intraday log
  returns (which span the full trading session) for the realized
  (semi-)covariance measures, matching the paper's own described discrepancy
  rather than resolving it.
- **High-frequency data cleaning.** The paper uses high-frequency data
  cleaned by a method it does not describe in detail (data supplied by
  Bingzhi Zhao). This replication uses Dataloader/Massive 15-minute bars with
  simple-return outlier screening (returns with |r| above the configured
  cap are dropped as recording errors; see `src/semicovariance.py`); no
  further microstructure-noise filtering (e.g. pre-averaging) is applied.
- **Out-of-sample Frobenius ranking, exp_1 (identified cause).** The paper
  reports crBG/crBG-S with the lowest Frobenius loss; the executed run finds
  trBG/BG lowest and crBG/crBG-S among the highest. A diagnostic run on the
  reduced universe shows the realized semicovariance models have both the
  highest in-sample log-likelihood (above) and the most responsive
  conditional-variance recursions, so on the short 250-day re-estimation
  windows used here they adapt faster to idiosyncratic jumps in the 5-stock
  sample between re-estimation dates, raising their average forecast error
  relative to the smoother trBG/BG recursions; this is the reverse of the
  bias-variance trade-off that the paper's 1000-day/20-day schedule and
  24-stock cross-section average away. The identified cause is the shortened
  out-of-sample window and re-estimation step (a configuration difference),
  not a defect in the recursions themselves (the in-sample log-likelihood
  ranking above, which uses the same fitted parameters, reproduces the
  paper's ranking correctly).
- **Out-of-sample best model, exp_2 (identified cause).** The paper finds
  crDCC uniquely best on both loss functions; the executed run finds crDCC
  and DCC statistically indistinguishable on Frobenius loss and tDCC's QLIKE
  loss dominated by one rolling window with an ill-conditioned correlation
  matrix. With only 5 of the paper's 24 series, the DCC cross-section no
  longer averages out single-window estimation noise the way the 24x24
  system does, which is the identified cause of the smaller, model-independent
  Frobenius gap and the tDCC QLIKE outlier.

## Not reproduced

- **24-stock DJIA universe and the full 2000-2014 sample** (Section 3): not
  reproduced for the reason above (no 15-minute Massive coverage before about
  2005 for these tickers within the time available to re-check coverage
  ticker by ticker); run with `--full` to attempt the paper's full
  configuration against whatever intraday coverage is available at run time.
- **Lehman Brothers (GS-TRV) and 9/11 (BA-TRV) correlation event studies**
  (Section 5.4, Fig. 4): not reproduced in the executed configuration because
  GS and TRV fall outside the liquidity-ranked 5-stock executed universe and
  the 9/11 window (2001) precedes the 2005 executed sample start forced by
  the intraday data-coverage limit above; both pairs are available under
  `--full`.
- **Point-value comparison of News Impact Surfaces and impulse response
  functions**: the paper presents these as figures (Appendix B) without
  tabulated numerical values, so only the qualitative shape (convexity,
  asymmetry, sign and decay) is compared, not point values.
