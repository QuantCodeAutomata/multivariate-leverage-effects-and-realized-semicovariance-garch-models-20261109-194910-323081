"""Parameters for the replication of Bollerslev, Patton and Quaedvlieg (2020),
"Multivariate leverage effects and realized semicovariance GARCH models",
Journal of Econometrics 217, 411-430.

Every variable below is a module-level constant with a comment naming its
meaning, unit, and source (a paper section/equation, or "not specified in
the paper"). No classes or parameter dictionaries are used, per the
replication's configuration convention.
"""

# ---------------------------------------------------------------------------
# Universe and sample period (Section 3, "Data description")
# ---------------------------------------------------------------------------

# The paper's 24 DJIA constituents that traded continuously Jan 2000-Dec 2014.
DJIA_24_PAPER = [
    "AAPL", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS", "HD", "IBM",
    "INTC", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV",
    "UTX", "VZ", "WMT",
]  # Section 3, paper's universe (24 names)

MARKET_TICKER = "SPY"  # proxy for the paper's value-weighted S&P 500 index:
# Massive requires an index entitlement for "I:SPX" (HTTP 502 NOT_AUTHORIZED);
# the ETF tracking the S&P 500 is used instead, as allowed by the data skill.

UNIVERSE_SIZE_PAPER = 24  # number of individual stocks, Section 3
# The QuantX Dataloader has no 15-minute Massive bars before ~2005 for these
# tickers (verified by probing AAPL for 2000, 2003, 2005: 2000 and 2003 fail
# with "no ticker record", 2005 succeeds). The stock universe itself is cut to
# 8 names, ranked by the paper's own measure of market significance -- average
# daily dollar trading volume (price times volume) over the executed sample
# start -- to keep bivariate QMLE estimation, Monte-Carlo IRFs, and the 24 (or
# fewer) dimensional DCC optimizers within the time available for this
# replication. The cut preserves the paper's method, frequency and ordering.
UNIVERSE_SIZE_RUN = 5
DJIA_RUN = ["AAPL", "MSFT", "INTC", "CSCO", "GE"]  # top-5
# of DJIA_24_PAPER by mean (close * volume) daily dollar volume, 2010-2014;
# cut from the 8 originally selected to let bivariate QMLE on 8 model
# specifications, Monte-Carlo IRFs and rolling out-of-sample re-estimation
# finish for every name within the time available for this replication.

SAMPLE_START_PAPER = "2000-01-01"  # Section 3: sample start
SAMPLE_END_PAPER = "2014-12-31"  # Section 3: sample end (3773 trading days)
SAMPLE_START_RUN = "2005-01-01"  # earliest date with 15-minute Massive
# coverage for the executed tickers (data availability, not specified choice)
SAMPLE_END_RUN = "2014-12-31"  # kept at the paper's sample end

INTRADAY_FREQ = "15m"  # Section 3: "we rely [on] 15-minute returns in the
# construction of all the realized (semi)covariance matrices"
DAILY_FREQ = "1d"  # close-to-close daily returns used for covariance
# targeting and the daily-threshold (GJR-type) models, footnote 6

# ---------------------------------------------------------------------------
# Bivariate GARCH models (exp_1), Section 2.2 and Section 4
# ---------------------------------------------------------------------------

BIVARIATE_MODELS = ["BG", "tBG", "rBG", "trBG", "crBG", "tBG-S", "trBG-S", "crBG-S"]
# BG: Eq. (5); rBG: Eq. (6); crBG: Eq. (7); crBG-S (split M): Eq. (8);
# tBG: Eq. (9); trBG: Eq. (11); tBG-S/trBG-S (split M): Eq. (10) analogue.

QMLE_OPTIMIZER = "L-BFGS-B"  # scipy.optimize.minimize method; starting
# values and routine are not specified in the paper (uncertainty noted in
# the task's evidence ledger); L-BFGS-B is used for its bound support
# (0 <= alpha-type parameters, 0 <= beta < 1, alpha + beta < 1).
QMLE_MAX_ITER = 500  # scipy optimizer iteration cap; not specified in the paper

# News Impact Surface (Appendix B.1)
NIS_Z_GRID = 41  # number of grid points per axis for the NIS cross-sections;
# not specified in the paper

# Impulse response functions (Appendix B.2)
IRF_HORIZON = 30  # days; "IRFs up to a 30-day horizon", Appendix B.2
IRF_SHOCK_SIZE_SD = 3.0  # "initial shocks of size 3*sqrt(H-bar)", Appendix B.2
IRF_PATHS_PAPER = 1_000_000  # "we... simulate 1,000,000 paths", Appendix B.2
IRF_PATHS_RUN = 20_000  # reduced Monte Carlo path count: the full 1,000,000
# paths x 30 horizons x 4 shocks x 8 stocks exceeds the time available for
# this replication; 20,000 paths keep the simulation standard error small
# enough to see the sign and the relative ordering of the IRFs.

# Out-of-sample forecasting (Section 4.3 / Section 5.5)
OOS_WINDOW_PAPER = 1000  # "rolling sample windows of 1000 observations"
OOS_STEP_PAPER = 20  # "re-estimate... every 20th observation"
OOS_COUNT_PAPER = 2773  # "a total of 2773 out-of-sample forecast observations"
OOS_WINDOW_RUN = 250  # rolling window length (trading days) used in the
# executed run: proportionally shorter than the paper's 1000-day window to
# match the shorter 2005-2014 executed sample while leaving enough
# observations after the window for a meaningful forecast evaluation.
OOS_STEP_RUN = 50  # re-estimation frequency widened from the paper's 20 (kept
# in OOS_STEP_PAPER) so that the rolling QMLE re-estimation of all eight
# bivariate / five DCC specifications finishes within the time available for
# this replication; the window length, loss functions and MCS level are
# unchanged
MCS_LEVEL = 0.80  # "80% Model Confidence Set", Hansen et al. (2011)
MCS_BOOT_REPLICATIONS_PAPER_UNSPECIFIED = None  # the paper does not state the
# number of bootstrap replications or the block length used for the MCS
MCS_BOOT_REPLICATIONS_RUN = 300  # stationary-bootstrap replications used here
MCS_BLOCK_LENGTH_RUN = 10  # expected block length (trading days) of the
# stationary bootstrap (Politis and Romano, 1994); not specified in the paper

RIVERS_VUONG_LEVEL = 0.10  # "significant at the 10%-level", Table 3 note

# ---------------------------------------------------------------------------
# Univariate and DCC models (exp_2), Section 5
# ---------------------------------------------------------------------------

UNIVARIATE_MODELS = ["GARCH", "tGARCH", "rGARCH", "trGARCH", "crGARCH"]
DCC_MODELS = ["DCC", "tDCC", "rDCC", "trDCC", "crDCC"]
# DCC: Eq. (23); rDCC: Eq. (24); crDCC: Eq. (25); tDCC: Eq. (26); trDCC: Eq. (27)

DCC_QMLE_OPTIMIZER = "L-BFGS-B"
DCC_QMLE_MAX_ITER = 500

CRISIS_EVENT_LEHMAN = ("2008-01-01", "2008-12-31")  # Fig. 4 left panel,
# Goldman Sachs (GS) vs Travelers (TRV), around the Lehman Brothers bankruptcy
# of September 15, 2008
CRISIS_EVENT_911_PAIR = ("BA", "TRV")  # Fig. 4 right panel
CRISIS_EVENT_911_WINDOW = ("2001-08-01", "2001-12-31")  # the task's evidence
# notes the figure caption's "2011" is a typo for 2001 (9/11 occurred in
# 2001); the September 2001 attacks date is used here. Because the executed
# sample starts in 2005 (data-availability cut above), this event cannot be
# evaluated on the executed data and is reported under "Not reproduced".

CORR_DYNAMICS_PAIRS = [("GS", "TRV"), ("BA", "TRV")]  # Section 5.4 event studies;
# TRV is not in the liquidity-based 8-stock run universe, so these pairs are
# only available when --full uses the paper's 24-stock universe (see README).

RANDOM_SEED = 20200123  # paper's "Available online" date used as a fixed,
# arbitrary seed for reproducible Monte Carlo draws; not specified in the paper
