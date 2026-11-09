"""Experiment parameters derived from the SmartDelta user-path specification.

Every value carries a short comment giving its meaning, unit and the section of
the paper (SmartDelta "User path step by step") it originates from, or the note
"not specified in the paper" when the paper leaves it open.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Shared universe and sample period
# -----------------------------------------------------------------------------

# Tickers for the structured-product basket; up to 10 names are supported by the
# platform (Section 4 "Details that matter for product design": Up to 10 assets).
UNIVERSE: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

# Historical sample used for correlation / volatility estimation and for the
# backtest in experiment 2. The paper does not pin a specific window; the user
# selects it on the Backtest screen (Section 1, step 12 "Backtest - Optional
# Period"). A two-year span is used so that daily returns are enough for stable
# correlation estimates.
SAMPLE_START: str = "2022-01-03"
SAMPLE_END: str = "2024-12-31"

# -----------------------------------------------------------------------------
# Experiment 1 — Monte Carlo pricing engine
# -----------------------------------------------------------------------------

# Number of simulated correlated GBM paths. Section 4: "Monte Carlo 100,000
# paths".
N_PATHS_PAPER: int = 100_000

# Number of paths executed for every call unless --full is passed; equal to
# the paper configuration.
N_PATHS_RUN: int = 100_000

# Daily time-stepping of the GBM: 252 trading days per year, in days. Not
# specified in the paper; standard convention for daily-step GBM.
TRADING_DAYS_PER_YEAR: int = 252

# Random seed for reproducibility of the simulation. Not specified in the
# paper.
RANDOM_SEED: int = 20251109

# Maximum number of European barriers applied at maturity, as defined by the
# platform (Section 4: "Up to 5 barriers on the maturity date").
MAX_MATURITY_BARRIERS: int = 5

# Decimal places displayed for pricing and hedging outputs. Section 4:
# "8 decimal places".
DISPLAY_DECIMALS: int = 8

# Default interest-rate curve type: "flat" uses the user-defined flat rate,
# "treasury" bootstraps the US Treasury yield curve. Section 1, step 11
# "Price": "Rate Flat or a curve (Index Type -> Fetch Curve)".
DEFAULT_RATE_TYPE: str = "flat"

# Flat annual risk-free rate (continuous compounding, decimal). Not specified
# in the paper; the user enters it at pricing time.
DEFAULT_FLAT_RATE: float = 0.045

# Treasury constant-maturity series on FRED, in years. Section 1, step 11
# mentions a 30-year US Treasury curve; shorter tenors anchor the short end.
TREASURY_TENORS_YEARS: tuple[float, ...] = (
    1.0 / 12.0, 3.0 / 12.0, 6.0 / 12.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0
)
TREASURY_FRED_SERIES: tuple[str, ...] = (
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7",
    "DGS10", "DGS20", "DGS30",
)

# -----------------------------------------------------------------------------
# Synthetic product parameters (capital protection, no coupons)
# -----------------------------------------------------------------------------

# Nominal (par value) of the note, in the issue currency. Section 1, step 7
# "Structure - basics": "Par Value".
SYNTHETIC_PAR_VALUE: float = 100.0

# Time to maturity in years. The user picks Issue / Maturity dates (Section 1,
# step 7). A three-year note is used as a typical capital-protected tenor.
SYNTHETIC_MATURITY_YEARS: float = 3.0

# Capital protection level, fraction of par returned at maturity regardless of
# basket performance (Section 1, step 9 "Structure - Synthetic").
SYNTHETIC_PROTECTION: float = 1.00

# Participation rate in the upside of the basket. Section 4 "Upside
# participation and leverage": discussed but formula not given. The standard
# capital-protected-note formula is used: payoff = par * (protection +
# participation * max(0, basket_return)).
SYNTHETIC_PARTICIPATION: float = 1.00

# Target annual yield used by the "Find barrier" helper (Section 1, step 9).
SYNTHETIC_TARGET_YIELD: float = 0.04

# -----------------------------------------------------------------------------
# Autocall product parameters (yield generating)
# -----------------------------------------------------------------------------

# Time to maturity in years. Section 1, step 7 "Structure - basics".
AUTOCALL_MATURITY_YEARS: float = 2.0

# Observation frequency in months. Section 1, step 10 "Structure - Autocall":
# "Memory Feature, observation frequency or count, Autocall Barrier, Coupon
# Barrier, Coupon Rate".
AUTOCALL_OBS_FREQ_MONTHS: int = 3

# Autocall barrier as fraction of the initial spot of the single underlying.
# Section 1, step 7 "Structure - basics": "Autocall, a single name".
AUTOCALL_BARRIER: float = 1.00

# Coupon barrier as fraction of initial spot.
AUTOCALL_COUPON_BARRIER: float = 0.70

# European downside protection barrier at maturity, fraction of initial spot.
AUTOCALL_PROTECTION_BARRIER: float = 0.60

# Per-observation coupon rate (fraction of par). Section 1, step 10.
AUTOCALL_COUPON_RATE: float = 0.02

# Memory feature flag. Section 1, step 10.
AUTOCALL_MEMORY: bool = True

# Par value of the autocall note.
AUTOCALL_PAR_VALUE: float = 100.0

# -----------------------------------------------------------------------------
# Experiment 2 — Historical backtest and stress testing
# -----------------------------------------------------------------------------

# Front-end sales charge, decimal of initial principal. Section 1, step 12
# "Backtest": "Sales Change" (costs line). Not specified numerically.
SALES_CHARGE: float = 0.005

# Stress-test scenarios. Section 1, step 12 "Backtest": "stress test
# (No / Coupon Bumps)". The paper mentions two scenarios; the "Bump" scenario
# is interpreted as a one-off instantaneous shock applied to prices.
STRESS_BUMP_PCT: float = -0.20  # -20% price shock on the stress date
STRESS_BUMP_DAY_FRACTION: float = 0.5  # shock placed at the middle of the window

# Bank savings benchmark annual rate (Section 1, step 12 "Backtest": "Sensitive
# Change with a bank note"). Default fallback when the FRED savings series is
# unavailable.
BANK_BENCHMARK_RATE: float = 0.03

# Historical VaR confidence level (Section 1, step 31: "Returns, replication
# or hedge accuracy, risks and costs").
VAR_CONFIDENCE: float = 0.95

# Window for "gap risk" calculation, in trading days. Interpreted as the worst
# overnight loss observed over this trailing window.
GAP_RISK_WINDOW: int = 252

# -----------------------------------------------------------------------------
# Experiment 3 — Dynamic delta hedging and fractional rebalancing
# -----------------------------------------------------------------------------

# Rebalance tolerance, percentage drift of actual vs target exposure. Section 4
# "Hedge profile": "Rebalance Tolerance (8 or 10)". The unit is not specified;
# the paper comments (Section 4) list it alongside Commission and Slippage,
# which are percentages, so a percentage is used here.
REBALANCE_TOLERANCE_PCT: float = 0.08

# Commission rate per traded notional (decimal). Section 4 "Hedge profile".
COMMISSION_RATE: float = 0.0005

# Slippage rate per traded notional (decimal). Section 4 "Hedge profile".
SLIPPAGE_RATE: float = 0.0002

# Short borrow annualized rate (decimal). Section 4 "Hedge profile".
SHORT_BORROW_RATE: float = 0.01

# Margin rate (decimal). Section 4 "Hedge profile".
MARGIN_RATE: float = 0.06

# Wash-sale window in trading days (IRS 30-calendar-day rule mapped to roughly
# 21 trading days); applied as the lookback for detecting wash sales when the
# Wash Sale Tax flag is enabled in the Hedge profile.
WASH_SALE_WINDOW_DAYS: int = 21

# Short-term capital-gains tax rate used when wash-sale-adjusted loss is
# realized. Not specified in the paper.
CAP_GAINS_TAX_RATE: float = 0.25
