# Results — SmartDelta user path step by step

Paper: *SmartDelta — "1. User path step by step"* (product specification).

Every number in this file comes from an artefact written by the final run into
`results/`. Re-running an experiment overwrites its artefacts; the verdicts
below must then be checked against the fresh files and updated by hand.

## Experiment 1 — Monte Carlo pricing engine for Synthetic and Autocall structured products

Reproduces the pricing of the two product families described in Sections 1.7
to 1.11 and Section 4: Synthetic (capital protection) and Autocall
(yield-generating), with up to 10 underlyings, correlated geometric Brownian
motion, user-defined flat rate or bootstrapped US Treasury curve, and up to 5
European barriers at maturity.

Artefacts:

- `results/exp1_summary.json`      — configuration and all headline metrics
- `results/exp1_report.md`         — human-readable price, CI, yield, barrier
- `results/exp1_synthetic_payoff.png` — payoff visualisation
- `results/exp1_autocall_pv_hist.png` — PV distribution of the autocall note

Headline numbers (100,000 paths, flat rate, default basket; see
`results/exp1_summary.json`):

| Product     | Price per 100 | 95 % MC CI                | Expected annual yield | Prob. capital loss |
|-------------|---------------|---------------------------|-----------------------|--------------------|
| Synthetic   | 119.09244793  | [118.72102598, 119.46386988] | 0.10324329         | 0.00000000         |
| Autocall    |  99.33223323  | [ 99.25938413,  99.40508233] | expected life 0.77 y | 0.05627000         |

The paper document does not report headline numeric values for a specific
product; it describes the engine's inputs, methodology (100,000 paths,
Section 4), and the outputs that must be produced (price, expected yield,
risk metrics, payoff chart). The run produces every one of those outputs at
8-decimal-place precision (Section 4). Verdict: **reproduced** (structural
reproduction; the paper provides no reference numbers to compare against).

## Experiment 2 — Historical backtesting and stress testing of payoff strategies

Reproduces the Backtest workflow of Section 1.12: daily close prices for the
chosen basket over the backtest window, deduction of structural costs (sales
charge), stress scenarios ("No" / "Bump"), walk-forward mark-to-market,
tracking error vs. target exposure, VaR, gap risk, whip-saw cost, and a
Savings Benchmarking Ledger comparing the strategy to a bank note.

Artefacts:

- `results/exp2_summary.json`      — configuration and risk metrics
- `results/exp2_report.md`         — report
- `results/exp2_mtm_series.csv`    — daily mark-to-market series
- `results/exp2_mtm_vs_bank.png`   — strategy equity curve vs. bank benchmark
- `results/exp2_return_hist.png`   — daily-return distribution

Sample: 2022-01-03 .. 2024-12-31, 753 trading days. Headline numbers (see
`results/exp2_summary.json`):

| Metric                              | Value     |
|-------------------------------------|-----------|
| Initial value                       | 99.500000 |
| Terminal value                      | 194.099778 |
| Cumulative return                   | 0.950752  |
| Bank-note terminal (benchmark)      | 108.845753 |
| Excess vs. bank                     | 85.254025 |
| VaR 95 % (daily)                    | 0.018215  |
| Tracking error (annualised)         | 0.253290  |
| Max daily drift                     | 0.713777  |
| Gap risk                            | 0.056465  |
| Whip-saw cost (basket reversal)     | 10.368211 |
| Bump –20 % terminal                 | 155.279822 |
| Bump –10 % terminal                 | 174.689800 |
| Bump +10 % terminal                 | 213.509756 |

The paper document gives no numeric target for any metric on this screen. The
run writes every metric requested (tracking error, whip-saw cost, gap risk,
VaR, bank-benchmark comparison, stress scenarios). Verdict: **reproduced**
(structural reproduction).

## Experiment 3 — Dynamic delta hedging and fractional rebalancing simulation

Reproduces the Delta Hub rebalancing engine (Sections 1.23–1.28 and Section 4
"Hedge profile"): aggregated target delta per asset, drift vs. actual, orders
generated when the drift exceeds the tolerance, fractional shares at 8-decimal
precision, commission, slippage, short-borrow fee, margin, wash-sale and
capital-gains tax deductions, blocked trades when short borrow is unavailable.

Artefacts:

- `results/exp3_summary.json`      — configuration and execution summary
- `results/exp3_report.md`         — report
- `results/exp3_trades.csv`        — trade log at 8 decimal places
- `results/exp3_drift.csv`         — per-asset drift history
- `results/exp3_positions.csv`     — walk-forward positions
- `results/exp3_portfolio_value.png` — portfolio value over time
- `results/exp3_drift.png`         — worst drift vs. tolerance

Sample: 2022-01-03 .. 2024-12-31. Headline numbers (see
`results/exp3_summary.json`):

| Metric                              | Value         |
|-------------------------------------|---------------|
| Trades executed                     | 139           |
| Trades blocked (short borrow)       | 0             |
| Wash-sale trades flagged            | 22            |
| Total commission                    | 0.71456130    |
| Total slippage                      | 0.28582254    |
| Total short-borrow fees             | 0.00000000    |
| Total tax                           | 8.56082574    |
| Final cash                          | –60.30884779  |
| Final portfolio value               | 40.12611354   |

The paper document gives no numeric reference values; it specifies the
precision (8 decimal places), the frictions, and the flow. All of those are
produced. Verdict: **reproduced** (structural reproduction).

## Deviations from the paper

- The paper does not specify a numeric universe, sample period, flat rate,
  participation, barrier levels, coupon rate, autocall frequency, sales
  charge, tolerance, commission, slippage, short-borrow rate or tax rate.
  Each such parameter uses an explicit choice documented in `src/config.py`
  with the paper section that introduces it, or the note "not specified in
  the paper". Changing these values does not change the methodology.
- The paper's data source is not named precisely for the backtest ("Daily
  historical price data"); this implementation loads prices through the
  QuantX Dataloader (built on Massive) with the Yahoo backstop enabled. The
  `ticker_details` field of each load is recorded in the JSON summary, and
  any ticker whose `data_source` differs from the rest (for example Yahoo
  Finance) is listed there under `non_massive_source`.
- Section 1.12 mentions a "Sensitive Change with a bank note" with "Savings
  Benchmarking Ledger". The paper does not give a bank rate; the
  implementation uses a fixed annualised rate from `BANK_BENCHMARK_RATE`,
  compounded daily, as a transparent proxy. When FRED credentials are
  available the Savings Benchmarking Ledger can be fed by a FRED series via
  `src.rates`.
- "Rebalance Tolerance (8 or 10)" (Section 4) has no stated unit. It is
  interpreted as a fraction of the absolute target quantity, with 0.08
  (i.e. 8 %) chosen as the default, since Section 4 lists it alongside
  Commission and Slippage which are percentages.
- "Up to 5 barriers on the maturity date" (Section 4) is implemented as a list
  of European barriers with an explicit direction (±1), participation and cap;
  the paper does not specify a liquidation order, so barriers are applied
  additively on the terminal basket return.
- Wash-sale detection uses the IRS 30-calendar-day rule, approximated as 21
  trading days; the paper does not define a window.
- Treasury-curve bootstrap (Section 1.11) uses a continuous-compounding
  bisection against par-yield observations; the paper does not specify the
  bootstrap convention.

## Not reproduced

Nothing to report: the three experiments selected from the paper are each
implemented and produce the artefacts the paper requests.
