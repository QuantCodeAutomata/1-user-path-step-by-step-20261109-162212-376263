"""Dynamic delta hedging and fractional rebalancing simulation.

Reproduces the Delta Hub workflow (Section 1, steps 23-28) and Hedge profile
(Section 4) of the SmartDelta user path:

* a target delta exposure per asset is aggregated across live products;
* the drift to the actual held position is measured;
* a rebalancing order is generated when the drift exceeds a tolerance;
* fractional shares are produced to 8 decimal places (Section 4);
* execution frictions (commission, slippage, short borrow, margin, taxes)
  are deducted.

All quantities are carried as :mod:`decimal` Decimals with a 10-decimal context
so that the mandatory 8-decimal precision is never lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Iterable

import pandas as pd

from .config import (
    CAP_GAINS_TAX_RATE,
    COMMISSION_RATE,
    DISPLAY_DECIMALS,
    MARGIN_RATE,
    REBALANCE_TOLERANCE_PCT,
    SHORT_BORROW_RATE,
    SLIPPAGE_RATE,
    WASH_SALE_WINDOW_DAYS,
)

# Keep enough precision to carry 8-decimal quantities through intermediate math.
getcontext().prec = 28

Q8 = Decimal("0.00000001")  # 8 decimal places (Section 4 "8 decimal places")


def d(value) -> Decimal:
    """Convert ``value`` to :class:`Decimal` safely from ``float`` or ``str``."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def round8(value: Decimal) -> Decimal:
    """Round ``value`` to 8 decimal places using banker-safe HALF_UP."""
    return value.quantize(Q8, rounding=ROUND_HALF_UP)


@dataclass
class HedgeProfile:
    """Hedge profile as described in Section 4 "Hedge profile"."""

    commission_rate: Decimal = field(default_factory=lambda: d(COMMISSION_RATE))
    slippage_rate: Decimal = field(default_factory=lambda: d(SLIPPAGE_RATE))
    short_borrow_rate: Decimal = field(default_factory=lambda: d(SHORT_BORROW_RATE))
    margin_rate: Decimal = field(default_factory=lambda: d(MARGIN_RATE))
    rebalance_tolerance: Decimal = field(default_factory=lambda: d(REBALANCE_TOLERANCE_PCT))
    wash_sale_tax: bool = True
    cap_gains_tax_rate: Decimal = field(default_factory=lambda: d(CAP_GAINS_TAX_RATE))
    pre_approved: bool = False


@dataclass
class AssetState:
    """Live state of an asset in the Delta Hub."""

    asset: str
    target_qty: Decimal
    actual_qty: Decimal
    price: Decimal
    short_borrow_available: bool = True

    @property
    def drift_qty(self) -> Decimal:
        return self.target_qty - self.actual_qty

    @property
    def drift_pct(self) -> Decimal:
        if self.target_qty == 0:
            return abs(self.drift_qty) if self.actual_qty == 0 else Decimal("Infinity")
        return abs(self.drift_qty) / abs(self.target_qty)


@dataclass
class TradeExecution:
    """Executed rebalancing trade with all frictions applied."""

    date: pd.Timestamp
    asset: str
    quantity: Decimal           # positive = buy, negative = sell
    price: Decimal              # clean price before slippage
    executed_price: Decimal     # price after slippage
    commission: Decimal
    slippage_cost: Decimal
    borrow_fee: Decimal
    tax: Decimal
    cash_impact: Decimal        # negative = cash out
    wash_sale: bool
    status: str                 # "executed", "blocked_borrow"

    def as_dict(self) -> dict:
        return {
            "date": self.date,
            "asset": self.asset,
            "quantity": format(round8(self.quantity), "f"),
            "price": format(round8(self.price), "f"),
            "executed_price": format(round8(self.executed_price), "f"),
            "commission": format(round8(self.commission), "f"),
            "slippage_cost": format(round8(self.slippage_cost), "f"),
            "borrow_fee": format(round8(self.borrow_fee), "f"),
            "tax": format(round8(self.tax), "f"),
            "cash_impact": format(round8(self.cash_impact), "f"),
            "wash_sale": self.wash_sale,
            "status": self.status,
        }


def needs_rebalance(state: AssetState, profile: HedgeProfile) -> bool:
    """True when the drift exceeds the configured tolerance."""
    return state.drift_pct > profile.rebalance_tolerance


def check_wash_sale(
    asset: str,
    now: pd.Timestamp,
    quantity: Decimal,
    current_price: Decimal,
    history: Iterable[TradeExecution],
    window_days: int = WASH_SALE_WINDOW_DAYS,
) -> bool:
    """Wash-sale rule: a loss-making sell matched by a buy on the same asset
    within ``window_days`` trading days is treated as a wash sale.
    """
    if quantity >= 0:  # only sells trigger a wash sale
        return False
    for t in history:
        if t.asset != asset:
            continue
        if t.quantity <= 0:
            continue
        if (now - t.date).days <= window_days and current_price < t.executed_price:
            return True
    return False


def execute_rebalance(
    date: pd.Timestamp,
    state: AssetState,
    profile: HedgeProfile,
    history: Iterable[TradeExecution],
    holding_days: int = 1,
) -> TradeExecution | None:
    """Simulate a rebalancing trade.

    Returns ``None`` when no trade is needed. Returns a ``TradeExecution`` with
    status ``"blocked_borrow"`` when a short borrow is required but unavailable.
    """
    if not needs_rebalance(state, profile):
        return None

    qty = round8(state.drift_qty)
    price = state.price
    is_sell_short = qty < 0 and state.actual_qty + qty < 0
    if is_sell_short and not state.short_borrow_available:
        return TradeExecution(
            date=date,
            asset=state.asset,
            quantity=Decimal("0"),
            price=price,
            executed_price=price,
            commission=Decimal("0"),
            slippage_cost=Decimal("0"),
            borrow_fee=Decimal("0"),
            tax=Decimal("0"),
            cash_impact=Decimal("0"),
            wash_sale=False,
            status="blocked_borrow",
        )

    slippage_sign = Decimal("1") if qty > 0 else Decimal("-1")
    executed_price = price * (Decimal("1") + slippage_sign * profile.slippage_rate)
    notional = abs(qty) * executed_price
    commission = notional * profile.commission_rate
    slippage_cost = abs(qty) * abs(executed_price - price)

    borrow_fee = Decimal("0")
    if is_sell_short:
        short_qty = abs(state.actual_qty + qty)
        borrow_fee = short_qty * price * profile.short_borrow_rate * Decimal(holding_days) / Decimal("365")

    wash = False
    tax = Decimal("0")
    if qty < 0 and profile.wash_sale_tax:
        wash = check_wash_sale(state.asset, date, qty, price, history)
        if not wash:
            avg_cost = _average_cost(state.asset, history)
            if avg_cost is not None and price > avg_cost:
                gain = (price - avg_cost) * abs(qty)
                tax = gain * profile.cap_gains_tax_rate

    cash_impact = -qty * executed_price - commission - borrow_fee - tax

    return TradeExecution(
        date=date,
        asset=state.asset,
        quantity=qty,
        price=price,
        executed_price=executed_price,
        commission=commission,
        slippage_cost=slippage_cost,
        borrow_fee=borrow_fee,
        tax=tax,
        cash_impact=cash_impact,
        wash_sale=wash,
        status="executed",
    )


def _average_cost(asset: str, history: Iterable[TradeExecution]) -> Decimal | None:
    buys = [t for t in history if t.asset == asset and t.quantity > 0 and t.status == "executed"]
    if not buys:
        return None
    total_qty = sum((t.quantity for t in buys), Decimal("0"))
    total_cost = sum((t.quantity * t.executed_price for t in buys), Decimal("0"))
    if total_qty == 0:
        return None
    return total_cost / total_qty


# -----------------------------------------------------------------------------
# Multi-step simulation
# -----------------------------------------------------------------------------


def simulate_hedging(
    prices: pd.DataFrame,
    target_qty: pd.DataFrame,
    initial_qty: dict[str, float] | None = None,
    profile: HedgeProfile | None = None,
) -> dict:
    """Walk-forward delta hedging simulation.

    Parameters
    ----------
    prices : DataFrame of daily prices indexed by date, one column per asset.
    target_qty : DataFrame of target delta quantities, same shape/columns.
    initial_qty : optional starting actual positions per asset (default 0).
    profile : hedge profile; defaults to the parameters in :mod:`src.config`.

    Returns
    -------
    dict with keys ``trades`` (DataFrame), ``positions`` (DataFrame),
    ``drift`` (DataFrame), ``cash`` (Series), ``summary`` (dict).
    """
    if profile is None:
        profile = HedgeProfile()
    if prices.shape != target_qty.shape:
        raise ValueError("prices and target_qty must have identical shapes")
    if not (prices.index == target_qty.index).all():
        raise ValueError("prices and target_qty must share the same index")

    assets = list(prices.columns)
    actual = {a: d(initial_qty.get(a, 0.0) if initial_qty else 0.0) for a in assets}
    history: list[TradeExecution] = []
    drift_log: list[dict] = []
    position_log: list[dict] = []
    cash_log: list[Decimal] = []
    cash = Decimal("0")

    for date, row in prices.iterrows():
        for a in assets:
            state = AssetState(
                asset=a,
                target_qty=d(target_qty.loc[date, a]),
                actual_qty=actual[a],
                price=d(row[a]),
            )
            drift_log.append({
                "date": date, "asset": a,
                "target_qty": float(round8(state.target_qty)),
                "actual_qty": float(round8(state.actual_qty)),
                "drift_qty": float(round8(state.drift_qty)),
                "drift_pct": float(min(state.drift_pct, Decimal("9.999999"))),
            })

            trade = execute_rebalance(date, state, profile, history)
            if trade is None:
                continue
            history.append(trade)
            if trade.status == "executed":
                actual[a] = round8(actual[a] + trade.quantity)
                cash += trade.cash_impact

        for a in assets:
            position_log.append({
                "date": date, "asset": a, "qty": float(round8(actual[a])),
                "price": float(round8(d(row[a]))),
                "mv": float(round8(actual[a] * d(row[a]))),
            })
        cash_log.append(cash)

    trades_df = pd.DataFrame([t.as_dict() for t in history])
    positions_df = pd.DataFrame(position_log)
    drift_df = pd.DataFrame(drift_log)
    cash_series = pd.Series([float(round8(c)) for c in cash_log], index=prices.index,
                            name="cash")

    total_mv = positions_df.groupby("date")["mv"].sum()
    portfolio_value = total_mv + cash_series.reindex(total_mv.index).fillna(0.0)

    summary = {
        "n_trades": int((trades_df["status"] == "executed").sum()) if not trades_df.empty else 0,
        "n_blocked_borrow": int((trades_df["status"] == "blocked_borrow").sum()) if not trades_df.empty else 0,
        "n_wash_sales": int(trades_df["wash_sale"].sum()) if not trades_df.empty else 0,
        "total_commission": float(trades_df["commission"].astype(float).sum()) if not trades_df.empty else 0.0,
        "total_slippage": float(trades_df["slippage_cost"].astype(float).sum()) if not trades_df.empty else 0.0,
        "total_borrow_fee": float(trades_df["borrow_fee"].astype(float).sum()) if not trades_df.empty else 0.0,
        "total_tax": float(trades_df["tax"].astype(float).sum()) if not trades_df.empty else 0.0,
        "final_cash": float(cash_series.iloc[-1]),
        "final_portfolio_value": float(portfolio_value.iloc[-1]),
        "decimals": DISPLAY_DECIMALS,
    }

    return {
        "trades": trades_df,
        "positions": positions_df,
        "drift": drift_df,
        "cash": cash_series,
        "portfolio_value": portfolio_value,
        "summary": summary,
    }
