from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import (
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    MAX_DAILY_LOSS_PCT,
    MAX_DAILY_TRADES,
    MAX_POSITION_PERCENT,
    MAX_POSITIONS,
    MIN_CASH_TARGET_PCT,
    TRADING_START_DATE,
)
from core.alpaca_client import AlpacaClient, Position
from core.market_utils import count_trading_days_since


@dataclass
class PortfolioState:
    equity: float
    cash: float
    positions: list[Position]
    day_pnl: float
    day_pnl_pct: float
    open_symbols: set[str] = field(default_factory=set)
    trade_count_today: int = 0


class PortfolioManager:
    def __init__(self, client: AlpacaClient) -> None:
        self._client = client

    def get_portfolio_state(self) -> PortfolioState:
        account = self._client.get_account()
        positions = self._client.get_positions()
        equity = account["equity"]
        cash = account["cash"]

        # Day P&L: sum of unrealized + today's realized (approximated via equity change)
        unrealized_total = sum(p.unrealized_pl for p in positions)
        open_symbols = {p.symbol for p in positions}

        activity = self._client.get_daily_activity()
        trade_count = len(activity)

        # Approximate day P&L from account (alpaca provides last_equity)
        day_pnl = unrealized_total  # simplified; full impl would use portfolio history
        day_pnl_pct = (day_pnl / (equity - day_pnl)) * 100 if (equity - day_pnl) != 0 else 0

        return PortfolioState(
            equity=equity,
            cash=cash,
            positions=positions,
            day_pnl=day_pnl,
            day_pnl_pct=day_pnl_pct,
            open_symbols=open_symbols,
            trade_count_today=trade_count,
        )

    def can_trade(self, state: PortfolioState | None = None) -> bool:
        if state is None:
            state = self.get_portfolio_state()
        if state.trade_count_today >= MAX_DAILY_TRADES:
            print(f"Daily trade limit reached ({MAX_DAILY_TRADES})")
            return False
        if len(state.positions) >= MAX_POSITIONS:
            print(f"Max positions reached ({MAX_POSITIONS})")
            return False
        if state.day_pnl_pct <= -MAX_DAILY_LOSS_PCT * 100:
            print(f"Daily loss circuit breaker triggered ({state.day_pnl_pct:.2f}%)")
            return False
        return True

    def calculate_position_size(
        self, entry_price: float, stop_price: float, equity: float
    ) -> int:
        risk_per_trade = equity * 0.01  # risk 1% per trade
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return 0
        shares_by_risk = int(risk_per_trade / risk_per_share)

        max_value = equity * MAX_POSITION_PERCENT
        shares_by_cap = int(max_value / entry_price)

        qty = min(shares_by_risk, shares_by_cap)
        return max(1, qty)

    def get_stop_price(self, entry_price: float, atr_stop: float = 0.0) -> float:
        if atr_stop > 0 and atr_stop < entry_price:
            return atr_stop
        return round(entry_price * (1 - DEFAULT_STOP_LOSS_PCT), 2)

    def get_take_profit_price(self, entry_price: float, stop_price: float | None = None) -> float:
        if stop_price is not None:
            risk = entry_price - stop_price
            return round(entry_price + 2 * risk, 2)  # strict 2:1 R/R on actual stop
        return round(entry_price * (1 + DEFAULT_TAKE_PROFIT_PCT), 2)

    def get_target_position_count(self) -> int:
        days = count_trading_days_since(TRADING_START_DATE)
        if days <= 5:
            return 4
        if days <= 10:
            return 6
        return 8

    def is_underdeployed(self, state: PortfolioState) -> bool:
        cash_pct = state.cash / state.equity if state.equity else 0
        target = self.get_target_position_count()
        return len(state.positions) < target and cash_pct > MIN_CASH_TARGET_PCT

    def calculate_daily_pnl(self) -> dict:
        positions = self._client.get_positions()
        result = {p.symbol: p.unrealized_pl for p in positions}
        result["total"] = sum(result.values())
        return result

    def get_closed_trades_today(self) -> list[dict]:
        activity = self._client.get_daily_activity()
        sells = [a for a in activity if a.get("side") == "sell"]
        trades = []
        for s in sells:
            trades.append({
                "symbol": s["symbol"],
                "qty": s["qty"],
                "price": s["price"],
                "pnl": 0.0,      # Exact P&L requires tracking entry price
                "pnl_pct": 0.0,
            })
        return trades
