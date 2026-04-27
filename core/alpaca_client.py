from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    OrderRequest,
)
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config.settings import (
    ALPACA_API_KEY,
    ALPACA_BASE_URL,
    ALPACA_SECRET_KEY,
    DRY_RUN,
)


@dataclass
class Order:
    id: str
    symbol: str
    qty: float
    side: str
    status: str
    filled_avg_price: float | None = None


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float
    unrealized_plpc: float
    market_value: float


class AlpacaClient:
    def __init__(self) -> None:
        paper = "paper-api" in ALPACA_BASE_URL
        self._trading = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=paper,
        )
        self._data = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
        )

    def get_account(self) -> dict:
        acct = self._trading.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "day_trade_count": acct.daytrade_count,
            "portfolio_value": float(acct.portfolio_value),
        }

    def get_positions(self) -> list[Position]:
        raw = self._trading.get_all_positions()
        return [
            Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc),
                market_value=float(p.market_value),
            )
            for p in raw
        ]

    def get_bars(
        self, symbol: str, timeframe: str = "1Day", limit: int = 60
    ) -> pd.DataFrame:
        tf_map = {
            "1Day": TimeFrame.Day,
            "1Hour": TimeFrame.Hour,
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        }
        tf = tf_map.get(timeframe, TimeFrame.Day)
        end = datetime.now(tz=timezone.utc)
        # Fetch enough calendar days to account for weekends/holidays
        start = end - timedelta(days=limit * 2)

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        bars = self._data.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        df = df.sort_index().tail(limit).copy()
        df.columns = [c.lower() for c in df.columns]
        return df

    def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> Order:
        if DRY_RUN:
            print(
                f"[DRY RUN] BRACKET {symbol} qty={qty} entry={entry_price:.2f} "
                f"sl={stop_loss:.2f} tp={take_profit:.2f}"
            )
            return Order(id="dry-run", symbol=symbol, qty=qty, side="buy", status="dry_run")

        # Validate: only US equities (no options)
        asset = self._trading.get_asset(symbol)
        if asset.asset_class != AssetClass.US_EQUITY:
            raise ValueError(f"{symbol} is not a US equity — options not allowed")

        req = OrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=round(entry_price, 2),
            order_class="bracket",
            stop_loss={"stop_price": round(stop_loss, 2)},
            take_profit={"limit_price": round(take_profit, 2)},
        )
        order = self._trading.submit_order(req)
        return Order(
            id=str(order.id),
            symbol=order.symbol,
            qty=float(order.qty),
            side=order.side.value,
            status=order.status.value,
        )

    def submit_market_sell(self, symbol: str, qty: float) -> Order:
        if DRY_RUN:
            print(f"[DRY RUN] MARKET SELL {symbol} qty={qty}")
            return Order(id="dry-run", symbol=symbol, qty=qty, side="sell", status="dry_run")
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self._trading.submit_order(req)
        return Order(
            id=str(order.id),
            symbol=order.symbol,
            qty=float(order.qty),
            side="sell",
            status=order.status.value,
        )

    def cancel_all_orders(self) -> None:
        if DRY_RUN:
            print("[DRY RUN] cancel_all_orders")
            return
        self._trading.cancel_orders()

    def close_position(self, symbol: str) -> None:
        if DRY_RUN:
            print(f"[DRY RUN] close_position {symbol}")
            return
        self._trading.close_position(symbol)

    def get_open_orders(self) -> list[Order]:
        req = GetOrdersRequest(status="open")
        orders = self._trading.get_orders(filter=req)
        return [
            Order(
                id=str(o.id),
                symbol=o.symbol,
                qty=float(o.qty),
                side=o.side.value,
                status=o.status.value,
            )
            for o in orders
        ]

    def get_clock(self) -> dict:
        clock = self._trading.get_clock()
        return {
            "is_open": clock.is_open,
            "next_open": clock.next_open,
            "next_close": clock.next_close,
        }

    def get_calendar(self, start: date, end: date) -> list[dict]:
        from alpaca.trading.requests import GetCalendarRequest
        req = GetCalendarRequest(start=start, end=end)
        cal = self._trading.get_calendar(filters=req)
        return [{"date": c.date, "open": c.open, "close": c.close} for c in cal]

    def get_daily_activity(self) -> list[dict]:
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        today = date.today().isoformat()
        try:
            activities = self._trading.get_activities()
            today_fills = [
                a for a in activities
                if hasattr(a, "transaction_time")
                and str(a.transaction_time).startswith(today)
            ]
            return [
                {
                    "symbol": a.symbol,
                    "side": a.side.value if hasattr(a.side, "value") else str(a.side),
                    "qty": float(a.qty),
                    "price": float(a.price),
                }
                for a in today_fills
            ]
        except Exception:
            return []

    def get_position_entry_date(self, symbol: str) -> date | None:
        """Returns the date the current position was most recently opened via a filled buy order."""
        try:
            req = GetOrdersRequest(status="closed", symbols=[symbol], side=OrderSide.BUY, limit=10)
            orders = self._trading.get_orders(filter=req)
            filled = [o for o in orders if str(o.status) == "filled" and o.filled_at]
            if not filled:
                return None
            latest = max(filled, key=lambda o: o.filled_at)
            return latest.filled_at.date()
        except Exception:
            return None

    def wait_for_market_open(self, max_wait_seconds: int = 300) -> bool:
        for _ in range(max_wait_seconds // 10):
            clock = self.get_clock()
            if clock["is_open"]:
                return True
            print("Market not open yet, waiting 10 seconds...")
            time.sleep(10)
        return False
