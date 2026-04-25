"""Unit tests for backtest/transaction_costs.py"""
import pytest
from backtest.transaction_costs import calculate_slippage, calculate_regulatory_fees, round_trip_cost


class TestCalculateSlippage:
    def test_base_plus_default_spread(self):
        # No bid-ask provided → default 0.1% spread → 0.5×0.1% = 0.05% spread cost
        # Total = 0.05% base + 0.05% spread = 0.10%
        slip = calculate_slippage(price=100.0, volume=100, avg_daily_volume=5_000_000)
        assert abs(slip - 0.001) < 1e-9

    def test_custom_spread(self):
        # Spread = 0.2% → spread cost = 0.5 × 0.002 = 0.001 → total = 0.0005 + 0.001 = 0.0015
        slip = calculate_slippage(price=100.0, volume=100, avg_daily_volume=5_000_000,
                                  bid_ask_spread_pct=0.002)
        assert abs(slip - 0.0015) < 1e-9

    def test_no_market_impact_below_threshold(self):
        # 1000 shares out of 500_000 ADV → participation = 0.2% < 1% threshold
        slip = calculate_slippage(price=50.0, volume=1_000, avg_daily_volume=500_000)
        assert slip == 0.001  # base + default spread only

    def test_market_impact_triggered(self):
        # 100_000 shares out of 500_000 ADV → participation = 20% > 1%
        # market_impact = 0.1 × 0.20 = 0.02 → total would be 0.001 + 0.02 = 0.021 → capped at 0.02
        slip = calculate_slippage(price=50.0, volume=100_000, avg_daily_volume=500_000)
        assert slip == 0.02  # capped

    def test_market_impact_partial(self):
        # 10_000 out of 500_000 → participation = 2% > 1%
        # impact = 0.1 × 0.02 = 0.002 → total = 0.001 + 0.002 = 0.003
        slip = calculate_slippage(price=50.0, volume=10_000, avg_daily_volume=500_000)
        assert abs(slip - 0.003) < 1e-9

    def test_zero_avg_daily_volume(self):
        # Defensive: ADV = 0 should not raise, no market impact applied
        slip = calculate_slippage(price=50.0, volume=100, avg_daily_volume=0)
        assert slip == 0.001

    def test_cap_at_two_percent(self):
        slip = calculate_slippage(price=1.0, volume=10_000_000, avg_daily_volume=1_000)
        assert slip == 0.02

    def test_returns_float(self):
        slip = calculate_slippage(100.0, 100, 1_000_000)
        assert isinstance(slip, float)


class TestCalculateRegulatoryFees:
    def test_buy_side_no_fees(self):
        fee = calculate_regulatory_fees(price=150.0, qty=100, side="buy")
        assert fee == 0.0

    def test_sell_sec_and_taf(self):
        # price=150, qty=100 → trade_value=$15,000
        # SEC = 0.0000229 × 15000 = $0.3435
        # TAF = min(0.000166 × 100, 8.30) = min($0.0166, $8.30) = $0.0166
        # Total = $0.36 / $15,000 ≈ 0.0000240
        fee = calculate_regulatory_fees(price=150.0, qty=100, side="sell")
        expected = (0.0000229 * 15_000 + min(0.000166 * 100, 8.30)) / 15_000
        assert abs(fee - expected) < 1e-10

    def test_taf_cap(self):
        # qty=100_000 → TAF = 0.000166 × 100_000 = $16.60 > cap $8.30 → capped at $8.30
        fee = calculate_regulatory_fees(price=10.0, qty=100_000, side="sell")
        trade_value = 10.0 * 100_000
        expected_taf = 8.30
        expected_sec = 0.0000229 * trade_value
        expected = (expected_sec + expected_taf) / trade_value
        assert abs(fee - expected) < 1e-10

    def test_sell_case_insensitive(self):
        fee_lower = calculate_regulatory_fees(50.0, 100, "sell")
        fee_upper = calculate_regulatory_fees(50.0, 100, "SELL")
        assert fee_lower == fee_upper

    def test_zero_trade_value(self):
        fee = calculate_regulatory_fees(price=0.0, qty=0, side="sell")
        assert fee == 0.0


class TestRoundTripCost:
    def test_higher_than_one_way(self):
        one_way = calculate_slippage(100.0, 100, 5_000_000)
        rt = round_trip_cost(100.0, 100, 5_000_000)
        assert rt > one_way

    def test_includes_sell_fees(self):
        slip = calculate_slippage(100.0, 100, 5_000_000)
        fees = calculate_regulatory_fees(100.0, 100, "sell")
        expected = slip * 2 + fees
        assert abs(round_trip_cost(100.0, 100, 5_000_000) - expected) < 1e-12
