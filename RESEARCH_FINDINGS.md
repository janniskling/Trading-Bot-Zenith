# Zenith Trading Bot — Research Findings

## Project Goal

Build a trading bot that paper-trades a Triple-Confirmation EMA strategy
on US stocks/ETFs and benchmark performance against MSCI World (URTH)
over a 3-4 week observation period.

**Status:** Lernprojekt (learning project), not intended for live capital.

## Hypothesis (Original)

A combination of EMA-9/21 crossover, EMA-50 trend filter, volume confirmation,
RSI momentum filter, and ADX trend strength would produce alpha over MSCI World.

## Methodology

- Walk-Forward Backtesting framework with 2-year IS / 6-month OOS windows
- Realistic transaction cost model (slippage, SEC/TAF fees)
- Correlation filter (60-day rolling, threshold 0.70)
- Sector cap (max 4 positions per GICS sector)
- Anti-curve-fitting guard with parameter change tracking
- Test period: 2022–2024 (11 OOS windows)
- Universe: ~80 US stocks/ETFs across 8 sectors

## Results

### 6-Year Backtest (2022–2024 OOS)

| Metric | Baseline | +Costs | +Costs+Corr |
|--------|----------|--------|-------------|
| Sharpe | 0.84 | -0.36 | -0.34 |
| Beat URTH Rate | 18% | 18% | 18% |
| Win Rate | 15% | 10% | 11% |
| Total Trades | 223 | 223 | 221 |
| Mean Alpha vs URTH | -4.6% | -4.6% | -4.6% |

### Key Findings

**1. Strategy has no reliable edge over URTH buy-and-hold.**
Mean alpha of -4.6% over 11 OOS windows. Beat-URTH rate of 18% is
significantly below the 50% null hypothesis baseline.

**2. Transaction costs alone destroy the apparent edge.**
Baseline Sharpe 0.84 → with realistic costs Sharpe -0.36.
A drop of 1.20 Sharpe points indicates the "edge" was an artifact
of unrealistic backtest assumptions.

**3. Macro filter (SPY > EMA-200) works as designed.**
2022 bear market windows correctly produced 0 trades, preserving capital.

**4. Correlation filter low impact at this signal frequency.**
Only 23 firing events across 223 trades — signals are too sparse
for correlation clustering to be a major factor.

## What This Means

The empirical result is consistent with academic literature on
technical-indicator-based strategies in liquid US equities:
**no persistent alpha after costs.**

This is not a failure of implementation — the bot is working correctly.
It is a confirmation that EMA-crossover strategies on daily bars in
liquid US stocks do not produce edge in the modern market structure.

## Lessons Learned

1. **Walk-Forward validation is critical.** Without OOS testing, the
   baseline Sharpe of 0.84 would have looked promising.

2. **Transaction cost modeling reveals true edge.** The 1.20 Sharpe
   delta from costs alone is the diagnostic signature of a no-edge strategy.

3. **Filter funnel diagnostics are essential.** Discovered that ADX
   checked at exact crossover day was structurally broken — fixed with
   rolling lookback.

4. **Curve-fitting discipline matters.** Anti-curve-fitting guard
   prevented multiple temptations to "tune until profitable" —
   which would have produced a worse out-of-sample result.

## Next Phase: Live Paper-Trading Observation

Despite negative backtest alpha, the bot will run on Alpaca Paper
for 3-4 weeks to observe live behavior. Goals:

- Validate infrastructure under live market conditions
- Compare paper performance vs URTH in real-time
- Document any divergence between backtest and live results

**This is for learning, not for live capital deployment.**

## What Would Be Needed for a Strategy with Real Edge

Based on this research, a viable retail-quant strategy would likely require:

- Different signal source (factor investing, earnings drift, etc.)
- Different timeframe (intraday or weekly, not daily)
- Or: different asset class (crypto, options) with less efficient markets

For now, this codebase serves as a quantitative research framework
that can be applied to other strategy hypotheses.
