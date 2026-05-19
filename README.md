# Zenith Trading Bot

An algorithmic paper trading bot that implements a Triple-Confirmation EMA strategy on US stocks and ETFs. Built as a learning project to understand quantitative trading, backtesting methodology, and the real-world cost of technical indicator strategies.

## What it does

Zenith runs through structured daily market phases — premarket analysis, trade execution at market open, midday review, end-of-day close, and nightly reflection — using the Alpaca brokerage API for paper trading and Google Gemini AI for market commentary and research assistance.

The strategy combines:
- **EMA-9/21 crossover** as the primary signal
- **EMA-50 trend filter** to avoid counter-trend trades
- **RSI momentum filter** and **ADX trend strength** as confirmation gates
- **Volume confirmation** to filter low-conviction signals
- **Macro filter** (SPY > EMA-200) to sit out bear markets entirely

## Backtesting Framework

The project includes a Walk-Forward Backtesting engine with realistic transaction cost modeling (slippage, SEC/TAF fees), a correlation filter to reduce portfolio clustering, and a sector cap (max 4 positions per GICS sector). Tests covered 2022–2024 across ~80 US stocks/ETFs in 8 sectors.

**Result:** The strategy produces no reliable edge over MSCI World buy-and-hold after costs. Baseline Sharpe of 0.84 collapsed to -0.36 once realistic transaction costs were applied — a 1.20 Sharpe-point delta that confirms the "edge" was an artifact of cost-free backtesting assumptions.

This outcome is documented in [`RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md) and is considered a success: the infrastructure works correctly, and the empirical result is consistent with academic literature on EMA-based strategies in liquid US equities.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Brokerage API | Alpaca (paper trading) |
| AI Integration | Google Gemini |
| Data | yfinance, pandas, numpy |
| Config | YAML + .env |

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env  # add your Alpaca and Gemini API keys

python main.py --step premarket
python main.py --step market_open
python main.py --step midday
python main.py --step market_close
python main.py --step nightly_reflection
python main.py --step account_info   # quick connection test

python run_backtest.py               # run the backtesting suite
```

## Key Learnings

- Walk-forward validation is non-negotiable — in-sample results are meaningless without out-of-sample testing
- Transaction cost modeling is the single most important reality check for any strategy
- Filter funnel diagnostics revealed a structural bug (ADX checked at exact crossover day rather than via rolling lookback) — emphasizing the importance of per-signal auditing
- Anti-curve-fitting discipline prevented tuning toward a result, which would have made the OOS performance worse
