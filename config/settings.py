from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# --- API Keys ---
ALPACA_API_KEY: str = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY: str = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL: str = "gemini-2.0-flash"

TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID: str = os.environ["TELEGRAM_CHAT_ID"]

# --- Mode ---
DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"

# --- Portfolio Rules ---
MAX_POSITION_PERCENT: float = 0.15    # Max 15% of portfolio per position
MAX_POSITIONS: int = 10
DEFAULT_STOP_LOSS_PCT: float = 0.03   # 3% stop-loss
DEFAULT_TAKE_PROFIT_PCT: float = 0.06 # 6% take-profit (2:1 reward/risk)
MAX_DAILY_LOSS_PCT: float = 0.05      # Circuit breaker: halt if down 5%
MAX_DAILY_TRADES: int = 5             # Max buys + sells combined per day
MIN_AVG_DAILY_VOLUME: int = 500_000   # Minimum liquidity filter

# --- Capital Deployment Ramp ---
MIN_CASH_TARGET_PCT: float = 0.20     # Target: keep ≤ 20% cash idle
TRADING_START_DATE: str = os.getenv("TRADING_START_DATE", "")

# --- EMA Strategy Parameters ---
EMA_FAST: int = 9
EMA_SLOW: int = 21
EMA_TREND: int = 50
MOMENTUM_LOOKBACK: int = 20
RSI_PERIOD: int = 14
VOLUME_LOOKBACK: int = 20
VOLUME_CONFIRMATION_MULTIPLIER: float = 1.1
MIN_RSI: float = 40.0
MAX_RSI_ENTRY: float = 75.0
MAX_RSI_EXIT: float = 80.0
MIN_BARS_REQUIRED: int = 60

# --- Strategy Improvements ---
MIN_ADX: float = 20.0            # Trend strength filter — only trade real trends
ATR_STOP_MULTIPLIER: float = 1.5 # Stop-loss = 1.5 × ATR(14), adapts to volatility
MAX_HOLD_DAYS: int = 10          # Exit stagnant positions after 10 trading days
SPY_TREND_FILTER: bool = True    # Only trade when SPY > EMA-200 (bull market)

# --- File Paths ---
BASE_DIR: Path = Path(__file__).parent.parent
KNOWLEDGE_DIR: Path = BASE_DIR / "knowledge"
DAILY_LOG_PATH: Path = KNOWLEDGE_DIR / "daily_log.md"
LEARNINGS_PATH: Path = KNOWLEDGE_DIR / "learnings.md"
WATCHLIST_PATH: Path = KNOWLEDGE_DIR / "watchlist.md"
LOGS_DIR: Path = BASE_DIR / "logs"

KNOWLEDGE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
