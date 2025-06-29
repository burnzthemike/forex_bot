import os
from typing import List, Dict, Union

# ─── Environment / .env Integration ─────────────────────────────────────────────
# Make sure you have a `.env` file in the project root with these entries:
#   TWELVE_DATA_API_KEY=your_twelvedata_key
#   NEWSAPI_KEY=your_newsapi_key
#   MT5_PATH="C:\Program Files\MetaTrader 5\terminal64.exe"
#   MT5_LOGIN=12345678
#   MT5_PASSWORD=your_mt5_password
#   MT5_SERVER=YourBrokerServer
#
# You can load these via python-dotenv or ensure they are exported in your shell.
from dotenv import load_dotenv
load_dotenv()

# === API Keys & Endpoints ===
TWELVE_DATA_API_KEY: str = os.getenv("TWELVE_DATA_API_KEY", "a16a29e3d9a34acca2fd3d7435e42a62")
NEWSAPI_KEY: str      = os.getenv("NEWSAPI_KEY", "617ca539a455482c9a08f204f7af4d47")

BASE_URL: str        = "https://api.twelvedata.com"
NEWSAPI_URL: str     = "https://newsapi.org/v2/everything"

# === Instruments & Data Settings ===
CURRENCY_PAIRS: List[str] = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "USD/CAD"]
INTERVAL: str             = "1min"
OUTPUT_SIZE: int          = 100

# === Cache Settings ===
CACHE_EXPIRY_SECONDS: int = 60
CACHE_DIR: str            = "cache"

# === Trading Mode Flags ===
PAPER_TRADING: bool = True
LIVE_TRADING: bool  = os.getenv("LIVE_TRADING", "False").lower() in ("1", "true", "yes")

# === MetaTrader 5 Settings ===
MT5_PATH: str    = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
MT5_LOGIN: int   = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD: str= os.getenv("MT5_PASSWORD", "")
MT5_SERVER: str  = os.getenv("MT5_SERVER", "")

# Map our “EUR/USD” style to broker symbol “EURUSD”
SYMBOLS_MAPPING: Dict[str, str] = {
    "EUR/USD": "EURUSD",
    "USD/JPY": "USDJPY",
    "GBP/USD": "GBPUSD",
    "USD/CHF": "USDCHF",
    "USD/CAD": "USDCAD",
}

# === Strategy Parameters ===
EMA_FAST: int      = 12
EMA_SLOW: int      = 26
POSITION_SIZE: float = 10_000.0
TOP_PAIR_COUNT: int  = 3

# === Risk Management Settings ===
MAX_DRAWDOWN: float      = 0.10
MAX_RISK_PER_TRADE: float= 0.02
STOP_LOSS_PCT: float     = 0.005
TAKE_PROFIT_PCT: float   = 0.01

# === Reinforcement Learning Hyperparameters ===
ALPHA: float         = 0.1
GAMMA: float         = 0.95
EPSILON_START: float = 0.5
EPSILON_MIN: float   = 0.05
EPSILON_DECAY: float = 0.995

# === Reward Shaping Parameters ===
REWARD_CLIP_MIN: float         = -1.0
REWARD_CLIP_MAX: float         = 1.0
REWARD_SMOOTHING_ALPHA: float  = 0.3
DRAWDOWN_PENALTY_WEIGHT: float = 2.0

# === Feature Analysis Window Lengths ===
VOLATILITY_WINDOW: int = 14
MOMENTUM_WINDOW: int   = 10
SENTIMENT_WINDOW: int  = 1

# === Dashboard Settings ===
DASHBOARD_PORT: int    = 8050
DASHBOARD_REFRESH: int = 60

# === API Retry Settings ===
MAX_RETRIES: int = 3
RETRY_WAIT: int  = 2  # seconds
TIMEOUT: int     = 10 # seconds

# === Logging & File Settings ===
LOG_FILE: str     = "trading.log"
EQUITY_LOG: str   = "equity.csv"

# === Optimization Grid ===
OPTIMIZATION_GRID: Dict[str, List[Union[int, float]]] = {
    "EMA_FAST": [8, 12, 16],
    "EMA_SLOW": [24, 26, 30],
    "POSITION_SIZE": [5_000, 10_000, 15_000],
    "MAX_RISK_PER_TRADE": [0.01, 0.02, 0.03]
}

# === Trend Scanner Weights ===
TREND_SCANNER_WEIGHTS: Dict[str, float] = {
    "atr_weight": 0.5,
    "adx_weight": 0.4,
    "sentiment_weight": 0.1
}

# === Execution Simulator Settings ===
SPREAD_PIPS: float         = 0.8
SLIPPAGE_PIPS: float       = 0.2
COMMISSION_PER_TRADE: float= 2.5

# === Pip Sizes ===
PAIR_PIP_SIZES: Dict[str, float] = {
    "USD/JPY": 0.01,
    "EUR/JPY": 0.01,
    "GBP/JPY": 0.01,
    # default for others is 0.0001
}

# === Trade Duration Constraints ===
MAX_HOLD_TIME_MINUTES: int = 120
