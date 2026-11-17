from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "trading_setup.db"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


@dataclass
class Settings:
    """Application and strategy configuration."""

    dhan_client_id: str = ""
    dhan_access_token: str = ""
    dhan_api_key: str = ""
    dhan_api_secret: str = ""
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    # Risk & Small-SL Guardrails
    paper_initial_capital: float = 100_000.0
    default_risk_per_trade_inr: float = 1_000.0
    max_sl_pct_intraday: float = 0.65
    max_sl_pct_swing: float = 1.80
    min_rr_ratio: float = 2.0
    auto_paper_trade: bool = False
    auto_paper_min_grade: str = "A+"
    max_open_paper_positions: int = 6
    slippage_pct: float = 0.03
    brokerage_per_order: float = 20.0

    # Scanner defaults
    scan_universe_limit: int = 40
    watchlist_symbols: list[str] = field(
        default_factory=lambda: [
            "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
            "SBIN", "AXISBANK", "KOTAKBANK", "BHARTIARTL", "LT",
            "BAJFINANCE", "TATAMOTORS", "TATASTEEL", "SUNPHARMA", "MARUTI",
            "ADANIENT", "ADANIPORTS", "TITAN", "TRENT", "HAL",
            "BEL", "PFC", "RECLTD", "INDUSINDBK", "JSWSTEEL",
            "HINDALCO", "TECHM", "WIPRO", "HCLTECH", "M&M",
        ]
    )

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ENV_PATH, override=True)
        client_id = (os.getenv("DHAN_CLIENT_ID") or "").strip()
        access_token = (
            os.getenv("DHAN_ACCESS_TOKEN")
            or os.getenv("DHAN_API_TOKEN")
            or ""
        ).strip()
        api_key = (os.getenv("DHAN_APP_ID") or os.getenv("DHAN_API_KEY") or "").strip()
        api_secret = (os.getenv("DHAN_APP_SECRET") or os.getenv("DHAN_API_SECRET") or "").strip()
        return cls(
            dhan_client_id=client_id,
            dhan_access_token=access_token,
            dhan_api_key=api_key,
            dhan_api_secret=api_secret,
            app_host=os.getenv("APP_HOST", "127.0.0.1"),
            app_port=int(os.getenv("APP_PORT", "8000")),
            paper_initial_capital=float(os.getenv("PAPER_INITIAL_CAPITAL_INR", "100000")),
            default_risk_per_trade_inr=float(os.getenv("DEFAULT_RISK_PER_TRADE_INR", "1000")),
            max_sl_pct_intraday=float(os.getenv("MAX_SL_PCT_INTRADAY", "0.65")),
            max_sl_pct_swing=float(os.getenv("MAX_SL_PCT_SWING", "1.80")),
            min_rr_ratio=float(os.getenv("MIN_RR_RATIO", "2.0")),
            auto_paper_trade=os.getenv("AUTO_PAPER_TRADE", "false").lower() == "true",
        )


settings = Settings.from_env()
