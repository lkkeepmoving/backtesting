"""
Schwab Historical Data Fetcher
Fetches daily OHLCV data and saves to data/storage/{SYMBOL}.csv
"""

import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Allow imports from the data/ directory when run directly
sys.path.insert(0, str(Path(__file__).parent))
from schwab_client import SchwabClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


def get_client() -> SchwabClient:
    app_key = os.getenv("SCHWAB_APP_KEY")
    secret = os.getenv("SCHWAB_SECRET")
    callback_url = os.getenv("SCHWAB_CALLBACK_URL")

    if not all([app_key, secret, callback_url]):
        raise ValueError("SCHWAB_APP_KEY, SCHWAB_SECRET, and SCHWAB_CALLBACK_URL must be set in .env")

    return SchwabClient(
        client_id=app_key,
        client_secret=secret,
        redirect_uri=callback_url,
    )


def fetch_and_save(symbol: str, start_date: str, end_date: str):
    """
    Fetch daily OHLCV data for a symbol and save to data/storage/{SYMBOL}.csv

    Args:
        symbol:     Ticker (e.g. "TSM")
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"
    """
    client = get_client()

    logger.info(f"Fetching {symbol} from {start_date} to {end_date}...")
    df = client.get_price_history(symbol, start_date, end_date)

    if df is None or df.empty:
        logger.error(f"No data returned for {symbol}.")
        return

    storage_dir = Path(__file__).parent / "storage"
    storage_dir.mkdir(exist_ok=True)

    out_path = storage_dir / f"{symbol}.csv"
    df.to_csv(out_path, index=False)

    logger.info(f"Saved {len(df)} rows to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    fetch_and_save("TSM", "2026-02-10", "2026-02-15")
