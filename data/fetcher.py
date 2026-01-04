"""
Historical Data Fetcher
Downloads daily bar data from TradeStation API and saves to local CSV files
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import List

from tradestation_client import TradeStationClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_symbols_from_file(file_path: str) -> List[str]:
    """
    Load symbols from CSV file

    Args:
        file_path: Path to CSV file with 'symbol' header

    Returns:
        List of symbols
    """
    symbols = []
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row.get('symbol', '').strip()
                if symbol:
                    symbols.append(symbol)

        logger.info(f"Loaded {len(symbols)} symbols from {file_path}")
        return symbols

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading symbols file: {e}")
        return []


def get_tradestation_config() -> dict:
    """
    Load TradeStation configuration from environment variables

    Returns:
        Config dict with api_key, api_secret, base_url
    """
    # Load .env file if it exists (check both current dir and parent dir)
    env_path = Path.cwd() / '.env'
    if not env_path.exists():
        # Try parent directory (project root)
        env_path = Path.cwd().parent / '.env'

    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key not in os.environ:
                        os.environ[key] = value

    api_key = os.environ.get('TRADESTATION_API_KEY')
    api_secret = os.environ.get('TRADESTATION_API_SECRET')
    base_url = os.environ.get('TRADESTATION_BASE_URL', 'https://api.tradestation.com/v3')

    if not api_key or not api_secret:
        raise ValueError(
            "Missing TradeStation credentials. Please set TRADESTATION_API_KEY and "
            "TRADESTATION_API_SECRET in .env file or environment variables"
        )

    return {
        'api_key': api_key,
        'api_secret': api_secret,
        'base_url': base_url
    }


def sanitize_filename(symbol: str) -> str:
    """
    Sanitize symbol name for use as filename
    Replaces special characters that might cause issues

    Args:
        symbol: Original symbol name (e.g., $VIX.X)

    Returns:
        Sanitized filename (e.g., VIX_X)
    """
    # Remove $ prefix and replace . with _
    filename = symbol.replace('$', '').replace('.', '_')
    return filename


def fetch_and_save_data(client: TradeStationClient, symbol: str, bar_count: int,
                        storage_dir: Path) -> bool:
    """
    Fetch data for a symbol and save to CSV

    Args:
        client: TradeStation API client
        symbol: Stock symbol
        bar_count: Number of bars to fetch
        storage_dir: Directory to save CSV files

    Returns:
        True if successful, False otherwise
    """
    try:
        # Fetch data
        logger.info(f"Fetching {bar_count} bars for {symbol}...")
        df = client.get_bars_by_bar_count(symbol, bar_count)

        if df is None or df.empty:
            logger.error(f"No data retrieved for {symbol}")
            return False

        # Sanitize symbol for filename
        sanitized_name = sanitize_filename(symbol)

        # Define output path
        output_path = storage_dir / f"{sanitized_name}.csv"

        # Remove old file if exists
        if output_path.exists():
            output_path.unlink()
            logger.debug(f"Removed old file: {output_path}")

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved {len(df)} bars to {output_path} (symbol: {symbol})")

        return True

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download historical daily bar data from TradeStation API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch data for symbols from core_symbols.csv (default 260 bars)
  python fetcher.py --symbols-file core_symbols.csv

  # Fetch 500 bars for symbols in file
  python fetcher.py --symbols-file core_symbols.csv --bars 500

  # Fetch data for specific symbols
  python fetcher.py --symbols AAPL MSFT TSLA

  # Fetch data for specific symbols with custom bar count
  python fetcher.py --symbols AAPL MSFT --bars 100
        """
    )

    # Symbol input options (mutually exclusive)
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        '--symbols-file',
        type=str,
        help='Path to CSV file with symbol list (header: symbol)'
    )
    symbol_group.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='List of symbols to fetch (e.g., AAPL MSFT TSLA)'
    )

    # Bar count option
    parser.add_argument(
        '--bars',
        type=int,
        default=260,
        help='Number of bars (trading days) to fetch (default: 260)'
    )

    args = parser.parse_args()

    # Get symbols
    if args.symbols_file:
        symbols = load_symbols_from_file(args.symbols_file)
        if not symbols:
            logger.error("No symbols loaded from file. Exiting.")
            sys.exit(1)
    else:
        symbols = args.symbols

    logger.info("=" * 60)
    logger.info("Historical Data Fetcher")
    logger.info("=" * 60)
    logger.info(f"Symbols to fetch: {len(symbols)}")
    logger.info(f"Bars per symbol: {args.bars}")
    logger.info("=" * 60)

    # Setup storage directory
    script_dir = Path(__file__).parent
    storage_dir = script_dir / 'storage'
    storage_dir.mkdir(exist_ok=True)

    # Initialize TradeStation client
    try:
        config = get_tradestation_config()
        client = TradeStationClient(config)
    except Exception as e:
        logger.error(f"Failed to initialize TradeStation client: {e}")
        sys.exit(1)

    # Fetch data for each symbol
    success_count = 0
    failed_symbols = []

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}")

        success = fetch_and_save_data(client, symbol, args.bars, storage_dir)

        if success:
            success_count += 1
        else:
            failed_symbols.append(symbol)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Successfully fetched: {success_count} symbols")
    logger.info(f"Failed: {len(failed_symbols)} symbols")

    if failed_symbols:
        logger.info(f"\nFailed symbols:")
        for symbol in failed_symbols:
            logger.info(f"  - {symbol}")

    logger.info(f"\nTotal bars per symbol: {args.bars}")
    logger.info(f"Storage location: {storage_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
