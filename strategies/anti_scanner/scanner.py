"""
Anti Pattern Scanner — Main CLI Entry Point

Scans historical daily price data for the Anti pattern:
  Phase 1: Divergence detection (trend exhaustion via RSI/MACD)
  Phase 2: Counter-trend shock detection (sigma spike / pct change + MACD new extreme)

Usage:
  python -m strategies.anti_scanner.scanner --date 2025-03-20 --lookback 252
  python -m strategies.anti_scanner.scanner --direction short --symbols AAPL,MSFT,NVDA
"""

import argparse
import csv
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'data'))

from indicators.rsi import compute_rsi_multiple
from indicators.modified_macd import compute_modified_macd, compute_macd_percentile
from strategies.anti_scanner.phase1 import run_phase1
from strategies.anti_scanner.phase2 import run_phase2
from strategies.anti_scanner.report import generate_report
from strategies.anti_scanner.csv_export import export_csv

logger = logging.getLogger(__name__)


def ensure_data(symbols: List[str], storage_dir: Path, bar_count: int) -> List[str]:
    """
    Check which symbols have local data. Fetch missing ones via TradeStation API.

    Args:
        symbols: List of stock tickers to check
        storage_dir: Path to data/storage/ directory
        bar_count: Number of bars to fetch for missing symbols

    Returns:
        List of symbols that have data available (local or freshly fetched)
    """
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Partition into sufficient vs needs-fetch
    # A symbol needs fetching if: no file, empty file, or too few bars
    min_bars = int(bar_count * 0.8)  # Allow some tolerance
    present = []
    missing = []
    for symbol in symbols:
        filename = sanitize_filename(symbol)
        csv_path = storage_dir / f"{filename}.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            try:
                row_count = sum(1 for _ in open(csv_path)) - 1  # subtract header
                if row_count >= min_bars:
                    present.append(symbol)
                else:
                    logger.info(f"{symbol}: only {row_count} bars locally (need {min_bars}+), will re-fetch")
                    missing.append(symbol)
            except Exception:
                missing.append(symbol)
        else:
            missing.append(symbol)

    if not missing:
        logger.info(f"All {len(present)} symbols have local data")
        return symbols

    print(f"\nData check: {len(present)} symbols cached, {len(missing)} need fetching")
    print(f"Missing: {', '.join(missing)}")
    print(f"Fetching {bar_count} bars per symbol from TradeStation API...")
    print()

    # Import TradeStation client (only when needed)
    try:
        from tradestation_client import TradeStationClient
        from fetcher import get_tradestation_config, fetch_and_save_data
    except ImportError as e:
        logger.error(f"Cannot import TradeStation modules: {e}")
        logger.error("Ensure data/tradestation_client.py and data/fetcher.py exist.")
        print(f"WARNING: Cannot fetch data. Skipping {len(missing)} symbols.")
        return present

    # Initialize client
    # The TradeStation client looks for .tradestation_tokens in os.getcwd().
    # fetcher.py is normally run from data/, so tokens live in data/.
    # Temporarily chdir so the client finds existing tokens.
    data_dir = storage_dir.parent
    original_cwd = os.getcwd()
    try:
        os.chdir(data_dir)
        config = get_tradestation_config()
        client = TradeStationClient(config)
    except Exception as e:
        logger.error(f"Failed to initialize TradeStation client: {e}")
        print(f"WARNING: Cannot authenticate with TradeStation. Skipping {len(missing)} symbols.")
        return present
    finally:
        os.chdir(original_cwd)

    # Fetch missing symbols
    fetched = []
    for i, symbol in enumerate(missing, 1):
        print(f"  Fetching [{i}/{len(missing)}] {symbol}...", end=" ")
        try:
            success = fetch_and_save_data(client, symbol, bar_count, storage_dir)
            if success:
                fetched.append(symbol)
                print("OK")
            else:
                print("FAILED")
        except Exception as e:
            print(f"ERROR: {e}")

        # Rate limit: brief pause between API calls
        if i < len(missing):
            time.sleep(0.5)

    print(f"\nFetched {len(fetched)}/{len(missing)} missing symbols")
    if len(fetched) < len(missing):
        skipped = set(missing) - set(fetched)
        print(f"Skipped: {', '.join(skipped)}")
    print()

    return present + fetched


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_symbols(symbols_str: str = None, symbols_file: str = None) -> List[str]:
    """
    Load symbol list from CLI arg or file.

    Args:
        symbols_str: Comma-separated symbols (takes priority)
        symbols_file: Path to CSV file with 'symbol' header

    Returns:
        List of symbol strings
    """
    if symbols_str:
        return [s.strip() for s in symbols_str.split(',') if s.strip()]

    if symbols_file:
        symbols = []
        with open(symbols_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row.get('symbol', '').strip()
                if symbol:
                    symbols.append(symbol)
        return symbols

    return []


def sanitize_filename(symbol: str) -> str:
    """Sanitize symbol for filename lookup (matches fetcher.py convention)."""
    return symbol.replace('$', '').replace('.', '_')


def load_price_data(symbol: str, storage_dir: Path) -> pd.DataFrame:
    """
    Load price data from local CSV storage.

    Args:
        symbol: Stock ticker
        storage_dir: Path to data/storage/ directory

    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume columns,
        sorted oldest to newest.

    Raises:
        FileNotFoundError: If CSV doesn't exist for this symbol
    """
    filename = sanitize_filename(symbol)
    csv_path = storage_dir / f"{filename}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"No data file for {symbol}: {csv_path}")

    df = pd.read_csv(csv_path)

    # Reverse to oldest-to-newest (storage is newest-to-oldest)
    df = df.iloc[::-1].reset_index(drop=True)

    return df


def find_date_index(dates: np.ndarray, target_date: str) -> int:
    """
    Find the index of target_date in the dates array, or the closest preceding date.

    Args:
        dates: Array of date strings (YYYY-MM-DD), sorted ascending
        target_date: Target date string

    Returns:
        Index of the matching or closest preceding date, or -1 if not found
    """
    for i in range(len(dates) - 1, -1, -1):
        if dates[i] <= target_date:
            return i
    return -1


def scan_symbol(
    symbol: str,
    df: pd.DataFrame,
    config: dict,
    scan_end_idx: int,
    lookback: int,
    directions: List[str],
) -> List[Dict[str, Any]]:
    """
    Scan a single symbol for Anti patterns.

    Args:
        symbol: Stock ticker
        df: Price DataFrame (oldest to newest)
        config: Full config dict
        scan_end_idx: Index of the last bar in the scan window
        lookback: Number of bars to scan back
        directions: List of directions to scan ("short", "long", or both)

    Returns:
        List of occurrence dicts
    """
    close = df['Close'].values
    dates = df['Date'].values

    # Compute indicators
    rsi_config = config.get('rsi', {})
    rsi_data = {}
    if rsi_config.get('enabled', False):
        rsi_data = compute_rsi_multiple(close, rsi_config.get('periods', []))

    macd_config = config.get('macd', {})
    macd_line, signal_line = compute_modified_macd(
        close,
        fast_period=macd_config.get('fast_period', 3),
        slow_period=macd_config.get('slow_period', 10),
        signal_period=macd_config.get('signal_period', 16),
    )

    macd_percentile = compute_macd_percentile(
        macd_line,
        percentile_lookback=macd_config.get('percentile_lookback', 252),
    )

    # Define scan window
    scan_start_idx = max(0, scan_end_idx - lookback + 1)

    all_occurrences = []

    for bar in range(scan_start_idx, scan_end_idx + 1):
        for direction in directions:
            # Phase 1: Divergence detection
            phase1_results = run_phase1(
                current_bar=bar,
                direction=direction,
                close=close,
                rsi_data=rsi_data,
                macd_line=macd_line,
                macd_percentile=macd_percentile,
                config=config,
            )

            if not phase1_results:
                continue

            # Phase 2: Shock detection
            phase2_config = config.get('phase2', {})
            occurrences = run_phase2(
                phase1_results=phase1_results,
                direction=direction,
                close=close,
                macd_line=macd_line,
                phase2_config=phase2_config,
            )

            all_occurrences.extend(occurrences)

    return all_occurrences


def deduplicate_occurrences(
    occurrences: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove exact duplicate occurrences that have the same shock_bar,
    direction, and config combination.

    Different Phase 1 bars can produce the same Phase 2 shock. This
    deduplicates by (shock_bar, direction, indicator_type, indicator_period,
    threshold, divergence_count, shock_method, shock_threshold).
    """
    seen = set()
    deduped = []

    for occ in occurrences:
        p1 = occ['phase1']
        p2 = occ['phase2']
        key = (
            p2['shock_bar'],
            occ['direction'],
            p1['indicator_type'],
            p1['indicator_period'],
            p1['threshold'],
            p1['divergence_count'],
            p2['shock_method'],
            p2['shock_threshold'],
        )
        if key not in seen:
            seen.add(key)
            deduped.append(occ)

    return deduped


def main():
    parser = argparse.ArgumentParser(
        description='Anti Pattern Scanner — Scan for trend-reversal divergence + shock patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan past year from a specific date, both directions
  python -m strategies.anti_scanner.scanner --date 2025-03-20 --lookback 252

  # Short setups only
  python -m strategies.anti_scanner.scanner --date 2025-03-20 --direction short

  # Specific symbols
  python -m strategies.anti_scanner.scanner --date 2025-03-20 --symbols AAPL,MSFT,NVDA

  # Custom config
  python -m strategies.anti_scanner.scanner --date 2025-03-20 --config path/to/config.yaml
        """
    )

    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    parser.add_argument('--date', type=str, default=yesterday,
                        help=f'End date of scan window (default: {yesterday})')
    parser.add_argument('--lookback', type=int, default=252,
                        help='Number of trading bars to scan back (default: 252)')
    parser.add_argument('--direction', type=str, default='both',
                        choices=['long', 'short', 'both'],
                        help='Scan direction (default: both)')
    parser.add_argument('--config', type=str,
                        default=str(Path(__file__).parent / 'config' / 'default.yaml'),
                        help='Path to indicator config file')
    parser.add_argument('--symbols-file', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'anti_symbols.csv'),
                        help='Path to symbol list CSV')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated symbols (overrides --symbols-file)')
    parser.add_argument('--output-dir', type=str,
                        default=str(PROJECT_ROOT / 'reports'),
                        help='Directory for output files')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed debug logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Load symbols
    symbols = load_symbols(args.symbols, args.symbols_file)
    if not symbols:
        logger.error("No symbols to scan. Provide --symbols or --symbols-file.")
        sys.exit(1)

    # Determine directions
    if args.direction == 'both':
        directions = ['short', 'long']
    else:
        directions = [args.direction]

    # Setup paths
    storage_dir = PROJECT_ROOT / 'data' / 'storage'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Ensure data is available locally, fetch missing symbols
    data_bars = config.get('data_bars', 500)
    symbols = ensure_data(symbols, storage_dir, data_bars)
    if not symbols:
        logger.error("No symbols with available data. Exiting.")
        sys.exit(1)

    # Print scan info
    print("=" * 65)
    print("ANTI PATTERN SCANNER")
    print("=" * 65)
    print(f"Scan date:    {args.date}")
    print(f"Lookback:     {args.lookback} bars")
    print(f"Direction:    {args.direction}")
    print(f"Symbols:      {len(symbols)}")
    print(f"Config:       {args.config}")
    print("=" * 65)
    print()

    # Scan each symbol
    all_occurrences = {}
    dates_by_symbol = {}

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Scanning {symbol}...", end=" ")

        try:
            df = load_price_data(symbol, storage_dir)
        except FileNotFoundError as e:
            print(f"SKIP (no data: {e})")
            continue

        dates = df['Date'].values
        dates_by_symbol[symbol] = dates

        # Find scan end index
        scan_end_idx = find_date_index(dates, args.date)
        if scan_end_idx < 0:
            print(f"SKIP (no data on or before {args.date})")
            continue

        # Scan
        occurrences = scan_symbol(
            symbol=symbol,
            df=df,
            config=config,
            scan_end_idx=scan_end_idx,
            lookback=args.lookback,
            directions=directions,
        )

        # Deduplicate
        occurrences = deduplicate_occurrences(occurrences)
        all_occurrences[symbol] = occurrences

        # Count unique (shock_bar, direction) pairs
        unique = set((o['phase2']['shock_bar'], o['direction']) for o in occurrences)
        print(f"{len(unique)} occurrences ({len(occurrences)} total configs)")

    # Generate report
    print()
    print("Generating reports...")

    report_content = generate_report(
        all_occurrences=all_occurrences,
        dates_by_symbol=dates_by_symbol,
        scan_date=args.date,
        lookback=args.lookback,
        direction=args.direction,
        config=config,
        symbols=symbols,
    )

    # Save .txt report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_dir / f"anti_scan_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(report_content)
    print(f"Report saved: {report_file}")

    # Save .csv export
    csv_file = output_dir / f"anti_scan_{timestamp}.csv"
    export_csv(
        filepath=str(csv_file),
        all_occurrences=all_occurrences,
        dates_by_symbol=dates_by_symbol,
        symbols=symbols,
    )
    print(f"CSV saved:    {csv_file}")

    # Print report to console
    print()
    print(report_content)


if __name__ == '__main__':
    main()
