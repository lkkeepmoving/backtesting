"""
Fetch latest daily trading volume for a list of symbols.

Reads a ticker CSV (any file with a 'symbol' header), pulls the most recent
daily bar from TradeStation for each, and writes a CSV sorted by volume.

Adds dollar_volume (Close * Volume) which is a better cross-sectional liquidity
proxy than raw share volume.
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

from tradestation_client import TradeStationClient
from fetcher import get_tradestation_config, load_symbols_from_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Fetch latest daily volume for symbols')
    parser.add_argument('--symbols-file', default='finviz_30b_large_cap_tickers.csv',
                        help='CSV file with a symbol header')
    parser.add_argument('--output', default='latest_volume.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    symbols = load_symbols_from_file(args.symbols_file)
    if not symbols:
        logger.error("No symbols loaded. Exiting.")
        sys.exit(1)

    config = get_tradestation_config()
    client = TradeStationClient(config)

    rows = []
    failed = []
    for i, symbol in enumerate(symbols, 1):
        # Fetch a couple of recent bars; take the newest (row 0, sorted desc).
        df = client.get_bars_by_bar_count(symbol, 2)
        if df is None or df.empty:
            logger.warning(f"[{i}/{len(symbols)}] {symbol}: no data")
            failed.append(symbol)
            continue

        latest = df.iloc[0]
        date = latest['Date']
        close = float(latest['Close'])
        volume = int(latest['Volume'])
        dollar_volume = close * volume
        rows.append({
            'symbol': symbol,
            'date': date,
            'close': round(close, 2),
            'volume': volume,
            'dollar_volume': round(dollar_volume, 0),
        })
        logger.info(f"[{i}/{len(symbols)}] {symbol}: {date} vol={volume:,} "
                    f"$vol={dollar_volume/1e9:.2f}B")
        time.sleep(0.05)  # gentle pacing

    # Sort by volume descending
    rows.sort(key=lambda r: r['volume'], reverse=True)

    out_path = Path(args.output)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['symbol', 'date', 'close',
                                               'volume', 'dollar_volume'])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("=" * 60)
    logger.info(f"Wrote {len(rows)} rows to {out_path}")
    if failed:
        logger.info(f"Failed ({len(failed)}): {', '.join(failed)}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
