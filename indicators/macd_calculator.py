"""
MACD (Moving Average Convergence Divergence) Calculator

Calculates MACD indicator with standard parameters (12, 26, 9):
- Fast EMA: 12 periods
- Slow EMA: 26 periods
- Signal Line: 9-period EMA of MACD line

Calculation Strategy:
1. Calculate EMA(12) and EMA(26) from 2023-01-03, let them converge through 2023-12-29
2. Start calculating MACD Line from 2024-01-02 (MACD = EMA(12) - EMA(26))
3. Use first 9 MACD values to initialize Signal Line with SMA
4. Continue Signal Line as EMA(9) of MACD
5. Calculate Histogram (MACD - Signal)

Output CSV columns:
- Date, EMA(12), EMA(26), MACD, Signal, Histogram
- Rows from 2023-01-03 onwards: EMA values populated
- Rows from 2024-01-02 to ~2024-01-16: EMA and MACD populated
- Rows from ~2024-01-17 onwards: All columns populated
"""

import argparse
import pandas as pd
from pathlib import Path


def calculate_ema_series(closes: pd.Series, period: int) -> list:
    """
    Calculate EMA for a series of close prices

    Args:
        closes: Series of close prices (oldest to newest)
        period: EMA period

    Returns:
        List of EMA values (one per input value after initial SMA)
    """
    multiplier = 2 / (period + 1)
    ema_values = []

    # First EMA = SMA of first N periods
    first_ema = round(closes.iloc[:period].mean(), 5)
    ema_values.append(first_ema)

    # Calculate remaining EMAs
    previous_ema = first_ema
    for i in range(period, len(closes)):
        current_close = closes.iloc[i]
        current_ema = round((current_close * multiplier) + (previous_ema * (1 - multiplier)), 5)
        ema_values.append(current_ema)
        previous_ema = current_ema

    return ema_values


def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MACD indicator

    Args:
        df: DataFrame with Date and Close columns, sorted newest to oldest

    Returns:
        DataFrame with Date, EMA(12), EMA(26), MACD, Signal, Histogram columns
    """
    # Reverse the dataframe to work oldest to newest
    df_reversed = df.iloc[::-1].reset_index(drop=True)

    # Calculate EMA(12) - needs first 12 days, starts from index 11
    ema12_values = calculate_ema_series(df_reversed['Close'], 12)

    # Calculate EMA(26) - needs first 26 days, starts from index 25
    ema26_values = calculate_ema_series(df_reversed['Close'], 26)

    # Build results
    results = []

    # Find the index for 2023-01-03 (desired output start)
    output_start_idx = None
    for i in range(len(df_reversed)):
        if df_reversed.iloc[i]['Date'] == '2023-01-03':
            output_start_idx = i
            break

    # Must have EMA(12) available (starts at index 11)
    if output_start_idx is None:
        output_start_idx = 11  # Fallback to when EMA(12) is available

    # Ensure we have EMA(12) data
    if output_start_idx < 11:
        output_start_idx = 11

    macd_values = []  # Store MACD values for signal line calculation

    for i in range(output_start_idx, len(df_reversed)):
        date = df_reversed.iloc[i]['Date']

        # Get EMA(12) if available (starts at index 11)
        ema12 = ema12_values[i - 11] if i >= 11 else None

        # Get EMA(26) if available (starts at index 25)
        ema26 = ema26_values[i - 25] if i >= 25 else None

        # Calculate MACD only from 2024-01-02 onwards AND if we have both EMAs
        macd = None
        if date >= '2024-01-02' and ema12 is not None and ema26 is not None:
            macd = round(ema12 - ema26, 5)
            macd_values.append(macd)

        results.append({
            'Date': date,
            'EMA(12)': ema12 if ema12 is not None else '',
            'EMA(26)': ema26 if ema26 is not None else '',
            'MACD': macd if macd is not None else '',
            'Signal': '',
            'Histogram': ''
        })

    # Calculate Signal Line (EMA(9) of MACD)
    if len(macd_values) >= 9:
        # First signal = SMA of first 9 MACD values
        first_signal = round(sum(macd_values[:9]) / 9, 5)
        signal_values = [first_signal]

        # Calculate remaining signal values as EMA(9)
        multiplier = 2 / (9 + 1)
        previous_signal = first_signal

        for i in range(9, len(macd_values)):
            current_macd = macd_values[i]
            current_signal = round((current_macd * multiplier) + (previous_signal * (1 - multiplier)), 5)
            signal_values.append(current_signal)
            previous_signal = current_signal

        # Update results with Signal and Histogram values
        # Find where MACD starts in results
        macd_start_idx = None
        for i, row in enumerate(results):
            if row['MACD'] != '':
                macd_start_idx = i
                break

        if macd_start_idx is not None:
            # First 8 MACD rows don't have signal yet (need 9 for SMA)
            # Starting from 9th MACD row (index macd_start_idx + 8), add signals
            for i, signal in enumerate(signal_values):
                result_idx = macd_start_idx + 8 + i
                if result_idx < len(results):
                    results[result_idx]['Signal'] = signal
                    macd_val = results[result_idx]['MACD']
                    histogram = round(macd_val - signal, 5)
                    results[result_idx]['Histogram'] = histogram

    result_df = pd.DataFrame(results)
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description='Calculate MACD (Moving Average Convergence Divergence) for symbol(s)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate MACD for TSLA (default)
  python macd_calculator.py

  # Calculate MACD for a specific symbol
  python macd_calculator.py --symbol AAPL

  # Calculate MACD for all symbols in a file
  python macd_calculator.py --symbols-file core_symbols.csv
        """
    )

    # Symbol input options (mutually exclusive)
    symbol_group = parser.add_mutually_exclusive_group()
    symbol_group.add_argument(
        '--symbol',
        type=str,
        help='Single stock symbol to process'
    )
    symbol_group.add_argument(
        '--symbols-file',
        type=str,
        help='Path to CSV file with symbol list (header: symbol)'
    )

    args = parser.parse_args()

    # Determine which symbols to process
    if args.symbols_file:
        # Load symbols from file
        import csv
        symbols = []
        try:
            with open(args.symbols_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get('symbol', '').strip()
                    if symbol:
                        symbols.append(symbol)
        except FileNotFoundError:
            print(f"Error: File not found: {args.symbols_file}")
            return
        except Exception as e:
            print(f"Error reading symbols file: {e}")
            return

        if not symbols:
            print("Error: No symbols found in file")
            return
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = ['TSLA']  # Default

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    storage_dir = project_root / 'data' / 'storage'
    output_dir = project_root / 'indicators' / 'output' / 'macd'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MACD Calculator")
    print("=" * 80)
    print(f"Parameters: Fast EMA(12), Slow EMA(26), Signal EMA(9)")
    print(f"Symbols to process: {len(symbols)}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    print()

    success_count = 0
    failed_symbols = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Processing {symbol}...")

        # Load data
        csv_path = storage_dir / f"{symbol}.csv"

        if not csv_path.exists():
            print(f"  ✗ Error: Data file not found")
            failed_symbols.append(symbol)
            continue

        try:
            # Read data
            df = pd.read_csv(csv_path)

            if 'Close' not in df.columns or 'Date' not in df.columns:
                print(f"  ✗ Error: CSV must have 'Date' and 'Close' columns")
                failed_symbols.append(symbol)
                continue

            # Calculate MACD
            result_df = calculate_macd(df)

            # Save to CSV
            output_file = output_dir / f"{symbol}_MACD.csv"
            result_df.to_csv(output_file, index=False)

            print(f"  ✓ Calculated {len(result_df)} bars, saved to {output_file.name}")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_symbols.append(symbol)

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully processed: {success_count} / {len(symbols)} symbols")
    if failed_symbols:
        print(f"Failed symbols: {', '.join(failed_symbols)}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
