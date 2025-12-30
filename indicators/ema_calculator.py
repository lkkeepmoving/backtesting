"""
EMA (Exponential Moving Average) Calculator

Calculates EMA for a given symbol and period, using SMA for the first value
and then applying the standard EMA formula iteratively.

Formula:
- First EMA = SMA of first N periods
- Subsequent EMA = (Close × multiplier) + (Previous EMA × (1 - multiplier))
  where multiplier = 2 / (period + 1)
"""

import argparse
import pandas as pd
from pathlib import Path


def calculate_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculate EMA for the given dataframe

    Args:
        df: DataFrame with Date and Close columns, sorted newest to oldest
        period: EMA period (e.g., 26 for EMA(26))

    Returns:
        DataFrame with Date and EMA columns, sorted oldest to newest
    """
    if len(df) < period:
        raise ValueError(f"Not enough data. Need at least {period} bars, but only have {len(df)}")

    # Reverse the dataframe to work oldest to newest
    df_reversed = df.iloc[::-1].reset_index(drop=True)

    # Calculate multiplier for EMA
    multiplier = 2 / (period + 1)

    # Initialize result list
    results = []

    # Calculate first EMA using SMA of first N periods
    first_n_closes = df_reversed.iloc[:period]['Close']
    first_ema = round(first_n_closes.mean(), 2)

    results.append({
        'Date': df_reversed.iloc[period - 1]['Date'],
        f'EMA({period})': first_ema
    })

    # Calculate EMA for remaining days
    previous_ema = first_ema

    for i in range(period, len(df_reversed)):
        current_close = df_reversed.iloc[i]['Close']
        current_ema = round((current_close * multiplier) + (previous_ema * (1 - multiplier)), 2)

        results.append({
            'Date': df_reversed.iloc[i]['Date'],
            f'EMA({period})': current_ema
        })

        previous_ema = current_ema

    # Convert to DataFrame
    result_df = pd.DataFrame(results)

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description='Calculate EMA (Exponential Moving Average) for a symbol',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate EMA(26) for TSLA (default)
  python ema_calculator.py

  # Calculate EMA(26) for a specific symbol
  python ema_calculator.py --symbol AAPL

  # Calculate EMA(12) for TSLA
  python ema_calculator.py --period 12

  # Calculate EMA(50) for NVDA
  python ema_calculator.py --symbol NVDA --period 50
        """
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='TSLA',
        help='Stock symbol (default: TSLA)'
    )

    parser.add_argument(
        '--period',
        type=int,
        default=26,
        help='EMA period (default: 26)'
    )

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    storage_dir = project_root / 'data' / 'storage'
    output_dir = project_root / 'indicators' / 'output'
    output_dir.mkdir(exist_ok=True)

    # Load data
    csv_path = storage_dir / f"{args.symbol}.csv"

    if not csv_path.exists():
        print(f"Error: Data file not found for {args.symbol}")
        print(f"Expected location: {csv_path}")
        return

    print("=" * 80)
    print(f"EMA Calculator - {args.symbol}")
    print("=" * 80)
    print(f"Period: EMA({args.period})")
    print(f"Data source: {csv_path}")
    print()

    # Read data
    df = pd.read_csv(csv_path)

    if 'Close' not in df.columns or 'Date' not in df.columns:
        print("Error: CSV must have 'Date' and 'Close' columns")
        return

    print(f"Total bars available: {len(df)}")
    print(f"Date range: {df.iloc[-1]['Date']} to {df.iloc[0]['Date']}")
    print()

    # Calculate EMA
    try:
        result_df = calculate_ema(df, args.period)

        print(f"EMA calculated for {len(result_df)} bars")
        print(f"First EMA date: {result_df.iloc[0]['Date']}")
        print(f"Last EMA date: {result_df.iloc[-1]['Date']}")
        print()

        # Save to CSV
        output_file = output_dir / f"{args.symbol}_EMA{args.period}.csv"
        result_df.to_csv(output_file, index=False)

        print(f"Output saved to: {output_file}")
        print()

        # Show preview
        print("=" * 80)
        print("PREVIEW (First 10 rows)")
        print("=" * 80)
        print(result_df.head(10).to_string(index=False))
        print()

        print("=" * 80)
        print("PREVIEW (Last 10 rows)")
        print("=" * 80)
        print(result_df.tail(10).to_string(index=False))
        print()

        print("=" * 80)

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
