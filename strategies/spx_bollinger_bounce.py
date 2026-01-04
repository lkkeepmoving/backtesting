"""
SPX Bollinger Bounce Pattern Scanner

Detects days when the SPX price bounces back above the lower Bollinger Band
after being below it on the previous trading day.

Bollinger Bands Configuration:
- Period: 20 days
- Standard Deviation: 2
- Middle Band: 20-period SMA
- Upper Band: SMA + 2 * StdDev
- Lower Band: SMA - 2 * StdDev

Trigger Day Definition:
- Day t-1: Close < Lower Bollinger Band
- Day t: Close > Lower Bollinger Band

This pattern identifies potential bounce opportunities when price re-enters
the Bollinger Bands from below the lower band.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, NamedTuple


class BollingerBounce(NamedTuple):
    """Represents a single Bollinger Bounce occurrence"""
    trigger_date: str
    close: float
    lower_band: float
    middle_band: float
    upper_band: float
    prev_close: float
    prev_lower_band: float


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20,
                              num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands for the given price data

    Args:
        df: DataFrame with Close prices (oldest to newest)
        period: SMA period (default: 20)
        num_std: Number of standard deviations (default: 2.0)

    Returns:
        DataFrame with added BB columns: SMA, Upper_Band, Lower_Band, StdDev
    """
    # Calculate rolling mean (middle band)
    df['SMA'] = df['Close'].rolling(window=period).mean()

    # Calculate rolling standard deviation (population std, ddof=0)
    df['StdDev'] = df['Close'].rolling(window=period).std(ddof=0)

    # Calculate upper and lower bands
    df['Upper_Band'] = df['SMA'] + (num_std * df['StdDev'])
    df['Lower_Band'] = df['SMA'] - (num_std * df['StdDev'])

    return df


def detect_bounces(df: pd.DataFrame) -> List[BollingerBounce]:
    """
    Detect Bollinger Bounce trigger days

    Args:
        df: DataFrame with Close and Bollinger Band data (oldest to newest)

    Returns:
        List of BollingerBounce occurrences
    """
    bounces = []

    # Start from index 1 to have a previous day to compare
    # Also skip rows where BB values are NaN (first 19 days)
    for i in range(1, len(df)):
        # Get current and previous day data
        current = df.iloc[i]
        previous = df.iloc[i - 1]

        # Skip if Bollinger Bands are not calculated yet (NaN values)
        if pd.isna(current['Lower_Band']) or pd.isna(previous['Lower_Band']):
            continue

        # Check trigger conditions:
        # 1. Previous day: Close < Lower Band
        # 2. Current day: Close > Lower Band
        if (previous['Close'] < previous['Lower_Band'] and
            current['Close'] > current['Lower_Band']):

            bounce = BollingerBounce(
                trigger_date=current['Date'],
                close=round(current['Close'], 2),
                lower_band=round(current['Lower_Band'], 2),
                middle_band=round(current['SMA'], 2),
                upper_band=round(current['Upper_Band'], 2),
                prev_close=round(previous['Close'], 2),
                prev_lower_band=round(previous['Lower_Band'], 2)
            )

            bounces.append(bounce)

    return bounces


def load_spx_data(data_path: Path) -> pd.DataFrame:
    """Load SPX data from CSV file"""
    if not data_path.exists():
        raise FileNotFoundError(f"SPX data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Reverse data to oldest-to-newest for calculation
    df = df.iloc[::-1].reset_index(drop=True)

    return df


def format_bounce(bounce: BollingerBounce, index: int) -> str:
    """Format a single bounce occurrence for display"""
    lines = []
    lines.append(f"  [{index}] Trigger Date: {bounce.trigger_date}")
    lines.append(f"      Close: {bounce.close:,.2f}")
    lines.append(f"      Lower Band: {bounce.lower_band:,.2f}")
    lines.append(f"      Middle Band: {bounce.middle_band:,.2f}")
    lines.append(f"      Upper Band: {bounce.upper_band:,.2f}")
    lines.append(f"      Previous Close: {bounce.prev_close:,.2f} (below Lower Band: {bounce.prev_lower_band:,.2f})")
    lines.append(f"      Bounce: {bounce.close - bounce.lower_band:,.2f} above Lower Band")
    return '\n'.join(lines)


def generate_report(bounces: List[BollingerBounce], df: pd.DataFrame,
                   data_file: Path) -> str:
    """Generate comprehensive report of Bollinger Bounce occurrences"""
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("SPX Bollinger Bounce Pattern Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Data Source: {data_file}")
    report_lines.append(f"Total Bars Analyzed: {len(df)}")
    report_lines.append(f"Date Range: {df.iloc[0]['Date']} to {df.iloc[-1]['Date']}")
    report_lines.append("")

    # Bollinger Band Configuration
    report_lines.append("Bollinger Band Configuration:")
    report_lines.append("  Period: 20 days")
    report_lines.append("  Standard Deviation: 2.0")
    report_lines.append("  Middle Band: 20-period SMA")
    report_lines.append("  Upper Band: SMA + 2 * StdDev")
    report_lines.append("  Lower Band: SMA - 2 * StdDev")
    report_lines.append("")

    # Pattern Definition
    report_lines.append("Trigger Day Definition:")
    report_lines.append("  Day t-1: Close < Lower Bollinger Band")
    report_lines.append("  Day t (Trigger): Close > Lower Bollinger Band")
    report_lines.append("")

    # Occurrences section
    report_lines.append("=" * 80)
    report_lines.append("TRIGGER DAY OCCURRENCES")
    report_lines.append("=" * 80)

    if not bounces:
        report_lines.append("\nNo Bollinger Bounce patterns detected in the data.")
    else:
        report_lines.append(f"\nTotal Trigger Days Found: {len(bounces)}")
        report_lines.append("")

        # Sort bounces by trigger date in reverse chronological order (newest first)
        sorted_bounces = sorted(bounces, key=lambda x: x.trigger_date, reverse=True)

        for idx, bounce in enumerate(sorted_bounces, 1):
            report_lines.append(format_bounce(bounce, idx))
            report_lines.append("")

    # Summary section
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Total Trigger Days: {len(bounces)}")

    if bounces:
        # Calculate statistics
        bounce_distances = [b.close - b.lower_band for b in bounces]
        avg_bounce = sum(bounce_distances) / len(bounce_distances)
        max_bounce = max(bounce_distances)
        min_bounce = min(bounce_distances)

        report_lines.append(f"Average Bounce Distance: {avg_bounce:.2f}")
        report_lines.append(f"Maximum Bounce Distance: {max_bounce:.2f}")
        report_lines.append(f"Minimum Bounce Distance: {min_bounce:.2f}")

    report_lines.append("=" * 80)

    return '\n'.join(report_lines)


def main():
    print("=" * 80)
    print("SPX Bollinger Bounce Pattern Scanner")
    print("=" * 80)

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_file = project_root / 'data' / 'storage' / 'SPX_X.csv'

    print(f"Data file: {data_file}")

    # Load SPX data
    try:
        df = load_spx_data(data_file)
        print(f"Loaded {len(df)} bars of SPX data")
        print(f"Date range: {df.iloc[0]['Date']} to {df.iloc[-1]['Date']}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Calculate Bollinger Bands
    print("\nCalculating Bollinger Bands (20-period, 2 StdDev)...")
    df = calculate_bollinger_bands(df, period=20, num_std=2.0)

    # Detect bounce patterns
    print("Scanning for Bollinger Bounce trigger days...")
    bounces = detect_bounces(df)

    print(f"\n✓ Found {len(bounces)} trigger days")

    # Generate report
    report_content = generate_report(bounces, df, data_file)

    # Save report
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = reports_dir / f'spx_bollinger_bounce_{timestamp}.txt'

    with open(report_file, 'w') as f:
        f.write(report_content)

    print(f"\nReport saved to: {report_file}")

    # Print preview
    print("\n" + "=" * 80)
    print("REPORT PREVIEW")
    print("=" * 80)
    print(report_content)


if __name__ == '__main__':
    main()
