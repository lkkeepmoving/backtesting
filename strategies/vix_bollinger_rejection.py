"""
VIX Bollinger Rejection Pattern Scanner

Detects days when VIX drops back below the upper Bollinger Band after being
above it on the previous trading day.

Bollinger Bands Configuration:
- Period: 20 days
- Standard Deviation: 2 (population std, ddof=0)
- Middle Band: 20-period SMA
- Upper Band: SMA + 2 * StdDev
- Lower Band: SMA - 2 * StdDev

Trigger Day Definition:
- Day t-1: VIX Close > Upper Bollinger Band
- Day t (Trigger): VIX Close < Upper Bollinger Band

This pattern identifies potential volatility retreat signals when VIX falls
back below the upper band after spiking above it.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, NamedTuple


class BollingerRejection(NamedTuple):
    """Represents a single Bollinger Rejection occurrence"""
    trigger_date: str
    close: float
    upper_band: float
    middle_band: float
    lower_band: float
    prev_close: float
    prev_upper_band: float


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


def detect_rejections(df: pd.DataFrame) -> List[BollingerRejection]:
    """
    Detect Bollinger Rejection trigger days

    Args:
        df: DataFrame with Close and Bollinger Band data (oldest to newest)

    Returns:
        List of BollingerRejection occurrences
    """
    rejections = []

    # Start from index 1 to have a previous day to compare
    # Also skip rows where BB values are NaN (first 19 days)
    for i in range(1, len(df)):
        # Get current and previous day data
        current = df.iloc[i]
        previous = df.iloc[i - 1]

        # Skip if Bollinger Bands are not calculated yet (NaN values)
        if pd.isna(current['Upper_Band']) or pd.isna(previous['Upper_Band']):
            continue

        # Check trigger conditions:
        # 1. Previous day: Close > Upper Band
        # 2. Current day: Close < Upper Band
        if (previous['Close'] > previous['Upper_Band'] and
            current['Close'] < current['Upper_Band']):

            rejection = BollingerRejection(
                trigger_date=current['Date'],
                close=round(current['Close'], 2),
                upper_band=round(current['Upper_Band'], 2),
                middle_band=round(current['SMA'], 2),
                lower_band=round(current['Lower_Band'], 2),
                prev_close=round(previous['Close'], 2),
                prev_upper_band=round(previous['Upper_Band'], 2)
            )

            rejections.append(rejection)

    return rejections


def load_vix_data(data_path: Path) -> pd.DataFrame:
    """Load VIX data from CSV file"""
    if not data_path.exists():
        raise FileNotFoundError(f"VIX data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Reverse data to oldest-to-newest for calculation
    df = df.iloc[::-1].reset_index(drop=True)

    return df


def format_rejection(rejection: BollingerRejection, index: int) -> str:
    """Format a single rejection occurrence for display"""
    lines = []
    lines.append(f"  [{index}] Trigger Date: {rejection.trigger_date}")
    lines.append(f"      Close: {rejection.close:.2f}")
    lines.append(f"      Upper Band: {rejection.upper_band:.2f}")
    lines.append(f"      Middle Band: {rejection.middle_band:.2f}")
    lines.append(f"      Lower Band: {rejection.lower_band:.2f}")
    lines.append(f"      Previous Close: {rejection.prev_close:.2f} (above Upper Band: {rejection.prev_upper_band:.2f})")
    lines.append(f"      Rejection: {rejection.prev_close - rejection.upper_band:.2f} drop from previous close to current upper band")
    return '\n'.join(lines)


def generate_report(rejections: List[BollingerRejection], df: pd.DataFrame,
                   data_file: Path) -> str:
    """Generate comprehensive report of Bollinger Rejection occurrences"""
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("VIX Bollinger Rejection Pattern Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Data Source: {data_file}")
    report_lines.append(f"Total Bars Analyzed: {len(df)}")
    report_lines.append(f"Date Range: {df.iloc[0]['Date']} to {df.iloc[-1]['Date']}")
    report_lines.append("")

    # Bollinger Band Configuration
    report_lines.append("Bollinger Band Configuration:")
    report_lines.append("  Period: 20 days")
    report_lines.append("  Standard Deviation: 2.0 (population std, ddof=0)")
    report_lines.append("  Middle Band: 20-period SMA")
    report_lines.append("  Upper Band: SMA + 2 * StdDev")
    report_lines.append("  Lower Band: SMA - 2 * StdDev")
    report_lines.append("")

    # Pattern Definition
    report_lines.append("Trigger Day Definition:")
    report_lines.append("  Day t-1: VIX Close > Upper Bollinger Band")
    report_lines.append("  Day t (Trigger): VIX Close < Upper Bollinger Band")
    report_lines.append("")

    # Occurrences section
    report_lines.append("=" * 80)
    report_lines.append("TRIGGER DAY OCCURRENCES")
    report_lines.append("=" * 80)

    if not rejections:
        report_lines.append("\nNo Bollinger Rejection patterns detected in the data.")
    else:
        report_lines.append(f"\nTotal Trigger Days Found: {len(rejections)}")
        report_lines.append("")

        # Sort rejections by trigger date in reverse chronological order (newest first)
        sorted_rejections = sorted(rejections, key=lambda x: x.trigger_date, reverse=True)

        for idx, rejection in enumerate(sorted_rejections, 1):
            report_lines.append(format_rejection(rejection, idx))
            report_lines.append("")

    # Summary section
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Total Trigger Days: {len(rejections)}")

    if rejections:
        # Calculate statistics
        vix_levels = [r.close for r in rejections]
        upper_bands = [r.upper_band for r in rejections]
        prev_closes = [r.prev_close for r in rejections]

        avg_vix = sum(vix_levels) / len(vix_levels)
        avg_upper = sum(upper_bands) / len(upper_bands)
        avg_prev = sum(prev_closes) / len(prev_closes)

        report_lines.append(f"Average VIX Close on Trigger Day: {avg_vix:.2f}")
        report_lines.append(f"Average Upper Band on Trigger Day: {avg_upper:.2f}")
        report_lines.append(f"Average Previous Day VIX Close: {avg_prev:.2f}")

    report_lines.append("=" * 80)

    return '\n'.join(report_lines)


def main():
    print("=" * 80)
    print("VIX Bollinger Rejection Pattern Scanner")
    print("=" * 80)

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_file = project_root / 'data' / 'storage' / 'VIX_X.csv'

    print(f"Data file: {data_file}")

    # Load VIX data
    try:
        df = load_vix_data(data_file)
        print(f"Loaded {len(df)} bars of VIX data")
        print(f"Date range: {df.iloc[0]['Date']} to {df.iloc[-1]['Date']}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Calculate Bollinger Bands
    print("\nCalculating Bollinger Bands (20-period, 2 StdDev, population std)...")
    df = calculate_bollinger_bands(df, period=20, num_std=2.0)

    # Detect rejection patterns
    print("Scanning for Bollinger Rejection trigger days...")
    rejections = detect_rejections(df)

    print(f"\n✓ Found {len(rejections)} trigger days")

    # Generate report
    report_content = generate_report(rejections, df, data_file)

    # Save report
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = reports_dir / f'vix_bollinger_rejection_{timestamp}.txt'

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
