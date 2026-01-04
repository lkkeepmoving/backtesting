"""
VIX Bollinger Rejection Confirmation Strategy

This strategy identifies VIX Bollinger Band upper-band rejections within a rolling
lookback window and triggers when price confirms the rejection with downside follow-through.

Strategy Logic:
For each trading day t:
1. Look back X days (days [t-9, t]) to find the most recent Bollinger rejection:
   - Day A: VIX Close > Upper BB
   - Day B (A+1): VIX Close < Upper BB

2. After identifying the most recent Day B, check if at any point between
   Day B (exclusive) and day t (inclusive), the close price drops below Day B's close.

3. If this condition is satisfied, generate a trigger on the first day where
   Close < Day B Close.

Bollinger Bands Configuration:
- Period: 20 days
- Standard Deviation: 2.0 (population std, ddof=0)
- Upper Band: SMA + 2 * StdDev

Parameters:
- Lookback window: X = 10 days (configurable)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, NamedTuple, Optional


class RejectionConfirmation(NamedTuple):
    """Represents a single confirmed rejection occurrence"""
    trigger_date: str
    trigger_close: float
    day_b_date: str
    day_b_close: float
    day_a_date: str
    day_a_close: float
    day_a_upper_band: float
    day_b_upper_band: float
    days_to_confirm: int  # Days from Day B to trigger


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


def find_most_recent_rejection(df: pd.DataFrame, current_idx: int,
                               lookback: int) -> Optional[tuple]:
    """
    Find the most recent Bollinger rejection within the lookback window

    Args:
        df: DataFrame with VIX data and Bollinger Bands
        current_idx: Current day index (t)
        lookback: Number of days to look back (X=10 means days [t-9, t])

    Returns:
        Tuple of (day_a_idx, day_b_idx) or None if no rejection found
    """
    # Lookback window: [t-9, t] for X=10
    start_idx = max(0, current_idx - (lookback - 1))

    # Search backwards from current_idx to start_idx for most recent rejection
    # We need i and i+1, so stop at current_idx - 1
    for i in range(current_idx - 1, start_idx - 1, -1):
        # Skip if next day is beyond current_idx
        if i + 1 > current_idx:
            continue

        day_a = df.iloc[i]
        day_b = df.iloc[i + 1]

        # Skip if BB not calculated
        if pd.isna(day_a['Upper_Band']) or pd.isna(day_b['Upper_Band']):
            continue

        # Check for rejection pattern:
        # Day A: Close > Upper Band
        # Day B: Close < Upper Band
        if (day_a['Close'] > day_a['Upper_Band'] and
            day_b['Close'] < day_b['Upper_Band']):
            return (i, i + 1)  # Return indices of Day A and Day B

    return None


def check_confirmation(df: pd.DataFrame, day_b_idx: int,
                      current_idx: int) -> Optional[int]:
    """
    Check if price has broken below Day B's close between Day B and current day

    Args:
        df: DataFrame with VIX data
        day_b_idx: Index of Day B (rejection day)
        current_idx: Current day index (t)

    Returns:
        Index of first day where Close < Day B Close, or None
    """
    day_b_close = df.iloc[day_b_idx]['Close']

    # Check from day after Day B to current day (inclusive)
    for i in range(day_b_idx + 1, current_idx + 1):
        if df.iloc[i]['Close'] < day_b_close:
            return i  # Return first trigger day

    return None


def detect_confirmations(df: pd.DataFrame, lookback: int = 10) -> List[RejectionConfirmation]:
    """
    Detect all confirmed rejection patterns

    Args:
        df: DataFrame with VIX data and Bollinger Bands (oldest to newest)
        lookback: Lookback window size (default: 10)

    Returns:
        List of RejectionConfirmation occurrences
    """
    confirmations = []
    triggered_on_days = set()  # Track which days already have triggers

    # Start scanning from index where we have enough data for lookback and BB
    start_scan = 20  # Need 20 days for BB calculation

    for t in range(start_scan, len(df)):
        # Find most recent rejection in lookback window
        rejection = find_most_recent_rejection(df, t, lookback)

        if rejection is None:
            continue

        day_a_idx, day_b_idx = rejection

        # Check for confirmation
        trigger_idx = check_confirmation(df, day_b_idx, t)

        if trigger_idx is not None:
            # Only record if we haven't already triggered on this day
            if trigger_idx not in triggered_on_days:
                triggered_on_days.add(trigger_idx)

                day_a = df.iloc[day_a_idx]
                day_b = df.iloc[day_b_idx]
                trigger = df.iloc[trigger_idx]

                days_to_confirm = trigger_idx - day_b_idx

                confirmation = RejectionConfirmation(
                    trigger_date=trigger['Date'],
                    trigger_close=round(trigger['Close'], 2),
                    day_b_date=day_b['Date'],
                    day_b_close=round(day_b['Close'], 2),
                    day_a_date=day_a['Date'],
                    day_a_close=round(day_a['Close'], 2),
                    day_a_upper_band=round(day_a['Upper_Band'], 2),
                    day_b_upper_band=round(day_b['Upper_Band'], 2),
                    days_to_confirm=days_to_confirm
                )

                confirmations.append(confirmation)

    return confirmations


def load_vix_data(data_path: Path) -> pd.DataFrame:
    """Load VIX data from CSV file"""
    if not data_path.exists():
        raise FileNotFoundError(f"VIX data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Reverse data to oldest-to-newest for calculation
    df = df.iloc[::-1].reset_index(drop=True)

    return df


def format_confirmation(conf: RejectionConfirmation, index: int) -> str:
    """Format a single confirmation occurrence for display"""
    lines = []
    lines.append(f"  [{index}] Trigger Date: {conf.trigger_date}")
    lines.append(f"      Trigger Close: {conf.trigger_close:.2f}")
    lines.append(f"      Day B (Rejection): {conf.day_b_date}, Close: {conf.day_b_close:.2f}, Upper Band: {conf.day_b_upper_band:.2f}")
    lines.append(f"      Day A (Above Band): {conf.day_a_date}, Close: {conf.day_a_close:.2f}, Upper Band: {conf.day_a_upper_band:.2f}")
    lines.append(f"      Days to Confirm: {conf.days_to_confirm} (from Day B to trigger)")
    lines.append(f"      Breakdown: {conf.trigger_close:.2f} < {conf.day_b_close:.2f} (dropped {conf.day_b_close - conf.trigger_close:.2f})")
    return '\n'.join(lines)


def generate_report(confirmations: List[RejectionConfirmation], df: pd.DataFrame,
                   data_file: Path, lookback: int) -> str:
    """Generate comprehensive report of confirmed rejection patterns"""
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("VIX Bollinger Rejection Confirmation Strategy Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Data Source: {data_file}")
    report_lines.append(f"Total Bars Analyzed: {len(df)}")
    report_lines.append(f"Date Range: {df.iloc[0]['Date']} to {df.iloc[-1]['Date']}")
    report_lines.append("")

    # Strategy Configuration
    report_lines.append("Strategy Configuration:")
    report_lines.append(f"  Lookback Window: {lookback} days")
    report_lines.append("  Bollinger Bands: 20-period, 2.0 StdDev (population std)")
    report_lines.append("")

    # Strategy Logic
    report_lines.append("Strategy Logic:")
    report_lines.append("  1. For each day t, look back X days [t-9, t] to find most recent rejection:")
    report_lines.append("     - Day A: VIX Close > Upper BB")
    report_lines.append("     - Day B (A+1): VIX Close < Upper BB (rejection)")
    report_lines.append("  2. Check if any day after Day B has Close < Day B Close")
    report_lines.append("  3. Trigger on first day where Close < Day B Close")
    report_lines.append("")

    # Occurrences section
    report_lines.append("=" * 80)
    report_lines.append("CONFIRMED REJECTION TRIGGERS")
    report_lines.append("=" * 80)

    if not confirmations:
        report_lines.append("\nNo confirmed rejection patterns detected in the data.")
    else:
        report_lines.append(f"\nTotal Triggers Found: {len(confirmations)}")
        report_lines.append("")

        # Sort by trigger date in reverse chronological order (newest first)
        sorted_confirmations = sorted(confirmations, key=lambda x: x.trigger_date, reverse=True)

        for idx, conf in enumerate(sorted_confirmations, 1):
            report_lines.append(format_confirmation(conf, idx))
            report_lines.append("")

    # Summary section
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Total Confirmed Triggers: {len(confirmations)}")

    if confirmations:
        # Calculate statistics
        avg_days = sum(c.days_to_confirm for c in confirmations) / len(confirmations)
        min_days = min(c.days_to_confirm for c in confirmations)
        max_days = max(c.days_to_confirm for c in confirmations)

        avg_trigger_vix = sum(c.trigger_close for c in confirmations) / len(confirmations)
        avg_day_b_vix = sum(c.day_b_close for c in confirmations) / len(confirmations)

        report_lines.append(f"Average Days to Confirm: {avg_days:.1f}")
        report_lines.append(f"Min Days to Confirm: {min_days}")
        report_lines.append(f"Max Days to Confirm: {max_days}")
        report_lines.append(f"Average Trigger VIX Level: {avg_trigger_vix:.2f}")
        report_lines.append(f"Average Day B VIX Level: {avg_day_b_vix:.2f}")

    report_lines.append("=" * 80)

    return '\n'.join(report_lines)


def main():
    print("=" * 80)
    print("VIX Bollinger Rejection Confirmation Strategy Scanner")
    print("=" * 80)

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_file = project_root / 'data' / 'storage' / 'VIX_X.csv'

    # Configuration
    lookback_days = 10

    print(f"Data file: {data_file}")
    print(f"Lookback window: {lookback_days} days")

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

    # Detect confirmed rejections
    print(f"Scanning for confirmed rejection patterns (lookback={lookback_days})...")
    confirmations = detect_confirmations(df, lookback=lookback_days)

    print(f"\n✓ Found {len(confirmations)} confirmed triggers")

    # Generate report
    report_content = generate_report(confirmations, df, data_file, lookback_days)

    # Save report
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = reports_dir / f'vix_bollinger_rejection_confirmation_{timestamp}.txt'

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
