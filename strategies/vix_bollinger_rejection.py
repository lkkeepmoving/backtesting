"""
VIX Bollinger Rejection Pattern Scanner

Detects VIX Bollinger Band rejections with intraday confirmation and measures
the corresponding SPX drawdown from recent highs.

Trigger Day Definition:
- Day A (previous day):
  * VIX Close > Upper Bollinger Band
  * VIX Close > VIX Open (bullish intraday move)

- Day B (trigger day):
  * VIX Close < Upper Bollinger Band (rejection)
  * VIX Close < VIX Open (bearish intraday move)

For each trigger, also calculate SPX drawdown from 10-day high.

Bollinger Bands Configuration:
- Period: 20 days
- Standard Deviation: 2.0 (population std, ddof=0)
- Upper Band: SMA + 2 * StdDev
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, NamedTuple, Optional


class BollingerRejection(NamedTuple):
    """Represents a single Bollinger Rejection occurrence with SPX drawdown"""
    trigger_date: str
    vix_close: float
    vix_open: float
    vix_upper_band: float
    day_a_date: str
    day_a_close: float
    day_a_open: float
    day_a_upper_band: float
    spx_close: float
    spx_highest_high: float
    highest_high_date: str
    spx_drawdown_points: float
    spx_drawdown_percent: float


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


def calculate_spx_drawdown(spx_df: pd.DataFrame, trigger_date: str,
                          lookback: int = 10) -> Optional[tuple]:
    """
    Calculate SPX drawdown from the highest high in the lookback window

    Args:
        spx_df: DataFrame with SPX data (oldest to newest)
        trigger_date: The trigger date from VIX
        lookback: Number of days to look back (default: 10)

    Returns:
        Tuple of (spx_close, highest_high, highest_high_date, drawdown_points, drawdown_pct)
        or None if date not found
    """
    # Find the trigger date in SPX data
    spx_trigger = spx_df[spx_df['Date'] == trigger_date]

    if spx_trigger.empty:
        return None

    trigger_idx = spx_trigger.index[0]

    # Define lookback window [t-9, t] - 10 days including trigger
    start_idx = max(0, trigger_idx - (lookback - 1))
    end_idx = trigger_idx + 1  # +1 for inclusive slice

    # Get the window data
    window = spx_df.iloc[start_idx:end_idx]

    # Find the highest High in the window
    highest_high = window['High'].max()
    highest_high_idx = window['High'].idxmax()
    highest_high_date = spx_df.loc[highest_high_idx, 'Date']

    # Get SPX close on trigger day
    spx_close = spx_df.loc[trigger_idx, 'Close']

    # Calculate drawdown
    drawdown_points = highest_high - spx_close
    drawdown_percent = (drawdown_points / highest_high) * 100

    return (spx_close, highest_high, highest_high_date, drawdown_points, drawdown_percent)


def detect_rejections(vix_df: pd.DataFrame, spx_df: pd.DataFrame,
                     lookback: int = 10) -> List[BollingerRejection]:
    """
    Detect Bollinger Rejection trigger days with SPX drawdown analysis

    Args:
        vix_df: DataFrame with VIX data and Bollinger Bands (oldest to newest)
        spx_df: DataFrame with SPX data (oldest to newest)
        lookback: SPX lookback window size (default: 10)

    Returns:
        List of BollingerRejection occurrences
    """
    rejections = []

    # Start from index 1 to have a previous day to compare
    # Also skip rows where BB values are NaN (first 19 days)
    for i in range(1, len(vix_df)):
        # Get current and previous day data
        day_b = vix_df.iloc[i]      # Trigger day (Day B)
        day_a = vix_df.iloc[i - 1]  # Previous day (Day A)

        # Skip if Bollinger Bands are not calculated yet (NaN values)
        if pd.isna(day_b['Upper_Band']) or pd.isna(day_a['Upper_Band']):
            continue

        # Check trigger conditions:
        # Day A: Close > Upper Band AND Close > Open
        # Day B: Close < Upper Band AND Close < Open
        if (day_a['Close'] > day_a['Upper_Band'] and
            day_a['Close'] > day_a['Open'] and
            day_b['Close'] < day_b['Upper_Band'] and
            day_b['Close'] < day_b['Open']):

            # Calculate SPX drawdown for this trigger
            spx_result = calculate_spx_drawdown(spx_df, day_b['Date'], lookback)

            if spx_result is None:
                print(f"Warning: No SPX data found for {day_b['Date']}")
                continue

            spx_close, highest_high, highest_high_date, dd_points, dd_pct = spx_result

            rejection = BollingerRejection(
                trigger_date=day_b['Date'],
                vix_close=round(day_b['Close'], 2),
                vix_open=round(day_b['Open'], 2),
                vix_upper_band=round(day_b['Upper_Band'], 2),
                day_a_date=day_a['Date'],
                day_a_close=round(day_a['Close'], 2),
                day_a_open=round(day_a['Open'], 2),
                day_a_upper_band=round(day_a['Upper_Band'], 2),
                spx_close=round(spx_close, 2),
                spx_highest_high=round(highest_high, 2),
                highest_high_date=highest_high_date,
                spx_drawdown_points=round(dd_points, 2),
                spx_drawdown_percent=round(dd_pct, 2)
            )

            rejections.append(rejection)

    return rejections


def load_data(vix_path: Path, spx_path: Path) -> tuple:
    """Load VIX and SPX data"""
    if not vix_path.exists():
        raise FileNotFoundError(f"VIX data file not found: {vix_path}")
    if not spx_path.exists():
        raise FileNotFoundError(f"SPX data file not found: {spx_path}")

    vix_df = pd.read_csv(vix_path)
    spx_df = pd.read_csv(spx_path)

    # Reverse to oldest-to-newest for calculation
    vix_df = vix_df.iloc[::-1].reset_index(drop=True)
    spx_df = spx_df.iloc[::-1].reset_index(drop=True)

    return vix_df, spx_df


def format_rejection(rejection: BollingerRejection, index: int) -> str:
    """Format a single rejection occurrence for display"""
    lines = []
    lines.append(f"  [{index}] Trigger Date: {rejection.trigger_date}")
    lines.append(f"      VIX Close: {rejection.vix_close:.2f} (Open: {rejection.vix_open:.2f}, Upper Band: {rejection.vix_upper_band:.2f})")
    lines.append(f"      VIX Intraday: Close < Open by {rejection.vix_open - rejection.vix_close:.2f} (bearish)")
    lines.append(f"      Day A ({rejection.day_a_date}): VIX Close: {rejection.day_a_close:.2f} (Open: {rejection.day_a_open:.2f}, Upper Band: {rejection.day_a_upper_band:.2f})")
    lines.append(f"      Day A Intraday: Close > Open by {rejection.day_a_close - rejection.day_a_open:.2f} (bullish)")
    lines.append(f"      SPX Close: {rejection.spx_close:,.2f}")
    lines.append(f"      SPX 10-Day High: {rejection.spx_highest_high:,.2f} (on {rejection.highest_high_date})")
    lines.append(f"      SPX Drawdown: {rejection.spx_drawdown_points:,.2f} points ({rejection.spx_drawdown_percent:.2f}%)")
    return '\n'.join(lines)


def generate_report(rejections: List[BollingerRejection], vix_df: pd.DataFrame,
                   spx_df: pd.DataFrame, vix_path: Path, spx_path: Path,
                   lookback: int) -> str:
    """Generate comprehensive report of VIX rejection patterns"""
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("VIX Bollinger Rejection Pattern Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"VIX Data: {vix_path}")
    report_lines.append(f"SPX Data: {spx_path}")
    report_lines.append(f"VIX Bars: {len(vix_df)}, Date Range: {vix_df.iloc[0]['Date']} to {vix_df.iloc[-1]['Date']}")
    report_lines.append(f"SPX Bars: {len(spx_df)}, Date Range: {spx_df.iloc[0]['Date']} to {spx_df.iloc[-1]['Date']}")
    report_lines.append("")

    # Strategy Configuration
    report_lines.append("Strategy Configuration:")
    report_lines.append("  VIX Bollinger Bands: 20-period, 2.0 StdDev (population std)")
    report_lines.append(f"  SPX Lookback Window: {lookback} days")
    report_lines.append("")

    # Strategy Logic
    report_lines.append("Trigger Day Definition:")
    report_lines.append("  Day A (previous day):")
    report_lines.append("    - VIX Close > Upper Bollinger Band")
    report_lines.append("    - VIX Close > VIX Open (bullish intraday move)")
    report_lines.append("  Day B (trigger day):")
    report_lines.append("    - VIX Close < Upper Bollinger Band (rejection)")
    report_lines.append("    - VIX Close < VIX Open (bearish intraday move)")
    report_lines.append("")
    report_lines.append("SPX Drawdown Analysis:")
    report_lines.append(f"  - For each trigger, measure SPX drawdown from {lookback}-day high")
    report_lines.append("")

    # Occurrences section
    report_lines.append("=" * 80)
    report_lines.append("TRIGGER DAY OCCURRENCES")
    report_lines.append("=" * 80)

    if not rejections:
        report_lines.append("\nNo VIX Bollinger Rejection patterns detected in the data.")
    else:
        report_lines.append(f"\nTotal Trigger Days Found: {len(rejections)}")
        report_lines.append("")

        # Sort rejections by trigger_date in reverse chronological order (newest first)
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
        avg_vix_b = sum(r.vix_close for r in rejections) / len(rejections)
        avg_vix_a = sum(r.day_a_close for r in rejections) / len(rejections)

        avg_spx_dd_points = sum(r.spx_drawdown_points for r in rejections) / len(rejections)
        avg_spx_dd_pct = sum(r.spx_drawdown_percent for r in rejections) / len(rejections)
        max_spx_dd_pct = max(r.spx_drawdown_percent for r in rejections)
        min_spx_dd_pct = min(r.spx_drawdown_percent for r in rejections)

        report_lines.append("")
        report_lines.append("VIX Statistics:")
        report_lines.append(f"  Average VIX on Day B (trigger): {avg_vix_b:.2f}")
        report_lines.append(f"  Average VIX on Day A (spike): {avg_vix_a:.2f}")
        report_lines.append("")
        report_lines.append("SPX Drawdown Statistics:")
        report_lines.append(f"  Average Drawdown: {avg_spx_dd_points:,.2f} points ({avg_spx_dd_pct:.2f}%)")
        report_lines.append(f"  Max Drawdown: {max_spx_dd_pct:.2f}%")
        report_lines.append(f"  Min Drawdown: {min_spx_dd_pct:.2f}%")

    report_lines.append("=" * 80)

    return '\n'.join(report_lines)


def main():
    print("=" * 80)
    print("VIX Bollinger Rejection Pattern Scanner")
    print("=" * 80)

    # Setup paths
    project_root = Path(__file__).parent.parent
    vix_path = project_root / 'data' / 'storage' / 'VIX_X.csv'
    spx_path = project_root / 'data' / 'storage' / 'SPX_X.csv'

    # Configuration
    lookback_days = 10

    print(f"VIX Data: {vix_path}")
    print(f"SPX Data: {spx_path}")
    print(f"SPX Lookback: {lookback_days} days")
    print()

    # Load data
    try:
        vix_df, spx_df = load_data(vix_path, spx_path)
        print(f"Loaded VIX data: {len(vix_df)} bars")
        print(f"Loaded SPX data: {len(spx_df)} bars")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Calculate VIX Bollinger Bands
    print("\nCalculating VIX Bollinger Bands (20-period, 2 StdDev, population std)...")
    vix_df = calculate_bollinger_bands(vix_df, period=20, num_std=2.0)

    # Detect rejections with SPX drawdown
    print("Scanning for VIX Bollinger rejection patterns with SPX drawdown analysis...")
    rejections = detect_rejections(vix_df, spx_df, lookback=lookback_days)

    print(f"\n✓ Found {len(rejections)} VIX rejection trigger days")

    # Generate report
    report_content = generate_report(rejections, vix_df, spx_df, vix_path, spx_path, lookback_days)

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
