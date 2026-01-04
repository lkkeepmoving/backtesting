"""
VIX Spike SPX Drawdown Strategy

This strategy identifies VIX volatility spikes (VIX close above upper Bollinger Band
with bullish intraday action) and measures the corresponding SPX drawdown from the recent high.

Strategy Logic:
1. Identify trigger days:
   - VIX Close > Upper Bollinger Band
   - VIX Close > VIX Open (bullish intraday VIX move)
2. For each trigger day t, look back 10 SPX trading days [t-9, t]
3. Find the highest High in that 10-day SPX window
4. Calculate drawdown: Highest High - SPX Close on trigger day t

This provides insight into the relationship between VIX spikes and market drawdowns.

Bollinger Bands Configuration (for VIX):
- Period: 20 days
- Standard Deviation: 2.0 (population std, ddof=0)
- Upper Band: SMA + 2 * StdDev
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, NamedTuple, Optional


class VixSpikeDrawdown(NamedTuple):
    """Represents a VIX spike event with corresponding SPX drawdown"""
    trigger_date: str
    vix_close: float
    vix_upper_band: float
    spx_close: float
    spx_highest_high: float
    highest_high_date: str
    drawdown_points: float
    drawdown_percent: float


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


def identify_vix_spikes(vix_df: pd.DataFrame) -> List[int]:
    """
    Identify days where VIX closes above upper Bollinger Band and Close > Open

    Args:
        vix_df: DataFrame with VIX data and Bollinger Bands (oldest to newest)

    Returns:
        List of indices where VIX > Upper Band and Close > Open
    """
    spike_indices = []

    for i in range(len(vix_df)):
        row = vix_df.iloc[i]

        # Skip if BB not calculated
        if pd.isna(row['Upper_Band']):
            continue

        # Check if VIX close > upper band AND close > open
        if row['Close'] > row['Upper_Band'] and row['Close'] > row['Open']:
            spike_indices.append(i)

    return spike_indices


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


def analyze_vix_spikes(vix_df: pd.DataFrame, spx_df: pd.DataFrame,
                      lookback: int = 10) -> List[VixSpikeDrawdown]:
    """
    Analyze VIX spikes and calculate corresponding SPX drawdowns

    Args:
        vix_df: DataFrame with VIX data and Bollinger Bands
        spx_df: DataFrame with SPX data
        lookback: SPX lookback window size (default: 10)

    Returns:
        List of VixSpikeDrawdown events
    """
    events = []

    # Identify VIX spike days
    spike_indices = identify_vix_spikes(vix_df)

    print(f"Found {len(spike_indices)} VIX spike days")

    for idx in spike_indices:
        vix_row = vix_df.iloc[idx]
        trigger_date = vix_row['Date']

        # Calculate SPX drawdown for this trigger
        spx_result = calculate_spx_drawdown(spx_df, trigger_date, lookback)

        if spx_result is None:
            print(f"Warning: No SPX data found for {trigger_date}")
            continue

        spx_close, highest_high, highest_high_date, dd_points, dd_pct = spx_result

        event = VixSpikeDrawdown(
            trigger_date=trigger_date,
            vix_close=round(vix_row['Close'], 2),
            vix_upper_band=round(vix_row['Upper_Band'], 2),
            spx_close=round(spx_close, 2),
            spx_highest_high=round(highest_high, 2),
            highest_high_date=highest_high_date,
            drawdown_points=round(dd_points, 2),
            drawdown_percent=round(dd_pct, 2)
        )

        events.append(event)

    return events


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


def format_event(event: VixSpikeDrawdown, index: int) -> str:
    """Format a single VIX spike event for display"""
    lines = []
    lines.append(f"  [{index}] Trigger Date: {event.trigger_date}")
    lines.append(f"      VIX Close: {event.vix_close:.2f} (Upper Band: {event.vix_upper_band:.2f})")
    lines.append(f"      VIX Spike: {event.vix_close - event.vix_upper_band:.2f} above upper band")
    lines.append(f"      SPX Close: {event.spx_close:,.2f}")
    lines.append(f"      SPX 10-Day High: {event.spx_highest_high:,.2f} (on {event.highest_high_date})")
    lines.append(f"      Drawdown: {event.drawdown_points:,.2f} points ({event.drawdown_percent:.2f}%)")
    return '\n'.join(lines)


def generate_report(events: List[VixSpikeDrawdown], vix_df: pd.DataFrame,
                   spx_df: pd.DataFrame, vix_path: Path, spx_path: Path,
                   lookback: int) -> str:
    """Generate comprehensive report of VIX spike events"""
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("VIX Spike SPX Drawdown Strategy Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"VIX Data: {vix_path}")
    report_lines.append(f"SPX Data: {spx_path}")
    report_lines.append(f"VIX Bars: {len(vix_df)}, Date Range: {vix_df.iloc[0]['Date']} to {vix_df.iloc[-1]['Date']}")
    report_lines.append(f"SPX Bars: {len(spx_df)}, Date Range: {spx_df.iloc[0]['Date']} to {spx_df.iloc[-1]['Date']}")
    report_lines.append("")

    # Strategy Configuration
    report_lines.append("Strategy Configuration:")
    report_lines.append(f"  VIX Bollinger Bands: 20-period, 2.0 StdDev (population std)")
    report_lines.append(f"  SPX Lookback Window: {lookback} days")
    report_lines.append("")

    # Strategy Logic
    report_lines.append("Strategy Logic:")
    report_lines.append("  1. Identify trigger days:")
    report_lines.append("     - VIX Close > Upper Bollinger Band")
    report_lines.append("     - VIX Close > VIX Open (bullish intraday VIX move)")
    report_lines.append(f"  2. For each trigger day t, look back {lookback} SPX trading days [t-9, t]")
    report_lines.append("  3. Find the highest High in the SPX lookback window")
    report_lines.append("  4. Calculate drawdown: Highest High - SPX Close on trigger day")
    report_lines.append("")

    # Events section
    report_lines.append("=" * 80)
    report_lines.append("VIX SPIKE EVENTS")
    report_lines.append("=" * 80)

    if not events:
        report_lines.append("\nNo VIX spike events detected in the data.")
    else:
        report_lines.append(f"\nTotal VIX Spike Events: {len(events)}")
        report_lines.append("")

        # Sort by trigger date in reverse chronological order (newest first)
        sorted_events = sorted(events, key=lambda x: x.trigger_date, reverse=True)

        for idx, event in enumerate(sorted_events, 1):
            report_lines.append(format_event(event, idx))
            report_lines.append("")

    # Summary section
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Total VIX Spike Events: {len(events)}")

    if events:
        # Calculate statistics
        avg_vix = sum(e.vix_close for e in events) / len(events)
        max_vix = max(e.vix_close for e in events)
        min_vix = min(e.vix_close for e in events)

        avg_dd_points = sum(e.drawdown_points for e in events) / len(events)
        avg_dd_pct = sum(e.drawdown_percent for e in events) / len(events)
        max_dd_points = max(e.drawdown_points for e in events)
        max_dd_pct = max(e.drawdown_percent for e in events)
        min_dd_points = min(e.drawdown_points for e in events)
        min_dd_pct = min(e.drawdown_percent for e in events)

        report_lines.append("")
        report_lines.append("VIX Statistics:")
        report_lines.append(f"  Average VIX Level: {avg_vix:.2f}")
        report_lines.append(f"  Max VIX Level: {max_vix:.2f}")
        report_lines.append(f"  Min VIX Level: {min_vix:.2f}")
        report_lines.append("")
        report_lines.append("SPX Drawdown Statistics:")
        report_lines.append(f"  Average Drawdown: {avg_dd_points:,.2f} points ({avg_dd_pct:.2f}%)")
        report_lines.append(f"  Max Drawdown: {max_dd_points:,.2f} points ({max_dd_pct:.2f}%)")
        report_lines.append(f"  Min Drawdown: {min_dd_points:,.2f} points ({min_dd_pct:.2f}%)")

    report_lines.append("=" * 80)

    return '\n'.join(report_lines)


def main():
    print("=" * 80)
    print("VIX Spike SPX Drawdown Strategy Scanner")
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

    # Analyze VIX spikes and SPX drawdowns
    print("Analyzing VIX spikes and calculating SPX drawdowns...")
    events = analyze_vix_spikes(vix_df, spx_df, lookback=lookback_days)

    print(f"\n✓ Analyzed {len(events)} VIX spike events")

    # Generate report
    report_content = generate_report(events, vix_df, spx_df, vix_path, spx_path, lookback_days)

    # Save report
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = reports_dir / f'vix_spike_spx_drawdown_{timestamp}.txt'

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
