"""
MACD Wind Tunnel Pattern Scanner

Detects the MACD Wind Tunnel bullish pattern based on MACD and Signal Line
relationship over four consecutive trading days.

Pattern Definition (4-day pattern):

Day 0 (Prerequisite):
- MACD > Signal (setup condition)

Day 1 (Bearish Crossover):
- MACD crosses down below Signal
- MACD < Signal

Day 2 (Consolidation in Positive Territory - "Wind Tunnel"):
- MACD < Signal (still below)
- MACD > 0 (stays in positive territory)

Day 3 (Bullish Crossover - SIGNAL):
- MACD crosses back up above Signal
- MACD > Signal
- This confirms the pattern and generates the bullish signal

The pattern represents a temporary dip below the signal line while staying
positive, followed by a bullish reversal - like passing through a "wind tunnel".
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, NamedTuple


class WindTunnelOccurrence(NamedTuple):
    """Represents a single MACD Wind Tunnel pattern occurrence"""
    signal_date: str
    days_data: List[Dict]  # List of 4 days (Day 0-3) with MACD/Signal data


def load_macd_data(symbol: str, macd_dir: Path) -> pd.DataFrame:
    """Load MACD data for a symbol"""
    csv_path = macd_dir / f"{symbol}_MACD.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    # Filter out rows where MACD or Signal are empty/missing
    df = df[df['MACD'] != ''].copy()
    df = df[df['Signal'] != ''].copy()

    # Convert MACD and Signal to float
    df['MACD'] = pd.to_numeric(df['MACD'], errors='coerce')
    df['Signal'] = pd.to_numeric(df['Signal'], errors='coerce')

    # Drop rows with NaN values
    df = df.dropna(subset=['MACD', 'Signal'])

    return df


def detect_pattern(df: pd.DataFrame, start_date: str = '2024-08-01') -> List[WindTunnelOccurrence]:
    """
    Detect all occurrences of the MACD Wind Tunnel pattern

    Args:
        df: DataFrame with columns Date, MACD, Signal (sorted newest to oldest)
        start_date: Only detect patterns from this date onwards

    Returns:
        List of WindTunnelOccurrence objects
    """
    occurrences = []

    # We need at least 4 rows for the pattern
    if len(df) < 4:
        return occurrences

    # Scan through data (oldest to newest in CSV)
    # We check if rows i, i+1, i+2, i+3 form the pattern
    # Where i is Day 0 (oldest date) and i+3 is Day 3 (signal day, newest date)
    for i in range(len(df) - 3):
        day0 = df.iloc[i]      # Day 0: Prerequisite day (oldest date)
        day1 = df.iloc[i + 1]  # Day 1: Bearish crossover day
        day2 = df.iloc[i + 2]  # Day 2: In tunnel day
        day3 = df.iloc[i + 3]  # Day 3: Signal day (newest date)

        # Only check patterns from start_date onwards (check the signal date)
        if day3['Date'] < start_date:
            continue

        # Day 0: MACD > Signal (prerequisite)
        day0_setup = day0['MACD'] > day0['Signal']

        # Day 1: Bearish crossover (MACD crosses down below Signal)
        day1_crossover = (day0['MACD'] > day0['Signal'] and
                         day1['MACD'] < day1['Signal'])

        # Day 2: In tunnel (MACD < Signal, but MACD > 0)
        day2_in_tunnel = (day2['MACD'] < day2['Signal'] and
                         day2['MACD'] > 0)

        # Day 3: Bullish crossover (MACD crosses back up above Signal)
        day3_crossover = (day2['MACD'] < day2['Signal'] and
                         day3['MACD'] > day3['Signal'])

        # All conditions must be true
        if (day0_setup and day1_crossover and day2_in_tunnel and day3_crossover):
            # Found a pattern occurrence
            days_data = [
                {
                    'date': day0['Date'],
                    'macd': day0['MACD'],
                    'signal': day0['Signal'],
                    'label': 'Setup'
                },
                {
                    'date': day1['Date'],
                    'macd': day1['MACD'],
                    'signal': day1['Signal'],
                    'label': 'Bearish Crossover'
                },
                {
                    'date': day2['Date'],
                    'macd': day2['MACD'],
                    'signal': day2['Signal'],
                    'label': 'In Tunnel'
                },
                {
                    'date': day3['Date'],
                    'macd': day3['MACD'],
                    'signal': day3['Signal'],
                    'label': 'Bullish Crossover'
                }
            ]

            occurrences.append(WindTunnelOccurrence(
                signal_date=day3['Date'],
                days_data=days_data
            ))

    return occurrences


def format_occurrence(occurrence: WindTunnelOccurrence, index: int) -> str:
    """Format a single occurrence for display"""
    lines = []
    lines.append(f"  [{index}] Signal Date: {occurrence.signal_date}")

    for i, day_data in enumerate(occurrence.days_data):
        macd_vs_signal = "MACD > Signal" if day_data['macd'] > day_data['signal'] else "MACD < Signal"
        macd_vs_zero = f"MACD > 0" if day_data['macd'] > 0 else f"MACD < 0"

        line = (
            f"      Day {i} ({day_data['date']}): "
            f"MACD={day_data['macd']:.5f} Signal={day_data['signal']:.5f} "
            f"({macd_vs_signal}, {macd_vs_zero}) - {day_data['label']}"
        )

        if i == 3:
            line += " ⚡ SIGNAL"

        lines.append(line)

    lines.append("")  # Empty line after each occurrence
    return "\n".join(lines)


def generate_report(all_results: Dict[str, List[WindTunnelOccurrence]],
                   macd_dir: Path, start_date: str) -> str:
    """
    Generate the full report text

    Args:
        all_results: Dict[symbol] -> List[occurrences]
        macd_dir: Path to MACD data directory
        start_date: Start date for pattern detection
    """
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("MACD Wind Tunnel Pattern Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Data Source: {macd_dir}/")
    report_lines.append(f"Symbols Scanned: {len(all_results)}")
    report_lines.append(f"Pattern Detection Start Date: {start_date}")
    report_lines.append("")

    # Pattern description
    report_lines.append("Pattern Definition:")
    report_lines.append("  Day 0: MACD > Signal (Setup)")
    report_lines.append("  Day 1: MACD crosses below Signal (Bearish Crossover)")
    report_lines.append("  Day 2: MACD < Signal, MACD > 0 (In Tunnel)")
    report_lines.append("  Day 3: MACD crosses above Signal (Bullish Crossover - SIGNAL)")
    report_lines.append("")

    # Pattern occurrences section
    report_lines.append("=" * 80)
    report_lines.append("PATTERN OCCURRENCES")
    report_lines.append("=" * 80)
    report_lines.append("")

    symbols_with_pattern = {s: o for s, o in all_results.items() if len(o) > 0}

    if not symbols_with_pattern:
        report_lines.append("No patterns found in any symbol.")
        report_lines.append("")
    else:
        # Sort symbols by number of occurrences (descending)
        sorted_symbols = sorted(symbols_with_pattern.items(),
                              key=lambda x: len(x[1]),
                              reverse=True)

        for symbol, occurrences in sorted_symbols:
            plural = "occurrence" if len(occurrences) == 1 else "occurrences"
            report_lines.append(f"Symbol: {symbol} ({len(occurrences)} {plural})")
            # Sort occurrences by signal_date in reverse chronological order
            sorted_occurrences = sorted(occurrences, key=lambda x: x.signal_date, reverse=True)
            for idx, occurrence in enumerate(sorted_occurrences, 1):
                report_lines.append(format_occurrence(occurrence, idx))

    # Summary section
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)

    total_occurrences = sum(len(o) for o in all_results.values())
    symbols_with_count = len(symbols_with_pattern)
    total_symbols = len(all_results)

    report_lines.append(f"Total Symbols with Pattern: {symbols_with_count} / {total_symbols}")
    report_lines.append(f"Total Pattern Occurrences: {total_occurrences}")
    report_lines.append("")

    if symbols_with_pattern:
        report_lines.append("Symbols with Pattern:")
        for symbol, occurrences in sorted(symbols_with_pattern.items(),
                                         key=lambda x: len(x[1]),
                                         reverse=True):
            count = len(occurrences)
            plural = "occurrence" if count == 1 else "occurrences"
            report_lines.append(f"  {symbol}: {count} {plural}")
        report_lines.append("")

    symbols_without = [s for s, o in all_results.items() if len(o) == 0]
    if symbols_without:
        report_lines.append(f"Symbols without Pattern: {len(symbols_without)}")
        report_lines.append(f"  {', '.join(sorted(symbols_without))}")
        report_lines.append("")

    report_lines.append("=" * 80)

    return "\n".join(report_lines)


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    macd_dir = project_root / 'indicators' / 'output' / 'macd'
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)

    # Pattern detection start date
    start_date = '2024-08-01'

    print("=" * 80)
    print("Scanning for MACD Wind Tunnel Patterns")
    print("=" * 80)
    print(f"MACD data directory: {macd_dir}")
    print(f"Start date: {start_date}")
    print()

    # Get all MACD files
    macd_files = sorted(macd_dir.glob("*_MACD.csv"))
    symbols = [f.stem.replace('_MACD', '') for f in macd_files]

    if not symbols:
        print("No MACD files found in directory!")
        return

    print(f"Scanning {len(symbols)} symbols...")
    print()

    # Scan each symbol
    all_results = {}
    for symbol in symbols:
        df = load_macd_data(symbol, macd_dir)
        if df is not None and len(df) > 0:
            occurrences = detect_pattern(df, start_date)
            all_results[symbol] = occurrences

            if occurrences:
                print(f"  {symbol}: {len(occurrences)} pattern(s) found")
        else:
            all_results[symbol] = []
            print(f"  {symbol}: Could not load data or no valid MACD/Signal values")

    print()
    print("Scan complete!")
    print()

    # Generate report
    report_text = generate_report(all_results, macd_dir, start_date)

    # Save report to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = reports_dir / f"macd_wind_tunnel_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"Report saved to: {report_path}")
    print()
    print("=" * 80)
    print("REPORT PREVIEW")
    print("=" * 80)
    print(report_text)


if __name__ == '__main__':
    main()
