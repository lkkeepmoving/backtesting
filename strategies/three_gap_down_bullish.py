"""
Three Consecutive Gap-Down Bullish Days Pattern Scanner

Scans historical data for the pattern where three consecutive days each have:
1. Gap down: Open < Previous Close
2. Bullish: Close > Previous Close
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple


class PatternOccurrence:
    """Represents a single pattern occurrence"""
    def __init__(self, signal_date: str, days_data: List[Dict]):
        self.signal_date = signal_date
        self.days_data = days_data  # List of 3 days with their OHLC data


def load_symbol_data(symbol: str, storage_dir: Path) -> pd.DataFrame:
    """Load CSV data for a symbol"""
    csv_path = storage_dir / f"{symbol}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    return df


def detect_pattern(df: pd.DataFrame) -> List[PatternOccurrence]:
    """
    Detect all occurrences of the three consecutive gap-down bullish days pattern

    Args:
        df: DataFrame with columns Date, Open, High, Low, Close, Volume
            Must be sorted newest to oldest

    Returns:
        List of PatternOccurrence objects
    """
    occurrences = []

    # We need at least 4 rows (1 day before + 3 pattern days)
    if len(df) < 4:
        return occurrences

    # Iterate through data (newest to oldest)
    # We need to check if rows i, i+1, i+2 form the pattern
    # Where i is the most recent day of the pattern (signal day)
    for i in range(len(df) - 3):
        # Days in the pattern (newest to oldest)
        day3 = df.iloc[i]      # Signal day (most recent)
        day2 = df.iloc[i + 1]  # Middle day
        day1 = df.iloc[i + 2]  # First day of pattern
        day0 = df.iloc[i + 3]  # Day before pattern starts

        # Check pattern conditions for each day
        # Day 1: Gap down from day0, close higher than day0's close
        day1_gap_down = day1['Open'] < day0['Close']
        day1_bullish = day1['Close'] > day0['Close']

        # Day 2: Gap down from day1, close higher than day1's close
        day2_gap_down = day2['Open'] < day1['Close']
        day2_bullish = day2['Close'] > day1['Close']

        # Day 3: Gap down from day2, close higher than day2's close
        day3_gap_down = day3['Open'] < day2['Close']
        day3_bullish = day3['Close'] > day2['Close']

        # All conditions must be true
        if (day1_gap_down and day1_bullish and
            day2_gap_down and day2_bullish and
            day3_gap_down and day3_bullish):

            # Found a pattern occurrence
            days_data = [
                {
                    'date': day1['Date'],
                    'open': day1['Open'],
                    'close': day1['Close'],
                    'prev_close': day0['Close'],
                    'gap_down': True,
                    'bullish': True
                },
                {
                    'date': day2['Date'],
                    'open': day2['Open'],
                    'close': day2['Close'],
                    'prev_close': day1['Close'],
                    'gap_down': True,
                    'bullish': True
                },
                {
                    'date': day3['Date'],
                    'open': day3['Open'],
                    'close': day3['Close'],
                    'prev_close': day2['Close'],
                    'gap_down': True,
                    'bullish': True
                }
            ]

            occurrences.append(PatternOccurrence(
                signal_date=day3['Date'],
                days_data=days_data
            ))

    return occurrences


def format_occurrence_details(occurrence: PatternOccurrence, index: int) -> str:
    """Format a single occurrence for display"""
    lines = []
    lines.append(f"  [{index}] Signal Date: {occurrence.signal_date}")

    for i, day_data in enumerate(occurrence.days_data, 1):
        line = (
            f"      Day {i} ({day_data['date']}): "
            f"Open=${day_data['open']:.2f} < Prev Close=${day_data['prev_close']:.2f} | "
            f"Close=${day_data['close']:.2f} > Prev Close (Bullish)"
        )
        if i == 3:
            line += " ⚡ SIGNAL"
        lines.append(line)

    return "\n".join(lines)


def generate_report(results: Dict[str, List[PatternOccurrence]], storage_dir: Path) -> str:
    """Generate the full report text"""
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("Three Consecutive Gap-Down Bullish Days - Pattern Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Data Source: {storage_dir}/")
    report_lines.append(f"Symbols Scanned: {len(results)}")

    # Get date range from first symbol with data
    date_range = ""
    for symbol, occurrences in results.items():
        csv_path = storage_dir / f"{symbol}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                newest = df.iloc[0]['Date']
                oldest = df.iloc[-1]['Date']
                bar_count = len(df)
                date_range = f"{oldest} to {newest} ({bar_count} bars)"
                break

    report_lines.append(f"Date Range: {date_range}")
    report_lines.append("")

    # Pattern occurrences section
    report_lines.append("=" * 80)
    report_lines.append("PATTERN OCCURRENCES")
    report_lines.append("=" * 80)
    report_lines.append("")

    symbols_with_pattern = {s: o for s, o in results.items() if len(o) > 0}

    if not symbols_with_pattern:
        report_lines.append("No patterns found in any symbol.")
        report_lines.append("")
    else:
        for symbol, occurrences in sorted(symbols_with_pattern.items()):
            report_lines.append(f"Symbol: {symbol}")
            for idx, occurrence in enumerate(occurrences, 1):
                report_lines.append(format_occurrence_details(occurrence, idx))
                report_lines.append("")

            report_lines.append(f"  Total occurrences for {symbol}: {len(occurrences)}")
            report_lines.append("")

    # Summary section
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)

    total_occurrences = sum(len(o) for o in results.values())
    symbols_with_count = len(symbols_with_pattern)
    total_symbols = len(results)

    report_lines.append(f"Total Symbols with Pattern: {symbols_with_count} / {total_symbols}")
    report_lines.append(f"Total Pattern Occurrences: {total_occurrences}")
    report_lines.append("")

    if symbols_with_pattern:
        report_lines.append("Symbols with Pattern:")
        for symbol, occurrences in sorted(symbols_with_pattern.items(),
                                         key=lambda x: len(x[1]), reverse=True):
            count = len(occurrences)
            plural = "occurrence" if count == 1 else "occurrences"
            report_lines.append(f"  {symbol}: {count} {plural}")
        report_lines.append("")

    symbols_without = [s for s, o in results.items() if len(o) == 0]
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
    storage_dir = project_root / 'data' / 'storage'
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Scanning for Three Consecutive Gap-Down Bullish Days Pattern")
    print("=" * 80)
    print(f"Data directory: {storage_dir}")
    print()

    # Get all CSV files
    csv_files = sorted(storage_dir.glob("*.csv"))
    symbols = [f.stem for f in csv_files]

    if not symbols:
        print("No CSV files found in storage directory!")
        return

    print(f"Scanning {len(symbols)} symbols...")
    print()

    # Scan each symbol
    results = {}
    for symbol in symbols:
        df = load_symbol_data(symbol, storage_dir)
        if df is not None:
            occurrences = detect_pattern(df)
            results[symbol] = occurrences
            if occurrences:
                print(f"  {symbol}: {len(occurrences)} pattern(s) found")
        else:
            results[symbol] = []
            print(f"  {symbol}: Could not load data")

    print()
    print("Scan complete!")
    print()

    # Generate report
    report_text = generate_report(results, storage_dir)

    # Save report to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = reports_dir / f"three_gap_down_bullish_{timestamp}.txt"
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
