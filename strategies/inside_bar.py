"""
Inside Bar Pattern Scanner

Detects three levels of inside bar patterns:
1. Single Inside Bar: Today's range is within yesterday's range
2. Two Consecutive Inside Bars: Two days in a row, each inside the previous
3. Three Consecutive Inside Bars: Three days in a row, each inside the previous

Pattern Definition:
An Inside Bar occurs when:
- High[today] < High[yesterday]
- Low[today] > Low[yesterday]
- High[today] <= 0.90 * Highest_High[previous 14 days]
  (i.e., today's high is at least 10% lower than the highest high of the previous 14 trading days)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, NamedTuple


class InsideBarOccurrence(NamedTuple):
    """Represents a single inside bar pattern occurrence"""
    signal_date: str
    pattern_type: str  # 'single', 'two', 'three'
    days_data: List[Dict]  # List of day data (Day 0 + pattern days)
    highest_high_date: str  # Date of highest high in previous 14 days
    highest_high_value: float  # Value of highest high
    drop_percentage: float  # Percentage drop from highest high


def load_symbol_data(symbol: str, storage_dir: Path) -> pd.DataFrame:
    """Load CSV data for a symbol"""
    csv_path = storage_dir / f"{symbol}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    return df


def is_inside_bar(current_day: pd.Series, previous_day: pd.Series) -> bool:
    """
    Check if current_day is an inside bar relative to previous_day

    Returns True if:
    - current_day High < previous_day High
    - current_day Low > previous_day Low
    """
    return (current_day['High'] < previous_day['High'] and
            current_day['Low'] > previous_day['Low'])


def get_14day_high_details(df: pd.DataFrame, current_index: int) -> Dict:
    """
    Get details about the highest high over the previous 14 trading days

    Args:
        df: DataFrame sorted newest to oldest
        current_index: Index of the current day

    Returns:
        Dict with keys: passes_filter (bool), highest_high_date (str),
        highest_high_value (float), drop_percentage (float)
    """
    # Need at least 14 days before current day
    if current_index + 14 >= len(df):
        return {'passes_filter': False}

    current_high = df.iloc[current_index]['High']

    # Get the previous 14 days (indices current_index+1 to current_index+14)
    prev_14_days = df.iloc[current_index + 1:current_index + 15]

    # Find the highest high and which day it occurred
    highest_high_idx = prev_14_days['High'].idxmax()
    highest_high_value = prev_14_days.loc[highest_high_idx, 'High']
    highest_high_date = prev_14_days.loc[highest_high_idx, 'Date']

    # Calculate percentage drop
    drop_percentage = ((highest_high_value - current_high) / highest_high_value) * 100

    # Check if current high is at least 10% lower
    passes_filter = current_high <= highest_high_value * 0.90

    return {
        'passes_filter': passes_filter,
        'highest_high_date': highest_high_date,
        'highest_high_value': highest_high_value,
        'drop_percentage': drop_percentage
    }


def detect_patterns(df: pd.DataFrame) -> Dict[str, List[InsideBarOccurrence]]:
    """
    Detect all three levels of inside bar patterns

    Args:
        df: DataFrame with columns Date, Open, High, Low, Close, Volume
            Must be sorted newest to oldest

    Returns:
        Dict with keys 'single', 'two', 'three' containing lists of occurrences
    """
    results = {
        'single': [],
        'two': [],
        'three': []
    }

    if len(df) < 2:
        return results

    # Scan for patterns (newest to oldest)
    for i in range(len(df) - 1):
        current = df.iloc[i]
        prev = df.iloc[i + 1]

        # Check if current day is an inside bar
        if is_inside_bar(current, prev):
            # Apply 14-day high filter and get details
            high_details = get_14day_high_details(df, i)
            if not high_details['passes_filter']:
                continue

            # Found at least a single inside bar that passes the filter
            days_data = [
                {
                    'date': prev['Date'],
                    'high': prev['High'],
                    'low': prev['Low'],
                    'is_inside': False
                },
                {
                    'date': current['Date'],
                    'high': current['High'],
                    'low': current['Low'],
                    'is_inside': True
                }
            ]

            results['single'].append(InsideBarOccurrence(
                signal_date=current['Date'],
                pattern_type='single',
                days_data=days_data,
                highest_high_date=high_details['highest_high_date'],
                highest_high_value=high_details['highest_high_value'],
                drop_percentage=high_details['drop_percentage']
            ))

            # Check for two consecutive inside bars
            if i + 2 < len(df):
                prev2 = df.iloc[i + 2]
                if is_inside_bar(prev, prev2):
                    # Found two consecutive inside bars
                    days_data = [
                        {
                            'date': prev2['Date'],
                            'high': prev2['High'],
                            'low': prev2['Low'],
                            'is_inside': False
                        },
                        {
                            'date': prev['Date'],
                            'high': prev['High'],
                            'low': prev['Low'],
                            'is_inside': True,
                            'label': 'Inside Bar 1'
                        },
                        {
                            'date': current['Date'],
                            'high': current['High'],
                            'low': current['Low'],
                            'is_inside': True,
                            'label': 'Inside Bar 2'
                        }
                    ]

                    results['two'].append(InsideBarOccurrence(
                        signal_date=current['Date'],
                        pattern_type='two',
                        days_data=days_data,
                        highest_high_date=high_details['highest_high_date'],
                        highest_high_value=high_details['highest_high_value'],
                        drop_percentage=high_details['drop_percentage']
                    ))

                    # Check for three consecutive inside bars
                    if i + 3 < len(df):
                        prev3 = df.iloc[i + 3]
                        if is_inside_bar(prev2, prev3):
                            # Found three consecutive inside bars
                            days_data = [
                                {
                                    'date': prev3['Date'],
                                    'high': prev3['High'],
                                    'low': prev3['Low'],
                                    'is_inside': False
                                },
                                {
                                    'date': prev2['Date'],
                                    'high': prev2['High'],
                                    'low': prev2['Low'],
                                    'is_inside': True,
                                    'label': 'Inside Bar 1'
                                },
                                {
                                    'date': prev['Date'],
                                    'high': prev['High'],
                                    'low': prev['Low'],
                                    'is_inside': True,
                                    'label': 'Inside Bar 2'
                                },
                                {
                                    'date': current['Date'],
                                    'high': current['High'],
                                    'low': current['Low'],
                                    'is_inside': True,
                                    'label': 'Inside Bar 3'
                                }
                            ]

                            results['three'].append(InsideBarOccurrence(
                                signal_date=current['Date'],
                                pattern_type='three',
                                days_data=days_data,
                                highest_high_date=high_details['highest_high_date'],
                                highest_high_value=high_details['highest_high_value'],
                                drop_percentage=high_details['drop_percentage']
                            ))

    return results


def format_occurrence(occurrence: InsideBarOccurrence, index: int) -> str:
    """Format a single occurrence for display"""
    lines = []
    lines.append(f"  [{index}] Signal Date: {occurrence.signal_date}")

    # Add 14-day highest high details
    lines.append(f"      14-Day Highest High: {occurrence.highest_high_date} at ${occurrence.highest_high_value:.2f}")
    lines.append(f"      Drop from Peak: {occurrence.drop_percentage:.2f}%")
    lines.append("")

    for i, day_data in enumerate(occurrence.days_data):
        if i == 0:
            # Day 0 (reference day)
            line = f"      Day 0 ({day_data['date']}): High=${day_data['high']:.2f} Low=${day_data['low']:.2f}"
        else:
            # Inside bar days
            label = day_data.get('label', '')
            if label:
                line = f"      Day {i} ({day_data['date']}): High=${day_data['high']:.2f} Low=${day_data['low']:.2f} ({label})"
            else:
                line = f"      Day {i} ({day_data['date']}): High=${day_data['high']:.2f} Low=${day_data['low']:.2f} ⚡ INSIDE BAR"

            # Add signal marker to the last day
            if i == len(occurrence.days_data) - 1:
                line += " ⚡ SIGNAL"

        lines.append(line)

    lines.append("")  # Empty line after each occurrence
    return "\n".join(lines)


def generate_report(all_results: Dict[str, Dict[str, List[InsideBarOccurrence]]],
                   storage_dir: Path) -> str:
    """
    Generate the full report text

    Args:
        all_results: Dict[symbol] -> Dict[pattern_type] -> List[occurrences]
        storage_dir: Path to data storage
    """
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("Inside Bar Pattern Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Data Source: {storage_dir}/")
    report_lines.append(f"Symbols Scanned: {len(all_results)}")

    # Get date range from first symbol
    date_range = ""
    for symbol in all_results.keys():
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

    # Generate sections for each pattern type
    pattern_types = [
        ('single', 'SINGLE INSIDE BAR OCCURRENCES'),
        ('two', 'TWO CONSECUTIVE INSIDE BARS OCCURRENCES'),
        ('three', 'THREE CONSECUTIVE INSIDE BARS OCCURRENCES')
    ]

    for pattern_key, section_title in pattern_types:
        report_lines.append("=" * 80)
        report_lines.append(section_title)
        report_lines.append("=" * 80)
        report_lines.append("")

        # Collect symbols with this pattern type
        symbols_with_pattern = {}
        for symbol, patterns in all_results.items():
            occurrences = patterns.get(pattern_key, [])
            if occurrences:
                symbols_with_pattern[symbol] = occurrences

        if not symbols_with_pattern:
            report_lines.append("No occurrences found.")
            report_lines.append("")
        else:
            # Sort symbols by number of occurrences (descending)
            sorted_symbols = sorted(symbols_with_pattern.items(),
                                   key=lambda x: len(x[1]),
                                   reverse=True)

            for symbol, occurrences in sorted_symbols:
                report_lines.append(f"Symbol: {symbol} ({len(occurrences)} occurrence{'s' if len(occurrences) > 1 else ''})")
                for idx, occurrence in enumerate(occurrences, 1):
                    report_lines.append(format_occurrence(occurrence, idx))

            # Section summary
            total_occurrences = sum(len(occs) for occs in symbols_with_pattern.values())
            report_lines.append(f"Summary: {total_occurrences} total occurrence{'s' if total_occurrences > 1 else ''} across {len(symbols_with_pattern)} symbol{'s' if len(symbols_with_pattern) > 1 else ''}")
            report_lines.append("")

    # Overall summary
    report_lines.append("=" * 80)
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Total Symbols Scanned: {len(all_results)}")
    report_lines.append("")

    for pattern_key, pattern_name in [('single', 'SINGLE INSIDE BAR'),
                                       ('two', 'TWO CONSECUTIVE INSIDE BARS'),
                                       ('three', 'THREE CONSECUTIVE INSIDE BARS')]:
        total_occurrences = 0
        symbols_with_pattern = []
        symbol_counts = []

        for symbol, patterns in all_results.items():
            occurrences = patterns.get(pattern_key, [])
            if occurrences:
                total_occurrences += len(occurrences)
                symbols_with_pattern.append(symbol)
                symbol_counts.append((symbol, len(occurrences)))

        report_lines.append(f"{pattern_name}:")
        report_lines.append(f"  Total Occurrences: {total_occurrences}")
        report_lines.append(f"  Symbols with pattern: {len(symbols_with_pattern)} / {len(all_results)}")

        if symbol_counts:
            # Sort by count descending and show top 5
            symbol_counts.sort(key=lambda x: x[1], reverse=True)
            top_5 = symbol_counts[:5]
            top_5_str = ", ".join([f"{sym} ({cnt})" for sym, cnt in top_5])
            report_lines.append(f"  Top symbols: {top_5_str}")

        report_lines.append("")

    # Symbols with no patterns
    symbols_without_patterns = [
        symbol for symbol, patterns in all_results.items()
        if not any(patterns.values())
    ]

    if symbols_without_patterns:
        report_lines.append(f"Symbols with NO patterns: {', '.join(sorted(symbols_without_patterns))}")
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
    print("Scanning for Inside Bar Patterns")
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
    all_results = {}
    for symbol in symbols:
        df = load_symbol_data(symbol, storage_dir)
        if df is not None:
            patterns = detect_patterns(df)
            all_results[symbol] = patterns

            # Print summary
            single_count = len(patterns['single'])
            two_count = len(patterns['two'])
            three_count = len(patterns['three'])

            if single_count or two_count or three_count:
                parts = []
                if single_count:
                    parts.append(f"Single={single_count}")
                if two_count:
                    parts.append(f"Two={two_count}")
                if three_count:
                    parts.append(f"Three={three_count}")
                print(f"  {symbol}: {', '.join(parts)}")
        else:
            all_results[symbol] = {'single': [], 'two': [], 'three': []}
            print(f"  {symbol}: Could not load data")

    print()
    print("Scan complete!")
    print()

    # Generate report
    report_text = generate_report(all_results, storage_dir)

    # Save report to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = reports_dir / f"inside_bar_{timestamp}.txt"
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
