"""
CSV Export for Anti Pattern Scanner

One row per (symbol, shock_date, direction, config_combination).
No deduplication — preserves full parameter sensitivity data.
"""

import csv
import numpy as np
from typing import List, Dict, Any


CSV_COLUMNS = [
    'symbol',
    'direction',
    'divergence_type',
    'divergence_count',
    'divergence_indicator_period',
    'divergence_threshold',
    'divergence_anchor_date',
    'divergence_anchor_indicator_value',
    'divergence_anchor_close',
    'divergence_end_date',
    'divergence_end_indicator_value',
    'divergence_end_close',
    'divergence_mid_date',
    'divergence_mid_indicator_value',
    'divergence_mid_close',
    'shock_date',
    'shock_close',
    'shock_method',
    'shock_sigma_value',
    'shock_pct_change',
    'shock_sigma_threshold',
    'shock_pct_threshold',
    'macd_at_shock',
    'macd_is_new_extreme',
    'macd_extreme_lookback',
]


def occurrence_to_row(
    symbol: str,
    occ: Dict[str, Any],
    dates: np.ndarray,
) -> Dict[str, Any]:
    """
    Convert an occurrence dict to a flat CSV row dict.

    Args:
        symbol: Stock ticker
        occ: Occurrence dict with 'phase1', 'phase2', 'direction' keys
        dates: Array of date strings for this symbol

    Returns:
        Dict with CSV column keys
    """
    p1 = occ['phase1']
    p2 = occ['phase2']
    div = p1['divergence']
    pivot_bars = div['pivot_bars']

    def get_date(bar):
        if dates is not None and 0 <= bar < len(dates):
            return dates[bar]
        return ''

    # Anchor = first pivot, end = last pivot
    anchor_idx = 0
    end_idx = len(pivot_bars) - 1

    row = {
        'symbol': symbol,
        'direction': occ['direction'],
        'divergence_type': p1['indicator_type'],
        'divergence_count': p1['divergence_count'],
        'divergence_indicator_period': p1['indicator_period'],
        'divergence_threshold': p1['threshold'],
        'divergence_anchor_date': get_date(pivot_bars[anchor_idx]),
        'divergence_anchor_indicator_value': div['pivot_indicator_values'][anchor_idx],
        'divergence_anchor_close': div['pivot_close_values'][anchor_idx],
        'divergence_end_date': get_date(pivot_bars[end_idx]),
        'divergence_end_indicator_value': div['pivot_indicator_values'][end_idx],
        'divergence_end_close': div['pivot_close_values'][end_idx],
        'divergence_mid_date': '',
        'divergence_mid_indicator_value': '',
        'divergence_mid_close': '',
        'shock_date': get_date(p2['shock_bar']),
        'shock_close': p2['close_at_shock'],
        'shock_method': p2['shock_method'],
        'shock_sigma_value': p2['sigma_value'] if p2['sigma_value'] is not None else '',
        'shock_pct_change': p2['pct_change'],
        'shock_sigma_threshold': p2['shock_threshold'] if p2['shock_method'] == 'sigma_spike' else '',
        'shock_pct_threshold': p2['shock_threshold'] if p2['shock_method'] == 'price_pct_change' else '',
        'macd_at_shock': p2['macd_at_shock'],
        'macd_is_new_extreme': p2['macd_is_new_extreme'],
        'macd_extreme_lookback': p2['macd_extreme_lookback'],
    }

    # Fill mid pivot for 3-divergence
    if len(pivot_bars) == 3:
        row['divergence_mid_date'] = get_date(pivot_bars[1])
        row['divergence_mid_indicator_value'] = div['pivot_indicator_values'][1]
        row['divergence_mid_close'] = div['pivot_close_values'][1]

    return row


def export_csv(
    filepath: str,
    all_occurrences: Dict[str, List[Dict[str, Any]]],
    dates_by_symbol: Dict[str, np.ndarray],
    symbols: List[str],
):
    """
    Export all occurrences to CSV. One row per config combination (no dedup).

    Args:
        filepath: Output CSV file path
        all_occurrences: Dict mapping symbol -> list of occurrence dicts
        dates_by_symbol: Dict mapping symbol -> dates array
        symbols: List of all symbols scanned
    """
    rows = []
    for symbol in symbols:
        occs = all_occurrences.get(symbol, [])
        dates = dates_by_symbol.get(symbol)
        for occ in occs:
            rows.append(occurrence_to_row(symbol, occ, dates))

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
