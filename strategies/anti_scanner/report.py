"""
Human-Readable Report Generator for Anti Pattern Scanner

Generates a .txt report with deduplicated occurrences grouped by
(symbol, shock_date, direction), listing all triggering configs.
"""

import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict


def _format_config_label(occ: Dict[str, Any]) -> str:
    """Format a single config combination as a readable label."""
    p1 = occ['phase1']
    p2 = occ['phase2']

    indicator = p1['indicator_type']
    period = p1['indicator_period']
    threshold = p1['threshold']
    div_count = p1['divergence_count']

    method = p2['shock_method']
    if method == "sigma_spike":
        shock_label = f"sigma_spike >= {p2['shock_threshold']}"
    else:
        pct = p2['shock_threshold'] * 100
        shock_label = f"price_pct >= {pct:.0f}%"

    return f"{indicator}({period}), threshold {threshold}, {div_count}-divergence, {shock_label}"


def _format_divergence(occ: Dict[str, Any], dates: np.ndarray) -> List[str]:
    """Format divergence pivot details."""
    lines = []
    p1 = occ['phase1']
    div = p1['divergence']
    indicator_type = p1['indicator_type']

    for i, bar in enumerate(div['pivot_bars']):
        date = dates[bar] if bar < len(dates) else "N/A"
        ind_val = div['pivot_indicator_values'][i]
        close_val = div['pivot_close_values'][i]

        if i == 0:
            role = "(anchor)"
        elif i == len(div['pivot_bars']) - 1:
            role = "(divergent)"
        else:
            role = ""

        if indicator_type == "MACD" and 'macd_raw_values' in p1:
            raw = p1['macd_raw_values'][i]
            raw_str = f"  MACD={raw:.4f}" if raw is not None else ""
            lines.append(f"       {date}  pctl={ind_val:.0f}{raw_str}  Close=${close_val:.2f}   {role}")
        else:
            lines.append(f"       {date}  {indicator_type}={ind_val:.1f}  Close=${close_val:.2f}   {role}")

    return lines


def generate_report(
    all_occurrences: Dict[str, List[Dict[str, Any]]],
    dates_by_symbol: Dict[str, Any],
    scan_date: str,
    lookback: int,
    direction: str,
    config: dict,
    symbols: List[str],
) -> str:
    """
    Generate human-readable report.

    Args:
        all_occurrences: Dict mapping symbol -> list of occurrence dicts
        dates_by_symbol: Dict mapping symbol -> dates array
        scan_date: End date of scan window
        lookback: Number of bars scanned
        direction: "long", "short", or "both"
        config: Full config dict
        symbols: List of all symbols scanned

    Returns:
        Report string
    """
    lines = []

    # Count total occurrences (deduplicated by symbol, shock_bar, direction)
    total_deduped = 0
    symbols_with_hits = 0
    for symbol in symbols:
        occs = all_occurrences.get(symbol, [])
        if occs:
            # Deduplicate by (shock_bar, direction)
            unique = set()
            for o in occs:
                unique.add((o['phase2']['shock_bar'], o['direction']))
            total_deduped += len(unique)
            symbols_with_hits += 1

    # === Header ===
    lines.append("=" * 65)
    lines.append("ANTI PATTERN SCANNER -- RESULTS")
    lines.append("=" * 65)
    lines.append(f"Scan Date:    {scan_date}")
    lines.append(f"Lookback:     {lookback} bars")
    lines.append(f"Direction:    {direction}")
    lines.append(f"Symbols:      {len(symbols)} scanned")
    lines.append("")

    # Indicator Parameters
    lines.append("Indicator Parameters:")
    rsi_cfg = config.get('rsi', {})
    if rsi_cfg.get('enabled'):
        lines.append("  RSI:")
        lines.append(f"    Periods:            {rsi_cfg.get('periods', [])}")
        thresholds_str = ", ".join(f"{u}/{l}" for u, l in rsi_cfg.get('thresholds', []))
        lines.append(f"    Thresholds:         [{thresholds_str}]")

    macd_cfg = config.get('macd', {})
    if macd_cfg.get('enabled'):
        lines.append("  MACD:")
        lines.append(f"    Fast/Slow/Signal:   {macd_cfg.get('fast_period')}"
                     f" / {macd_cfg.get('slow_period')}"
                     f" / {macd_cfg.get('signal_period')} (SMA)")
        lines.append(f"    Percentile lookback: {macd_cfg.get('percentile_lookback')} bars")
        pctl_str = ", ".join(f"{u}/{l}" for u, l in macd_cfg.get('percentile_thresholds', []))
        lines.append(f"    Pctl thresholds:    [{pctl_str}]")

    div_cfg = config.get('divergence', {})
    lines.append("  Divergence:")
    lines.append(f"    Lookback window:    {div_cfg.get('lookback_window')} bars")
    lines.append(f"    Pivot lookback:     {div_cfg.get('pivot_lookback')} bars")
    lines.append(f"    Min separation:     {div_cfg.get('min_separation')} bars")
    lines.append(f"    Divergence counts:  {div_cfg.get('divergence_counts')}")

    p2_cfg = config.get('phase2', {})
    lines.append("  Phase 2:")
    lines.append(f"    Scan window:        {p2_cfg.get('window')} bars after divergence")
    lines.append(f"    Shock methods:      {', '.join(p2_cfg.get('shock_methods', []))}")
    sigma_cfg = p2_cfg.get('sigma_spike', {})
    lines.append(f"    Sigma thresholds:   {sigma_cfg.get('thresholds')} (lookback: {sigma_cfg.get('lookback')})")
    pct_cfg = p2_cfg.get('price_pct_change', {})
    pct_display = [f"{t*100:.0f}%" for t in pct_cfg.get('thresholds', [])]
    lines.append(f"    Pct thresholds:     [{', '.join(pct_display)}]")
    lines.append(f"    MACD extreme lookback: {p2_cfg.get('macd_extreme', {}).get('lookback')} bars")

    lines.append("")
    lines.append(f"Total occurrences found: {total_deduped} across {symbols_with_hits} symbols")
    lines.append("=" * 65)

    # === Per-Symbol Sections ===
    for symbol in symbols:
        occs = all_occurrences.get(symbol, [])
        dates = dates_by_symbol.get(symbol)

        if not occs:
            lines.append(f"\n{symbol} -- 0 occurrences")
            continue

        # Group by (shock_bar, direction) for deduplication
        grouped = defaultdict(list)
        for o in occs:
            key = (o['phase2']['shock_bar'], o['direction'])
            grouped[key].append(o)

        sorted_keys = sorted(grouped.keys(), key=lambda k: k[0])
        lines.append(f"\n{symbol} -- {len(sorted_keys)} occurrences")
        lines.append("-" * 65)

        for idx, key in enumerate(sorted_keys, 1):
            shock_bar, occ_direction = key
            group = grouped[key]
            representative = group[0]

            shock_date = dates[shock_bar] if dates is not None and shock_bar < len(dates) else "N/A"
            p2 = representative['phase2']

            lines.append(f"\n  #{idx} [{occ_direction.upper()}] Shock date: {shock_date}")

            # List all configs that triggered
            lines.append("     Triggered by configs:")
            for o in group:
                label = _format_config_label(o)
                lines.append(f"       - {label}")

            # Divergence details (from representative)
            lines.append("     Divergence:")
            div_lines = _format_divergence(representative, dates)
            for dl in div_lines:
                lines.append(dl)

            # Shock details
            lines.append("     Shock:")
            lines.append(f"       Date:       {shock_date}")
            lines.append(f"       Close:      ${p2['close_at_shock']:.2f}")
            lines.append(f"       Pct change: {p2['pct_change']:+.1%}")
            if p2['sigma_value'] is not None:
                lines.append(f"       Sigma:      {p2['sigma_value']:+.1f}s")
            lines.append(f"       MACD fast:  {p2['macd_at_shock']:.4f}"
                         f" (new {p2['macd_extreme_lookback']}-bar"
                         f" {'low' if occ_direction == 'short' else 'high'} +)")

        lines.append("")
        lines.append("-" * 65)

    # === Summary ===
    lines.append("")
    lines.append("=" * 65)
    lines.append("SUMMARY")
    lines.append("=" * 65)
    lines.append(f"{'Symbol':<10} {'Direction':<12} {'Count':<8} Shock Dates")

    symbols_with_zero = []
    for symbol in symbols:
        occs = all_occurrences.get(symbol, [])
        if not occs:
            symbols_with_zero.append(symbol)
            continue

        grouped = defaultdict(list)
        for o in occs:
            key = (o['phase2']['shock_bar'], o['direction'])
            grouped[key].append(o)

        # Group by direction for summary
        by_dir = defaultdict(list)
        for (shock_bar, d), group in grouped.items():
            dates = dates_by_symbol.get(symbol)
            shock_date = dates[shock_bar] if dates is not None and shock_bar < len(dates) else "N/A"
            by_dir[d].append(shock_date)

        for d in sorted(by_dir.keys()):
            shock_dates = sorted(by_dir[d])
            lines.append(f"{symbol:<10} {d.upper():<12} {len(shock_dates):<8} {', '.join(shock_dates)}")

    if symbols_with_zero:
        lines.append(f"\nSymbols with 0 occurrences ({len(symbols_with_zero)}): {', '.join(symbols_with_zero)}")

    lines.append("=" * 65)

    return '\n'.join(lines)
