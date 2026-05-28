"""
RSI Divergence Scanner

Standalone tool that scans a stock list for confirmed RSI bullish 2-divergences
as of a specific date (or date range).

A divergence qualifies when:
  - The last RSI pivot trough falls exactly `last_pivot_right_bars` trading days
    before --date (default: 1, i.e. yesterday)
  - --date itself is the confirmation bar (Day D): it must be strictly higher
    than the last pivot's price, completing the trough confirmation

Output: a clean candidate list with pivot dates, RSI values, and prices.
        No forward return analysis — Day D is today, the future is unknown.

Single-date mode:
  python tools/rsi_divergence_scanner/scanner.py --date 2026-05-26
  python tools/rsi_divergence_scanner/scanner.py --date 2026-05-26 --symbols AAPL,MSFT,NVDA
  python tools/rsi_divergence_scanner/scanner.py --date 2026-05-26 --symbols-file data/anti_symbols.csv
  python tools/rsi_divergence_scanner/scanner.py --date 2026-05-26 --config tools/rsi_divergence_scanner/config.yaml

Date-range mode:
  python tools/rsi_divergence_scanner/scanner.py --start-date 2026-05-01 --end-date 2026-05-26
  python tools/rsi_divergence_scanner/scanner.py --start-date 2026-05-01 --end-date 2026-05-26 --symbols AAPL,MSFT
"""

import argparse
import csv
import logging
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# RSI Calculation  (standalone)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's smoothing RSI.  First `period` values are NaN.
    close must be sorted oldest→newest.
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    deltas = np.diff(close)
    gains  = np.where(deltas > 0,  deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    def _rsi_val(ag: float, al: float) -> float:
        if al == 0.0:
            return 100.0
        return 100.0 - (100.0 / (1.0 + ag / al))

    rsi[period] = _rsi_val(avg_gain, avg_loss)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i])  / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi[i + 1] = _rsi_val(avg_gain, avg_loss)

    return rsi


# ═════════════════════════════════════════════════════════════════════════════
# Pivot Detection  (standalone)
# ═════════════════════════════════════════════════════════════════════════════

def _find_pivot_lows(
    values: np.ndarray,
    start: int,
    end: int,
    pivot_lookback: int,
) -> List[int]:
    """
    Return indices of pivot lows in values[start:end+1].
    Bar j is a pivot low if values[j] <= values[j±k] for all k in 1..pivot_lookback.
    """
    pivots = []
    scan_start = max(start, pivot_lookback)
    scan_end   = min(end, len(values) - 1 - pivot_lookback)

    for j in range(scan_start, scan_end + 1):
        if np.isnan(values[j]):
            continue
        ok = True
        for k in range(1, pivot_lookback + 1):
            if np.isnan(values[j - k]) or np.isnan(values[j + k]):
                ok = False
                break
            if values[j] > values[j - k] or values[j] > values[j + k]:
                ok = False
                break
        if ok:
            pivots.append(j)

    return pivots


def _no_lower_between(rsi: np.ndarray, a: int, b: int, ref: float) -> bool:
    """
    True if no RSI value strictly between indices a and b falls below `ref`.
    Trivially True when there are no bars between a and b.
    """
    if b - a <= 1:
        return True
    segment = rsi[a + 1:b]
    valid   = segment[~np.isnan(segment)]
    return len(valid) == 0 or float(np.min(valid)) >= ref


def _no_overbought_in_range(rsi: np.ndarray, a: int, b: int, upper_thresh: float) -> bool:
    """
    True if no RSI value in rsi[a..b] (inclusive) exceeds upper_thresh.
    Ensures no overbought recovery occurred anywhere inside the divergence window.
    """
    segment = rsi[a:b + 1]
    valid   = segment[~np.isnan(segment)]
    return len(valid) == 0 or float(np.max(valid)) <= upper_thresh


def _has_price_bounce(close: np.ndarray, a: int, b: int) -> bool:
    """
    True if close price recovered above close[a] at any bar strictly between a and b.
    Ensures the two troughs belong to separate price cycles, not one continuous selloff.
    Uses close (not low) so intraday wicks don't count as a recovery.
    """
    if b - a <= 1:
        return False
    segment = close[a + 1:b]
    return len(segment) > 0 and float(np.max(segment)) > float(close[a])


# ═════════════════════════════════════════════════════════════════════════════
# 2-Divergence Detection  (standalone, bullish only)
# ═════════════════════════════════════════════════════════════════════════════

def _detect_2div(
    current_bar: int,
    close: np.ndarray,
    low: np.ndarray,
    rsi: np.ndarray,
    lookback_window: int,
    lower_threshold: float,
    upper_threshold: float,
    wing_bars: int,
    last_pivot_right_bars: int,
    min_separation: int,
    strict_threshold: bool,
    pivot_source: str = "close",
) -> List[Dict[str, Any]]:
    """
    Detect bullish RSI 2-divergences whose Day D is exactly current_bar.

    last pivot = current_bar - last_pivot_right_bars
    All `last_pivot_right_bars` bars between last_pivot and current_bar (inclusive
    of current_bar) must be strictly higher than last_pivot's price.

    Returns a list of divergence dicts (usually 0 or 1).
    """
    price_arr = low if pivot_source == "low" else close
    n = len(price_arr)

    last_pivot = current_bar - last_pivot_right_bars

    # ── Guard: enough room to the left of last_pivot ──────────────────────────
    if last_pivot < wing_bars or current_bar >= n:
        return []

    # ── Last pivot: right-side check ──────────────────────────────────────────
    if np.isnan(price_arr[last_pivot]):
        return []
    for k in range(1, last_pivot_right_bars + 1):
        idx = last_pivot + k
        if idx >= n or np.isnan(price_arr[idx]):
            return []
        if price_arr[idx] <= price_arr[last_pivot]:
            return []   # not strictly higher → trough not confirmed

    # ── Last pivot: left-side check ───────────────────────────────────────────
    for k in range(1, wing_bars + 1):
        idx = last_pivot - k
        if idx < 0 or np.isnan(price_arr[idx]):
            return []
        if price_arr[last_pivot] > price_arr[idx]:
            return []   # a bar to the left is lower → not a local minimum

    # ── RSI at last pivot ─────────────────────────────────────────────────────
    if np.isnan(rsi[last_pivot]):
        return []
    last_rsi_ok = rsi[last_pivot] <= lower_threshold
    if strict_threshold and not last_rsi_ok:
        return []

    # ── Find anchor pivots (symmetric wing_bars, fully in the past) ───────────
    hist_end   = last_pivot - min_separation
    hist_start = max(0, last_pivot - lookback_window)

    if hist_end < wing_bars:
        return []

    anchor_candidates = _find_pivot_lows(price_arr, hist_start, hist_end, wing_bars)
    if not anchor_candidates:
        return []

    qualifying_anchors = {
        p for p in anchor_candidates
        if not np.isnan(rsi[p]) and rsi[p] <= lower_threshold
    }

    results = []
    b = last_pivot

    for a in anchor_candidates:
        # Threshold gate
        if strict_threshold:
            if a not in qualifying_anchors:
                continue
        else:
            # At least one of (anchor, last) must be in the oversold zone
            if a not in qualifying_anchors and not last_rsi_ok:
                continue

        # Divergence conditions:
        #   price: lower low  (last pivot price < anchor pivot price)
        #   RSI:   higher low (last pivot RSI  > anchor pivot RSI)
        if not (price_arr[b] < price_arr[a]):
            continue
        if not (rsi[b] > rsi[a]):
            continue
        if not _no_lower_between(rsi, a, b, rsi[a]):
            continue
        if not _no_overbought_in_range(rsi, a, b, upper_threshold):
            continue
        if not _has_price_bounce(close, a, b):
            continue

        results.append({
            'pivot_bars':  [a, b],
            'pivot_rsi':   [float(rsi[a]),   float(rsi[b])],
            'pivot_close': [float(close[a]), float(close[b])],
            'pivot_low':   [float(low[a]),   float(low[b])],
            'last_pivot_bar': b,
            'day_d_bar':      current_bar,
        })

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Date Utilities  (standalone)
# ═════════════════════════════════════════════════════════════════════════════

def _find_run_date_bar(dates: np.ndarray, run_date_str: str) -> Optional[int]:
    """
    Return the bar index whose Date string matches run_date_str (YYYY-MM-DD),
    or None if not present.
    """
    for i, d in enumerate(dates):
        if str(d) == run_date_str:
            return i
    return None


def _get_trading_sessions(start_str: str, end_str: str) -> List[str]:
    """
    Return all NYSE trading sessions in [start_str, end_str] (inclusive),
    as sorted YYYY-MM-DD strings. Weekends and holidays are automatically
    excluded. Falls back to every weekday in the range if exchange_calendars
    is unavailable.
    """
    try:
        import exchange_calendars as xcals
        import pandas as _pd
        nyse     = xcals.get_calendar("XNYS")
        sessions = nyse.sessions_in_range(start_str, end_str)
        return [s.strftime("%Y-%m-%d") for s in sessions]
    except Exception:
        pass

    # Fallback: every weekday in the range (no holiday filtering)
    from datetime import timedelta
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end   = datetime.strptime(end_str,   "%Y-%m-%d").date()
    days  = []
    cur   = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return days


def _nearby_trading_dates(dates: np.ndarray, run_date_str: str, n: int = 5) -> List[str]:
    """
    Return up to n dates from `dates` that are closest to run_date_str.
    Used to suggest valid alternatives when run_date has no data.
    """
    try:
        target = datetime.strptime(run_date_str, "%Y-%m-%d").date()
    except ValueError:
        return []

    dated = []
    for d in dates:
        try:
            dated.append(datetime.strptime(str(d), "%Y-%m-%d").date())
        except ValueError:
            continue

    dated.sort(key=lambda x: abs((x - target).days))
    return [d.strftime("%Y-%m-%d") for d in dated[:n] if d.strftime("%Y-%m-%d") != run_date_str]


# ═════════════════════════════════════════════════════════════════════════════
# Data Loading  (standalone)
# ═════════════════════════════════════════════════════════════════════════════

def _sanitize(symbol: str) -> str:
    """Convert symbol to safe filename (matches fetcher.py convention)."""
    return symbol.replace('$', '').replace('.', '_')


def _load_price_data(symbol: str, storage_dir: Path) -> pd.DataFrame:
    """
    Load OHLCV CSV from storage and reverse to oldest→newest order.
    All available bars are loaded — daily CSVs are small and loading everything
    ensures run_date can be any date in the history, not just the most recent window.
    Raises FileNotFoundError if the file doesn't exist.
    """
    path = storage_dir / f"{_sanitize(symbol)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data file for {symbol}: {path}")
    df = pd.read_csv(path)
    df = df.iloc[::-1].reset_index(drop=True)   # storage is newest-first → reverse
    return df


def _load_symbols(symbols_str: Optional[str], symbols_file: Optional[str]) -> List[str]:
    if symbols_str:
        return [s.strip() for s in symbols_str.split(',') if s.strip()]
    if symbols_file and Path(symbols_file).exists():
        syms = []
        with open(symbols_file) as f:
            for row in csv.DictReader(f):
                s = row.get('symbol', '').strip()
                if s:
                    syms.append(s)
        return syms
    return []


# ═════════════════════════════════════════════════════════════════════════════
# Symbol Scanner
# ═════════════════════════════════════════════════════════════════════════════

def scan_symbol(
    symbol: str,
    df: pd.DataFrame,
    config: dict,
    run_date_str: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Scan a single symbol for qualifying 2-divergences where Day D = run_date.

    Returns:
        (results, had_run_date)
        results       — list of qualifying divergence dicts (empty if none)
        had_run_date  — True if the symbol's data contained the run_date bar
    """
    close  = df['Close'].values
    open_  = df['Open'].values
    low    = df['Low'].values
    dates  = df['Date'].values

    run_date_bar = _find_run_date_bar(dates, run_date_str)
    if run_date_bar is None:
        return [], False

    div_cfg = config['divergence']
    lookback_window       = div_cfg['lookback_window']
    wing_bars             = div_cfg['wing_bars']
    last_pivot_right_bars = div_cfg.get('last_pivot_right_bars', 1)
    min_separation        = div_cfg['min_separation']
    strict_threshold      = div_cfg.get('strict_threshold', False)
    pivot_source          = div_cfg.get('pivot_source', 'close')

    rsi_cfg   = config['rsi']
    periods:         List[int]       = rsi_cfg.get('periods', [14])
    threshold_pairs: List[List[int]] = rsi_cfg.get('thresholds', [[70, 30]])

    seen: Dict[tuple, Dict[str, Any]] = {}

    for period in periods:
        rsi_values = _compute_rsi(close, period)

        for pair in threshold_pairs:
            upper_thresh, lower_thresh = pair[0], pair[1]

            divs = _detect_2div(
                current_bar=run_date_bar,
                close=close,
                low=low,
                rsi=rsi_values,
                lookback_window=lookback_window,
                lower_threshold=lower_thresh,
                upper_threshold=upper_thresh,
                wing_bars=wing_bars,
                last_pivot_right_bars=last_pivot_right_bars,
                min_separation=min_separation,
                strict_threshold=strict_threshold,
                pivot_source=pivot_source,
            )

            for div in divs:
                key = (period, lower_thresh, tuple(div['pivot_bars']))
                if key not in seen:
                    seen[key] = {
                        'symbol':          symbol,
                        'rsi_period':      period,
                        'threshold_upper': upper_thresh,
                        'threshold_lower': lower_thresh,
                        'pivot_source':    pivot_source,
                        'div':             div,
                        'close':           close,
                        'open':            open_,
                        'low':             low,
                        'dates':           dates,
                    }

    return list(seen.values()), True


# ═════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═════════════════════════════════════════════════════════════════════════════

def generate_report(
    all_results: List[Dict[str, Any]],
    run_date_str: str,
    config: dict,
    n_scanned: int,
    n_skipped: int,
) -> str:
    lines: List[str] = []

    div_cfg = config.get('divergence', {})
    rsi_cfg = config.get('rsi', {})
    multiplier = config.get('rsi_warmup_multiplier', 5)
    lookback_window = div_cfg.get('lookback_window', 40)
    max_period = max(rsi_cfg.get('periods', [14]))
    data_bars = lookback_window + max_period * multiplier

    n_qualified = len(all_results)
    thresh_str = ", ".join(
        f"{u}/{l}" for u, l in rsi_cfg.get('thresholds', [])
    )

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append("  RSI DIVERGENCE SCANNER")
    lines.append("=" * 72)
    lines.append(f"  Run date:   {run_date_str}  (Day D — divergence confirmed as of this date)")
    lines.append(f"  Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Scanned:    {n_scanned} symbol(s)  |  Skipped (no data for date): {n_skipped}")
    lines.append("")
    lines.append("  Config:")
    lines.append(f"    RSI periods:          {rsi_cfg.get('periods', [])}")
    lines.append(f"    Thresholds (U/L):     [{thresh_str}]")
    lines.append(f"    Lookback window:      {lookback_window} bars")
    lines.append(f"    Wing bars:            {div_cfg.get('wing_bars')} bars each side")
    lines.append(f"    Last pivot right:     {div_cfg.get('last_pivot_right_bars', 1)} bar(s)")
    lines.append(f"    Min separation:       {div_cfg.get('min_separation')} bars")
    lines.append(f"    Strict threshold:     {div_cfg.get('strict_threshold', False)}")
    lines.append(f"    Pivot source:         {div_cfg.get('pivot_source', 'close')}")
    lines.append(f"    Min bars for RSI:     {data_bars}  "
                 f"(lookback {lookback_window} + period {max_period} × {multiplier})  "
                 f"[all available history is loaded]")
    lines.append("=" * 72)
    lines.append("")

    if not all_results:
        lines.append("  No qualifying divergences found.")
        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    lines.append(f"  {n_qualified} qualifying divergence(s) found:")
    lines.append("")

    # ── Per-result blocks ─────────────────────────────────────────────────────
    for rec in all_results:
        symbol       = rec['symbol']
        period       = rec['rsi_period']
        lower_thresh = rec['threshold_lower']
        upper_thresh = rec['threshold_upper']
        p_source     = rec['pivot_source']
        div          = rec['div']
        dates        = rec['dates']
        close        = rec['close']
        low          = rec['low']

        pivot_bars:  List[int]   = div['pivot_bars']
        pivot_rsi:   List[float] = div['pivot_rsi']
        pivot_close: List[float] = div['pivot_close']
        pivot_low:   List[float] = div['pivot_low']
        day_d_bar:   int         = div['day_d_bar']

        a_bar, b_bar = pivot_bars

        a_date = str(dates[a_bar]) if a_bar < len(dates) else "???"
        b_date = str(dates[b_bar]) if b_bar < len(dates) else "???"
        d_date = str(dates[day_d_bar]) if day_d_bar < len(dates) else "???"

        low_marker   = " ◀pivot" if p_source == "low"   else ""
        close_marker = " ◀pivot" if p_source == "close" else ""

        lines.append("─" * 72)
        lines.append(
            f"  {symbol:<8}  RSI({period})  threshold ≤ {lower_thresh}"
            f"  [no RSI > {upper_thresh} between pivots]  [pivot: {p_source}]"
        )
        lines.append(
            f"    Anchor  {a_date}"
            f"  Low=${pivot_low[0]:>8.2f}{low_marker}"
            f"  Close=${pivot_close[0]:>8.2f}{close_marker}"
            f"  RSI={pivot_rsi[0]:.1f}"
        )
        lines.append(
            f"    Last    {b_date}"
            f"  Low=${pivot_low[1]:>8.2f}{low_marker}"
            f"  Close=${pivot_close[1]:>8.2f}{close_marker}"
            f"  RSI={pivot_rsi[1]:.1f}"
        )
        lines.append(f"    Day D   {d_date}  ← confirmed today")
        lines.append("")

    lines.append("=" * 72)
    lines.append(f"  {n_qualified} qualifying divergence(s)  |  {n_scanned} symbol(s) scanned")
    lines.append("=" * 72)

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Range Report Generation
# ═════════════════════════════════════════════════════════════════════════════

def generate_range_report(
    results_by_date: Dict[str, List[Dict[str, Any]]],
    start_date_str: str,
    end_date_str: str,
    trading_sessions: List[str],
    config: dict,
    n_symbols: int,
    n_skipped_symbols: int,
) -> str:
    """
    Generate a report for a date-range scan, with results grouped by Day D.
    """
    lines: List[str] = []

    div_cfg = config.get('divergence', {})
    rsi_cfg = config.get('rsi', {})

    n_trading_days = len(trading_sessions)
    all_hits       = [r for recs in results_by_date.values() for r in recs]
    n_qualified    = len(all_hits)
    n_hit_days     = sum(1 for recs in results_by_date.values() if recs)
    thresh_str     = ", ".join(f"{u}/{l}" for u, l in rsi_cfg.get('thresholds', []))

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append("  RSI DIVERGENCE SCANNER  —  DATE RANGE")
    lines.append("=" * 72)
    lines.append(f"  Range:      {start_date_str} → {end_date_str}")
    lines.append(f"  Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        f"  Scanned:    {n_trading_days} trading day(s)  ×  "
        f"{n_symbols - n_skipped_symbols} symbol(s)"
        + (f"  |  {n_skipped_symbols} symbol(s) skipped (no data file)" if n_skipped_symbols else "")
    )
    lines.append("")
    lines.append("  Config:")
    lines.append(f"    RSI periods:          {rsi_cfg.get('periods', [])}")
    lines.append(f"    Thresholds (U/L):     [{thresh_str}]")
    lines.append(f"    Lookback window:      {div_cfg.get('lookback_window')} bars")
    lines.append(f"    Wing bars:            {div_cfg.get('wing_bars')} bars each side")
    lines.append(f"    Last pivot right:     {div_cfg.get('last_pivot_right_bars', 1)} bar(s)")
    lines.append(f"    Min separation:       {div_cfg.get('min_separation')} bars")
    lines.append(f"    Strict threshold:     {div_cfg.get('strict_threshold', False)}")
    lines.append(f"    Pivot source:         {div_cfg.get('pivot_source', 'close')}")
    lines.append("=" * 72)
    lines.append("")

    if not all_hits:
        lines.append("  No qualifying divergences found in this date range.")
        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    lines.append(
        f"  {n_qualified} qualifying divergence(s) across "
        f"{n_hit_days} of {n_trading_days} trading day(s):"
    )
    lines.append("")

    p_source = div_cfg.get('pivot_source', 'close')
    low_marker   = " ◀pivot" if p_source == "low"   else ""
    close_marker = " ◀pivot" if p_source == "close" else ""

    # ── Per-day blocks (chronological) ───────────────────────────────────────
    for date_str in trading_sessions:
        recs = results_by_date.get(date_str)
        if not recs:
            continue

        lines.append(f"─── {date_str} " + "─" * (72 - 8 - len(date_str)))
        lines.append("")

        for rec in recs:
            symbol       = rec['symbol']
            period       = rec['rsi_period']
            lower_thresh = rec['threshold_lower']
            upper_thresh = rec['threshold_upper']
            div          = rec['div']
            dates        = rec['dates']
            close        = rec['close']
            low          = rec['low']

            pivot_bars:  List[int]   = div['pivot_bars']
            pivot_rsi:   List[float] = div['pivot_rsi']
            pivot_close: List[float] = div['pivot_close']
            pivot_low:   List[float] = div['pivot_low']
            day_d_bar:   int         = div['day_d_bar']

            a_bar, b_bar = pivot_bars
            a_date = str(dates[a_bar]) if a_bar < len(dates) else "???"
            b_date = str(dates[b_bar]) if b_bar < len(dates) else "???"
            d_date = str(dates[day_d_bar]) if day_d_bar < len(dates) else "???"

            lines.append(
                f"  {symbol:<8}  RSI({period})  threshold ≤ {lower_thresh}"
                f"  [no RSI > {upper_thresh} between pivots]  [pivot: {p_source}]"
            )
            lines.append(
                f"    Anchor  {a_date}"
                f"  Low=${pivot_low[0]:>8.2f}{low_marker}"
                f"  Close=${pivot_close[0]:>8.2f}{close_marker}"
                f"  RSI={pivot_rsi[0]:.1f}"
            )
            lines.append(
                f"    Last    {b_date}"
                f"  Low=${pivot_low[1]:>8.2f}{low_marker}"
                f"  Close=${pivot_close[1]:>8.2f}{close_marker}"
                f"  RSI={pivot_rsi[1]:.1f}"
            )
            lines.append(f"    Day D   {d_date}  ← confirmed")
            lines.append("")

    lines.append("=" * 72)
    lines.append(
        f"  {n_qualified} qualifying divergence(s)  |  "
        f"{n_trading_days} trading day(s)  |  "
        f"{n_symbols - n_skipped_symbols} symbol(s) scanned"
    )
    lines.append("=" * 72)

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def _validate_date_arg(date_str: str, flag: str) -> date:
    """Parse and return a date from a YYYY-MM-DD string, exiting on error."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        sys.exit(f"Error: {flag} must be in YYYY-MM-DD format, got: {date_str!r}")


def _check_not_holiday(date_str: str, day_name: str) -> None:
    """
    Exit with a clear message if date_str is not a NYSE trading session.
    Uses date_to_session(direction=...) to find adjacent sessions — unlike
    previous_session/next_session, this accepts any date including holidays.
    Silently passes through if exchange_calendars is unavailable.
    """
    _holiday_msg: Optional[str] = None
    try:
        import exchange_calendars as xcals
        import pandas as _pd
        nyse = xcals.get_calendar("XNYS")
        if not nyse.is_session(date_str):
            ts    = _pd.Timestamp(date_str)
            prior = nyse.date_to_session(ts, direction="previous")
            nxt   = nyse.date_to_session(ts, direction="next")
            _holiday_msg = (
                f"Error: {date_str} ({day_name}) is not a NYSE trading session "
                f"— it is likely a market holiday.\n"
                f"  Previous trading day: {prior.strftime('%Y-%m-%d')}\n"
                f"  Next trading day:     {nxt.strftime('%Y-%m-%d')}"
            )
    except Exception:
        pass
    if _holiday_msg:
        sys.exit(_holiday_msg)


def main() -> None:
    default_config       = str(Path(__file__).parent / 'config.yaml')
    default_symbols_file = str(PROJECT_ROOT / 'data' / 'anti_symbols.csv')
    default_output_dir   = str(PROJECT_ROOT / 'reports')

    parser = argparse.ArgumentParser(
        description='RSI Divergence Scanner — find stocks with confirmed bullish RSI '
                    '2-divergences as of a specific date or date range.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Single-date mode:
  python tools/rsi_divergence_scanner/scanner.py --date 2026-05-26
  python tools/rsi_divergence_scanner/scanner.py --date 2026-05-26 --symbols AAPL,MSFT,NVDA
  python tools/rsi_divergence_scanner/scanner.py --date 2026-05-26 --symbols-file data/core_symbols.csv

Date-range mode:
  python tools/rsi_divergence_scanner/scanner.py --start-date 2026-05-01 --end-date 2026-05-26
  python tools/rsi_divergence_scanner/scanner.py --start-date 2026-05-01 --end-date 2026-05-26 --symbols AAPL,MSFT
        """,
    )

    # ── Date arguments (mutually exclusive modes) ─────────────────────────────
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        '--date',
        help='Single run date in YYYY-MM-DD format.',
    )
    date_group.add_argument(
        '--start-date',
        help='Start of date range in YYYY-MM-DD format (use with --end-date).',
    )
    parser.add_argument(
        '--end-date',
        help='End of date range in YYYY-MM-DD format (use with --start-date).',
    )

    parser.add_argument(
        '--config', default=default_config,
        help=f'Config YAML file (default: {default_config})',
    )
    parser.add_argument(
        '--symbols-file', default=default_symbols_file,
        help=f'CSV with a "symbol" column (default: {default_symbols_file})',
    )
    parser.add_argument(
        '--symbols', default=None,
        help='Comma-separated list of symbols (overrides --symbols-file)',
    )
    parser.add_argument(
        '--output-dir', default=default_output_dir,
        help=f'Directory to save the report (default: {default_output_dir})',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable DEBUG-level logging',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
    )

    # ── Determine mode ────────────────────────────────────────────────────────
    range_mode = args.start_date is not None

    if range_mode:
        if not args.end_date:
            sys.exit("Error: --end-date is required when using --start-date.")

        start_date = _validate_date_arg(args.start_date, "--start-date")
        end_date   = _validate_date_arg(args.end_date,   "--end-date")

        if start_date > end_date:
            sys.exit(
                f"Error: --start-date ({args.start_date}) must be "
                f"on or before --end-date ({args.end_date})."
            )

        trading_sessions = _get_trading_sessions(args.start_date, args.end_date)
        if not trading_sessions:
            sys.exit(
                f"Error: No NYSE trading sessions found between "
                f"{args.start_date} and {args.end_date}."
            )

    else:
        run_date = _validate_date_arg(args.date, "--date")
        day_name = run_date.strftime("%A")
        if run_date.weekday() >= 5:
            sys.exit(
                f"Error: {args.date} is a {day_name} (weekend). "
                f"Markets are closed. Please use a weekday date."
            )
        _check_not_holiday(args.date, day_name)

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    div_cfg    = config['divergence']
    rsi_cfg    = config['rsi']
    multiplier = config.get('rsi_warmup_multiplier', 5)

    lookback_window = div_cfg['lookback_window']
    max_period      = max(rsi_cfg.get('periods', [14]))
    data_bars       = lookback_window + max_period * multiplier

    # ── Load symbols ──────────────────────────────────────────────────────────
    symbols = _load_symbols(args.symbols, args.symbols_file)
    if not symbols:
        sys.exit("Error: No symbols found. Use --symbols or --symbols-file.")

    storage_dir = PROJECT_ROOT / 'data' / 'storage'
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # DATE-RANGE MODE
    # ══════════════════════════════════════════════════════════════════════════
    if range_mode:
        print("=" * 65)
        print("RSI DIVERGENCE SCANNER  —  DATE RANGE")
        print("=" * 65)
        print(f"Range:      {args.start_date} → {args.end_date}")
        print(f"Sessions:   {len(trading_sessions)} trading day(s)")
        print(f"Symbols:    {len(symbols)}")
        print(f"Data bars:  {data_bars}  per symbol")
        print(f"Config:     {args.config}")
        print("=" * 65)
        print()

        # results_by_date[date_str] = list of result dicts
        results_by_date: Dict[str, List[Dict[str, Any]]] = {d: [] for d in trading_sessions}
        n_skipped = 0

        for i, symbol in enumerate(symbols, 1):
            print(f"[{i:>3}/{len(symbols)}] {symbol:<12}", end=" ", flush=True)
            try:
                df = _load_price_data(symbol, storage_dir)
            except FileNotFoundError:
                n_skipped += 1
                print("SKIP  (no data file)")
                continue
            except Exception as exc:
                n_skipped += 1
                print(f"ERROR  ({exc})")
                logger.debug("", exc_info=True)
                continue

            symbol_hits = 0
            for date_str in trading_sessions:
                try:
                    results, _ = scan_symbol(symbol, df, config, date_str)
                except Exception as exc:
                    logger.debug("scan_symbol error for %s on %s: %s", symbol, date_str, exc)
                    continue
                if results:
                    results_by_date[date_str].extend(results)
                    symbol_hits += len(results)

            if symbol_hits:
                print(f"{symbol_hits:>2} signal(s)  ✓")
            else:
                print("  —")

        print()

        # ── Range report ──────────────────────────────────────────────────────
        report = generate_range_report(
            results_by_date=results_by_date,
            start_date_str=args.start_date,
            end_date_str=args.end_date,
            trading_sessions=trading_sessions,
            config=config,
            n_symbols=len(symbols),
            n_skipped_symbols=n_skipped,
        )

        timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = output_dir / f"rsi_divergence_{args.start_date}_{args.end_date}_{timestamp}.txt"
        report_path.write_text(report)

        print(f"Report saved: {report_path}")
        print()
        print(report)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # SINGLE-DATE MODE  (unchanged behaviour)
    # ══════════════════════════════════════════════════════════════════════════
    run_date_str = args.date
    day_name     = datetime.strptime(run_date_str, "%Y-%m-%d").strftime("%A")

    print("=" * 65)
    print("RSI DIVERGENCE SCANNER")
    print("=" * 65)
    print(f"Date:       {run_date_str}  ({day_name})")
    print(f"Symbols:    {len(symbols)}")
    print(f"Data bars:  {data_bars}  per symbol")
    print(f"Config:     {args.config}")
    print("=" * 65)
    print()

    all_results:         List[Dict[str, Any]] = []
    n_skipped:           int  = 0
    n_no_date:           int  = 0
    any_symbol_had_date: bool = False
    fallback_dates:      List[str] = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i:>3}/{len(symbols)}] {symbol:<12}", end=" ", flush=True)
        try:
            df = _load_price_data(symbol, storage_dir)
        except FileNotFoundError:
            n_skipped += 1
            print("SKIP  (no data file)")
            continue
        except Exception as exc:
            n_skipped += 1
            print(f"ERROR  ({exc})")
            logger.debug("", exc_info=True)
            continue

        try:
            results, had_date = scan_symbol(symbol, df, config, run_date_str)
        except Exception as exc:
            n_skipped += 1
            print(f"ERROR  ({exc})")
            logger.debug("", exc_info=True)
            continue

        if had_date:
            any_symbol_had_date = True
        else:
            n_no_date += 1
            if not fallback_dates and 'Date' in df.columns:
                fallback_dates = _nearby_trading_dates(df['Date'].values, run_date_str)
            print("SKIP  (date not in data)")
            continue

        if results:
            all_results.extend(results)
            print(f"{len(results):>2} divergence(s)  ✓")
        else:
            print("  —")

    print()

    if not any_symbol_had_date:
        hint = ""
        if fallback_dates:
            hint = f"\n  Nearest available trading dates: {', '.join(fallback_dates)}"
        sys.exit(
            f"Error: No symbol had data for {run_date_str}. "
            f"This may be a market holiday or a date beyond the loaded history.{hint}"
        )

    n_scanned = len(symbols) - n_skipped
    report    = generate_report(all_results, run_date_str, config, n_scanned, n_no_date)

    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f"rsi_divergence_{run_date_str}_{timestamp}.txt"
    report_path.write_text(report)

    print(f"Report saved: {report_path}")
    print()
    print(report)


if __name__ == '__main__':
    main()
