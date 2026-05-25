"""
Entry Analyzer — RSI Bullish Divergence Entry Point Evaluator

Standalone tool (no dependency on anti_scanner) that:
  1. Detects bullish RSI divergences in historical daily price data
  2. Computes Day D  = trading day immediately after the last divergence pivot
  3. Evaluates two entry ideas starting from Day D:
       Idea 1: Buy at close of the first red/flat candle (close <= open)
               within `idea1.window` bars after Day D
       Idea 2: Buy at close of the first bar whose close exceeds the
               reference price within `idea2.window` bars after Day D
               — 2-divergence: reference = anchor (Pivot 1) close
               — 3-divergence: reference = middle (Pivot 2) close
  4. Reports entry date, entry price, and forward returns at configurable horizons

Usage:
  python tools/entry_analyzer/entry_analyzer.py
  python tools/entry_analyzer/entry_analyzer.py --symbols AAPL,MSFT,NVDA
  python tools/entry_analyzer/entry_analyzer.py --config tools/entry_analyzer/config.yaml
  python tools/entry_analyzer/entry_analyzer.py --symbols-file data/anti_symbols.csv
"""

import argparse
import csv
import logging
import sys
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# RSI Calculation  (standalone — no import from indicators/)
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
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    def _rsi_from_avg(ag: float, al: float) -> float:
        if al == 0:
            return 100.0
        return 100.0 - (100.0 / (1.0 + ag / al))

    rsi[period] = _rsi_from_avg(avg_gain, avg_loss)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi[i + 1] = _rsi_from_avg(avg_gain, avg_loss)

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
    scan_end = min(end, len(values) - 1 - pivot_lookback)

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
    valid = segment[~np.isnan(segment)]
    return len(valid) == 0 or float(np.min(valid)) >= ref


def _has_price_bounce(close: np.ndarray, a: int, b: int) -> bool:
    """
    True if close price recovered above close[a] at any bar strictly between a and b.

    This ensures the two price troughs belong to separate price cycles.
    Without a bounce, pivots a and b may just be two lows inside the same
    continuous selloff — that is not a divergence setup, it is just a downtrend.
    Uses close (not low) so intraday wicks don't count as a recovery.
    """
    if b - a <= 1:
        return False
    segment = close[a + 1:b]
    return len(segment) > 0 and float(np.max(segment)) > float(close[a])


# ═════════════════════════════════════════════════════════════════════════════
# Bullish Divergence Detection  (standalone)
# ═════════════════════════════════════════════════════════════════════════════

def _detect_bullish_divergence(
    current_bar: int,        # Day D — the bar you are scanning on
    close: np.ndarray,
    low: np.ndarray,
    rsi: np.ndarray,
    lookback_window: int,
    lower_threshold: float,
    wing_bars: int,          # bars each side for historical pivots; left-only for last pivot
    min_separation: int,
    divergence_count: int,
    strict_threshold: bool,
    pivot_source: str = "low",
) -> List[Dict[str, Any]]:
    """
    Detect bullish RSI divergence whose last pivot is current_bar - 1 (Day D-1).

    Last pivot confirmation (asymmetric — mimics real-time observation):
      Left  : wing_bars bars to the left must all have a higher price than Day D-1
      Right : exactly 1 bar — Day D's price must be strictly higher than Day D-1's price
      → The trough at Day D-1 is confirmed the instant Day D closes; no future-bar leakage.

    Historical pivots (anchor for 2-div; anchor + middle for 3-div):
      Symmetric wing_bars on both left and right sides, located in the window
      [last_pivot - lookback_window,  last_pivot - min_separation].
    """
    price_arr = low if pivot_source == "low" else close
    n = len(price_arr)

    last_pivot = current_bar - 1          # Day D-1 is always the last pivot candidate

    # ── Guard: enough room for wing_bars to the left of last_pivot ────────────
    if last_pivot < wing_bars or current_bar >= n:
        return []

    # ── Last pivot: right-side check — Day D must show a strictly higher price ─
    if np.isnan(price_arr[last_pivot]) or np.isnan(price_arr[current_bar]):
        return []
    if price_arr[current_bar] <= price_arr[last_pivot]:
        return []      # Day D is not higher → Day D-1 is not a trough

    # ── Last pivot: left-side check — wing_bars bars must all be higher ────────
    for k in range(1, wing_bars + 1):
        idx = last_pivot - k
        if idx < 0 or np.isnan(price_arr[idx]):
            return []
        if price_arr[last_pivot] > price_arr[idx]:
            return []  # A bar to the left is lower → not a local minimum on the left

    # ── RSI at last pivot ──────────────────────────────────────────────────────
    if np.isnan(rsi[last_pivot]):
        return []
    last_rsi_ok = rsi[last_pivot] <= lower_threshold
    if strict_threshold and not last_rsi_ok:
        return []      # Strict mode: last pivot must be in the oversold zone

    # ── Find historical pivots (symmetric wing_bars, fully in the past) ────────
    hist_end   = last_pivot - min_separation   # at least min_separation before last pivot
    hist_start = max(0, last_pivot - lookback_window)

    if hist_end < wing_bars:
        return []

    hist_pivots = _find_pivot_lows(price_arr, hist_start, hist_end, wing_bars)
    if len(hist_pivots) < (divergence_count - 1):
        return []

    qualifying_hist = set(
        p for p in hist_pivots
        if not np.isnan(rsi[p]) and rsi[p] <= lower_threshold
    )

    results = []
    b = last_pivot

    # ── 2-divergence: anchor (historical) + last pivot ─────────────────────────
    if divergence_count == 2:
        for a in hist_pivots:
            if strict_threshold:
                if a not in qualifying_hist:
                    continue
            else:
                # At least one of (anchor, last) must be in the oversold zone
                if a not in qualifying_hist and not last_rsi_ok:
                    continue

            if (price_arr[b] < price_arr[a]                     # price: lower low
                    and rsi[b] > rsi[a]                          # RSI:   higher low
                    and _no_lower_between(rsi, a, b, rsi[a])     # clean RSI between
                    and _has_price_bounce(close, a, b)):          # separate price cycles
                results.append({
                    'count': 2,
                    'pivot_bars':  [a, b],
                    'pivot_rsi':   [float(rsi[a]),   float(rsi[b])],
                    'pivot_close': [float(close[a]), float(close[b])],
                    'pivot_low':   [float(low[a]),   float(low[b])],
                    'divergence_end_bar': b,
                })

    # ── 3-divergence: anchor + middle (both historical) + last pivot ───────────
    elif divergence_count == 3:
        if len(hist_pivots) < 2:
            return results

        for ia in range(len(hist_pivots)):
            for ib in range(ia + 1, len(hist_pivots)):
                a, mid = hist_pivots[ia], hist_pivots[ib]

                if (mid - a) < min_separation:
                    continue

                if strict_threshold:
                    if a not in qualifying_hist or mid not in qualifying_hist:
                        continue
                else:
                    if mid not in qualifying_hist:   # middle must qualify at minimum
                        continue

                if (price_arr[a] > price_arr[mid] > price_arr[b]        # three descending price lows
                        and rsi[a] < rsi[mid] < rsi[b]                   # three ascending RSI lows
                        and _no_lower_between(rsi, a,   mid, rsi[a])
                        and _no_lower_between(rsi, mid, b,   rsi[mid])
                        and _has_price_bounce(close, a,   mid)
                        and _has_price_bounce(close, mid, b)):
                    results.append({
                        'count': 3,
                        'pivot_bars':  [a, mid, b],
                        'pivot_rsi':   [float(rsi[a]),    float(rsi[mid]),   float(rsi[b])],
                        'pivot_close': [float(close[a]),  float(close[mid]), float(close[b])],
                        'pivot_low':   [float(low[a]),    float(low[mid]),   float(low[b])],
                        'divergence_end_bar': b,
                    })

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═════════════════════════════════════════════════════════════════════════════

def _sanitize(symbol: str) -> str:
    """Convert symbol to safe filename (matches fetcher.py convention)."""
    return symbol.replace('$', '').replace('.', '_')


def _load_price_data(symbol: str, storage_dir: Path) -> pd.DataFrame:
    """
    Load OHLCV CSV from storage, reverse to oldest→newest order.
    Raises FileNotFoundError if the file doesn't exist.
    """
    path = storage_dir / f"{_sanitize(symbol)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data file for {symbol}: {path}")
    df = pd.read_csv(path)
    df = df.iloc[::-1].reset_index(drop=True)  # storage is newest-first
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
# Divergence Scanning — finds every unique bullish divergence in the series
# ═════════════════════════════════════════════════════════════════════════════

def scan_symbol(
    symbol: str,
    df: pd.DataFrame,
    config: dict,
    scan_bars: int = 0,
) -> List[Dict[str, Any]]:
    """
    Sweep bars to collect all unique bullish RSI divergences.

    Args:
        scan_bars: If > 0, only scan the most recent scan_bars bars for divergences.
                   RSI is still computed on all available data for accuracy.
                   If 0, scan all bars.

    Deduplicates by (rsi_period, lower_threshold, pivot_bars tuple) so the same
    divergence found from multiple current_bar positions is only returned once.

    Returns list of records sorted by divergence_end_bar descending (most recent first).
    """
    close = df['Close'].values
    open_ = df['Open'].values
    low   = df['Low'].values
    dates = df['Date'].values

    div_cfg = config['divergence']
    lookback_window  = div_cfg['lookback_window']
    wing_bars        = div_cfg['wing_bars']
    min_separation   = div_cfg['min_separation']
    strict_threshold = div_cfg.get('strict_threshold', False)
    pivot_source     = div_cfg.get('pivot_source', 'low')   # "low" or "close"

    rsi_cfg = config['rsi']
    periods: List[int] = rsi_cfg.get('periods', [])
    threshold_pairs: List[List[int]] = rsi_cfg.get('thresholds', [])
    div_counts: List[int] = rsi_cfg.get('bullish_divergence_counts', [])

    # Determine the bar range to scan for divergences.
    # RSI is computed on the full array; only the scan window is restricted.
    n = len(close)
    scan_start = max(0, n - scan_bars) if scan_bars > 0 else 0

    seen: Dict[tuple, Dict[str, Any]] = {}

    for period in periods:
        rsi_values = _compute_rsi(close, period)

        for pair in threshold_pairs:
            upper_thresh, lower_thresh = pair[0], pair[1]

            for div_count in div_counts:
                for bar in range(scan_start, n):
                    divs = _detect_bullish_divergence(
                        current_bar=bar,
                        close=close,
                        low=low,
                        rsi=rsi_values,
                        lookback_window=lookback_window,
                        lower_threshold=lower_thresh,
                        wing_bars=wing_bars,
                        min_separation=min_separation,
                        divergence_count=div_count,
                        strict_threshold=strict_threshold,
                        pivot_source=pivot_source,
                    )
                    for div in divs:
                        key = (period, lower_thresh, tuple(div['pivot_bars']))
                        if key not in seen:
                            seen[key] = {
                                'symbol': symbol,
                                'rsi_period': period,
                                'threshold_upper': upper_thresh,
                                'threshold_lower': lower_thresh,
                                'pivot_source': pivot_source,
                                'div': div,
                                'close': close,
                                'open': open_,
                                'low': low,
                                'dates': dates,
                            }

    records = list(seen.values())
    records.sort(key=lambda r: r['div']['divergence_end_bar'], reverse=True)
    return records


# ═════════════════════════════════════════════════════════════════════════════
# Entry Logic
# ═════════════════════════════════════════════════════════════════════════════

def _entry_idea1(
    close: np.ndarray,
    open_: np.ndarray,
    day_d_bar: int,
    window: int,
) -> Optional[Tuple[int, float]]:
    """
    Idea 1: First bar in [day_d_bar, day_d_bar + window) where close <= open.
    Returns (bar_index, entry_close_price) or None if none found.
    """
    limit = min(day_d_bar + window, len(close))
    for bar in range(day_d_bar, limit):
        if close[bar] <= open_[bar]:
            return (bar, float(close[bar]))
    return None


def _entry_idea2(
    close: np.ndarray,
    day_d_bar: int,
    reference_price: float,
    window: int,
) -> Optional[Tuple[int, float]]:
    """
    Idea 2: First bar in [day_d_bar, day_d_bar + window) where close > reference_price.
    Returns (bar_index, entry_close_price) or None if none found.
    """
    limit = min(day_d_bar + window, len(close))
    for bar in range(day_d_bar, limit):
        if close[bar] > reference_price:
            return (bar, float(close[bar]))
    return None


def _forward_returns(
    close: np.ndarray,
    entry_bar: int,
    entry_price: float,
    horizons: List[int],
) -> Dict[int, Optional[float]]:
    """
    % return vs entry_price at each horizon (bars after entry_bar).
    Returns None for horizons with insufficient data.
    """
    n = len(close)
    out: Dict[int, Optional[float]] = {}
    for h in horizons:
        target = entry_bar + h
        if target < n:
            out[h] = (close[target] - entry_price) / entry_price * 100.0
        else:
            out[h] = None
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═════════════════════════════════════════════════════════════════════════════

def _fmt_ret(val: Optional[float]) -> str:
    if val is None:
        return "  N/A  "
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


def _ret_row(horizons: List[int], returns: Dict[int, Optional[float]]) -> str:
    parts = [f"+{h}bar: {_fmt_ret(returns.get(h))}" for h in horizons]
    return "  ".join(parts)


def generate_report(
    all_records: Dict[str, List[Dict[str, Any]]],
    config: dict,
    symbols: List[str],
    scan_bars: int = 252,
) -> str:
    lines: List[str] = []

    entry_cfg = config.get('entry', {})
    idea1_cfg = entry_cfg.get('idea1', {})
    idea2_cfg = entry_cfg.get('idea2', {})
    idea1_enabled: bool = idea1_cfg.get('enabled', True)
    idea2_enabled: bool = idea2_cfg.get('enabled', True)
    idea1_window: int = idea1_cfg.get('window', 10)
    idea2_window: int = idea2_cfg.get('window', 20)
    horizons: List[int] = config.get('forward_bars', [1, 3, 5, 10])

    rsi_cfg = config.get('rsi', {})
    div_cfg = config.get('divergence', {})

    # ── Header ──────────────────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append("  ENTRY ANALYZER — RSI Bullish Divergence Entry Evaluation")
    lines.append("=" * 72)
    lines.append(f"  Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Symbols:    {len(symbols)}")
    lines.append(f"  Lookback:   {scan_bars} bars  (most recent)")
    lines.append("")
    lines.append("  RSI Config:")
    lines.append(f"    Periods:            {rsi_cfg.get('periods', [])}")
    thresh_str = ", ".join(f"{u}/{l}" for u, l in rsi_cfg.get('thresholds', []))
    lines.append(f"    Thresholds (U/L):   [{thresh_str}]")
    lines.append(f"    Divergence counts:  {rsi_cfg.get('bullish_divergence_counts', [])}")
    lines.append("  Divergence Config:")
    lines.append(f"    Lookback window:    {div_cfg.get('lookback_window')} bars  "
                 f"(search range for historical pivots before last pivot)")
    lines.append(f"    Wing bars:          {div_cfg.get('wing_bars')} bars each side  "
                 f"(historical pivots: symmetric left+right;  last pivot: left only)")
    lines.append(f"    Last pivot right:   1 bar fixed  (Day D confirms Day D-1 is a trough)")
    lines.append(f"    Min separation:     {div_cfg.get('min_separation')} bars")
    lines.append(f"    Strict threshold:   {div_cfg.get('strict_threshold', False)}")
    lines.append(f"    Pivot source:       {div_cfg.get('pivot_source', 'low')}  "
                 f"(price series used to find troughs)")
    lines.append("  Entry Config:")
    lines.append(f"    Idea 1 window:      {idea1_window} bars  (first close <= open after Day D)")
    lines.append(f"    Idea 2 window:      {idea2_window} bars  (first close > reference after Day D)")
    lines.append(f"    Forward horizons:   {horizons} bars")
    lines.append("=" * 72)
    lines.append("")

    total_divs = sum(len(v) for v in all_records.values())
    syms_with_hits = sum(1 for v in all_records.values() if v)
    lines.append(f"  Total divergences found: {total_divs} across {syms_with_hits} / {len(symbols)} symbols")
    lines.append("")

    # Accumulators for summary stats
    idea1_all: List[Dict[int, Optional[float]]] = []
    idea2_all: List[Dict[int, Optional[float]]] = []
    idea1_no_entry = 0
    idea2_no_entry = 0

    # ── Per-Symbol Sections ──────────────────────────────────────────────────
    for symbol in symbols:
        records = all_records.get(symbol, [])

        lines.append("═" * 72)
        if not records:
            lines.append(f"  {symbol}  —  0 divergences")
            lines.append("")
            continue

        lines.append(f"  {symbol}  —  {len(records)} divergence(s)  [most recent first]")
        lines.append("")

        for rec_idx, rec in enumerate(records, 1):
            div    = rec['div']
            close  = rec['close']
            open_  = rec['open']
            low    = rec['low']
            dates  = rec['dates']
            period       = rec['rsi_period']
            lower_thresh = rec['threshold_lower']
            upper_thresh = rec['threshold_upper']

            pivot_bars:  List[int]   = div['pivot_bars']
            pivot_rsi:   List[float] = div['pivot_rsi']
            pivot_close: List[float] = div['pivot_close']
            pivot_low:   List[float] = div['pivot_low']
            div_count:   int         = div['count']
            p_source:    str         = rec.get('pivot_source', 'low')
            last_pivot_bar: int = div['divergence_end_bar']
            day_d_bar:      int = last_pivot_bar + 1

            # Pivot label names
            if div_count == 2:
                pivot_roles = ["anchor", "last"]
            elif div_count == 3:
                pivot_roles = ["anchor", "middle", "last"]
            else:
                pivot_roles = [f"P{i+1}" for i in range(div_count)]

            # Divergence header
            count_label = {2: "Classic 2-Divergence", 3: "Triple 3-Divergence"}.get(
                div_count, f"{div_count}-Divergence"
            )
            lines.append(
                f"  ── #{rec_idx}  RSI({period})  {count_label}"
                f"  [threshold ≤ {lower_thresh}] ──"
            )

            # Pivot table — show Low and Close so you can verify on chart
            for i, (pbar, prsi, pclose, plow) in enumerate(
                zip(pivot_bars, pivot_rsi, pivot_close, pivot_low)
            ):
                pdate = dates[pbar] if pbar < len(dates) else "???"
                role  = pivot_roles[i] if i < len(pivot_roles) else f"P{i+1}"
                # Mark which price was used to anchor the pivot
                low_marker   = " ◀pivot" if p_source == "low"   else ""
                close_marker = " ◀pivot" if p_source == "close" else ""
                lines.append(
                    f"       Pivot {i+1} ({role}):  {pdate}"
                    f"  Low=${plow:>8.2f}{low_marker}"
                    f"  Close=${pclose:>8.2f}{close_marker}"
                    f"  RSI={prsi:.1f}"
                )

            # Day D
            if day_d_bar < len(dates):
                day_d_date = dates[day_d_bar]
                lines.append(f"       Day D:               {day_d_date}  ← setup confirmed, start scanning here")
            else:
                lines.append(f"       Day D:               beyond available data (bar {day_d_bar})")
                lines.append("")
                continue

            lines.append("")

            # Reference price for Idea 2
            if div_count == 2:
                ref_price = pivot_close[0]   # anchor close
                ref_label = f"anchor close (${ref_price:.2f})"
            elif div_count == 3:
                ref_price = pivot_close[1]   # middle close
                ref_label = f"middle close (${ref_price:.2f})"
            else:
                ref_price = pivot_close[-2]
                ref_label = f"2nd-to-last close (${ref_price:.2f})"

            # ── Entry Idea 1 ──────────────────────────────────────────────
            if idea1_enabled:
                result1 = _entry_idea1(close, open_, day_d_bar, idea1_window)
                if result1:
                    e1_bar, e1_price = result1
                    e1_date = dates[e1_bar] if e1_bar < len(dates) else "???"
                    e1_offset = e1_bar - day_d_bar
                    fwd1 = _forward_returns(close, e1_bar, e1_price, horizons)
                    idea1_all.append(fwd1)

                    lines.append(f"  ┌─ Idea 1: First Red/Flat Candle  (window={idea1_window} bars)")
                    lines.append(f"  │  BUY DATE:     {e1_date}  [Day D +{e1_offset}]")
                    lines.append(f"  │  BUY PRICE:    ${e1_price:.2f}  (at close)")
                    lines.append(f"  │  Returns:      {_ret_row(horizons, fwd1)}")
                    lines.append(f"  └{'─' * 65}")
                else:
                    idea1_no_entry += 1
                    lines.append(f"  ┌─ Idea 1: First Red/Flat Candle  (window={idea1_window} bars)")
                    lines.append(
                        f"  │  NO ENTRY — no red/flat candle within {idea1_window} bars of Day D"
                    )
                    lines.append(f"  └{'─' * 65}")

            # ── Entry Idea 2 ──────────────────────────────────────────────
            if idea2_enabled:
                result2 = _entry_idea2(close, day_d_bar, ref_price, idea2_window)
                if result2:
                    e2_bar, e2_price = result2
                    e2_date = dates[e2_bar] if e2_bar < len(dates) else "???"
                    e2_offset = e2_bar - day_d_bar
                    fwd2 = _forward_returns(close, e2_bar, e2_price, horizons)
                    idea2_all.append(fwd2)

                    lines.append(f"  ┌─ Idea 2: Close > {ref_label}  (window={idea2_window} bars)")
                    lines.append(f"  │  BUY DATE:     {e2_date}  [Day D +{e2_offset}]")
                    lines.append(f"  │  BUY PRICE:    ${e2_price:.2f}  (at close)")
                    lines.append(f"  │  Returns:      {_ret_row(horizons, fwd2)}")
                    lines.append(f"  └{'─' * 65}")
                else:
                    idea2_no_entry += 1
                    lines.append(f"  ┌─ Idea 2: Close > {ref_label}  (window={idea2_window} bars)")
                    lines.append(
                        f"  │  NO ENTRY — price did not exceed {ref_label} within {idea2_window} bars"
                    )
                    lines.append(f"  └{'─' * 65}")

            lines.append("")

        lines.append("")

    # ── Summary ──────────────────────────────────────────────────────────────
    lines.append("═" * 72)
    lines.append("  AGGREGATE SUMMARY")
    lines.append("═" * 72)
    lines.append("")

    def _avg(returns_list: List[Dict[int, Optional[float]]], h: int) -> Optional[float]:
        vals = [r[h] for r in returns_list if r.get(h) is not None]
        return sum(vals) / len(vals) if vals else None

    def _win_rate(returns_list: List[Dict[int, Optional[float]]], h: int) -> Optional[float]:
        vals = [r[h] for r in returns_list if r.get(h) is not None]
        return 100.0 * sum(1 for v in vals if v > 0) / len(vals) if vals else None

    for label, entries, no_entry_count in [
        ("Idea 1 — Red/Flat Candle Entry", idea1_all, idea1_no_entry),
        ("Idea 2 — Breakout Entry", idea2_all, idea2_no_entry),
    ]:
        total = len(entries) + no_entry_count
        lines.append(f"  {label}:")
        lines.append(f"    Entries triggered:  {len(entries)} / {total}")
        if entries:
            # Average return row
            avg_parts = "  ".join(
                f"+{h}bar: {_fmt_ret(_avg(entries, h))}" for h in horizons
            )
            lines.append(f"    Avg return:        {avg_parts}")
            # Win-rate row
            wr_parts = "  ".join(
                f"+{h}bar: {f'{_win_rate(entries, h):.0f}%' if _win_rate(entries, h) is not None else 'N/A':>7}"
                for h in horizons
            )
            lines.append(f"    Win rate (>0%):    {wr_parts}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    default_config = str(Path(__file__).parent / 'config.yaml')
    default_symbols_file = str(PROJECT_ROOT / 'data' / 'anti_symbols.csv')

    parser = argparse.ArgumentParser(
        description='Entry Analyzer — RSI bullish divergence entry point evaluator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/entry_analyzer/entry_analyzer.py
  python tools/entry_analyzer/entry_analyzer.py --symbols AAPL,MSFT,TSLA
  python tools/entry_analyzer/entry_analyzer.py --symbols-file data/anti_symbols.csv
  python tools/entry_analyzer/entry_analyzer.py --config tools/entry_analyzer/config.yaml
        """,
    )
    parser.add_argument(
        '--config', default=default_config,
        help=f'Config YAML file (default: {default_config})',
    )
    parser.add_argument(
        '--symbols-file', default=default_symbols_file,
        help=f'CSV with symbol column (default: {default_symbols_file})',
    )
    parser.add_argument(
        '--symbols', default=None,
        help='Comma-separated symbols (overrides --symbols-file)',
    )
    parser.add_argument(
        '--lookback', type=int, default=252,
        help='Number of most-recent bars to scan for divergences (default: 252). '
             'RSI is still computed on all loaded data for accuracy.',
    )
    parser.add_argument(
        '--output-dir', default=str(PROJECT_ROOT / 'reports'),
        help='Directory to save the report (default: reports/)',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print DEBUG-level logging',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
    )

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load symbols
    symbols = _load_symbols(args.symbols, args.symbols_file)
    if not symbols:
        logger.error("No symbols found. Use --symbols or --symbols-file.")
        sys.exit(1)

    storage_dir = PROJECT_ROOT / 'data' / 'storage'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 65)
    print("ENTRY ANALYZER")
    print("=" * 65)
    print(f"Symbols:    {len(symbols)}")
    print(f"Lookback:   {args.lookback} bars")
    print(f"Config:     {args.config}")
    print(f"Storage:    {storage_dir}")
    print("=" * 65)
    print()

    # Scan each symbol
    all_records: Dict[str, List[Dict[str, Any]]] = {}

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i:>3}/{len(symbols)}] {symbol:<12}", end=" ", flush=True)
        try:
            df = _load_price_data(symbol, storage_dir)
            records = scan_symbol(symbol, df, config, scan_bars=args.lookback)
            all_records[symbol] = records
            print(f"{len(records):>3} divergence(s)")
        except FileNotFoundError:
            print("SKIP  (no data file)")
            all_records[symbol] = []
        except Exception as exc:
            print(f"ERROR  ({exc})")
            logger.debug("", exc_info=True)
            all_records[symbol] = []

    # Generate report
    print()
    report = generate_report(all_records, config, symbols, scan_bars=args.lookback)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f"entry_analysis_{timestamp}.txt"
    report_path.write_text(report)

    print(f"Report saved: {report_path}")
    print()
    print(report)


if __name__ == '__main__':
    main()
