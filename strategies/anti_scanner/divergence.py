"""
Core Divergence Detection Algorithm

Detects bearish and bullish divergences between price and an indicator
(RSI or MACD). Supports 2-divergence (classic), 3-divergence (triple),
and 4-divergence (quadruple).

The same algorithm is used for both RSI and MACD divergence — only the
indicator values and threshold interpretation differ.
"""

import numpy as np
from typing import List, Dict, Any, Optional


def find_pivot_highs(values: np.ndarray, start: int, end: int,
                     pivot_lookback: int) -> List[int]:
    """
    Find all pivot highs in values[start:end+1].

    A bar j is a pivot high if values[j] >= values[j-k] and values[j] >= values[j+k]
    for all k in 1..pivot_lookback.

    Args:
        values: Indicator values array
        start: Start index (inclusive)
        end: End index (inclusive)
        pivot_lookback: Bars on each side to confirm pivot

    Returns:
        List of indices that are pivot highs
    """
    pivots = []
    # Ensure we have room for pivot_lookback on both sides
    scan_start = max(start, pivot_lookback)
    scan_end = min(end, len(values) - 1 - pivot_lookback)

    for j in range(scan_start, scan_end + 1):
        if np.isnan(values[j]):
            continue
        is_pivot = True
        for k in range(1, pivot_lookback + 1):
            if np.isnan(values[j - k]) or np.isnan(values[j + k]):
                is_pivot = False
                break
            if values[j] < values[j - k] or values[j] < values[j + k]:
                is_pivot = False
                break
        if is_pivot:
            pivots.append(j)

    return pivots


def find_pivot_lows(values: np.ndarray, start: int, end: int,
                    pivot_lookback: int) -> List[int]:
    """
    Find all pivot lows in values[start:end+1].

    A bar j is a pivot low if values[j] <= values[j-k] and values[j] <= values[j+k]
    for all k in 1..pivot_lookback.
    """
    pivots = []
    scan_start = max(start, pivot_lookback)
    scan_end = min(end, len(values) - 1 - pivot_lookback)

    for j in range(scan_start, scan_end + 1):
        if np.isnan(values[j]):
            continue
        is_pivot = True
        for k in range(1, pivot_lookback + 1):
            if np.isnan(values[j - k]) or np.isnan(values[j + k]):
                is_pivot = False
                break
            if values[j] > values[j - k] or values[j] > values[j + k]:
                is_pivot = False
                break
        if is_pivot:
            pivots.append(j)

    return pivots


def _no_higher_between(indicator: np.ndarray, a: int, b: int, ref_value: float) -> bool:
    """
    Return True if no indicator value strictly between indices a and b
    (exclusive) exceeds ref_value (ignoring NaNs).
    Trivially True when there are no bars between a and b.
    """
    if b - a <= 1:
        return True
    segment = indicator[a + 1:b]
    valid = segment[~np.isnan(segment)]
    return len(valid) == 0 or float(np.max(valid)) <= ref_value


def _no_lower_between(indicator: np.ndarray, a: int, b: int, ref_value: float) -> bool:
    """
    Return True if no indicator value strictly between indices a and b
    (exclusive) falls below ref_value (ignoring NaNs).
    Trivially True when there are no bars between a and b.
    """
    if b - a <= 1:
        return True
    segment = indicator[a + 1:b]
    valid = segment[~np.isnan(segment)]
    return len(valid) == 0 or float(np.min(valid)) >= ref_value


def detect_bearish_divergence(
    current_bar: int,
    close: np.ndarray,
    indicator: np.ndarray,
    lookback_window: int,
    threshold: float,
    pivot_lookback: int,
    min_separation: int,
    divergence_count: int,
    raw_indicator: Optional[np.ndarray] = None,
    strict_threshold: bool = False,
) -> List[Dict[str, Any]]:
    """
    Detect bearish divergence: price makes higher highs while indicator makes lower highs.
    Signals uptrend exhaustion (used for short setups).

    Args:
        current_bar: The bar index being evaluated
        close: Close price array
        indicator: Indicator values used for threshold qualification only when
                   raw_indicator is provided (e.g. MACD percentile); otherwise
                   used for all purposes (e.g. RSI values).
        lookback_window: Bars to look back from current_bar
        threshold: Upper threshold (RSI value or MACD percentile)
        pivot_lookback: Bars on each side to confirm a pivot
        min_separation: Minimum bars between two pivots
        divergence_count: 2 for classic, 3 for triple divergence
        raw_indicator: Optional raw values (e.g. MACD fast line) used for pivot
                       detection, divergence direction, and between-checks.
                       When None, `indicator` is used for all purposes (RSI path).
                       When provided, `indicator` is used only for threshold
                       qualification — preventing rolling-window percentile
                       artefacts from contaminating the divergence logic.
        strict_threshold: If True, every pivot in the pattern must qualify
                          (i.e. indicator >= threshold). If False (default),
                          only the anchor (2-point), anchor+D2 (3-point), or
                          anchor+D2+D3 (4-point) must qualify; the final
                          divergent pivot may be inside the threshold.

    Returns:
        List of divergence records. Each record contains pivot details.
    """
    # ind: the values used for pivot detection, divergence direction, between-checks.
    # For RSI this equals `indicator`; for MACD this is the raw fast line.
    ind = raw_indicator if raw_indicator is not None else indicator

    window_start = max(0, current_bar - lookback_window)
    window_end = current_bar

    # Early exit: check if ANY indicator (threshold scale) value in window >= threshold
    window_values = indicator[window_start:window_end + 1]
    valid_values = window_values[~np.isnan(window_values)]
    if len(valid_values) == 0 or np.max(valid_values) < threshold:
        return []

    # Find pivot highs on ind (raw values — avoids rolling-window percentile drift)
    pivots = find_pivot_highs(ind, window_start, window_end, pivot_lookback)
    if len(pivots) < divergence_count:
        return []

    # Threshold qualification uses indicator (percentile / RSI threshold scale)
    qualifying = [p for p in pivots if indicator[p] >= threshold]
    if not qualifying:
        return []

    divergences = []

    if divergence_count == 2:
        for i_a in range(len(pivots)):
            for i_b in range(i_a + 1, len(pivots)):
                a, b = pivots[i_a], pivots[i_b]
                if (b - a) < min_separation:
                    continue
                # Threshold qualification gate
                if strict_threshold:
                    if a not in qualifying or b not in qualifying:
                        continue
                else:
                    if a not in qualifying and b not in qualifying:
                        continue
                # Price higher high AND ind lower high
                # Guard: no bar between a and b may have ind > ind[a]
                if (close[b] > close[a] and ind[b] < ind[a] and
                        _no_higher_between(ind, a, b, ind[a])):
                    divergences.append({
                        'type': 'bearish',
                        'count': 2,
                        'pivot_bars': [a, b],
                        'pivot_indicator_values': [float(indicator[a]), float(indicator[b])],
                        'pivot_close_values': [float(close[a]), float(close[b])],
                        'divergence_end_bar': b,
                    })

    elif divergence_count == 3:
        for i_a in range(len(pivots)):
            for i_b in range(i_a + 1, len(pivots)):
                if (pivots[i_b] - pivots[i_a]) < min_separation:
                    continue
                for i_c in range(i_b + 1, len(pivots)):
                    a, b, c = pivots[i_a], pivots[i_b], pivots[i_c]
                    if (c - b) < min_separation:
                        continue
                    # Threshold qualification gate
                    if strict_threshold:
                        if a not in qualifying or b not in qualifying or c not in qualifying:
                            continue
                    else:
                        # Default: D1+D2 must qualify (D3 may be inside threshold)
                        if b not in qualifying:
                            continue
                    # Progressively higher highs in price, lower highs in ind
                    # Guard: no bar between a↔b may exceed ind[a],
                    #         no bar between b↔c may exceed ind[b]
                    if (close[a] < close[b] < close[c] and
                            ind[a] > ind[b] > ind[c] and
                            _no_higher_between(ind, a, b, ind[a]) and
                            _no_higher_between(ind, b, c, ind[b])):
                        divergences.append({
                            'type': 'bearish',
                            'count': 3,
                            'pivot_bars': [a, b, c],
                            'pivot_indicator_values': [float(indicator[a]),
                                                       float(indicator[b]),
                                                       float(indicator[c])],
                            'pivot_close_values': [float(close[a]),
                                                   float(close[b]),
                                                   float(close[c])],
                            'divergence_end_bar': c,
                        })

    elif divergence_count == 4:
        for i_a in range(len(pivots)):
            for i_b in range(i_a + 1, len(pivots)):
                if (pivots[i_b] - pivots[i_a]) < min_separation:
                    continue
                for i_c in range(i_b + 1, len(pivots)):
                    if (pivots[i_c] - pivots[i_b]) < min_separation:
                        continue
                    for i_d in range(i_c + 1, len(pivots)):
                        a, b, c, d = pivots[i_a], pivots[i_b], pivots[i_c], pivots[i_d]
                        if (d - c) < min_separation:
                            continue
                        # Threshold qualification gate
                        if strict_threshold:
                            if (a not in qualifying or b not in qualifying or
                                    c not in qualifying or d not in qualifying):
                                continue
                        else:
                            # Default: D2+D3 must qualify (D1 implied since D1>D2;
                            # D4 may be inside threshold)
                            if b not in qualifying or c not in qualifying:
                                continue
                        # Progressively higher highs in price, lower highs in ind
                        # Guard: no bar between consecutive pivots may exceed
                        #         the earlier pivot's ind value
                        if (close[a] < close[b] < close[c] < close[d] and
                                ind[a] > ind[b] > ind[c] > ind[d] and
                                _no_higher_between(ind, a, b, ind[a]) and
                                _no_higher_between(ind, b, c, ind[b]) and
                                _no_higher_between(ind, c, d, ind[c])):
                            divergences.append({
                                'type': 'bearish',
                                'count': 4,
                                'pivot_bars': [a, b, c, d],
                                'pivot_indicator_values': [float(indicator[a]),
                                                           float(indicator[b]),
                                                           float(indicator[c]),
                                                           float(indicator[d])],
                                'pivot_close_values': [float(close[a]),
                                                       float(close[b]),
                                                       float(close[c]),
                                                       float(close[d])],
                                'divergence_end_bar': d,
                            })

    return divergences


def detect_bullish_divergence(
    current_bar: int,
    close: np.ndarray,
    indicator: np.ndarray,
    lookback_window: int,
    threshold: float,
    pivot_lookback: int,
    min_separation: int,
    divergence_count: int,
    raw_indicator: Optional[np.ndarray] = None,
    strict_threshold: bool = False,
) -> List[Dict[str, Any]]:
    """
    Detect bullish divergence: price makes lower lows while indicator makes higher lows.
    Signals downtrend exhaustion (used for long setups).

    Args:
        Same as detect_bearish_divergence, but threshold is the LOWER threshold.
        raw_indicator: See detect_bearish_divergence. When provided (MACD path),
                       used for pivot detection, divergence direction, and
                       between-checks instead of `indicator`.
        strict_threshold: If True, every pivot must qualify (indicator <= threshold).
                          If False (default), only anchor+D2 (3-point) or
                          anchor+D2+D3 (4-point) must qualify; the final
                          divergent pivot may be inside the threshold.

    Returns:
        List of divergence records.
    """
    ind = raw_indicator if raw_indicator is not None else indicator

    window_start = max(0, current_bar - lookback_window)
    window_end = current_bar

    # Early exit: check if ANY indicator (threshold scale) value in window <= threshold
    window_values = indicator[window_start:window_end + 1]
    valid_values = window_values[~np.isnan(window_values)]
    if len(valid_values) == 0 or np.min(valid_values) > threshold:
        return []

    # Find pivot lows on ind (raw values)
    pivots = find_pivot_lows(ind, window_start, window_end, pivot_lookback)
    if len(pivots) < divergence_count:
        return []

    # Threshold qualification uses indicator (percentile / RSI threshold scale)
    qualifying = [p for p in pivots if indicator[p] <= threshold]
    if not qualifying:
        return []

    divergences = []

    if divergence_count == 2:
        for i_a in range(len(pivots)):
            for i_b in range(i_a + 1, len(pivots)):
                a, b = pivots[i_a], pivots[i_b]
                if (b - a) < min_separation:
                    continue
                # Threshold qualification gate
                if strict_threshold:
                    if a not in qualifying or b not in qualifying:
                        continue
                else:
                    if a not in qualifying and b not in qualifying:
                        continue
                # Price lower low AND ind higher low
                # Guard: no bar between a and b may have ind < ind[a]
                if (close[b] < close[a] and ind[b] > ind[a] and
                        _no_lower_between(ind, a, b, ind[a])):
                    divergences.append({
                        'type': 'bullish',
                        'count': 2,
                        'pivot_bars': [a, b],
                        'pivot_indicator_values': [float(indicator[a]), float(indicator[b])],
                        'pivot_close_values': [float(close[a]), float(close[b])],
                        'divergence_end_bar': b,
                    })

    elif divergence_count == 3:
        for i_a in range(len(pivots)):
            for i_b in range(i_a + 1, len(pivots)):
                if (pivots[i_b] - pivots[i_a]) < min_separation:
                    continue
                for i_c in range(i_b + 1, len(pivots)):
                    a, b, c = pivots[i_a], pivots[i_b], pivots[i_c]
                    if (c - b) < min_separation:
                        continue
                    # Threshold qualification gate
                    if strict_threshold:
                        if a not in qualifying or b not in qualifying or c not in qualifying:
                            continue
                    else:
                        # Default: D1+D2 must qualify (D3 may be inside threshold)
                        if b not in qualifying:
                            continue
                    # Progressively lower lows in price, higher lows in ind
                    # Guard: no bar between a↔b may fall below ind[a],
                    #         no bar between b↔c may fall below ind[b]
                    if (close[a] > close[b] > close[c] and
                            ind[a] < ind[b] < ind[c] and
                            _no_lower_between(ind, a, b, ind[a]) and
                            _no_lower_between(ind, b, c, ind[b])):
                        divergences.append({
                            'type': 'bullish',
                            'count': 3,
                            'pivot_bars': [a, b, c],
                            'pivot_indicator_values': [float(indicator[a]),
                                                       float(indicator[b]),
                                                       float(indicator[c])],
                            'pivot_close_values': [float(close[a]),
                                                   float(close[b]),
                                                   float(close[c])],
                            'divergence_end_bar': c,
                        })

    elif divergence_count == 4:
        for i_a in range(len(pivots)):
            for i_b in range(i_a + 1, len(pivots)):
                if (pivots[i_b] - pivots[i_a]) < min_separation:
                    continue
                for i_c in range(i_b + 1, len(pivots)):
                    if (pivots[i_c] - pivots[i_b]) < min_separation:
                        continue
                    for i_d in range(i_c + 1, len(pivots)):
                        a, b, c, d = pivots[i_a], pivots[i_b], pivots[i_c], pivots[i_d]
                        if (d - c) < min_separation:
                            continue
                        # Threshold qualification gate
                        if strict_threshold:
                            if (a not in qualifying or b not in qualifying or
                                    c not in qualifying or d not in qualifying):
                                continue
                        else:
                            # Default: D2+D3 must qualify (D1 implied since D1<D2;
                            # D4 may be inside threshold)
                            if b not in qualifying or c not in qualifying:
                                continue
                        # Progressively lower lows in price, higher lows in ind
                        # Guard: no bar between consecutive pivots may fall below
                        #         the earlier pivot's ind value
                        if (close[a] > close[b] > close[c] > close[d] and
                                ind[a] < ind[b] < ind[c] < ind[d] and
                                _no_lower_between(ind, a, b, ind[a]) and
                                _no_lower_between(ind, b, c, ind[b]) and
                                _no_lower_between(ind, c, d, ind[c])):
                            divergences.append({
                                'type': 'bullish',
                                'count': 4,
                                'pivot_bars': [a, b, c, d],
                                'pivot_indicator_values': [float(indicator[a]),
                                                           float(indicator[b]),
                                                           float(indicator[c]),
                                                           float(indicator[d])],
                                'pivot_close_values': [float(close[a]),
                                                       float(close[b]),
                                                       float(close[c]),
                                                       float(close[d])],
                                'divergence_end_bar': d,
                            })

    return divergences
