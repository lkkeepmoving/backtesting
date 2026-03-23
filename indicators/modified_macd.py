"""
Modified MACD Calculator for the Anti Strategy

Uses SMA (not EMA) with parameters 3/10/16:
  MACD fast line = SMA(close, fast_period) - SMA(close, slow_period)
  Signal line    = SMA(MACD_fast_line, signal_period)

Also computes rolling percentile rank of the MACD fast line for
threshold-based divergence detection.
"""

import numpy as np
from typing import Tuple


def compute_modified_macd(
    close: np.ndarray,
    fast_period: int = 3,
    slow_period: int = 10,
    signal_period: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute modified MACD using SMA.

    Args:
        close: Array of close prices, sorted oldest to newest
        fast_period: Fast SMA period (default: 3)
        slow_period: Slow SMA period (default: 10)
        signal_period: Signal line SMA period (default: 16)

    Returns:
        Tuple of (macd_fast_line, signal_line) arrays.
        Both have same length as close. Invalid values are NaN.
    """
    n = len(close)
    macd_line = np.full(n, np.nan)
    signal_line = np.full(n, np.nan)

    # Compute SMA-based MACD fast line
    # First valid value at index (slow_period - 1)
    for i in range(slow_period - 1, n):
        fast_sma = np.mean(close[i - fast_period + 1 : i + 1])
        slow_sma = np.mean(close[i - slow_period + 1 : i + 1])
        macd_line[i] = fast_sma - slow_sma

    # Compute signal line = SMA of MACD fast line
    # First valid signal at index (slow_period - 1) + (signal_period - 1)
    first_valid_macd = slow_period - 1
    for i in range(first_valid_macd + signal_period - 1, n):
        signal_line[i] = np.mean(macd_line[i - signal_period + 1 : i + 1])

    return macd_line, signal_line


def compute_macd_percentile(
    macd_line: np.ndarray,
    percentile_lookback: int = 252,
) -> np.ndarray:
    """
    Compute rolling percentile rank of the MACD fast line.

    For each bar, computes what percentile the current MACD value falls in
    relative to the previous `percentile_lookback` MACD values.

    Args:
        macd_line: MACD fast line values (may contain NaN)
        percentile_lookback: Number of bars for percentile baseline (default: 252)

    Returns:
        Array of percentile values (0-100). NaN where insufficient data.
    """
    n = len(macd_line)
    percentile = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(macd_line[i]):
            continue

        # Collect valid MACD values in the lookback window
        start = max(0, i - percentile_lookback)
        window = macd_line[start:i + 1]
        valid = window[~np.isnan(window)]

        if len(valid) < 2:
            continue

        current = macd_line[i]
        # Percentile rank: fraction of values <= current
        percentile[i] = (np.sum(valid <= current) / len(valid)) * 100.0

    return percentile
