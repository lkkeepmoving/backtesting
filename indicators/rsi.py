"""
RSI (Relative Strength Index) Calculator

Computes RSI using Wilder's smoothing method.

Formula:
- First average gain/loss = simple mean of first N periods
- Subsequent averages = (prev_avg * (period - 1) + current_value) / period
- RS = avg_gain / avg_loss
- RSI = 100 - (100 / (1 + RS))
"""

import numpy as np


def compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """
    Compute RSI using Wilder's smoothing method.

    Args:
        close: Array of close prices, sorted oldest to newest
        period: RSI period (e.g., 14)

    Returns:
        Array of RSI values (same length as close). First `period` values are NaN.
    """
    n = len(close)
    rsi = np.full(n, np.nan)

    if n < period + 1:
        return rsi

    # Calculate price changes
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # First average gain/loss = simple mean of first `period` changes
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # First RSI value at index `period`
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Subsequent values using Wilder's smoothing
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def compute_rsi_multiple(close: np.ndarray, periods: list) -> dict:
    """
    Compute RSI for multiple periods.

    Args:
        close: Array of close prices, sorted oldest to newest
        periods: List of RSI periods (e.g., [7, 14, 21])

    Returns:
        Dict mapping period -> RSI array
    """
    return {period: compute_rsi(close, period) for period in periods}
