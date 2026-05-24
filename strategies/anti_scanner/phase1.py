"""
Phase 1: Divergence Detection Orchestrator

Sweeps across all configured parameter combinations for RSI and MACD
divergence detection. Phase 1 passes if ANY divergence is found.
"""

import numpy as np
from typing import List, Dict, Any

from .divergence import detect_bearish_divergence, detect_bullish_divergence


def run_phase1(
    current_bar: int,
    direction: str,
    close: np.ndarray,
    rsi_data: dict,
    macd_line: np.ndarray,
    macd_percentile: np.ndarray,
    config: dict,
) -> List[Dict[str, Any]]:
    """
    Run Phase 1 divergence detection across all parameter combinations.

    Args:
        current_bar: The bar index being evaluated
        direction: "short" or "long"
        close: Close price array
        rsi_data: Dict mapping RSI period -> RSI values array
        macd_line: Modified MACD fast line array
        macd_percentile: MACD percentile array
        config: Full config dict

    Returns:
        List of Phase 1 results, each containing divergence details
        and the config that triggered it.
    """
    results = []
    div_config = config['divergence']
    lookback_window = div_config['lookback_window']
    pivot_lookback = div_config['pivot_lookback']
    min_separation = div_config['min_separation']
    divergence_counts = div_config['divergence_counts']
    strict_threshold = div_config.get('strict_threshold', False)

    detect_fn = (detect_bearish_divergence if direction == "short"
                 else detect_bullish_divergence)

    # --- RSI Divergence ---
    rsi_config = config.get('rsi', {})
    if rsi_config.get('enabled', False):
        for period in rsi_config.get('periods', []):
            rsi_values = rsi_data.get(period)
            if rsi_values is None:
                continue

            for threshold_pair in rsi_config.get('thresholds', []):
                upper_thresh, lower_thresh = threshold_pair
                threshold = upper_thresh if direction == "short" else lower_thresh

                for div_count in divergence_counts:
                    divergences = detect_fn(
                        current_bar=current_bar,
                        close=close,
                        indicator=rsi_values,
                        lookback_window=lookback_window,
                        threshold=threshold,
                        pivot_lookback=pivot_lookback,
                        min_separation=min_separation,
                        divergence_count=div_count,
                        strict_threshold=strict_threshold,
                    )

                    for div in divergences:
                        results.append({
                            'divergence': div,
                            'indicator_type': 'RSI',
                            'indicator_period': str(period),
                            'threshold': f"{upper_thresh}/{lower_thresh}",
                            'divergence_count': div_count,
                        })

    # --- MACD Divergence ---
    macd_config = config.get('macd', {})
    if macd_config.get('enabled', False):
        macd_label = (f"MACD({macd_config.get('fast_period', 3)}/"
                      f"{macd_config.get('slow_period', 10)}/"
                      f"{macd_config.get('signal_period', 16)})")

        for threshold_pair in macd_config.get('percentile_thresholds', []):
            upper_pctl, lower_pctl = threshold_pair
            threshold = upper_pctl if direction == "short" else lower_pctl

            for div_count in divergence_counts:
                divergences = detect_fn(
                    current_bar=current_bar,
                    close=close,
                    indicator=macd_percentile,
                    lookback_window=lookback_window,
                    threshold=threshold,
                    pivot_lookback=pivot_lookback,
                    min_separation=min_separation,
                    divergence_count=div_count,
                    raw_indicator=macd_line,  # pivots, divergence direction, and
                                              # between-checks use raw MACD to avoid
                                              # rolling-percentile floor/ceiling artefacts
                    strict_threshold=strict_threshold,
                )

                for div in divergences:
                    # Store both percentile and raw MACD values for reporting
                    raw_macd_values = [float(macd_line[b]) if not np.isnan(macd_line[b]) else None
                                       for b in div['pivot_bars']]
                    results.append({
                        'divergence': div,
                        'indicator_type': 'MACD',
                        'indicator_period': macd_label,
                        'threshold': f"pctl_{upper_pctl}/{lower_pctl}",
                        'divergence_count': div_count,
                        'macd_raw_values': raw_macd_values,
                    })

    return results
