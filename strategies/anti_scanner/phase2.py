"""
Phase 2: Counter-Trend Shock Detection

Detects strong counter-trend price moves (sigma spike or pct change)
combined with MACD new extreme confirmation. Both conditions must be true.
"""

import numpy as np
from typing import List, Dict, Any


def detect_shock(
    divergence_end_bar: int,
    direction: str,
    close: np.ndarray,
    macd_line: np.ndarray,
    phase2_config: dict,
    shock_method: str,
    shock_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Detect counter-trend shock after a Phase 1 divergence.

    Args:
        divergence_end_bar: The bar index of the rightmost divergence pivot
        direction: "short" or "long"
        close: Close price array
        macd_line: Modified MACD fast line array
        phase2_config: Phase 2 config section
        shock_method: "sigma_spike" or "price_pct_change"
        shock_threshold: Threshold value for the shock method

    Returns:
        List of shock records for each bar where both shock + MACD extreme triggered.
    """
    window = phase2_config.get('window', 20)
    macd_extreme_lookback = phase2_config.get('macd_extreme', {}).get('lookback', 40)
    sigma_lookback = phase2_config.get('sigma_spike', {}).get('lookback', 20)

    scan_start = divergence_end_bar + 1
    scan_end = min(divergence_end_bar + window, len(close) - 1)

    results = []

    for bar in range(scan_start, scan_end + 1):
        if bar < 1:
            continue

        # --- Compute daily return and pct change ---
        pct_change = (close[bar] - close[bar - 1]) / close[bar - 1]

        # --- Compute sigma spike ---
        sigma_value = np.nan
        if bar >= sigma_lookback + 1:
            returns_window = np.diff(close[bar - sigma_lookback - 1: bar]) / close[bar - sigma_lookback - 1: bar - 1]
            sigma = np.std(returns_window[:-1])  # std of returns BEFORE current bar
            if sigma > 0:
                current_return = (close[bar] - close[bar - 1]) / close[bar - 1]
                sigma_value = current_return / sigma

        # --- Step 2A: Shock detection ---
        shock_detected = False

        if shock_method == "sigma_spike":
            if not np.isnan(sigma_value):
                if direction == "short" and sigma_value <= -shock_threshold:
                    shock_detected = True
                elif direction == "long" and sigma_value >= shock_threshold:
                    shock_detected = True

        elif shock_method == "price_pct_change":
            if direction == "short" and pct_change <= -shock_threshold:
                shock_detected = True
            elif direction == "long" and pct_change >= shock_threshold:
                shock_detected = True

        if not shock_detected:
            continue

        # --- Step 2B: MACD new extreme confirmation ---
        if np.isnan(macd_line[bar]):
            continue

        macd_window_start = max(0, bar - macd_extreme_lookback)
        macd_window = macd_line[macd_window_start:bar]  # Exclude current bar
        valid_macd = macd_window[~np.isnan(macd_window)]

        if len(valid_macd) == 0:
            continue

        is_new_extreme = False
        if direction == "short":
            is_new_extreme = macd_line[bar] <= np.min(valid_macd)
        elif direction == "long":
            is_new_extreme = macd_line[bar] >= np.max(valid_macd)

        if not is_new_extreme:
            continue

        # Both conditions met
        results.append({
            'shock_bar': bar,
            'shock_method': shock_method,
            'shock_threshold': shock_threshold,
            'sigma_value': float(sigma_value) if not np.isnan(sigma_value) else None,
            'pct_change': float(pct_change),
            'close_at_shock': float(close[bar]),
            'macd_at_shock': float(macd_line[bar]),
            'macd_is_new_extreme': True,
            'macd_extreme_lookback': macd_extreme_lookback,
        })

    return results


def run_phase2(
    phase1_results: List[Dict[str, Any]],
    direction: str,
    close: np.ndarray,
    macd_line: np.ndarray,
    phase2_config: dict,
) -> List[Dict[str, Any]]:
    """
    Run Phase 2 for all Phase 1 results across all shock parameter combinations.

    Args:
        phase1_results: List of Phase 1 result dicts (from phase1.run_phase1)
        direction: "short" or "long"
        close: Close price array
        macd_line: Modified MACD fast line array
        phase2_config: Phase 2 config section

    Returns:
        List of complete occurrence records (Phase 1 + Phase 2 combined).
    """
    shock_methods = phase2_config.get('shock_methods', [])
    occurrences = []

    for p1 in phase1_results:
        divergence_end_bar = p1['divergence']['divergence_end_bar']

        for method in shock_methods:
            if method == "sigma_spike":
                thresholds = phase2_config.get('sigma_spike', {}).get('thresholds', [])
            elif method == "price_pct_change":
                thresholds = phase2_config.get('price_pct_change', {}).get('thresholds', [])
            else:
                continue

            for threshold in thresholds:
                shocks = detect_shock(
                    divergence_end_bar=divergence_end_bar,
                    direction=direction,
                    close=close,
                    macd_line=macd_line,
                    phase2_config=phase2_config,
                    shock_method=method,
                    shock_threshold=threshold,
                )

                for shock in shocks:
                    occurrences.append({
                        'phase1': p1,
                        'phase2': shock,
                        'direction': direction,
                    })

    return occurrences
