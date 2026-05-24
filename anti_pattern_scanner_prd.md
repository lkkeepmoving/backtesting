# Anti Pattern Scanner — Product Requirements Document

## 1. Overview

### 1.1 What is the Anti Strategy?

The Anti strategy is a **trend-reversal pullback pattern** that identifies moments where an established trend is exhausting and a strong counter-trend shock occurs. It works in both directions (long and short).

The core thesis: when an existing trend shows momentum divergence (Phase 1), followed by a strong counter-trend move that creates a new MACD extreme (Phase 2), a high-probability reversal setup is forming. The trader then manually evaluates chart context to find the optimal entry point.

**Short Anti setup (uptrend exhaustion):**

```
Price:    /\      /\
         /  \    /  \     ← price makes higher highs
        /    \  /    \
       /      \/      \
                       \    ← strong counter-trend shock (Phase 2)
                        \
                     "anti"  ← the pullback entry zone (manual evaluation)
```

- Phase 1: Price makes higher highs, but RSI/MACD makes lower highs → bearish divergence (trend exhaustion)
- Phase 2: A strong down move occurs (sigma spike or large % drop) + MACD makes a new low → counter-trend shock confirmed

**Long Anti setup (downtrend exhaustion):** Mirror image — price makes lower lows, RSI/MACD makes higher lows → bullish divergence, followed by a strong up move + MACD new high.

### 1.2 Purpose of This Tool

This is a **pattern scanner**, not a backtesting engine. The script scans historical daily price data for stocks that exhibit the Anti pattern (Phase 1 divergence → Phase 2 counter-trend shock) and outputs a structured report of all qualifying occurrences.

The user then visually reviews each occurrence on ThinkorSwim / TradeStation charts to:

- Validate whether the Anti pattern leads to tradeable setups
- Build intuition about post-pattern price action
- Determine what a good entry point looks like

This is a **hypothesis validation tool** — before investing in a full automated backtesting system, the user wants to eyeball real historical occurrences and assess the pattern's merit.

### 1.3 Key Design Principles

- **All parameters are configurable** with sensible defaults. Nothing is hardcoded.
- **Config file** for indicator parameters (change rarely). **CLI flags** for per-run settings (change every run).
- **Both directions** (long and short) are supported simultaneously.
- **Comprehensive logging** — report ALL qualifying occurrences with full metadata for manual chart review and debugging.

---

## 2. Architecture

### 2.1 High-Level Flow

```
CLI invocation
    │
    ├── Load config file (indicator parameters)
    ├── Parse CLI flags (date, lookback, direction, symbols)
    │
    ├── For each symbol in universe:
    │       │
    │       ├── Pull daily bar data from TradeStation API
    │       │     (enough bars for indicator warm-up + percentile + scan window)
    │       │
    │       ├── Compute indicators:
    │       │     ├── RSI (multiple periods)
    │       │     ├── Modified MACD fast line (3/10 SMA, 16 signal)
    │       │     └── MACD percentile history (252-bar rolling)
    │       │
    │       ├── For each bar in scan window:
    │       │     │
    │       │     ├── Phase 1: Divergence Detection
    │       │     │     ├── RSI divergence check (if enabled)
    │       │     │     └── MACD divergence check (if enabled)
    │       │     │
    │       │     └── If Phase 1 passes:
    │       │           └── Phase 2: Counter-Trend Shock Detection
    │       │                 ├── Shock detection (sigma spike / price pct)
    │       │                 └── MACD new extreme confirmation
    │       │                 └── If Phase 2 passes → LOG occurrence
    │       │
    │       └── Collect all occurrences for this symbol
    │
    ├── Generate human-readable report (.txt)
    ├── Generate machine-readable data (.csv)
    └── Print report to console
```

### 2.2 Data Requirements

For each symbol, the script needs enough daily bars to cover:

| Component | Bars Needed | Notes |
|---|---|---|
| MACD percentile lookback | 252 | ~1 year of history for percentile baseline |
| SMA warm-up (10 SMA) | 10 | For first valid MACD fast line value |
| Scan window | User-specified (default 252) | Bars to scan for patterns |
| Divergence lookback | 40 | Lookback window from each scanned bar |
| Buffer | ~50 | Safety margin |

**Total: ~500 daily bars** should be pulled per symbol. This provides comfortable headroom for all computations.

**Key advantage:** The modified MACD uses **SMA** (not EMA), so warm-up is exact — no convergence period needed. The 10-period SMA produces a valid value on its 10th bar.

### 2.3 File Structure

```
anti_scanner/
├── anti_scanner.py          # Main CLI entry point
├── config/
│   └── default.yaml         # Default indicator parameters
├── indicators/
│   ├── rsi.py               # RSI computation
│   ├── macd.py              # Modified MACD computation + percentile
│   └── divergence.py        # Divergence detection (shared by RSI & MACD)
├── phases/
│   ├── phase1.py            # Divergence detection orchestrator
│   └── phase2.py            # Shock detection + MACD extreme confirmation
├── data/
│   └── tradestation.py      # TradeStation API data fetcher
├── output/
│   ├── report.py            # Human-readable report generator
│   └── csv_export.py        # CSV export
├── symbols/
│   └── default.txt          # Default symbol list (one per line)
└── results/                 # Output directory for reports
```

---

## 3. CLI Interface

### 3.1 Usage

```bash
# Typical usage — scan past year from yesterday, both directions
python anti_scanner.py --date 2025-03-20 --lookback 252 --direction both

# Scan specific date range
python anti_scanner.py --date 2025-03-20 --lookback 120

# Long setups only
python anti_scanner.py --date 2025-03-20 --direction long

# Use custom config and symbol list
python anti_scanner.py --config config/aggressive.yaml --symbols-file symbols/sp500.txt

# Override symbols inline
python anti_scanner.py --symbols AAPL,MSFT,NVDA,SPY,QQQ
```

### 3.2 CLI Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--date` | string (YYYY-MM-DD) | Yesterday's date | The end date of the scan window |
| `--lookback` | int | 252 | Number of trading bars to scan back from `--date` |
| `--direction` | string | `"both"` | `"long"`, `"short"`, or `"both"` |
| `--config` | string | `"config/default.yaml"` | Path to indicator config file |
| `--symbols-file` | string | `"symbols/default.txt"` | Path to symbol list file (one ticker per line) |
| `--symbols` | string | None | Comma-separated symbols (overrides `--symbols-file`) |
| `--output-dir` | string | `"results/"` | Directory for output files |
| `--verbose` | flag | False | Print detailed debug logging during scan |

---

## 4. Configuration File

### 4.1 Default Config (`config/default.yaml`)

```yaml
# =============================================================
# Anti Pattern Scanner — Indicator Configuration
# =============================================================
# All parameters are configurable. Modify and re-run as needed.
# Parameters with multiple values create a sweep — the scanner
# tests all combinations and tags results accordingly.
# =============================================================

# --- Data ---
data_bars: 500                  # Daily bars to pull per symbol

# --- Phase 1: Divergence Detection ---

# Shared divergence parameters
divergence:
  lookback_window: 40           # Bars to look back for divergence from current bar
  pivot_lookback: 3             # Bars on each side to confirm a pivot high/low
  min_separation: 5             # Minimum bars between two pivots in a divergence pair
  divergence_counts: [2, 3]     # 2 = classic divergence, 3 = triple divergence

# RSI Divergence
rsi:
  enabled: true
  periods: [7, 14, 21]                        # RSI periods to sweep
  thresholds:                                  # [upper, lower] pairs for short/long setups
    - [70, 30]
    - [80, 20]
    - [90, 10]

# Modified MACD Divergence
macd:
  enabled: true
  fast_period: 3                # SMA fast period (fixed per Anti strategy definition)
  slow_period: 10               # SMA slow period (fixed per Anti strategy definition)
  signal_period: 16             # Signal line SMA period (fixed per Anti strategy definition)
  use_sma: true                 # Use SMA instead of EMA (per Anti strategy definition)
  divergence_source: "fast_line"  # Use MACD fast line (not histogram) for divergence
  percentile_lookback: 252      # Bars for computing MACD percentile baseline
  percentile_thresholds:        # [upper_percentile, lower_percentile] pairs
    - [80, 20]
    - [90, 10]
    - [95, 5]

# --- Phase 2: Counter-Trend Shock ---

phase2:
  window: 20                    # Max bars after divergence to scan for shock

  shock_methods: ["sigma_spike", "price_pct_change"]  # Methods to test

  sigma_spike:
    thresholds: [2.0, 2.5, 3.0]      # Sigma multiples
    lookback: 20                       # Bars for computing daily return std dev

  price_pct_change:
    thresholds: [0.02, 0.03, 0.05]   # Absolute pct change (2%, 3%, 5%)

  macd_extreme:
    lookback: 40                       # Bars to check for MACD new low/high
```

---

## 5. Indicator Specifications

### 5.1 RSI (Relative Strength Index)

Standard RSI computation using Wilder's smoothing method.

**Inputs:** Close prices, RSI period  
**Output:** RSI value for each bar (0–100 scale)

The RSI period is swept across configured values (default: 7, 14, 21).

### 5.2 Modified MACD

This is NOT the standard MACD. Per the Anti strategy definition:

```
MACD fast line = SMA(close, fast_period) - SMA(close, slow_period)
Signal line    = SMA(MACD_fast_line, signal_period)
```

**Default parameters:** fast=3, slow=10, signal=16  
**Key difference from standard MACD:** Uses SMA instead of EMA. This means:
- No warm-up convergence issues
- The slow SMA (period 10) needs exactly 10 bars to produce its first valid value
- The signal line needs an additional 16 bars after the first valid MACD fast line value

**MACD Percentile Computation:**

For threshold-based divergence detection, the MACD fast line values are normalized to percentiles:

```python
# For each bar, compute the percentile of the MACD fast line
# relative to the past `percentile_lookback` bars (default: 252)

macd_percentile[bar] = percentile_rank(
    value=macd_fast_line[bar],
    distribution=macd_fast_line[bar - percentile_lookback : bar]
)
# Result: 0–100 scale, where 95 means "MACD is higher than 95% of recent values"
```

This makes MACD thresholds comparable across instruments regardless of price level.

### 5.3 Sigma Spike

Measures how extreme a daily return is relative to recent volatility.

```python
daily_return[bar] = (close[bar] - close[bar - 1]) / close[bar - 1]
sigma[bar] = std_dev(daily_return[bar - sigma_lookback : bar])
sigma_spike[bar] = daily_return[bar] / sigma[bar]
```

A sigma spike of -2.5 means "today's return was 2.5 standard deviations below the recent average daily return."

---

## 6. Phase 1: Divergence Detection — Detailed Algorithm

### 6.1 Overview

Phase 1 detects trend exhaustion through indicator divergence. Two types of divergence are supported (independently enabled):

- **RSI divergence** — RSI values diverge from price
- **MACD divergence** — Modified MACD fast line values diverge from price

Phase 1 **passes** if EITHER RSI divergence OR MACD divergence is detected (or both). All detected divergences are logged.

### 6.2 Divergence Detection Algorithm

The same core algorithm is used for both RSI and MACD divergence. Only the indicator values and threshold mechanism differ.

#### Bearish Divergence (Short Setup)

Detects uptrend exhaustion: price making higher highs while the indicator makes lower highs.

```
DETECT_BEARISH_DIVERGENCE(
    current_bar,
    close[],                     # Close price array
    indicator[],                 # RSI values or MACD fast line values
    lookback_window,             # Default: 40
    threshold,                   # Upper threshold (RSI value or MACD percentile)
    pivot_lookback,              # Default: 3
    min_separation,              # Default: 5
    divergence_count             # 2 or 3
):

STEP 1: Define the analysis window
    window_start = current_bar - lookback_window
    window_end = current_bar

STEP 2: Early-exit optimization
    Scan the indicator values in [window_start, window_end].
    If NO indicator value >= threshold → return NO_DIVERGENCE (skip expensive pivot detection).
    Rationale: Most bars won't have extreme indicator values. This skips pivot
    detection on the vast majority of bars, providing a significant speedup.

STEP 3: Find all indicator PIVOT HIGHS in the window
    Bar j is a PIVOT HIGH if:
        indicator[j] >= indicator[j - k] for all k in 1..pivot_lookback  AND
        indicator[j] >= indicator[j + k] for all k in 1..pivot_lookback
    Bars too close to window edges to have full pivot_lookback on both sides are excluded.

STEP 4: Filter pivots — at least one pivot must have indicator >= threshold
    qualifying_pivots = [p for p in pivots if indicator[p] >= threshold]
    If len(qualifying_pivots) == 0 → return NO_DIVERGENCE

STEP 5: Check for divergence based on divergence_count

    IF divergence_count == 2:
        For each pair (A, B) where A and B are pivots, A is chronologically before B,
        and (B - A) >= min_separation:
            At least one of A or B must be in qualifying_pivots (indicator >= threshold).
            Check BOTH conditions:
                a) close[B] > close[A]          (price made higher high)
                b) indicator[B] < indicator[A]   (indicator made lower high)
            If both true → this is a valid 2-divergence.
            Record: {pivot_bars: [A, B], pivot_indicator_values, pivot_close_values}

    IF divergence_count == 3:
        For each triple (A, B, C) where A < B < C chronologically,
        (B - A) >= min_separation AND (C - B) >= min_separation:
            At least one of A, B, C must be in qualifying_pivots.
            Check ALL conditions:
                a) close[A] < close[B] < close[C]                (progressively higher highs)
                b) indicator[A] > indicator[B] > indicator[C]     (progressively lower highs)
            If all true → this is a valid 3-divergence.
            Record: {pivot_bars: [A, B, C], pivot_indicator_values, pivot_close_values}

STEP 6: Return results
    Return ALL found divergences (for logging/debugging).
    If at least one exists → Phase 1 PASSES for this bar.
    The rightmost pivot bar of each divergence is recorded as `divergence_end_bar`
    (used as the start point for Phase 2 scanning).
```

#### Bullish Divergence (Long Setup)

Mirror image of bearish divergence:

- Step 2: Check if any indicator value **<=** lower threshold
- Step 3: Find indicator **PIVOT LOWS** instead of highs
- Step 4: Filter pivots where indicator **<=** lower threshold
- Step 5:
  - 2-divergence: `close[B] < close[A]` (lower low) AND `indicator[B] > indicator[A]` (higher low)
  - 3-divergence: `close[A] > close[B] > close[C]` AND `indicator[A] < indicator[B] < indicator[C]`

### 6.3 RSI Divergence Specifics

- **Indicator values:** RSI(period) where period ∈ {7, 14, 21}
- **Threshold:** RSI upper/lower value (e.g., 70/30, 80/20, 90/10)
- **Threshold application:** At least one pivot in the divergence chain must have RSI >= upper threshold (short) or <= lower threshold (long)

### 6.4 MACD Divergence Specifics

- **Indicator values:** Modified MACD fast line = SMA(close, 3) - SMA(close, 10)
- **Threshold:** Percentile-based. The MACD fast line value is converted to a percentile rank relative to the past 252 bars of MACD fast line values.
- **Threshold application:** At least one pivot in the divergence chain must have MACD percentile >= upper threshold (short) or <= lower threshold (long). E.g., if the upper percentile threshold is 80, at least one MACD pivot must be in the top 20% of its recent 252-bar distribution.

---

## 7. Phase 2: Counter-Trend Shock Detection — Detailed Algorithm

### 7.1 Overview

Phase 2 detects the "strong counter-trend shock" that follows the divergence. It has two sub-components that BOTH must be true:

- **2A — Shock Detection:** A large price move against the prior trend
- **2B — MACD New Extreme Confirmation:** The MACD fast line reaches a new extreme

### 7.2 Dynamic Lookback Window

**Critical:** Phase 2 scanning begins AFTER the Phase 1 divergence ends. The start of the Phase 2 scan window is determined by the divergence:

```
phase2_scan_start = divergence_end_bar + 1   (the bar after the rightmost divergence pivot)
phase2_scan_end   = divergence_end_bar + phase2_window   (default: 20 bars)
```

This ensures Phase 1 (trend exhaustion) is chronologically prior to Phase 2 (counter-trend shock).

### 7.3 Algorithm

```
DETECT_COUNTER_TREND_SHOCK(
    divergence_end_bar,
    direction,                     # "short" or "long"
    close[],
    daily_returns[],
    macd_fast_line[],
    phase2_window,                 # Default: 20
    shock_methods,                 # ["sigma_spike", "price_pct_change"]
    sigma_threshold,               # Default: 2.0
    sigma_lookback,                # Default: 20
    pct_threshold,                 # Default: 0.03
    macd_extreme_lookback          # Default: 40
):

    scan_start = divergence_end_bar + 1
    scan_end   = divergence_end_bar + phase2_window

    For each bar in [scan_start, scan_end]:

        # --- Step 2A: Shock Detection ---

        shock_detected = False
        shock_details = {}

        IF "sigma_spike" in shock_methods:
            sigma = std_dev(daily_returns[bar - sigma_lookback : bar])
            sigma_value = daily_returns[bar] / sigma

            IF direction == "short" AND sigma_value <= -sigma_threshold:
                shock_detected = True
                shock_details["sigma_spike"] = sigma_value

            IF direction == "long" AND sigma_value >= sigma_threshold:
                shock_detected = True
                shock_details["sigma_spike"] = sigma_value

        IF "price_pct_change" in shock_methods:
            pct = (close[bar] - close[bar - 1]) / close[bar - 1]

            IF direction == "short" AND pct <= -pct_threshold:
                shock_detected = True
                shock_details["price_pct"] = pct

            IF direction == "long" AND pct >= pct_threshold:
                shock_detected = True
                shock_details["price_pct"] = pct

        IF NOT shock_detected → continue to next bar

        # --- Step 2B: MACD New Extreme Confirmation ---

        macd_window = macd_fast_line[bar - macd_extreme_lookback : bar]
        macd_current = macd_fast_line[bar]

        IF direction == "short":
            is_new_extreme = macd_current <= min(macd_window)
        IF direction == "long":
            is_new_extreme = macd_current >= max(macd_window)

        IF shock_detected AND is_new_extreme:
            → Phase 2 PASSES
            Record: {
                shock_bar, shock_date,
                shock_method(s) that triggered,
                sigma_spike_value (if applicable),
                pct_change_value (always computed for context),
                macd_value_at_shock,
                macd_is_new_extreme: True
            }

    If no bar in the scan window triggers both 2A and 2B → Phase 2 FAILS
```

### 7.4 Notes

- If multiple shock methods are configured, ANY of them triggering counts as a shock. The output logs which method(s) triggered.
- Even if only sigma_spike is configured as the shock method, the percent change is always computed and logged for context (and vice versa). This helps the user when reviewing charts.
- Multiple shocks may be found within the Phase 2 window. ALL are logged.

---

## 8. Parameter Sweep & Result Tagging

### 8.1 How Parameter Sweeps Work

Multiple config values (e.g., RSI periods: [7, 14, 21]) create a sweep. The scanner runs ALL parameter combinations. Each qualifying occurrence is tagged with the specific parameter set that triggered it.

For example, a single scan might check:
- RSI periods: 3 values × RSI thresholds: 3 pairs × divergence counts: 2 values = 18 RSI configs
- MACD percentile thresholds: 3 pairs × divergence counts: 2 values = 6 MACD configs
- Shock methods: 2 × sigma thresholds: 3 + pct thresholds: 3 = combination matrix

### 8.2 Deduplication

The same (symbol, shock_date) pair may be flagged by multiple parameter configurations. In the output:

- **Human-readable report (.txt):** Deduplicate by (symbol, shock_date, direction). Show the occurrence once, with a list of all configs that triggered it.
- **Machine-readable CSV:** One row per (symbol, shock_date, direction, config_combination). No deduplication — preserves full parameter sensitivity data.

---

## 9. Output Specification

### 9.1 File Outputs

Each run produces two files in the output directory:

```
results/
├── anti_scan_YYYY-MM-DD_direction.txt     # Human-readable report
└── anti_scan_YYYY-MM-DD_direction.csv     # Machine-readable CSV
```

Where `YYYY-MM-DD` is the `--date` flag value and `direction` is `long`, `short`, or `both`.

### 9.2 Human-Readable Report Format (.txt)

```
═══════════════════════════════════════════════════════════════
ANTI PATTERN SCANNER — RESULTS
═══════════════════════════════════════════════════════════════
Scan Date:    2025-03-20
Lookback:     252 bars (from ~2024-03-15)
Direction:    both
Symbols:      30 scanned
Config:       config/default.yaml

Indicator Parameters:
  RSI:
    Periods:            [7, 14, 21]
    Thresholds:         [70/30, 80/20, 90/10]
  MACD:
    Fast/Slow/Signal:   3 / 10 / 16 (SMA)
    Percentile lookback: 252 bars
    Pctl thresholds:    [80/20, 90/10, 95/5]
  Divergence:
    Lookback window:    40 bars
    Pivot lookback:     3 bars
    Min separation:     5 bars
    Divergence counts:  [2, 3]
  Phase 2:
    Scan window:        20 bars after divergence
    Shock methods:      sigma_spike, price_pct_change
    Sigma thresholds:   [2.0, 2.5, 3.0] (lookback: 20)
    Pct thresholds:     [2%, 3%, 5%]
    MACD extreme lookback: 40 bars

Total occurrences found: 12 across 8 symbols
═══════════════════════════════════════════════════════════════

AAPL — 3 occurrences
───────────────────────────────────────────────────────────────

  #1 [SHORT] Shock date: 2024-07-15
     Triggered by configs:
       - RSI(14), threshold 70/30, 2-divergence, sigma_spike ≥ 2.0
       - RSI(14), threshold 70/30, 2-divergence, sigma_spike ≥ 2.5
     Divergence:
       2024-06-10  RSI=74  Close=$195.20   (anchor)
       2024-07-02  RSI=67  Close=$201.50   (divergent)
     Shock:
       Date:       2024-07-15
       Close:      $188.30
       Pct change: -3.4%
       Sigma:      -2.8σ
       MACD fast:  -1.42 (new 40-bar low ✓)

  #2 [SHORT] Shock date: 2024-09-22
     Triggered by configs:
       - RSI(14), threshold 70/30, 3-divergence, price_pct ≥ 3%
     Divergence:
       2024-08-05  RSI=76  Close=$210.00   (anchor)
       2024-08-20  RSI=71  Close=$215.30
       2024-09-10  RSI=65  Close=$219.80   (divergent)
     Shock:
       Date:       2024-09-22
       Close:      $205.10
       Pct change: -4.1%
       Sigma:      -2.3σ
       MACD fast:  -2.05 (new 40-bar low ✓)

  #3 [LONG] Shock date: 2025-01-08
     Triggered by configs:
       - MACD pctl 80/20, 2-divergence, sigma_spike ≥ 2.0
     Divergence:
       2024-12-01  MACD=-1.85 (pctl: 12)  Close=$178.40   (anchor)
       2024-12-20  MACD=-1.20 (pctl: 18)  Close=$172.10   (divergent)
     Shock:
       Date:       2025-01-08
       Close:      $185.60
       Pct change: +3.8%
       Sigma:      +2.5σ
       MACD fast:  2.10 (new 40-bar high ✓)

───────────────────────────────────────────────────────────────

MSFT — 1 occurrence
───────────────────────────────────────────────────────────────

  #1 [SHORT] Shock date: 2024-11-05
     ...

───────────────────────────────────────────────────────────────

SPY — 0 occurrences

GOOG — 0 occurrences

...

═══════════════════════════════════════════════════════════════
SUMMARY
═══════════════════════════════════════════════════════════════
Symbol   Direction   Count   Shock Dates
AAPL     SHORT       2       2024-07-15, 2024-09-22
AAPL     LONG        1       2025-01-08
MSFT     SHORT       1       2024-11-05
NVDA     SHORT       2       2024-08-12, 2025-02-03
...

Symbols with 0 occurrences (22): SPY, GOOG, AMZN, META, ...
═══════════════════════════════════════════════════════════════
```

### 9.3 CSV Format (.csv)

One row per (symbol, shock_date, direction, specific config). No deduplication.

| Column | Type | Description |
|---|---|---|
| `symbol` | string | Stock ticker |
| `direction` | string | "short" or "long" |
| `divergence_type` | string | "RSI" or "MACD" |
| `divergence_count` | int | 2 or 3 |
| `divergence_indicator_period` | string | RSI period (e.g., "14") or "MACD(3/10/16)" |
| `divergence_threshold` | string | e.g., "70/30" or "pctl_80/20" |
| `divergence_anchor_date` | date | Date of leftmost pivot |
| `divergence_anchor_indicator_value` | float | RSI or MACD value at anchor |
| `divergence_anchor_close` | float | Close price at anchor |
| `divergence_end_date` | date | Date of rightmost pivot |
| `divergence_end_indicator_value` | float | RSI or MACD value at rightmost pivot |
| `divergence_end_close` | float | Close price at rightmost pivot |
| `divergence_mid_date` | date or empty | Date of middle pivot (3-divergence only) |
| `divergence_mid_indicator_value` | float or empty | Middle pivot indicator value |
| `divergence_mid_close` | float or empty | Middle pivot close price |
| `shock_date` | date | Date the counter-trend shock occurred |
| `shock_close` | float | Close price on shock date |
| `shock_method` | string | "sigma_spike" or "price_pct_change" |
| `shock_sigma_value` | float | Sigma spike value (always computed) |
| `shock_pct_change` | float | Percent change (always computed) |
| `shock_sigma_threshold` | float | Sigma threshold used |
| `shock_pct_threshold` | float | Pct threshold used |
| `macd_at_shock` | float | MACD fast line value on shock date |
| `macd_is_new_extreme` | bool | Whether MACD was a new N-bar extreme |
| `macd_extreme_lookback` | int | Lookback used for extreme check |

---

## 10. Data Source — TradeStation API

### 10.1 Requirements

The script pulls daily OHLCV (Open, High, Low, Close, Volume) bar data from the TradeStation API.

- **Bars needed per symbol:** Configurable via `data_bars` in config (default: 500)
- **End date:** Determined by the `--date` CLI flag
- **Authentication:** The user has existing TradeStation API access. The data fetcher module should handle auth (OAuth2 or API key — follow existing TradeStation API patterns).

### 10.2 Data Fetcher Interface

```python
def fetch_daily_bars(symbol: str, end_date: str, num_bars: int) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from TradeStation API.

    Args:
        symbol: Stock ticker (e.g., "AAPL")
        end_date: Last date to include (YYYY-MM-DD)
        num_bars: Number of trading days to fetch

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
        Sorted by date ascending.
    """
```

The data fetcher is a separate module so it can be swapped out for other data sources (CSV files, Schwab API, etc.) without changing any scanner logic.

### 10.3 Rate Limiting

When scanning 30+ symbols, the script should respect TradeStation API rate limits. Implement:

- Configurable delay between API calls (default: 0.5 seconds)
- Retry logic with exponential backoff on rate limit errors
- Progress logging ("Fetching AAPL... 5/30")

---

## 11. Implementation Notes

### 11.1 Scanning Logic — Order of Operations

For each bar in the scan window, the scanner runs Phase 1 and then conditionally Phase 2. The full sweep iterates over all configured parameter combinations:

```python
for symbol in symbols:
    bars = fetch_daily_bars(symbol, end_date, data_bars)
    compute_all_indicators(bars)  # RSI (multiple periods), MACD, percentiles

    for bar in scan_window:
        for direction in configured_directions:

            # Phase 1: Try all divergence configs
            phase1_results = []

            if rsi_enabled:
                for rsi_period in rsi_periods:
                    for threshold_pair in rsi_thresholds:
                        for div_count in divergence_counts:
                            result = detect_divergence(
                                bar, direction, indicator="RSI",
                                values=rsi[rsi_period],
                                close=bars.close,
                                threshold=threshold_pair,
                                divergence_count=div_count,
                                ...
                            )
                            if result.found:
                                phase1_results.append(result)

            if macd_enabled:
                for threshold_pair in macd_percentile_thresholds:
                    for div_count in divergence_counts:
                        result = detect_divergence(
                            bar, direction, indicator="MACD",
                            values=macd_fast_line,
                            close=bars.close,
                            threshold=threshold_pair,  # percentile-based
                            divergence_count=div_count,
                            ...
                        )
                        if result.found:
                            phase1_results.append(result)

            # Phase 2: For each Phase 1 hit, check for shock
            for p1 in phase1_results:
                for shock_method in shock_methods:
                    for shock_threshold in get_thresholds(shock_method):
                        result = detect_shock(
                            divergence_end_bar=p1.divergence_end_bar,
                            direction=direction,
                            shock_method=shock_method,
                            shock_threshold=shock_threshold,
                            ...
                        )
                        if result.found:
                            log_occurrence(symbol, direction, p1, result)
```

### 11.2 Performance Considerations

- **Early-exit optimization in Phase 1:** Before pivot detection, check if ANY indicator value in the lookback window exceeds the threshold. If not, skip entirely. This avoids expensive pivot computation on most bars.
- **Precompute indicators:** RSI (all periods) and MACD should be computed once per symbol for the full data range, then indexed into during the scan loop. Do not recompute per bar.
- **Precompute percentiles:** MACD percentile ranks should be computed as a rolling percentile over the full data range once, then looked up per bar.
- **For 30 symbols:** Runtime should be seconds. For 500 symbols: expect minutes. The bottleneck will be API calls, not computation.

### 11.3 Dependencies

```
pandas
numpy
pyyaml
requests (for TradeStation API)
argparse (stdlib)
```

No backtesting frameworks needed. This is a pure scanner.

---

## 12. Future Extensions (Out of Scope for v1)

These are NOT part of the current implementation but are noted for future planning:

- **Phase 3 — Entry point detection:** Algorithmic detection of the "reluctant bounce" pullback near the Keltner Channel midline. Deferred until the user manually evaluates enough occurrences to define the entry pattern.
- **Phase 4 — Exit strategy backtesting:** ATR-based stops, trailing stops, target-based exits. Deferred; user plans to use 30% trailing stop on options.
- **Full backtesting engine:** Walk-forward optimization, Monte Carlo robustness testing, cross-instrument validation. Only worth building after the Anti pattern is validated via this scanner.
- **Volume confirmation filter:** Volume spike on shock day as an additional Phase 2 filter.
- **Multi-timeframe analysis:** Daily for trend/divergence, intraday for entry timing.
- **Real-time alerting:** Run the scanner on live data and send alerts when patterns form.

---

## 13. Glossary

| Term | Definition |
|---|---|
| **Anti strategy** | A trend-reversal pattern that enters on the first pullback after a strong counter-trend shock following trend exhaustion |
| **Bearish divergence** | Price makes higher highs while an indicator makes lower highs — signals uptrend exhaustion |
| **Bullish divergence** | Price makes lower lows while an indicator makes higher lows — signals downtrend exhaustion |
| **Modified MACD** | MACD using SMA (not EMA) with parameters 3/10/16 — faster than standard 12/26/9 |
| **MACD fast line** | SMA(close, 3) - SMA(close, 10) — the primary momentum signal |
| **Pivot high** | A local maximum in indicator values, confirmed by N bars on each side being lower |
| **Pivot low** | A local minimum in indicator values, confirmed by N bars on each side being higher |
| **Sigma spike** | Daily return measured in standard deviations of recent returns |
| **Counter-trend shock** | A strong price move against the prevailing trend |
| **Phase 1** | Divergence detection — identifies trend exhaustion |
| **Phase 2** | Shock detection — identifies the strong counter-trend move |
| **2-divergence** | Classic divergence between two indicator pivots |
| **3-divergence** | Triple divergence across three progressively diverging indicator pivots |
| **Anchor pivot** | The first (leftmost) pivot in a divergence chain — establishes the momentum baseline |
| **Divergent pivot** | The subsequent pivot(s) that show weakening momentum despite price continuation |
