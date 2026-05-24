# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a quantitative trading backtesting framework focused on technical indicators and pattern detection strategies. The framework follows a pipeline architecture: Fetch historical data → Calculate indicators → Detect patterns → Generate reports.

## Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup TradeStation API credentials (required for data fetching)
# Create .env file with:
# TRADESTATION_API_KEY=your_api_key
# TRADESTATION_API_SECRET=your_api_secret
# TRADESTATION_BASE_URL=https://api.tradestation.com/v3
```

### Data Fetching
```bash
# Fetch historical data for all symbols in core_symbols.csv
cd data
python fetcher.py --symbols-file core_symbols.csv --bars 260

# Fetch for specific symbols with custom bar count
python fetcher.py --symbols AAPL MSFT TSLA --bars 500
```

### Calculate Indicators
```bash
# Calculate MACD for all symbols
python indicators/macd_calculator.py --symbols-file data/core_symbols.csv

# Calculate MACD for a single symbol
python indicators/macd_calculator.py --symbol TSLA

# Calculate EMA for a specific symbol and period
python indicators/ema_calculator.py --symbol AAPL --period 26
```

### Run Pattern Scanners
```bash
# MACD-based pattern
python strategies/macd_wind_tunnel.py

# Price action patterns
python strategies/inside_bar.py
python strategies/three_gap_down_bullish.py

# Bollinger Band strategies (SPX and VIX)
python strategies/spx_bollinger_bounce.py
python strategies/vix_bollinger_rejection.py
python strategies/vix_bollinger_rejection_confirmation.py
python strategies/vix_spike_spx_drawdown.py
```

## Architecture

### Data Pipeline Flow
1. **Data Layer** (`data/`): Fetches historical OHLCV data from TradeStation API
   - `tradestation_client.py`: OAuth2 authentication, token management, API calls
   - `fetcher.py`: CLI for batch downloading historical bars
   - `storage/`: CSV files for each symbol (Date, Open, High, Low, Close, Volume)
   - Token persistence: `.tradestation_tokens` (JSON file, gitignored)

2. **Indicators Layer** (`indicators/`): Calculates technical indicators from raw price data
   - `ema_calculator.py`: Exponential Moving Average with SMA initialization
   - `macd_calculator.py`: MACD (12, 26, 9) with convergence period from 2023-01-03
   - `output/macd/`: Generated indicator CSV files (e.g., `AAPL_MACD.csv`)

3. **Strategies Layer** (`strategies/`): Pattern detection and signal generation
   - Each strategy scans indicator/price data for specific patterns
   - Generates timestamped reports in `reports/` directory
   - All strategies use NamedTuple for pattern occurrences

4. **Reports Layer** (`reports/`): Generated pattern analysis reports
   - Timestamped text files with detailed pattern breakdowns
   - Statistics and summary sections

### Key Architectural Patterns

**CSV Data Format**:
- Price data is stored **newest-to-oldest** (first row = most recent)
- Strategies reverse this to **oldest-to-newest** for calculations
- All dates in `YYYY-MM-DD` format

**TradeStation Authentication**:
- Interactive OAuth2 flow on first run (opens browser)
- Token refresh happens automatically using refresh token
- Tokens cached in `.tradestation_tokens` at project root

**Indicator Calculation Strategy**:
- MACD uses specific convergence period: EMAs from 2023-01-03, MACD values from 2024-01-02
- Signal line starts after 9 MACD values are available
- All MACD/EMA values rounded to 5 decimal places
- Bollinger Bands use population standard deviation (ddof=0)

**Pattern Detection**:
- Strategies iterate through data chronologically (oldest → newest)
- Multi-day patterns validate conditions across consecutive days
- Results sorted reverse-chronologically in reports (newest first)
- Each pattern has a NamedTuple defining its structure

**Symbol Filename Sanitization**:
- Special symbols like `$VIX.X` are stored as `VIX_X.csv`
- `$` prefix removed, `.` replaced with `_`

## File Locations

- **Raw price data**: `data/storage/{SYMBOL}.csv`
- **Symbol list**: `data/core_symbols.csv` (CSV with 'symbol' header)
- **MACD output**: `indicators/output/macd/{SYMBOL}_MACD.csv`
- **Strategy reports**: `reports/{strategy_name}_{timestamp}.txt`
- **Credentials**: `.env` (gitignored)
- **API tokens**: `data/.tradestation_tokens` (gitignored, auto-managed)

## Strategy Examples

**MACD Wind Tunnel** (4-day pattern):
- Day 0: MACD > Signal (Setup)
- Day 1: MACD crosses below Signal (Bearish Crossover)
- Day 2: MACD < Signal, MACD > 0 (In Tunnel)
- Day 3: MACD crosses above Signal (Bullish Crossover - SIGNAL)

**VIX Bollinger Rejection** (2-day pattern with SPX correlation):
- Day A: VIX Close > Upper Band AND Close > Open
- Day B: VIX Close < Upper Band AND Close < Open
- Measures SPX drawdown from 10-day high on trigger

**Inside Bar** (consolidation pattern):
- Detects single, double, or triple inside bar formations
- Applies 14-day high filter (10% drop requirement)

## Common Development Patterns

**Adding a New Strategy**:
1. Create file in `strategies/` directory
2. Define NamedTuple for pattern occurrence structure
3. Load data from `data/storage/` or `indicators/output/`
4. Implement pattern detection function (iterate oldest → newest)
5. Create report generation function
6. Save timestamped report to `reports/` directory

**Data Dependencies**:
- Strategies using MACD must run `macd_calculator.py` first
- All strategies require historical data fetched via `fetcher.py`
- Multi-symbol strategies can load symbols from `data/core_symbols.csv`
