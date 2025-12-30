# Backtesting Framework

A quantitative trading backtesting framework with technical indicators and pattern detection strategies.

## Features

### Data Management
- **Historical Data Fetcher**: Download daily bar data from TradeStation API
- Supports batch fetching for multiple symbols
- Configurable bar count (default: 750 bars)

### Technical Indicators

#### EMA (Exponential Moving Average)
- Configurable period calculation
- SMA initialization for first value
- Precise convergence tracking

#### MACD (Moving Average Convergence Divergence)
- Standard parameters: EMA(12), EMA(26), Signal EMA(9)
- Convergence period from 2023-01-03
- Signal line calculation starting from 2024-01-02
- All values rounded to 5 decimal places

### Pattern Detection Strategies

#### MACD Wind Tunnel Pattern
4-day bullish pattern detection:
- **Day 0**: MACD > Signal (Setup)
- **Day 1**: MACD crosses below Signal (Bearish Crossover)
- **Day 2**: MACD < Signal, MACD > 0 (In Tunnel)
- **Day 3**: MACD crosses above Signal (Bullish Crossover - SIGNAL)

#### Inside Bar Pattern
Detects three levels of inside bar consolidation:
- Single Inside Bar
- Two Consecutive Inside Bars
- Three Consecutive Inside Bars

With 14-day high filter (10% drop requirement)

#### Three Gap-Down Bullish Pattern
Detects three consecutive days where:
- Each day gaps down (Open < Previous Close)
- Each day closes bullish (Close > Previous Close)

## Project Structure

```
backtesting/
├── data/
│   ├── fetcher.py              # TradeStation API data fetcher
│   └── storage/                # Historical price data (CSV)
├── indicators/
│   ├── ema_calculator.py       # EMA calculation
│   ├── macd_calculator.py      # MACD calculation
│   └── output/
│       └── macd/               # MACD indicator data (CSV)
├── strategies/
│   ├── macd_wind_tunnel.py     # MACD Wind Tunnel pattern scanner
│   ├── inside_bar.py           # Inside Bar pattern scanner
│   └── three_gap_down_bullish.py # Gap-down bullish pattern scanner
└── reports/                    # Generated pattern reports
```

## Usage

### Fetch Historical Data
```bash
# Fetch 750 bars for all symbols in core_symbols.csv
cd data
python3 fetcher.py --symbols-file core_symbols.csv --bars 750

# Fetch data for specific symbols
python3 fetcher.py --symbols AAPL MSFT TSLA --bars 500
```

### Calculate Indicators

#### EMA
```bash
# Calculate EMA(26) for AAPL
python3 indicators/ema_calculator.py --symbol AAPL --period 26

# Calculate EMA(12) for SPY
python3 indicators/ema_calculator.py --symbol SPY --period 12
```

#### MACD
```bash
# Calculate MACD for all symbols
python3 indicators/macd_calculator.py --symbols-file core_symbols.csv

# Calculate MACD for a single symbol
python3 indicators/macd_calculator.py --symbol TSLA
```

### Run Pattern Scanners

#### MACD Wind Tunnel
```bash
python3 strategies/macd_wind_tunnel.py
```

#### Inside Bar
```bash
python3 strategies/inside_bar.py
```

#### Three Gap-Down Bullish
```bash
python3 strategies/three_gap_down_bullish.py
```

## Configuration

Create a `.env` file in the project root with your TradeStation API credentials:

```
TRADESTATION_API_KEY=your_api_key_here
TRADESTATION_API_SECRET=your_api_secret_here
TRADESTATION_BASE_URL=https://api.tradestation.com/v3
```

## Requirements

- Python 3.8+
- pandas
- TradeStation API credentials

## Data Flow

1. **Fetch**: Download historical price data via TradeStation API
2. **Calculate**: Compute technical indicators (EMA, MACD)
3. **Detect**: Scan for patterns using calculated indicators
4. **Report**: Generate comprehensive reports with pattern occurrences

## Output

All pattern scanners generate detailed reports including:
- Pattern occurrence dates
- Multi-day breakdown with indicator values
- Summary statistics
- Symbols with/without patterns

Reports are saved to the `reports/` directory with timestamps.

## License

MIT
