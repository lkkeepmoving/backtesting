# Quick Start Guide

## 1. Setup Credentials

You can either:

**Option A:** Copy credentials from your existing project
```bash
cp /Users/kailu/Desktop/Projects/quantitative-trading/.env .env
```

**Option B:** Create new .env file
```bash
cp .env.example .env
# Then edit .env and add your TradeStation credentials
```

## 2. Add Your Symbols

Edit `data/core_symbols.csv`:

```csv
symbol
AAPL
MSFT
GOOGL
AMZN
TSLA
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Fetch Data

```bash
cd data
python fetcher.py --symbols-file core_symbols.csv --bars 260
```

## 5. Check Your Data

```bash
ls -lh data/storage/
head data/storage/AAPL.csv
```

You should see CSV files for each symbol with columns:
`Date,Open,High,Low,Close,Volume`

## Next Steps

- Start building strategy scripts in the `strategies/` folder
- Each strategy can read from `data/storage/*.csv`
- Analyze patterns, generate signals, and evaluate performance
