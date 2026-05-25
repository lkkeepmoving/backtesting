"""
API Data Comparator: TradeStation vs Charles Schwab
Fetches ~200 daily bars from both APIs for a given symbol (or list of symbols)
and reports any OHLC discrepancies on shared dates.

Usage:
    # Single symbol
    python tools/compare_apis.py --symbol AAPL

    # All symbols from a CSV file
    python tools/compare_apis.py --symbols-file data/core_symbols.csv
    python tools/compare_apis.py --symbols-file data/anti_symbols.csv
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

# Allow imports from data/ when run from tools/
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from tradestation_client import TradeStationClient
from schwab_client import SchwabClient

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent.parent / ".env")

BAR_COUNT    = 200
OHLC_FIELDS  = ["Open", "High", "Low", "Close"]
TOLERANCE_PCT = 0.01  # only flag if abs diff > 1% of TradeStation value

WIDTH = 66


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------

def get_tradestation_client() -> TradeStationClient:
    api_key    = os.getenv("TRADESTATION_API_KEY")
    api_secret = os.getenv("TRADESTATION_API_SECRET")
    base_url   = os.getenv("TRADESTATION_BASE_URL", "https://api.tradestation.com/v3")
    if not all([api_key, api_secret]):
        raise ValueError("TRADESTATION_API_KEY and TRADESTATION_API_SECRET must be set in .env")
    return TradeStationClient(config={"api_key": api_key, "api_secret": api_secret, "base_url": base_url})


def get_schwab_client() -> SchwabClient:
    app_key      = os.getenv("SCHWAB_APP_KEY")
    secret       = os.getenv("SCHWAB_SECRET")
    callback_url = os.getenv("SCHWAB_CALLBACK_URL")
    if not all([app_key, secret, callback_url]):
        raise ValueError("SCHWAB_APP_KEY, SCHWAB_SECRET, and SCHWAB_CALLBACK_URL must be set in .env")
    return SchwabClient(client_id=app_key, client_secret=secret, redirect_uri=callback_url)


# ---------------------------------------------------------------------------
# Symbol loading
# ---------------------------------------------------------------------------

def load_symbols_from_file(path: str) -> List[str]:
    df  = pd.read_csv(path)
    col = df.columns[0]
    return df[col].dropna().str.strip().tolist()


# ---------------------------------------------------------------------------
# Per-symbol fetch + compare (no printing of discrepancies here)
# ---------------------------------------------------------------------------

def fetch_and_compare(
    symbol: str,
    ts_client: TradeStationClient,
    sw_client: SchwabClient,
) -> dict:
    """Fetch and compare one symbol. Returns result dict."""

    result = {
        "symbol":        symbol,
        "status":        None,
        "shared_bars":   0,
        "ts_range":      "",
        "sw_range":      "",
        "discrepancies": [],
    }

    ts_df = ts_client.get_bars_by_bar_count(symbol, BAR_COUNT)
    if ts_df is None or ts_df.empty:
        result["status"] = "TS_ERROR"
        return result
    result["ts_range"] = f"{ts_df['Date'].iloc[-1]} → {ts_df['Date'].iloc[0]}"

    sw_df = sw_client.get_bars_by_bar_count(symbol, BAR_COUNT)
    if sw_df is None or sw_df.empty:
        result["status"] = "SW_ERROR"
        return result
    result["sw_range"] = f"{sw_df['Date'].iloc[-1]} → {sw_df['Date'].iloc[0]}"

    # Align on shared dates
    shared_dates     = set(ts_df["Date"]) & set(sw_df["Date"])
    result["shared_bars"] = len(shared_dates)

    ts_shared = ts_df[ts_df["Date"].isin(shared_dates)].set_index("Date")
    sw_shared = sw_df[sw_df["Date"].isin(shared_dates)].set_index("Date")

    discrepancies = []
    for date in sorted(shared_dates, reverse=True):
        for field in OHLC_FIELDS:
            ts_val = ts_shared.loc[date, field]
            sw_val = sw_shared.loc[date, field]
            if ts_val != 0 and abs(ts_val - sw_val) / abs(ts_val) > TOLERANCE_PCT:
                discrepancies.append({
                    "Symbol":       symbol,
                    "Date":         date,
                    "Field":        field,
                    "TradeStation": ts_val,
                    "Schwab":       sw_val,
                    "Diff":         sw_val - ts_val,
                    "Diff%":        (sw_val - ts_val) / abs(ts_val) * 100,
                })

    result["discrepancies"] = discrepancies
    result["status"] = "MISMATCH" if discrepancies else "OK"
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(results: List[dict], report_path: Optional[Path] = None):
    def emit(line: str = ""):
        print(line)
        if fh:
            fh.write(line + "\n")

    fh = open(report_path, "w") if report_path else None
    try:
        _render_results(results, emit)
    finally:
        if fh:
            fh.close()


def _render_results(results: List[dict], emit):
    # ── Section 1: Fetch log ─────────────────────────────────────────────
    emit(f"\n{'─' * WIDTH}")
    emit(f"  FETCH LOG")
    emit(f"{'─' * WIDTH}")
    for r in results:
        icon = {"OK": "✅", "MISMATCH": "⚠️ ", "TS_ERROR": "❌", "SW_ERROR": "❌"}.get(r["status"], "?")
        if r["status"] in ("TS_ERROR", "SW_ERROR"):
            src = "TradeStation" if r["status"] == "TS_ERROR" else "Schwab"
            emit(f"  {icon} {r['symbol']:<6}  {src} fetch FAILED")
        else:
            emit(f"  {icon} {r['symbol']:<6}  "
                 f"TS: {r['ts_range']}   "
                 f"Schwab: {r['sw_range']}   "
                 f"Shared: {r['shared_bars']} bars")

    # ── Section 2: All discrepancies ─────────────────────────────────────
    all_discs = [d for r in results for d in r["discrepancies"]]

    emit(f"\n{'─' * WIDTH}")
    emit(f"  DISCREPANCIES  (threshold: >{TOLERANCE_PCT*100:.0f}% diff)")
    emit(f"{'─' * WIDTH}")

    if not all_discs:
        emit(f"  ✅  No discrepancies found across all symbols.")
    else:
        col_w = {"Symbol": 8, "Date": 12, "Field": 7,
                 "TradeStation": 14, "Schwab": 12, "Diff": 10, "Diff%": 8}
        header = (
            f"  {'Symbol':<{col_w['Symbol']}}"
            f"{'Date':<{col_w['Date']}}"
            f"{'Field':<{col_w['Field']}}"
            f"{'TradeStation':>{col_w['TradeStation']}}"
            f"{'Schwab':>{col_w['Schwab']}}"
            f"{'Diff':>{col_w['Diff']}}"
            f"{'Diff%':>{col_w['Diff%']}}"
        )
        emit(header)
        emit("  " + "─" * (sum(col_w.values()) + 2))
        for d in all_discs:
            emit(
                f"  {d['Symbol']:<{col_w['Symbol']}}"
                f"{d['Date']:<{col_w['Date']}}"
                f"{d['Field']:<{col_w['Field']}}"
                f"{d['TradeStation']:>{col_w['TradeStation']}.4f}"
                f"{d['Schwab']:>{col_w['Schwab']}.4f}"
                f"{d['Diff']:>{col_w['Diff']}.4f}"
                f"{d['Diff%']:>{col_w['Diff%']}.2f}%"
            )

    # ── Section 3: Summary ───────────────────────────────────────────────
    ok       = sum(1 for r in results if r["status"] == "OK")
    mismatch = sum(1 for r in results if r["status"] == "MISMATCH")
    error    = sum(1 for r in results if r["status"] in ("TS_ERROR", "SW_ERROR"))

    emit(f"\n{'─' * WIDTH}")
    emit(f"  SUMMARY  ({len(results)} symbols)")
    emit(f"{'─' * WIDTH}")

    col_w = {"Symbol": 8, "Status": 12, "Shared": 8, "Discrepancies": 14}
    header = (
        f"  {'Symbol':<{col_w['Symbol']}}"
        f"{'Status':<{col_w['Status']}}"
        f"{'Shared':>{col_w['Shared']}}"
        f"{'Discrepancies':>{col_w['Discrepancies']}}"
    )
    emit(header)
    emit("  " + "─" * (sum(col_w.values()) + 2))
    for r in results:
        icon   = {"OK": "✅", "MISMATCH": "⚠️ ", "TS_ERROR": "❌", "SW_ERROR": "❌"}.get(r["status"], "?")
        disc   = str(len(r["discrepancies"])) if r["discrepancies"] else "—"
        shared = str(r["shared_bars"]) if r["shared_bars"] else "—"
        emit(
            f"  {r['symbol']:<{col_w['Symbol']}}"
            f"{icon + ' ' + r['status']:<{col_w['Status'] + 1}}"
            f"{shared:>{col_w['Shared']}}"
            f"{disc:>{col_w['Discrepancies']}}"
        )

    emit(f"\n  ✅ Clean: {ok}   ⚠️  Mismatches: {mismatch}   ❌ Errors: {error}")
    emit(f"{'─' * WIDTH}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare daily OHLC data between TradeStation and Schwab APIs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", "-s", help="Single ticker, e.g. AAPL")
    group.add_argument("--symbols-file", "-f", help="CSV with symbol column, e.g. data/core_symbols.csv")
    args = parser.parse_args()

    if args.symbol:
        symbols = [args.symbol]
    else:
        path = Path(args.symbols_file)
        if not path.exists():
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
        symbols = load_symbols_from_file(str(path))
        print(f"Loaded {len(symbols)} symbols from {path.name}: {', '.join(symbols)}")

    print("\nInitializing TradeStation client...", end=" ", flush=True)
    ts_client = get_tradestation_client()
    print("OK")

    print("Initializing Schwab client...     ", end=" ", flush=True)
    sw_client = get_schwab_client()
    print("OK")

    print(f"\nFetching & comparing {len(symbols)} symbol(s)...", flush=True)
    results = []
    for symbol in symbols:
        print(f"  {symbol}...", end=" ", flush=True)
        result = fetch_and_compare(symbol, ts_client, sw_client)
        results.append(result)
        print("done")

    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"api_compare_{timestamp}.txt"

    print_results(results, report_path=report_path)
    print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
