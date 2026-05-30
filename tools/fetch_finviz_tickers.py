#!/usr/bin/env python3
"""Fetch all large-cap tickers from Finviz screener and save as CSV."""

import csv
import ssl
import time
import urllib.request
import urllib.error
import re
import sys
from typing import NamedTuple

# Bypass SSL verification (Finviz cert chain issue on macOS Python 3.9)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

BASE_URL = "https://finviz.com/screener.ashx?v=111&f=cap_largeover&o=-marketcap"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
OUTPUT_FILE = "data/finviz_large_cap_tickers.csv"
DELAY = 1.5  # seconds between requests

# Each screener row td has: data-boxover-ticker="X" ... data-boxover-country="Y" ... data-boxover-value="Z"
ROW_PATTERN = re.compile(
    r'data-boxover-ticker="([A-Z0-9.\-]+)"[^>]*'
    r'data-boxover-company="[^"]*"[^>]*'
    r'data-boxover-industry="[^"]*"[^>]*'
    r'data-boxover-country="([^"]*)"[^>]*'
    r'data-boxover-value="([^"]*)"'
)


class Stock(NamedTuple):
    symbol: str
    country: str
    market_cap: str


def fetch_page(r: int) -> str:
    url = BASE_URL + (f"&r={r}" if r > 1 else "")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15, context=SSL_CTX) as resp:
        return resp.read().decode("utf-8", errors="replace")


def extract_stocks(html: str) -> list[Stock]:
    return [Stock(symbol=m[0], country=m[1], market_cap=m[2]) for m in ROW_PATTERN.findall(html)]


def extract_total(html: str) -> int:
    match = re.search(r'"result_count":(\d+)', html)
    return int(match.group(1)) if match else 948


def main():
    print("Fetching page 1...", flush=True)
    html = fetch_page(1)
    total = extract_total(html)
    stocks = extract_stocks(html)
    print(f"  Total tickers reported: {total}, found on page 1: {len(stocks)}")

    r = 21
    page = 2
    while r <= total:
        print(f"Fetching page {page} (r={r})...", flush=True)
        time.sleep(DELAY)
        try:
            html = fetch_page(r)
            found = extract_stocks(html)
            print(f"  Found {len(found)} stocks")
            stocks.extend(found)
        except urllib.error.URLError as e:
            print(f"  ERROR on page {page}: {e}", file=sys.stderr)
        r += 20
        page += 1

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_stocks: list[Stock] = []
    for s in stocks:
        if s.symbol not in seen:
            seen.add(s.symbol)
            unique_stocks.append(s)

    print(f"\nTotal unique tickers collected: {len(unique_stocks)}")

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "country", "market_cap"])
        for s in unique_stocks:
            writer.writerow([s.symbol, s.country, s.market_cap])

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
