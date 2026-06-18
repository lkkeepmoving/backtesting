"""
Filter a finviz-style ticker CSV by the options-spread drop-candidate rule,
writing the KEPT symbols to a new CSV (same columns: symbol,country,market_cap).

A symbol is a DROP candidate (removed) at threshold X when -- mirroring the HTML
report's default behavior -- neither tenor is observed tight and at least one is
actually wide:
    drop = (A>X or A no-quote) AND (B>X or B no-quote) AND (A>X or B>X)
With --strict, only "both sides observed wide" are dropped.

Usage:
  python3 filter_tickers_by_spread.py \
      --tickers finviz_30b_large_cap_tickers.csv \
      --dte-csv options_liquidity_60dte.csv \
      --monthly-csv options_liquidity_monthly.csv \
      --threshold 12 \
      --output finviz_30b_large_cap_tickers_filtered_12pct.csv
"""

import argparse
import csv
from pathlib import Path


def load_spreads(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                out[r['symbol']] = float(r['spread_pct'])
            except (TypeError, ValueError, KeyError):
                out[r['symbol']] = None   # present but no quote
    return out


def is_drop(a, b, t, include_missing):
    """a, b are spread% (float) or None (no quote)."""
    a_miss, b_miss = a is None, b is None
    a_wide = (a is not None and a > t)
    b_wide = (b is not None and b > t)
    if not include_missing:
        return a_wide and b_wide
    return (a_wide or b_wide) and (a_wide or a_miss) and (b_wide or b_miss)


def main():
    ap = argparse.ArgumentParser(description='Filter tickers by options-spread drop rule')
    ap.add_argument('--tickers', default='finviz_30b_large_cap_tickers.csv')
    ap.add_argument('--dte-csv', default='options_liquidity_60dte.csv')
    ap.add_argument('--monthly-csv', default='options_liquidity_monthly.csv')
    ap.add_argument('--threshold', type=float, default=12.0)
    ap.add_argument('--strict', action='store_true',
                    help='Only drop when BOTH sides are observed wide (ignore no-quote)')
    ap.add_argument('--output', default='finviz_30b_large_cap_tickers_filtered_12pct.csv')
    args = ap.parse_args()

    a = load_spreads(args.dte_csv)
    b = load_spreads(args.monthly_csv)
    include_missing = not args.strict

    with open(args.tickers) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    kept, dropped = [], []
    for r in rows:
        s = r['symbol']
        if is_drop(a.get(s), b.get(s), args.threshold, include_missing):
            dropped.append(s)
        else:
            kept.append(r)

    out = Path(args.output)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)

    print(f"Threshold {args.threshold}% | rule={'strict' if args.strict else 'include-no-quote'}")
    print(f"Input  : {len(rows)} symbols ({args.tickers})")
    print(f"Dropped: {len(dropped)}")
    print(f"Kept   : {len(kept)} -> {out}")
    print("\nDropped symbols:")
    print(" ".join(sorted(dropped)))


if __name__ == '__main__':
    main()
