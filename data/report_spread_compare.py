"""
Side-by-side SPREAD% comparison of two options_liquidity runs, with the full
inputs (expiration, DTE, strike, bid/ask/mid) shown for BOTH tenors so every
spread% is manually verifiable from its own row.

Reads two CSVs produced by options_liquidity.py -- one from the "closest to N
DTE" probe (A) and one from the "nearest monthly" probe (B). Open interest is
intentionally omitted.

Usage:
  python3 report_spread_compare.py \
      --dte-csv options_liquidity_60dte.csv --target-dte 60 \
      --monthly-csv options_liquidity_monthly.csv --min-dte 14 \
      --output options_liquidity_spread_compare.txt
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

# One row per symbol: identity, then the A block, then the B block, then verdict.
ROW = ("{sym:<6} {mcap:>9}  "
       "{ea:<10} {da:>3} {ka:>8} {ba:>7} {aa:>7} {ma:>8} {sa:>7}   "
       "{eb:<10} {db:>3} {kb:>8} {bb:>7} {ab:>7} {mb:>8} {sb:>7}   {tighter:<8}")


def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[r['symbol']] = r
    return d


def fnum(r, key='spread_pct'):
    try:
        return float(r[key])
    except (TypeError, ValueError, KeyError):
        return None


def g(r, key):
    """Field value or '-' (handles missing dict / empty string)."""
    v = (r or {}).get(key)
    return v if (v is not None and v != '') else '-'


def side_fields(r, prefix):
    return {
        f'e{prefix}': g(r, 'expiration'),
        f'd{prefix}': g(r, 'dte'),
        f'k{prefix}': g(r, 'atm_strike'),
        f'b{prefix}': g(r, 'opt_bid'),
        f'a{prefix}': g(r, 'opt_ask'),
        f'm{prefix}': g(r, 'opt_mid'),
        f's{prefix}': g(r, 'spread_pct'),
    }


def main():
    ap = argparse.ArgumentParser(description='Compare ATM spread% (with inputs) across two probes')
    ap.add_argument('--dte-csv', default='options_liquidity_60dte.csv')
    ap.add_argument('--monthly-csv', default='options_liquidity_monthly.csv')
    ap.add_argument('--target-dte', type=int, default=60)
    ap.add_argument('--min-dte', type=int, default=14)
    ap.add_argument('--output', default='options_liquidity_spread_compare.txt')
    args = ap.parse_args()

    a = load(args.dte_csv)       # A = closest-to-N-DTE
    b = load(args.monthly_csv)   # B = nearest monthly
    symbols = list(a.keys())

    defn = f"""\
================================================================================
 OPTIONS SPREAD% COMPARISON (with inputs)  --  two expiration choices
================================================================================
 Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}
 Universe  : {len(symbols)} symbols (finviz $30B+ large caps)

 Both blocks probe the AT-THE-MONEY (ATM) CALL (strike nearest the underlying's
 last price). Only the EXPIRATION choice differs:
   A = "~{args.target_dte} DTE" : the listed expiration CLOSEST TO {args.target_dte} days out.
   B = "MONTHLY"   : the NEAREST standard monthly expiration >= {args.min_dte} days out.

 Each block shows the exact inputs so SPR% is checkable by hand:
   EXP  = expiration date          K    = ATM strike used
   DTE  = days to expiration       BID/ASK = quoted bid & ask
   MID  = (BID+ASK)/2              SPR% = (ASK-BID)/MID*100   ('-' = no quote)
 TIGHTER = which tenor has the lower SPR%. Rows sorted by the better (lower) SPR%.
================================================================================
"""

    rows = []
    a_tighter = b_tighter = 0
    for s in symbols:
        s1, s2 = fnum(a.get(s, {})), fnum(b.get(s, {}))
        if s1 is not None and s2 is not None:
            if s1 < s2:
                tighter, _ = f"~{args.target_dte}DTE", a_tighter
                a_tighter += 1
            elif s2 < s1:
                tighter = 'MONTHLY'
                b_tighter += 1
            else:
                tighter = 'same'
        else:
            tighter = '-'
        best = min([x for x in (s1, s2) if x is not None], default=1e9)
        rec = {'sym': s, 'mcap': g(a.get(s, {}), 'market_cap'), 'tighter': tighter, '_best': best}
        rec.update(side_fields(a.get(s), 'a'))
        rec.update(side_fields(b.get(s), 'b'))
        rows.append(rec)
    rows.sort(key=lambda r: r['_best'])

    group = (f"{'':<16}  {'|--------- A: ~'+str(args.target_dte)+' DTE ----------':<54}"
             f"{'|--------- B: NEAREST MONTHLY ----':<54}")
    col = ROW.format(sym='SYMBOL', mcap='MKT_CAP',
                     ea='EXP_A', da='DTE', ka='STRIKE', ba='BID', aa='ASK', ma='MID', sa='SPR%',
                     eb='EXP_B', db='DTE', kb='STRIKE', bb='BID', ab='ASK', mb='MID', sb='SPR%',
                     tighter='TIGHTER')

    lines = [defn, ""]
    lines.append(f" SUMMARY:  ~{args.target_dte}DTE tighter on {a_tighter} | "
                 f"MONTHLY tighter on {b_tighter}")
    lines.append("")
    lines.append(group)
    lines.append(col)
    lines.append("-" * len(col))
    for r in rows:
        lines.append(ROW.format(**{k: v for k, v in r.items() if k != '_best'}))

    out = Path(args.output)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(rows)} symbols) | "
          f"~{args.target_dte}DTE tighter {a_tighter}, MONTHLY tighter {b_tighter}")


if __name__ == '__main__':
    main()
