"""
Render options_liquidity.csv as a human-readable text report.

Embeds the metric definitions at the top, then lists every symbol grouped by
verdict (TRADEABLE -> SKIP -> NO OPTIONS -> NO DATA), sorted by spread% within
each group.

Usage:
  python3 report_options_liquidity.py
  python3 report_options_liquidity.py --input options_liquidity.csv --output options_liquidity.txt --dte 60
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

DEFINITIONS = """\
================================================================================
 OPTIONS LIQUIDITY REPORT  --  is a stock's options worth trading?
================================================================================
 Generated : {generated}
 Universe  : {n} symbols (finviz $30B+ large caps)
 Probe     : the AT-THE-MONEY (ATM) CALL at the listed expiration CLOSEST TO
             {dte} DTE. ATM = the listed strike nearest the underlying's last
             traded price. All option figures are for that single contract.

 COLUMN DEFINITIONS
 ------------------
 SPREAD%  = (Ask - Bid) / Mid * 100,  where Mid = (Bid + Ask) / 2.
            The round-trip cost to cross the option's bid/ask, as a percent of
            its mid price. LOWER = tighter = cheaper to trade.
            Rating:  <=3 excellent | <=7 good | <=12 marginal | else avoid

 OI       = DailyOpenInterest of that same ATM ~{dte}-DTE call: the number of
            outstanding contracts at that one strike + expiration.
            HIGHER = deeper standing liquidity.
            Rating:  >=1000 strong | >=250 ok | >=50 thin | else illiquid
            NOTE: OI is for a SINGLE strike on an expiration that is not yet the
            front month, so it tends to UNDERSTATE a name's true option liquidity.
            When SPREAD% and OI disagree, SPREAD% (the actual cost) is the more
            reliable signal.

 VERDICT  = TRADEABLE  if SPREAD% <= 10 AND OI >= 250
            SKIP       if options exist but fail either gate
            NO OPTIONS if the underlying lists no options at all
            NO DATA    if the option quote could not be retrieved

 DTE/STRIKE = days-to-expiration and strike actually probed (for reference).
================================================================================
"""

HEADER = ("{sym:<6} {ctry:<14} {mcap:>10} {sp:>8} {spr:<10} "
          "{oi:>8} {oir:<9} {dte:>4} {strike:>9}")


def fmt_row(r):
    return HEADER.format(
        sym=r.get('symbol', '') or '',
        ctry=(r.get('country', '') or '')[:14],
        mcap=r.get('market_cap', '') or '',
        sp=(r.get('spread_pct') or '-'),
        spr=(r.get('spread_rating') or '') or '-',
        oi=(r.get('open_interest') or '-'),
        oir=(r.get('oi_rating') or '') or '-',
        dte=(r.get('dte') or '-'),
        strike=(r.get('atm_strike') or '-'),
    )


def main():
    ap = argparse.ArgumentParser(description='Render options liquidity CSV as text')
    ap.add_argument('--input', default='options_liquidity.csv')
    ap.add_argument('--output', default='options_liquidity.txt')
    ap.add_argument('--dte', type=int, default=60)
    args = ap.parse_args()

    with open(args.input) as f:
        rows = list(csv.DictReader(f))

    def spread_key(r):
        try:
            return float(r['spread_pct'])
        except (TypeError, ValueError):
            return 1e9

    col_header = HEADER.format(sym='SYMBOL', ctry='COUNTRY', mcap='MKT_CAP',
                               sp='SPREAD%', spr='SPRD_RATE', oi='OI',
                               oir='OI_RATE', dte='DTE', strike='STRIKE')

    lines = [DEFINITIONS.format(generated=datetime.now().strftime('%Y-%m-%d %H:%M'),
                                n=len(rows), dte=args.dte)]

    groups = [
        ('TRADEABLE', 'spread <= 10% AND OI >= 250'),
        ('SKIP', 'options exist but fail a gate'),
        ('NO OPTIONS', 'no listed options'),
        ('NO DATA', 'quote unavailable'),
    ]
    counts = {g: 0 for g, _ in groups}
    for r in rows:
        counts[r.get('verdict')] = counts.get(r.get('verdict'), 0) + 1

    # Summary line
    lines.append(" SUMMARY:  " + " | ".join(
        f"{g} {counts.get(g, 0)}" for g, _ in groups) + "\n")

    for verdict, desc in groups:
        sub = sorted([r for r in rows if r.get('verdict') == verdict], key=spread_key)
        lines.append("")
        lines.append("=" * 80)
        lines.append(f" {verdict}  ({len(sub)})   -- {desc}")
        lines.append("=" * 80)
        lines.append(col_header)
        lines.append("-" * 80)
        if not sub:
            lines.append(" (none)")
        for r in sub:
            lines.append(fmt_row(r))

    out = Path(args.output)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(rows)} symbols)")


if __name__ == '__main__':
    main()
