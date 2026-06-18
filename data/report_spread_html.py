"""
Render the two options_liquidity runs (A = ~N DTE, B = nearest monthly) as a
self-contained, interactive HTML page: sortable columns, a symbol filter, and
spread% cells color-coded by rating. Shows the full inputs (exp/dte/strike/
bid/ask/mid) for both tenors so every spread% is checkable by hand.

Usage:
  python3 report_spread_html.py \
      --dte-csv options_liquidity_60dte.csv --target-dte 60 \
      --monthly-csv options_liquidity_monthly.csv --min-dte 14 \
      --output options_liquidity_spread_compare.html
"""

import argparse
import csv
import html
from datetime import datetime
from pathlib import Path


def load(path):
    with open(path) as f:
        return {r['symbol']: r for r in csv.DictReader(f)}


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def rating_class(spr):
    if spr is None:
        return 'none'
    if spr <= 3:
        return 'excellent'
    if spr <= 7:
        return 'good'
    if spr <= 12:
        return 'marginal'
    return 'avoid'


def cell(value, numeric=False, cls='', sortval=None):
    """Render a <td> with a data-v sort key."""
    disp = '-' if value in (None, '', '-') else value
    if sortval is None:
        if numeric:
            n = fnum(disp)
            sortval = n if n is not None else 1e12  # missing sorts last asc
        else:
            sortval = str(disp)
    cls_attr = f' class="{cls}"' if cls else ''
    return f'<td{cls_attr} data-v="{html.escape(str(sortval))}">{html.escape(str(disp))}</td>'


def side_cells(r):
    spr = fnum((r or {}).get('spread_pct'))
    return ''.join([
        cell((r or {}).get('expiration')),
        cell((r or {}).get('dte'), numeric=True),
        cell((r or {}).get('atm_strike'), numeric=True),
        cell((r or {}).get('opt_bid'), numeric=True),
        cell((r or {}).get('opt_ask'), numeric=True),
        cell((r or {}).get('opt_mid'), numeric=True),
        cell((r or {}).get('spread_pct'), numeric=True, cls=f'spr {rating_class(spr)}'),
    ])


def main():
    ap = argparse.ArgumentParser(description='Render spread comparison as interactive HTML')
    ap.add_argument('--dte-csv', default='options_liquidity_60dte.csv')
    ap.add_argument('--monthly-csv', default='options_liquidity_monthly.csv')
    ap.add_argument('--target-dte', type=int, default=60)
    ap.add_argument('--min-dte', type=int, default=14)
    ap.add_argument('--output', default='options_liquidity_spread_compare.html')
    args = ap.parse_args()

    a, b = load(args.dte_csv), load(args.monthly_csv)
    symbols = list(a.keys())

    body_rows = []
    a_tighter = b_tighter = 0
    for s in symbols:
        ra, rb = a.get(s, {}), b.get(s, {})
        s1, s2 = fnum(ra.get('spread_pct')), fnum(rb.get('spread_pct'))
        if s1 is not None and s2 is not None:
            if s1 < s2:
                tighter = f'~{args.target_dte}DTE'; a_tighter += 1
            elif s2 < s1:
                tighter = 'MONTHLY'; b_tighter += 1
            else:
                tighter = 'same'
        else:
            tighter = '-'
        best = min([x for x in (s1, s2) if x is not None], default=1e12)
        tcls = 'tA' if tighter.endswith('DTE') else ('tB' if tighter == 'MONTHLY' else '')
        row = (
            f'<tr>'
            + cell(s, sortval=s)
            + cell(ra.get('market_cap') or rb.get('market_cap'),
                   numeric=True, sortval=fnum((ra.get('market_cap') or '0').rstrip('B')) or 0)
            + side_cells(ra)
            + side_cells(rb)
            + cell(tighter, cls=tcls)
            + f'</tr>'
        )
        # carry best for default sort
        body_rows.append((best, row))
    body_rows.sort(key=lambda t: t[0])
    rows_html = "\n".join(r for _, r in body_rows)

    sub = ('EXP', 'DTE', 'STRIKE', 'BID', 'ASK', 'MID', 'SPR%')
    head_a = "".join(f'<th class="sortable">{h}</th>' for h in sub)
    head_b = head_a

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Options Spread% Comparison</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 18px; color:#1f2328; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .meta {{ color:#57606a; font-size: 13px; margin-bottom: 10px; }}
  .legend span {{ display:inline-block; padding:2px 8px; border-radius:4px; margin-right:6px; font-size:12px; }}
  .controls {{ margin: 12px 0; }}
  input#filter {{ padding:6px 10px; font-size:14px; width:240px; border:1px solid #d0d7de; border-radius:6px; }}
  table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
  th, td {{ padding: 4px 8px; text-align: right; white-space: nowrap; border-bottom:1px solid #eaeef2; }}
  td:first-child, th:first-child {{ text-align: left; font-weight:600; }}
  thead th {{ position: sticky; top: 0; background:#f6f8fa; cursor: pointer; border-bottom:2px solid #d0d7de; }}
  thead tr.groups th {{ background:#eef1f5; cursor: default; font-weight:700; }}
  .grpA {{ border-left:3px solid #0969da; }}
  .grpB {{ border-left:3px solid #8250df; }}
  tr:hover td {{ background:#fbfdff; }}
  td.spr {{ font-weight:600; }}
  .excellent {{ background:#c6efce; color:#0a6b1f; }}
  .good      {{ background:#e3f4d7; color:#3d6b15; }}
  .marginal  {{ background:#fff1c2; color:#7a5b00; }}
  .avoid     {{ background:#ffd0d0; color:#9c0006; }}
  .none      {{ background:#f0f0f0; color:#999; }}
  td.tA {{ color:#0969da; font-weight:600; }}
  td.tB {{ color:#8250df; font-weight:600; }}
  th.sortable:hover {{ background:#eaeef2; }}
</style>
</head>
<body>
  <h1>Options Spread% Comparison</h1>
  <div class="meta">
    Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &middot; {len(symbols)} symbols &middot;
    A = ATM call closest to <b>{args.target_dte} DTE</b> &middot;
    B = ATM call at the <b>nearest monthly</b> (&ge; {args.min_dte} days).<br>
    SPR% = (Ask&minus;Bid)/Mid&times;100, Mid=(Bid+Ask)/2. Tighter: ~{args.target_dte}DTE on
    <b>{a_tighter}</b>, monthly on <b>{b_tighter}</b>.
    Quotes captured after hours unless re-run intraday (cells may show 0/&minus;).
  </div>
  <div class="legend">
    Spread rating:
    <span class="excellent">excellent &le;3%</span>
    <span class="good">good &le;7%</span>
    <span class="marginal">marginal &le;12%</span>
    <span class="avoid">avoid &gt;12%</span>
    <span class="none">no quote</span>
  </div>
  <div class="controls">
    <input id="filter" type="text" placeholder="Filter symbols (e.g. AAPL, NVDA)…">
    &nbsp;&nbsp;Drop-candidates: SPR% &gt;
    <input id="thresh" type="number" step="0.5" min="0" placeholder="e.g. 12" style="width:80px;"> %
    on <b>both</b> sides
    <label style="margin-left:8px;"><input type="checkbox" id="inclMissing" checked>
      also include no-quote rows</label>
    <button id="clearThresh" type="button">clear</button>
    <span id="count" style="margin-left:12px; color:#57606a;"></span>
    <div style="font-size:12px; color:#57606a; margin-top:4px;">
      With a threshold set: shows rows where BOTH A &amp; B SPR% &gt; X, plus (when
      checked) rows where one side &gt; X and the other has NO quote. Excludes
      one-tight/one-missing and both-missing (no real wide-spread evidence).
    </div>
  </div>
  <table id="t">
    <thead>
      <tr class="groups">
        <th></th><th></th>
        <th class="grpA" colspan="7">A: ~{args.target_dte} DTE</th>
        <th class="grpB" colspan="7">B: NEAREST MONTHLY</th>
        <th></th>
      </tr>
      <tr>
        <th class="sortable">SYMBOL</th>
        <th class="sortable">MKT_CAP</th>
        {head_a}
        {head_b}
        <th class="sortable">TIGHTER</th>
      </tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>
<script>
  const table = document.getElementById('t');
  const tbody = table.tBodies[0];
  // Sorting: header row is the SECOND row in thead
  const headerRow = table.tHead.rows[1];
  [...headerRow.cells].forEach((th, idx) => {{
    th.addEventListener('click', () => {{
      const asc = th.dataset.dir !== 'asc';
      [...headerRow.cells].forEach(c => c.dataset.dir = '');
      th.dataset.dir = asc ? 'asc' : 'desc';
      const rows = [...tbody.rows];
      rows.sort((r1, r2) => {{
        const v1 = r1.cells[idx].dataset.v, v2 = r2.cells[idx].dataset.v;
        const n1 = parseFloat(v1), n2 = parseFloat(v2);
        const bothNum = !isNaN(n1) && !isNaN(n2);
        const cmp = bothNum ? n1 - n2 : String(v1).localeCompare(String(v2));
        return asc ? cmp : -cmp;
      }});
      rows.forEach(r => tbody.appendChild(r));
    }});
  }});
  // Combined filter: symbol substring AND "both A & B SPR% > threshold".
  // SPR% columns are index 8 (A) and 15 (B). Missing quotes carry the 1e12
  // sentinel in data-v, so they are excluded whenever a threshold is set.
  const A_SPR = 8, B_SPR = 15, SENTINEL = 1e11;
  const filterInput = document.getElementById('filter');
  const thresh = document.getElementById('thresh');
  const inclMissing = document.getElementById('inclMissing');
  const countEl = document.getElementById('count');
  const total = tbody.rows.length;
  function applyFilters() {{
    const q = filterInput.value.trim().toUpperCase();
    const t = parseFloat(thresh.value);
    const hasT = !isNaN(t);
    let shown = 0;
    [...tbody.rows].forEach(r => {{
      let ok = r.cells[0].textContent.toUpperCase().includes(q);
      if (ok && hasT) {{
        const a = parseFloat(r.cells[A_SPR].dataset.v);
        const b = parseFloat(r.cells[B_SPR].dataset.v);
        const aMiss = isNaN(a) || a >= SENTINEL;
        const bMiss = isNaN(b) || b >= SENTINEL;
        const aWide = !aMiss && a > t;     // observed wide
        const bWide = !bMiss && b > t;
        // strict: both sides observed wide.
        // with-missing: at least one side wide, and neither side observed tight
        //   -> covers "both > X" and "one > X while the other has no quote",
        //      but excludes one-tight+one-missing and both-missing.
        const strict = aWide && bWide;
        const withMissing = (aWide || bWide) && (aWide || aMiss) && (bWide || bMiss);
        ok = inclMissing.checked ? withMissing : strict;
      }}
      r.style.display = ok ? '' : 'none';
      if (ok) shown++;
    }});
    countEl.textContent = hasT
      ? `Showing ${{shown}} of ${{total}} drop-candidates (both SPR% > ${{t}}%`
        + (inclMissing.checked ? `, or one > ${{t}}% & other no-quote)` : `)`)
      : `Showing ${{shown}} of ${{total}}`;
  }}
  filterInput.addEventListener('input', applyFilters);
  thresh.addEventListener('input', applyFilters);
  inclMissing.addEventListener('change', applyFilters);
  document.getElementById('clearThresh').addEventListener('click', () => {{
    thresh.value = ''; applyFilters();
  }});
  applyFilters();
</script>
</body>
</html>
"""
    out = Path(args.output)
    out.write_text(doc)
    print(f"Wrote {out} ({len(symbols)} symbols)")


if __name__ == '__main__':
    main()
