"""
VIX Divergence Scan
===================

Finds days where SPY and/or QQQ rose by at least a threshold while VIX did NOT
fall much ("the rally happened but fear didn't subside"), then reports forward
performance of SPY, QQQ and VIX over the next 1/2/3/5 trading days.

Thesis: an up day in the index that is NOT accompanied by the usual VIX decline
signals hesitation; the move may continue higher over the following days.

Filter semantics
----------------
  --spy-up X        SPY close-to-close return >= X%   (optional)
  --qqq-up Y        QQQ close-to-close return >= Y%   (optional)
                    If both are given the conditions are AND-ed.
                    At least one of the two is required.
  --vix-threshold Z VIX "fell by less than Z%", i.e. vix_ret > -Z%.
                    This is the divergence condition (default 5.0).

All returns are close-to-close. A signal is "entered" at the run day's close,
so forward performance over N days = close[t+N] / close[t] - 1.

Usage
-----
  # SPY up >= 1%, VIX fell less than 5%, full history
  python3 strategies/vix_divergence_scan.py --spy-up 1 --vix-threshold 5

  # SPY up >= 1% AND QQQ up >= 1%, last 500 bars, export CSV
  python3 strategies/vix_divergence_scan.py --spy-up 1 --qqq-up 1 --bars 500 --csv

  # QQQ only, within a date window
  python3 strategies/vix_divergence_scan.py --qqq-up 1.5 --start 2023-01-01 --end 2024-12-31
"""

import argparse
import csv
import os
import statistics
import sys
from datetime import datetime
from html import escape
from typing import Dict, List, NamedTuple, Optional

STORAGE = os.path.join(os.path.dirname(__file__), "..", "data", "storage")
REPORTS = os.path.join(os.path.dirname(__file__), "..", "reports")
HORIZONS = [1, 2, 3, 4, 5]


class Signal(NamedTuple):
    """A matching divergence day with its forward performance."""
    date: str
    spy: float
    qqq: float
    vix: float
    spy_prev5: Optional[float]  # SPY return over the 5 days BEFORE the signal day
    qqq_prev5: Optional[float]  # QQQ return over the 5 days BEFORE the signal day
    spy_fwd: Dict[int, Optional[float]]
    qqq_fwd: Dict[int, Optional[float]]
    vix_fwd: Dict[int, Optional[float]]


def load_closes(symbol: str) -> List[tuple]:
    """Return [(date, close), ...] sorted oldest -> newest."""
    path = os.path.join(STORAGE, f"{symbol}.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((r["Date"], float(r["Close"])))
    rows.sort(key=lambda x: x[0])
    return rows


def pct(a: float, b: float) -> Optional[float]:
    return (a / b - 1.0) * 100.0 if b else None


def forward(series: Dict[str, float], dates: List[str], i: int, n: int) -> Optional[float]:
    """Close-to-close % return from dates[i] to dates[i+n]."""
    if i + n >= len(dates):
        return None
    d0, dn = dates[i], dates[i + n]
    if d0 not in series or dn not in series:
        return None
    return pct(series[dn], series[d0])


def scan(spy_up: Optional[float], qqq_up: Optional[float], vix_threshold: float,
         start: Optional[str], end: Optional[str], bars: Optional[int]):
    """
    Returns (signals, baseline_dates, market, scanned).

    signals        : list[Signal] matching the full filter (incl. VIX divergence)
    baseline_dates : dates matching the SPY/QQQ up-filter ONLY (ignoring VIX) --
                     the control group; signals are a subset of these.
    market         : (spy_closes, qqq_closes, all_common_dates)
    scanned        : every trading day actually examined after the range gate
                     (used to report the concrete date range).
    """
    spy = dict(load_closes("SPY"))
    qqq = dict(load_closes("QQQ"))
    vix = dict(load_closes("VIX_X"))

    dates = sorted(set(spy) & set(qqq) & set(vix))

    # restrict by --bars (most recent N) if requested
    if bars is not None and bars < len(dates):
        dates_window = set(dates[-bars:])
    else:
        dates_window = None

    signals: List[Signal] = []
    baseline_dates: List[str] = []
    scanned: List[str] = []  # every trading day actually examined (after range gate)

    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i - 1]
        if start and d < start:
            continue
        if end and d > end:
            continue
        if dates_window is not None and d not in dates_window:
            continue
        scanned.append(d)

        spy_r = pct(spy[d], spy[dp])
        qqq_r = pct(qqq[d], qqq[dp])
        vix_r = pct(vix[d], vix[dp])

        # ---- index up-filter (AND if both specified) ----
        if spy_up is not None and spy_r < spy_up:
            continue
        if qqq_up is not None and qqq_r < qqq_up:
            continue

        # day passes the up-filter -> part of the control/baseline group
        baseline_dates.append(d)

        # ---- VIX divergence: fell by LESS than vix_threshold ----
        if vix_r <= -vix_threshold:
            continue

        # backward-looking: return over the 5 trading days BEFORE the signal day
        # = close[i-1] / close[i-6] - 1 (excludes the signal day itself)
        spy_prev5 = pct(spy[dates[i - 1]], spy[dates[i - 6]]) if i >= 6 else None
        qqq_prev5 = pct(qqq[dates[i - 1]], qqq[dates[i - 6]]) if i >= 6 else None

        signals.append(Signal(
            date=d, spy=spy_r, qqq=qqq_r, vix=vix_r,
            spy_prev5=spy_prev5, qqq_prev5=qqq_prev5,
            spy_fwd={n: forward(spy, dates, i, n) for n in HORIZONS},
            qqq_fwd={n: forward(qqq, dates, i, n) for n in HORIZONS},
            vix_fwd={n: forward(vix, dates, i, n) for n in HORIZONS},
        ))

    return signals, baseline_dates, (spy, qqq, dates), scanned


def _fwd_for_dates(series: Dict[str, float], dates: List[str], target_dates,
                   n: int) -> List[float]:
    idx = {d: k for k, d in enumerate(dates)}
    out = []
    for d in target_dates:
        i = idx[d]
        v = forward(series, dates, i, n)
        if v is not None:
            out.append(v)
    return out


def _fmt(v: Optional[float]) -> str:
    return f"{v:+7.2f}" if v is not None else "     --"


def print_table(signals: List[Signal]):
    cols = (f"{'Date':<11}{'SPYpre5':>8}{'QQQpre5':>8}  |"
            f"{'SPY%':>7}{'QQQ%':>7}{'VIX%':>7}  | "
            + "".join(f"SPY+{n:<4}" for n in HORIZONS) + " | "
            + "".join(f"QQQ+{n:<4}" for n in HORIZONS) + " | "
            + "".join(f"VIX+{n:<4}" for n in HORIZONS))
    print(cols)
    print("-" * len(cols))
    for s in signals:
        line = (f"{s.date:<11}{_fmt(s.spy_prev5):>8}{_fmt(s.qqq_prev5):>8}  |"
                f"{s.spy:>7.2f}{s.qqq:>7.2f}{s.vix:>7.2f}  | "
                + "".join(_fmt(s.spy_fwd[n]) + " " for n in HORIZONS) + "| "
                + "".join(_fmt(s.qqq_fwd[n]) + " " for n in HORIZONS) + "| "
                + "".join(_fmt(s.vix_fwd[n]) + " " for n in HORIZONS))
        print(line)
    print("-" * len(cols))


def _stats(vals):
    """(mean, median, win%, n) or None."""
    if not vals:
        return None
    win = sum(1 for v in vals if v > 0) / len(vals) * 100
    return (statistics.mean(vals), statistics.median(vals), win, len(vals))


def summary_rows(signals, baseline_dates, market):
    """Structured summary: [(label, n, div_stats, ctrl_stats), ...]."""
    spy, qqq, dates = market
    sig_dates = [s.date for s in signals]
    out = []
    for label, series in (("SPY", spy), ("QQQ", qqq)):
        for n in HORIZONS:
            out.append((label, n,
                        _stats(_fwd_for_dates(series, dates, sig_dates, n)),
                        _stats(_fwd_for_dates(series, dates, baseline_dates, n))))
    return out


def print_summary(signals, baseline_dates, market):
    spy, qqq, dates = market
    print(f"\n{len(baseline_dates)} days met the SPY/QQQ up-filter; of those, "
          f"{len(signals)} also had the VIX divergence.")
    print(f"  -> DIVERGENCE group = {len(signals)} days (the table above)")
    print(f"  -> CONTROL group    = all {len(baseline_dates)} up-days "
          f"(divergence is a subset; used for the edge comparison)\n")
    stats = _stats
    sig_dates = [s.date for s in signals]

    print(f"{'':>6} {'--- DIVERGENCE days ---':>30}   {'--- ALL up-days (control) ---':>32}")
    print(f"{'Horiz':>6} {'mean':>8}{'median':>8}{'win%':>7}{'n':>6}   "
          f"{'mean':>8}{'median':>8}{'win%':>7}{'n':>6}")
    for label, series in (("SPY", spy), ("QQQ", qqq)):
        print(f"  [{label} forward returns]")
        for n in HORIZONS:
            s = stats(_fwd_for_dates(series, dates, sig_dates, n))
            b = stats(_fwd_for_dates(series, dates, baseline_dates, n))
            sstr = (f"{s[0]:>8.2f}{s[1]:>8.2f}{s[2]:>6.0f}%{s[3]:>6}" if s
                    else f"{'--':>29}")
            bstr = (f"{b[0]:>8.2f}{b[1]:>8.2f}{b[2]:>6.0f}%{b[3]:>6}" if b
                    else f"{'--':>29}")
            print(f"  +{n:<4} {sstr}   {bstr}")
    print("\nEdge = DIVERGENCE mean minus CONTROL mean (positive => divergence "
          "outperforms plain up-days).")


def export_csv(signals: List[Signal], params: str) -> str:
    os.makedirs(REPORTS, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS, f"vix_divergence_{ts}.csv")
    header = (["Date", "SPY_prev5", "QQQ_prev5", "SPY%", "QQQ%", "VIX%"]
              + [f"SPY+{n}" for n in HORIZONS]
              + [f"QQQ+{n}" for n in HORIZONS]
              + [f"VIX+{n}" for n in HORIZONS])

    def f2(v):
        return f"{v:.2f}" if v is not None else ""

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# {params}"])
        w.writerow(header)
        for s in signals:
            row = [s.date, f2(s.spy_prev5), f2(s.qqq_prev5),
                   f"{s.spy:.2f}", f"{s.qqq:.2f}", f"{s.vix:.2f}"]
            for fwd in (s.spy_fwd, s.qqq_fwd, s.vix_fwd):
                row += [f2(fwd[n]) for n in HORIZONS]
            w.writerow(row)
    return path


def export_html(signals, baseline_dates, market, params, rng_label) -> str:
    os.makedirs(REPORTS, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS, f"vix_divergence_{ts}.html")

    def rcls(v):
        # color scale for forward returns
        if v is None:
            return "none"
        if v >= 1.0:
            return "pos2"
        if v > 0:
            return "pos1"
        if v <= -1.0:
            return "neg2"
        return "neg1"

    def cell(v, extra=""):
        if v is None:
            return f'<td class="none{extra}">--</td>'
        return f'<td class="{rcls(v)}{extra}" data-sort="{v:.4f}">{v:+.2f}</td>'

    # signal rows
    body = []
    for s in signals:
        tds = [f'<td class="date" data-sort="{s.date}">{s.date}</td>',
               cell(s.spy_prev5, " grpL"),
               cell(s.qqq_prev5),
               f'<td class="day grpD" data-sort="{s.spy:.4f}">{s.spy:+.2f}</td>',
               f'<td class="day" data-sort="{s.qqq:.4f}">{s.qqq:+.2f}</td>',
               f'<td class="day vix" data-sort="{s.vix:.4f}">{s.vix:+.2f}</td>']
        for fwd in (s.spy_fwd, s.qqq_fwd, s.vix_fwd):
            tds += [cell(fwd[n]) for n in HORIZONS]
        body.append("<tr>" + "".join(tds) + "</tr>")

    # summary table
    srows = summary_rows(signals, baseline_dates, market)

    def sfmt(st):
        if not st:
            return "<td>--</td><td>--</td><td>--</td><td>--</td>"
        m, med, win, n = st
        mc = "pos1" if m > 0 else "neg1"
        return (f'<td class="{mc}">{m:+.2f}</td><td>{med:+.2f}</td>'
                f'<td>{win:.0f}%</td><td>{n}</td>')

    summ = []
    for label, n, div, ctrl in srows:
        edge = (div[0] - ctrl[0]) if (div and ctrl) else None
        ec = "" if edge is None else (' class="pos1"' if edge > 0 else ' class="neg1"')
        estr = "--" if edge is None else f"{edge:+.2f}"
        summ.append(f"<tr><td>{label}</td><td>+{n}</td>{sfmt(div)}{sfmt(ctrl)}"
                    f"<td{ec}>{estr}</td></tr>")

    def fwd_th(grp):
        return "".join(
            f"<th class='sortable {grp}'>+{n}</th>" if i == 0
            else f"<th class='sortable'>+{n}</th>"
            for i, n in enumerate(HORIZONS))

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>VIX Divergence Scan</title>
<style>
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:18px; color:#1f2328; }}
  h2 {{ margin-bottom:2px; }}
  .meta {{ color:#57606a; font-size:13px; margin-bottom:12px; }}
  table {{ border-collapse:collapse; font-size:13px; margin-bottom:24px; }}
  th,td {{ padding:4px 9px; text-align:right; white-space:nowrap; border-bottom:1px solid #eaeef2; }}
  thead th {{ position:sticky; top:0; background:#f6f8fa; cursor:pointer; border-bottom:2px solid #d0d7de; }}
  thead tr.groups th {{ background:#eef1f5; cursor:default; font-weight:700; text-align:center; }}
  th.sortable:hover {{ background:#eaeef2; }}
  td.date {{ text-align:left; font-variant-numeric:tabular-nums; }}
  td.day {{ font-weight:600; background:#f3f6fb; }}
  td.day.vix {{ background:#f7f0fb; }}
  .grpL {{ border-left:3px solid #9a6700; }}
  .grpD {{ border-left:3px solid #57606a; }}
  .grpS {{ border-left:3px solid #0969da; }}
  .grpQ {{ border-left:3px solid #1a7f37; }}
  .grpV {{ border-left:3px solid #8250df; }}
  .pos2 {{ background:#c6efce; color:#0a6b1f; }}
  .pos1 {{ background:#e3f4d7; color:#3d6b15; }}
  .neg1 {{ background:#ffe3e3; color:#a32a2a; }}
  .neg2 {{ background:#ffd0d0; color:#9c0006; }}
  .none {{ background:#f0f0f0; color:#999; }}
  tr:hover td {{ filter:brightness(0.97); }}
  tr.hidden {{ display:none; }}
  .legend span {{ display:inline-block; padding:2px 8px; border-radius:4px; margin-right:6px; font-size:12px; }}
  .controls {{ background:#f6f8fa; border:1px solid #d0d7de; border-radius:8px;
               padding:10px 14px; margin:10px 0 16px; display:flex; flex-wrap:wrap;
               gap:14px; align-items:flex-end; font-size:13px; }}
  .controls label {{ display:flex; flex-direction:column; gap:3px; color:#57606a; }}
  .controls input {{ width:90px; padding:5px 8px; font-size:14px; border:1px solid #d0d7de;
                     border-radius:6px; }}
  .controls button {{ padding:6px 12px; font-size:13px; border:1px solid #d0d7de;
                      border-radius:6px; background:#fff; cursor:pointer; }}
  .controls button:hover {{ background:#eaeef2; }}
  #vcount {{ font-weight:700; color:#0a6b1f; }}
  .livebox {{ font-size:12px; color:#57606a; }}
  table.live td, table.live th {{ padding:3px 10px; }}
  table.live td.lwin {{ color:#57606a; font-size:11px; }}
</style></head><body>
<h2>VIX Divergence Scan</h2>
<div class="meta"><b>Filter:</b> {escape(params)}<br>
<b>Range:</b> {escape(rng_label)} &nbsp;|&nbsp; close-to-close, entry at signal-day close<br>
<b>{len(baseline_dates)}</b> days met the SPY/QQQ up-filter; of those,
<b>{len(signals)}</b> also had the VIX divergence (shown below).
The other {len(baseline_dates) - len(signals)} are the control-only days used in the summary.</div>
<div class="legend">
  <span class="pos2">&ge;+1%</span><span class="pos1">0 to +1%</span>
  <span class="neg1">-1 to 0%</span><span class="neg2">&le;-1%</span>
  <span class="none">n/a</span>
</div>

<div class="controls">
  <label>SPY rose &ge; (%)<input id="fSpy" type="number" step="0.1" placeholder="any"></label>
  <label>QQQ rose &ge; (%)<input id="fQqq" type="number" step="0.1" placeholder="any"></label>
  <label>Max VIX drop (%)<input id="fVix" type="number" step="0.1" placeholder="any"></label>
  <label>SPY prev-5d min (%)<input id="fSpyPreMin" type="number" step="0.1" placeholder="any"></label>
  <label>SPY prev-5d max (%)<input id="fSpyPreMax" type="number" step="0.1" placeholder="any"></label>
  <label>QQQ prev-5d min (%)<input id="fQqqPreMin" type="number" step="0.1" placeholder="any"></label>
  <label>QQQ prev-5d max (%)<input id="fQqqPreMax" type="number" step="0.1" placeholder="any"></label>
  <button id="reset">Reset</button>
  <div style="margin-left:auto;">Showing <span id="vcount">0</span> rows</div>
</div>
<div class="livebox" style="margin:-6px 0 14px;">
  <b>Max VIX drop</b> = the biggest VIX fall you'll allow (the divergence: VIX <em>didn't</em>
  drop much). The VIX% column is signed &mdash; a fall of 3% shows as &minus;3.00.
  <b>Example:</b> looking for &ldquo;VIX fell less than 3%&rdquo;? Type <b>3</b>.
  <span id="vixHint"></span> Blank = no VIX filter.<br>
  <b>SPY/QQQ prev-5d</b> = return over the 5 trading days <em>before</em> the signal day
  (excludes the signal day). Use min/max to bracket the lead-up, e.g. SPY prev-5d
  <b>max 0</b> finds divergences that fired right after a down week. Stats below update
  live for the visible rows.
</div>

<table class="live" id="liveStats">
<thead>
  <tr class="groups"><th>Visible-row stats</th>{"".join(f"<th>+{n}d</th>" for n in HORIZONS)}</tr>
</thead>
<tbody></tbody>
</table>

<table id="sig">
<thead>
  <tr class="groups"><th></th>
    <th class="grpL" colspan="2">Lead-up (prev 5d)</th>
    <th class="grpD" colspan="3">Signal day %</th>
    <th class="grpS" colspan="{len(HORIZONS)}">SPY forward</th>
    <th class="grpQ" colspan="{len(HORIZONS)}">QQQ forward</th>
    <th class="grpV" colspan="{len(HORIZONS)}">VIX forward</th></tr>
  <tr><th class="sortable">Date</th>
    <th class="sortable grpL">SPY -5d</th><th class="sortable">QQQ -5d</th>
    <th class="sortable grpD">SPY%</th><th class="sortable">QQQ%</th><th class="sortable">VIX%</th>
    {fwd_th("grpS")}{fwd_th("grpQ")}{fwd_th("grpV")}
  </tr>
</thead>
<tbody>{''.join(body)}</tbody>
</table>

<h2>Summary: divergence vs. control (all up-days)</h2>
<div class="meta">Computed for the file's <b>generation filter</b> ({escape(params)}) &mdash;
this table is static and does NOT change with the live filters above.
Edge = divergence mean &minus; control mean (positive &rArr; divergence outperforms plain up-days).</div>
<table>
<thead>
  <tr class="groups"><th colspan="2"></th><th colspan="4">DIVERGENCE days</th>
    <th colspan="4">ALL up-days (control)</th><th></th></tr>
  <tr><th>Idx</th><th>Horiz</th>
    <th>mean</th><th>median</th><th>win%</th><th>n</th>
    <th>mean</th><th>median</th><th>win%</th><th>n</th><th>edge</th></tr>
</thead>
<tbody>{''.join(summ)}</tbody>
</table>

<script>
const tb = document.querySelector('#sig tbody');
const allRows = [...tb.rows];
// column indices: 0=Date 1=SPYprev5 2=QQQprev5 3=SPY% 4=QQQ% 5=VIX% ; then forward blocks
const FWD = {{ SPY:{list(range(6, 6 + len(HORIZONS)))},
               QQQ:{list(range(6 + len(HORIZONS), 6 + 2 * len(HORIZONS)))},
               VIX:{list(range(6 + 2 * len(HORIZONS), 6 + 3 * len(HORIZONS)))} }};

function num(cell) {{ const v = parseFloat(cell.dataset.sort); return isNaN(v) ? null : v; }}

function applyFilters() {{
  const v = id => parseFloat(document.getElementById(id).value);
  const sMin = v('fSpy'), qMin = v('fQqq'), vFell = v('fVix');  // keep vix > -vFell
  const spMin = v('fSpyPreMin'), spMax = v('fSpyPreMax');
  const qpMin = v('fQqqPreMin'), qpMax = v('fQqqPreMax');
  let visible = 0;
  for (const r of allRows) {{
    const spy = num(r.cells[3]), qqq = num(r.cells[4]), vix = num(r.cells[5]);
    const spyPre = num(r.cells[1]), qqqPre = num(r.cells[2]);
    let ok = true;
    if (!isNaN(sMin) && !(spy >= sMin)) ok = false;
    if (!isNaN(qMin) && !(qqq >= qMin)) ok = false;
    if (!isNaN(vFell) && !(vix > -vFell)) ok = false;
    if (!isNaN(spMin) && !(spyPre !== null && spyPre >= spMin)) ok = false;
    if (!isNaN(spMax) && !(spyPre !== null && spyPre <= spMax)) ok = false;
    if (!isNaN(qpMin) && !(qqqPre !== null && qqqPre >= qpMin)) ok = false;
    if (!isNaN(qpMax) && !(qqqPre !== null && qqqPre <= qpMax)) ok = false;
    r.classList.toggle('hidden', !ok);
    if (ok) visible++;
  }}
  document.getElementById('vcount').textContent = visible;
  const hint = document.getElementById('vixHint');
  hint.textContent = isNaN(vFell) ? ''
    : `→ keeping days where VIX% ≥ −${{vFell}} (fell less than ${{vFell}}%, VIX-up days included).`;
  updateLiveStats();
}}

function updateLiveStats() {{
  const vis = allRows.filter(r => !r.classList.contains('hidden'));
  const tbody = document.querySelector('#liveStats tbody');
  tbody.innerHTML = '';
  for (const inst of ['SPY','QQQ','VIX']) {{
    const tr = document.createElement('tr');
    let cells = '<td style="text-align:left;font-weight:600;">' + inst + ' fwd</td>';
    for (const ci of FWD[inst]) {{
      const vals = vis.map(r => num(r.cells[ci])).filter(v => v !== null);
      if (!vals.length) {{ cells += '<td class="none">--</td>'; continue; }}
      const mean = vals.reduce((a,b)=>a+b,0)/vals.length;
      const win = vals.filter(v=>v>0).length/vals.length*100;
      const cls = mean > 0 ? 'pos1' : 'neg1';
      cells += `<td class="${{cls}}">${{mean>=0?'+':''}}${{mean.toFixed(2)}}<span class="lwin"> ${{win.toFixed(0)}}%·n${{vals.length}}</span></td>`;
    }}
    tr.innerHTML = cells;
    tbody.appendChild(tr);
  }}
}}

const FILTER_IDS = ['fSpy','fQqq','fVix','fSpyPreMin','fSpyPreMax','fQqqPreMin','fQqqPreMax'];
FILTER_IDS.forEach(id =>
  document.getElementById(id).addEventListener('input', applyFilters));
document.getElementById('reset').addEventListener('click', () => {{
  FILTER_IDS.forEach(id => document.getElementById(id).value = '');
  applyFilters();
}});

document.querySelectorAll('#sig thead th.sortable').forEach((th) => {{
  th.addEventListener('click', () => {{
    const rows = [...tb.rows];
    const asc = !(th.dataset.asc === 'true'); th.dataset.asc = asc;
    const idx = [...th.parentNode.children].indexOf(th);
    rows.sort((a, b) => {{
      const av = a.cells[idx].dataset.sort, bv = b.cells[idx].dataset.sort;
      const an = parseFloat(av), bn = parseFloat(bv);
      let c; if (!isNaN(an) && !isNaN(bn)) c = an - bn; else c = (av>bv?1:av<bv?-1:0);
      return asc ? c : -c;
    }});
    rows.forEach(r => tb.appendChild(r));
  }});
}});

applyFilters();  // initialize count + live stats
</script>
</body></html>"""
    with open(path, "w") as f:
        f.write(html)
    return path


def main():
    p = argparse.ArgumentParser(description="VIX divergence scan",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--spy-up", type=float, default=None,
                   help="SPY close-to-close return >= this %% (e.g. 1.0)")
    p.add_argument("--qqq-up", type=float, default=None,
                   help="QQQ close-to-close return >= this %% (e.g. 1.0)")
    p.add_argument("--vix-threshold", type=float, default=5.0,
                   help="VIX fell by LESS than this %% (divergence). Default 5.0")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    p.add_argument("--bars", type=int, default=None,
                   help="Scan only the most recent N bars (overrides start/end window)")
    p.add_argument("--csv", action="store_true", help="Export matches to reports/ (CSV)")
    p.add_argument("--html", action="store_true",
                   help="Export an interactive HTML report to reports/")
    args = p.parse_args()

    if args.spy_up is None and args.qqq_up is None:
        p.error("specify at least one of --spy-up or --qqq-up")

    conds = []
    if args.spy_up is not None:
        conds.append(f"SPY up >= {args.spy_up}%")
    if args.qqq_up is not None:
        conds.append(f"QQQ up >= {args.qqq_up}%")
    conds.append(f"VIX fell < {args.vix_threshold}%")
    params = "  AND  ".join(conds)

    signals, baseline_dates, market, scanned = scan(
        args.spy_up, args.qqq_up, args.vix_threshold,
        args.start, args.end, args.bars)

    # concrete scanned date range (actual data, not just the requested window)
    if scanned:
        rng_label = f"{scanned[0]} → {scanned[-1]} ({len(scanned)} trading days)"
    else:
        rng_label = "no trading days in range"

    print("=" * 70)
    print("VIX DIVERGENCE SCAN")
    print("=" * 70)
    print(f"Filter: {params}")
    print(f"Range : {rng_label}")
    print("Entry : close of signal day; forward returns are close-to-close\n")

    if not signals:
        print("No matching days.")
        return

    print_table(signals)
    print_summary(signals, baseline_dates, market)

    if args.csv:
        path = export_csv(signals, params)
        print(f"\nExported {len(signals)} rows -> {path}")
    if args.html:
        path = export_html(signals, baseline_dates, market, params, rng_label)
        print(f"\nHTML report -> {path}")


if __name__ == "__main__":
    main()
