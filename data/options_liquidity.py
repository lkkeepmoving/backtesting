"""
Options Liquidity Analyzer
==========================

Decides whether a stock's options are worth trading by probing the
at-the-money (ATM) contract at the expiration closest to a target DTE
(default 60 days).

Two metrics are computed and compared:

  1. Open Interest (DailyOpenInterest) at the ATM strike
        -> the user's proposal: how much standing size exists.
  2. Bid/Ask spread as % of mid at the ATM strike
        -> the recommended primary metric: the actual round-trip cost.

A combined verdict requires BOTH a reasonable spread and enough OI.

Data flow (TradeStation API) -- mirrors the proven approach in the
quantitative-trading repo:
  - GET /marketdata/quotes/{symbol}                  underlying last + spread
  - GET /marketdata/options/expirations/{underlying} pick ~target DTE expiration
  - GET /marketdata/options/strikes/{underlying}     strikes for that expiration
  - GET /marketdata/stream/options/quotes            ATM Bid/Ask/Volume/OI (one call)

NOTE: requires the `OptionSpreads` OAuth scope (see tradestation_client.py).
The chains endpoint does NOT return DailyOpenInterest; the per-contract
options/quotes stream does, which is why we use it here.

Usage:
  python3 options_liquidity.py --symbols AAPL UI MELI
  python3 options_liquidity.py --symbols-file finviz_30b_large_cap_tickers.csv --limit 30
  python3 options_liquidity.py --symbols-file finviz_30b_large_cap_tickers.csv --dte 60
"""

import argparse
import csv
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

from tradestation_client import TradeStationClient
from fetcher import get_tradestation_config, load_symbols_from_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---- Decision thresholds -------------------------------------------------

# Spread as % of mid (recommended primary metric)
SPREAD_EXCELLENT = 3.0
SPREAD_GOOD = 7.0
SPREAD_MARGINAL = 12.0

# Open interest at the ATM strike (user's proposal)
OI_STRONG = 1000
OI_OK = 250
OI_THIN = 50

# A symbol is "worth trading" if it passes BOTH gates.
VERDICT_MAX_SPREAD = 10.0
VERDICT_MIN_OI = 250


def spread_rating(spread_pct):
    if spread_pct is None:
        return 'n/a'
    if spread_pct <= SPREAD_EXCELLENT:
        return 'excellent'
    if spread_pct <= SPREAD_GOOD:
        return 'good'
    if spread_pct <= SPREAD_MARGINAL:
        return 'marginal'
    return 'avoid'


def oi_rating(oi):
    if oi is None:
        return 'n/a'
    if oi >= OI_STRONG:
        return 'strong'
    if oi >= OI_OK:
        return 'ok'
    if oi >= OI_THIN:
        return 'thin'
    return 'illiquid'


def verdict(spread_pct, oi):
    if spread_pct is None or oi is None:
        return 'NO DATA'
    if spread_pct <= VERDICT_MAX_SPREAD and oi >= VERDICT_MIN_OI:
        return 'TRADEABLE'
    return 'SKIP'


def _f(x):
    """Parse a possibly-None/str numeric to float, else None."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def option_root(symbol):
    """Root symbol used in option quote symbols (plain equities pass through)."""
    return symbol.replace('$', '').replace('.X', '')


def fmt_strike(strike):
    """Format a strike for an option symbol: 100.0->'100', 62.5->'62.5'."""
    if strike == int(strike):
        return str(int(strike))
    return f'{strike:.10f}'.rstrip('0').rstrip('.')


class OptionsAnalyzer:
    def __init__(self, client):
        self.client = client
        self.base = client.base_url
        self.options_403 = False
        self._auth_lock = threading.Lock()

    def _headers(self):
        with self._auth_lock:
            self.client._ensure_authenticated()
            return self.client._get_headers()

    def underlying_quote(self, symbol):
        """Return (last, bid, ask, volume) for the underlying, or Nones."""
        try:
            r = requests.get(f'{self.base}/marketdata/quotes/{symbol}',
                             headers=self._headers(), timeout=15)
            if r.status_code != 200:
                return None, None, None, None
            q = (r.json().get('Quotes') or [{}])[0]
            return _f(q.get('Last')), _f(q.get('Bid')), _f(q.get('Ask')), _f(q.get('Volume'))
        except requests.exceptions.RequestException as e:
            logger.debug(f'{symbol}: underlying quote error {e}')
            return None, None, None, None

    def fetch_expirations(self, symbol):
        """Return list of expiration dicts, or None on 403/error. [] = no options."""
        r = requests.get(f'{self.base}/marketdata/options/expirations/{symbol}',
                         headers=self._headers(), timeout=15)
        if r.status_code == 403:
            self.options_403 = True
            return None
        if r.status_code != 200:
            logger.warning(f'{symbol}: expirations status {r.status_code}')
            return None
        return r.json().get('Expirations') or []

    def pick_expiration(self, expirations, target_dte):
        """Return (iso_date, dte, mmddyyyy) closest to target, preferring Monthly."""
        today = datetime.now(timezone.utc).date()
        candidates = []
        for e in expirations:
            iso = (e.get('Date') or '')[:10]
            try:
                dt = datetime.strptime(iso, '%Y-%m-%d').date()
            except ValueError:
                continue
            dte = (dt - today).days
            if dte < 1:
                continue
            penalty = 0 if e.get('Type') == 'Monthly' else 5  # prefer monthlies
            candidates.append((abs(dte - target_dte) + penalty, iso, dte,
                               dt.strftime('%m-%d-%Y')))
        if not candidates:
            return None
        candidates.sort()
        _, iso, dte, mmddyyyy = candidates[0]
        return iso, dte, mmddyyyy

    def fetch_strikes(self, symbol, mmddyyyy):
        """Return sorted list of strike floats for an expiration, or []."""
        r = requests.get(f'{self.base}/marketdata/options/strikes/{symbol}',
                         headers=self._headers(), params={'expiration': mmddyyyy},
                         timeout=15)
        if r.status_code != 200:
            logger.warning(f'{symbol}: strikes status {r.status_code}')
            return []
        raw = r.json().get('Strikes') or []
        strikes = []
        for s in raw:
            v = _f(s[0]) if isinstance(s, list) and s else _f(s)
            if v is not None:
                strikes.append(v)
        return sorted(strikes)

    def option_quote(self, option_symbol, timeout=10):
        """Stream a single option quote; return first frame with Bid/Ask, or None."""
        url = f'{self.base}/marketdata/stream/options/quotes'
        # enableGreeks=true is REQUIRED for DailyOpenInterest to be present.
        params = {'legs[0].Symbol': option_symbol, 'legs[0].Ratio': 1,
                  'enableGreeks': 'true'}
        try:
            with requests.get(url, headers=self._headers(), params=params,
                              stream=True, timeout=(10, timeout)) as r:
                if r.status_code == 403:
                    self.options_403 = True
                    return None
                if r.status_code != 200:
                    logger.warning(f'{option_symbol}: quote status {r.status_code}')
                    return None
                start = time.time()
                for line in r.iter_lines():
                    if time.time() - start > timeout:
                        break
                    if not line:
                        continue
                    try:
                        msg = json.loads(line.decode())
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(msg, dict) and 'Bid' in msg and 'Ask' in msg:
                        return msg
                    if isinstance(msg, dict) and msg.get('Error'):
                        logger.debug(f'{option_symbol}: {msg.get("Error")}')
                        return None
        except requests.exceptions.RequestException as e:
            logger.debug(f'{option_symbol}: quote stream error {e}')
        return None

    def analyze(self, symbol, target_dte):
        last, u_bid, u_ask, u_vol = self.underlying_quote(symbol)
        u_spread_pct = None
        if last and u_bid and u_ask and last > 0:
            u_spread_pct = round((u_ask - u_bid) / last * 100, 2)

        row = {
            'symbol': symbol, 'underlying_last': last,
            'underlying_spread_pct': u_spread_pct,
            'expiration': None, 'dte': None, 'atm_strike': None,
            'opt_bid': None, 'opt_ask': None, 'opt_mid': None,
            'opt_spread_pct': None, 'open_interest': None, 'opt_volume': None,
            'bid_size': None, 'ask_size': None,
            'oi_rating': 'n/a', 'spread_rating': 'n/a', 'verdict': 'NO DATA',
        }

        exps = self.fetch_expirations(symbol)
        if exps is None:
            return row
        if not exps:
            row['verdict'] = 'NO OPTIONS'
            return row

        picked = self.pick_expiration(exps, target_dte)
        if not picked:
            return row
        iso, dte, mmddyyyy = picked
        row['expiration'], row['dte'] = iso, dte

        strikes = self.fetch_strikes(symbol, mmddyyyy)
        if not strikes or last is None:
            return row
        atm = min(strikes, key=lambda k: abs(k - last))
        row['atm_strike'] = atm

        yymmdd = datetime.strptime(iso, '%Y-%m-%d').strftime('%y%m%d')
        opt_symbol = f'{option_root(symbol)} {yymmdd}C{fmt_strike(atm)}'
        q = self.option_quote(opt_symbol)
        if not q:
            return row

        bid, ask = _f(q.get('Bid')), _f(q.get('Ask'))
        mid = _f(q.get('Mid'))
        if mid is None and bid is not None and ask is not None:
            mid = (bid + ask) / 2
        oi = q.get('DailyOpenInterest')
        vol = q.get('Volume')
        spread_pct = None
        if bid is not None and ask is not None and mid and mid > 0:
            spread_pct = round((ask - bid) / mid * 100, 2)

        row.update({
            'opt_bid': bid, 'opt_ask': ask,
            'opt_mid': round(mid, 4) if mid is not None else None,
            'opt_spread_pct': spread_pct,
            'open_interest': oi, 'opt_volume': vol,
            'bid_size': q.get('BidSize'), 'ask_size': q.get('AskSize'),
            'oi_rating': oi_rating(oi), 'spread_rating': spread_rating(spread_pct),
            'verdict': verdict(spread_pct, oi),
        })
        return row


def load_symbols_with_meta(file_path):
    """Load [{symbol, country, market_cap}, ...] from the finviz-style CSV."""
    rows = []
    with open(file_path) as f:
        for r in csv.DictReader(f):
            sym = (r.get('symbol') or '').strip()
            if sym:
                rows.append({'symbol': sym,
                             'country': (r.get('country') or '').strip(),
                             'market_cap': (r.get('market_cap') or '').strip()})
    return rows


def definitions_text(dte):
    return (
        "OPTIONS LIQUIDITY -- METRIC DEFINITIONS\n"
        "=======================================\n"
        f"All option metrics are for the AT-THE-MONEY (ATM) CALL at the listed\n"
        f"expiration CLOSEST TO {dte} DTE (days to expiration). ATM = the listed\n"
        "strike nearest the underlying's last traded price.\n\n"
        "spread_pct : (Ask - Bid) / Mid * 100, where Mid = (Bid + Ask) / 2.\n"
        "             The round-trip cost to cross the option's bid/ask, as a %\n"
        "             of its mid price. Lower = tighter = better.\n"
        "             Ratings: <=3 excellent, <=7 good, <=12 marginal, else avoid.\n\n"
        "open_interest : DailyOpenInterest of that same ATM call -- the number of\n"
        "             outstanding contracts at that one strike+expiration. Higher\n"
        "             = deeper standing liquidity.\n"
        "             Ratings: >=1000 strong, >=250 ok, >=50 thin, else illiquid.\n\n"
        "verdict   : TRADEABLE if spread_pct <= 10 AND open_interest >= 250.\n"
        "             SKIP       if options exist but fail either gate.\n"
        "             NO OPTIONS if the underlying lists no options at all.\n"
        "             NO DATA    if the option quote could not be retrieved.\n"
    )


def main():
    parser = argparse.ArgumentParser(description='Analyze options liquidity (ATM ~60 DTE)')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--symbols', nargs='+', help='Symbols, e.g. AAPL UI MELI')
    g.add_argument('--symbols-file', help='finviz-style CSV (symbol,country,market_cap)')
    parser.add_argument('--dte', type=int, default=60, help='Target days to expiration (default 60)')
    parser.add_argument('--limit', type=int, help='Only analyze first N symbols (testing)')
    parser.add_argument('--workers', type=int, default=5, help='Parallel workers (default 5)')
    parser.add_argument('--output', default='options_liquidity.csv')
    args = parser.parse_args()

    if args.symbols:
        meta = [{'symbol': s, 'country': '', 'market_cap': ''} for s in args.symbols]
    else:
        meta = load_symbols_with_meta(args.symbols_file)
    if args.limit:
        meta = meta[:args.limit]
    if not meta:
        logger.error('No symbols to analyze.')
        sys.exit(1)
    meta_by_sym = {m['symbol']: m for m in meta}

    print(definitions_text(args.dte))

    client = TradeStationClient(get_tradestation_config())
    analyzer = OptionsAnalyzer(client)

    total = len(meta)
    results = {}
    done = 0
    progress_lock = threading.Lock()

    def work(m):
        return analyzer.analyze(m['symbol'], args.dte)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, m): m['symbol'] for m in meta}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                logger.warning(f'{sym}: analyze error {e}')
                row = {'symbol': sym, 'opt_spread_pct': None, 'open_interest': None,
                       'verdict': 'NO DATA'}
            results[sym] = row
            with progress_lock:
                done += 1
                sp, oi = row.get('opt_spread_pct'), row.get('open_interest')
                logger.info(f"[{done}/{total}] {sym:6} "
                            f"spread={sp if sp is not None else '-'}% "
                            f"OI={oi if oi is not None else '-'} => {row.get('verdict')}")

    if analyzer.options_403:
        logger.error("Options endpoints returned 403 'Missing required scope'. Ensure the "
                     "OptionSpreads OAuth scope is set and re-run auth (rm data/.tradestation_tokens).")

    # Merge metadata + assemble rows in requested column order
    order = {'TRADEABLE': 0, 'SKIP': 1, 'NO OPTIONS': 2, 'NO DATA': 3}
    out_rows = []
    for m in meta:
        r = results.get(m['symbol'], {})
        out_rows.append({
            'symbol': m['symbol'],
            'country': m['country'],
            'market_cap': m['market_cap'],
            'spread_pct': r.get('opt_spread_pct'),
            'spread_rating': r.get('spread_rating', 'n/a'),
            'open_interest': r.get('open_interest'),
            'oi_rating': r.get('oi_rating', 'n/a'),
            'verdict': r.get('verdict', 'NO DATA'),
            'expiration': r.get('expiration'),
            'dte': r.get('dte'),
            'atm_strike': r.get('atm_strike'),
            'opt_bid': r.get('opt_bid'),
            'opt_ask': r.get('opt_ask'),
            'opt_mid': r.get('opt_mid'),
            'opt_volume': r.get('opt_volume'),
            'underlying_last': r.get('underlying_last'),
        })
    out_rows.sort(key=lambda r: (order.get(r['verdict'], 9),
                                 r['spread_pct'] if r['spread_pct'] is not None else 1e9))

    fields = ['symbol', 'country', 'market_cap', 'spread_pct', 'spread_rating',
              'open_interest', 'oi_rating', 'verdict', 'expiration', 'dte',
              'atm_strike', 'opt_bid', 'opt_ask', 'opt_mid', 'opt_volume',
              'underlying_last']
    out = Path(args.output)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    defs_path = out.with_name(out.stem + '_definitions.txt')
    defs_path.write_text(definitions_text(args.dte))

    def n(v):
        return sum(1 for r in out_rows if r['verdict'] == v)
    logger.info('=' * 64)
    logger.info(f"Analyzed {len(out_rows)} | TRADEABLE {n('TRADEABLE')} | SKIP {n('SKIP')} | "
                f"NO OPTIONS {n('NO OPTIONS')} | NO DATA {n('NO DATA')}")
    logger.info(f"Wrote {out}  and  {defs_path}")
    logger.info('=' * 64)


if __name__ == '__main__':
    main()
