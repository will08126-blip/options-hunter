"""
Options Hunter v1.0
© will0 — Personal Use Only

A web-based options analysis tool for 1-5 day swing trades.
Scores every call and put on a stock for a given expiration date
based on delta, theta decay, liquidity, DTE, and implied volatility.
"""

import os
from flask import Flask, render_template, jsonify, request
import yfinance as yf
import numpy as np
from scipy.stats import norm
import pandas as pd
from datetime import datetime, date
import socket
import threading
import webbrowser
import time

app = Flask(__name__)

# ─── SECTOR ETF MAPPING ───────────────────────────────────────────────────────
SECTOR_ETFS = {
    'Technology':             'XLK',
    'Healthcare':             'XLV',
    'Financial Services':     'XLF',
    'Consumer Cyclical':      'XLY',
    'Consumer Defensive':     'XLP',
    'Energy':                 'XLE',
    'Industrials':            'XLI',
    'Basic Materials':        'XLB',
    'Real Estate':            'XLRE',
    'Utilities':              'XLU',
    'Communication Services': 'XLC',
}

# ─── GREEK CALCULATIONS (Black-Scholes) ──────────────────────────────────────
def calculate_greeks(S, K, T, r, sigma, option_type):
    """
    S     = current stock price
    K     = strike price
    T     = time to expiration in years
    r     = risk-free rate
    sigma = implied volatility
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {'delta': 0.0, 'theta': 0.0, 'gamma': 0.0, 'vega': 0.0}
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
        vega  = float(S * norm.pdf(d1) * np.sqrt(T) / 100)

        if option_type == 'call':
            delta = float(norm.cdf(d1))
            theta = float((-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                          - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365)
        else:
            delta = float(norm.cdf(d1) - 1)
            theta = float((-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                          + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365)

        return {
            'delta': round(delta, 3),
            'theta': round(theta, 4),
            'gamma': round(gamma, 4),
            'vega':  round(vega,  4),
        }
    except Exception:
        return {'delta': 0.0, 'theta': 0.0, 'gamma': 0.0, 'vega': 0.0}


# ─── SCORING ENGINE ───────────────────────────────────────────────────────────
def score_option(row, greeks, dte):
    """
    Scores an option contract 0–100 for a 1–5 day swing trade.
    Higher = better. Returns (score, grade, reasons).
    """
    reasons = []

    delta = abs(greeks['delta'])
    theta = abs(greeks['theta'])
    bid   = float(row.get('bid', 0) or 0)
    ask   = float(row.get('ask', 0) or 0)
    mid   = (bid + ask) / 2 if ask > 0 else float(row.get('lastPrice', 0) or 0)
    vol   = int(row.get('volume', 0) or 0)
    oi    = int(row.get('openInterest', 0) or 0)
    iv    = float(row.get('impliedVolatility', 0) or 0)

    # ── 1. DELTA (25 pts) ───────────────────────────────────────────────────
    if 0.40 <= delta <= 0.65:
        d_score = 25
        reasons.append(('good', f'Delta {delta:.2f} — ideal range (0.40–0.65)'))
    elif 0.30 <= delta < 0.40 or 0.65 < delta <= 0.75:
        d_score = 15
        reasons.append(('ok', f'Delta {delta:.2f} — acceptable but not ideal'))
    elif 0.20 <= delta < 0.30 or 0.75 < delta <= 0.85:
        d_score = 8
        reasons.append(('warn', f'Delta {delta:.2f} — outside ideal range'))
    else:
        d_score = 3
        reasons.append(('bad', f'Delta {delta:.2f} — poor directional exposure'))

    # ── 2. THETA DECAY (25 pts) ─────────────────────────────────────────────
    if mid > 0:
        theta_pct = (theta / mid) * 100
        if theta_pct < 2:
            t_score = 25
            reasons.append(('good', f'Theta {theta_pct:.1f}%/day — very low decay'))
        elif theta_pct < 4:
            t_score = 20
            reasons.append(('good', f'Theta {theta_pct:.1f}%/day — low decay'))
        elif theta_pct < 6:
            t_score = 13
            reasons.append(('ok', f'Theta {theta_pct:.1f}%/day — moderate decay'))
        elif theta_pct < 10:
            t_score = 6
            reasons.append(('warn', f'Theta {theta_pct:.1f}%/day — high decay'))
        else:
            t_score = 1
            reasons.append(('bad', f'Theta {theta_pct:.1f}%/day — severe decay'))
    else:
        t_score = 0
        theta_pct = 0
        reasons.append(('bad', 'No price data for theta calc'))

    # ── 3. LIQUIDITY (25 pts) ───────────────────────────────────────────────
    liq = 0
    if   vol > 1000: liq += 10
    elif vol > 500:  liq += 8
    elif vol > 100:  liq += 6
    elif vol > 50:   liq += 4
    elif vol > 10:   liq += 2

    if   oi > 5000:  liq += 10
    elif oi > 1000:  liq += 8
    elif oi > 500:   liq += 6
    elif oi > 100:   liq += 4
    elif oi > 50:    liq += 2

    if ask > 0:
        spread_pct = ((ask - bid) / ask) * 100
        if   spread_pct < 5:  liq += 5
        elif spread_pct < 10: liq += 3
        elif spread_pct < 20: liq += 1

    liq = min(liq, 25)
    if   liq >= 20: reasons.append(('good', f'Volume {vol:,} / OI {oi:,} — great liquidity'))
    elif liq >= 12: reasons.append(('ok',   f'Volume {vol:,} / OI {oi:,} — decent liquidity'))
    else:           reasons.append(('bad',  f'Volume {vol:,} / OI {oi:,} — low liquidity, wide spreads'))

    # ── 4. DAYS TO EXPIRATION (15 pts) ──────────────────────────────────────
    if   14 <= dte <= 30: dte_score = 15; reasons.append(('good', f'{dte} DTE — ideal for 1–5 day swing'))
    elif 7  <= dte < 14 or 30 < dte <= 45: dte_score = 10; reasons.append(('ok', f'{dte} DTE — acceptable'))
    elif 5  <= dte < 7  or 45 < dte <= 60: dte_score = 5;  reasons.append(('warn', f'{dte} DTE — borderline'))
    else: dte_score = 2; reasons.append(('bad', f'{dte} DTE — not ideal for swing trade'))

    # ── 5. IMPLIED VOLATILITY (10 pts) ──────────────────────────────────────
    iv_pct = iv * 100
    if   iv_pct < 30:  iv_score = 10; reasons.append(('good', f'IV {iv_pct:.0f}% — cheap contract'))
    elif iv_pct < 50:  iv_score = 8;  reasons.append(('ok',   f'IV {iv_pct:.0f}% — moderate cost'))
    elif iv_pct < 80:  iv_score = 5;  reasons.append(('ok',   f'IV {iv_pct:.0f}% — elevated IV'))
    elif iv_pct < 120: iv_score = 2;  reasons.append(('warn', f'IV {iv_pct:.0f}% — expensive'))
    else:              iv_score = 0;  reasons.append(('bad',  f'IV {iv_pct:.0f}% — very expensive'))

    total = d_score + t_score + liq + dte_score + iv_score

    if   total >= 85: grade = 'A+'
    elif total >= 75: grade = 'A'
    elif total >= 65: grade = 'B+'
    elif total >= 55: grade = 'B'
    elif total >= 45: grade = 'C'
    else:             grade = 'D'

    return total, grade, reasons


# ─── MARKET & SECTOR SENTIMENT ────────────────────────────────────────────────
def get_sentiment(ticker_symbol):
    try:
        # ── Market sentiment ────────────────────────────────────────────────
        vix_data = yf.Ticker("^VIX").history(period="5d")
        spy_data = yf.Ticker("SPY").history(period="40d")
        qqq_data = yf.Ticker("QQQ").history(period="40d")

        vix_now   = float(vix_data['Close'].iloc[-1])
        spy_now   = float(spy_data['Close'].iloc[-1])
        spy_ema20 = float(spy_data['Close'].ewm(span=20).mean().iloc[-1])
        qqq_now   = float(qqq_data['Close'].iloc[-1])
        qqq_ema20 = float(qqq_data['Close'].ewm(span=20).mean().iloc[-1])

        spy_bull = spy_now > spy_ema20
        qqq_bull = qqq_now > qqq_ema20
        bulls    = sum([spy_bull, qqq_bull])

        if vix_now < 18 and bulls == 2:
            mkt_s = 'Bullish';  mkt_color = 'green'
            mkt_desc = f'VIX {vix_now:.1f} (calm) — SPY & QQQ above 20-day average'
        elif vix_now > 28 or bulls == 0:
            mkt_s = 'Bearish';  mkt_color = 'red'
            mkt_desc = f'VIX {vix_now:.1f} (elevated fear) — market under pressure'
        else:
            mkt_s = 'Neutral';  mkt_color = 'yellow'
            mkt_desc = f'VIX {vix_now:.1f} — mixed signals, proceed with caution'

        # ── Stock info & sector ─────────────────────────────────────────────
        stock  = yf.Ticker(ticker_symbol)
        info   = stock.info
        name   = info.get('longName', ticker_symbol)
        price  = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        sector = info.get('sector', 'Unknown')

        if not price:
            hist  = stock.history(period="1d")
            price = float(hist['Close'].iloc[-1]) if not hist.empty else 0

        # ── Sector sentiment ────────────────────────────────────────────────
        etf_sym  = SECTOR_ETFS.get(sector, 'SPY')
        etf_data = yf.Ticker(etf_sym).history(period="40d")
        etf_now  = float(etf_data['Close'].iloc[-1])
        etf_ema  = float(etf_data['Close'].ewm(span=20).mean().iloc[-1])
        etf_5d   = ((etf_now / float(etf_data['Close'].iloc[-5])) - 1) * 100

        if etf_now > etf_ema and etf_5d > 0:
            sec_s = 'Bullish'; sec_color = 'green'
        elif etf_now < etf_ema and etf_5d < 0:
            sec_s = 'Bearish'; sec_color = 'red'
        else:
            sec_s = 'Neutral'; sec_color = 'yellow'

        sec_desc = f'{sector} ({etf_sym}) {etf_5d:+.1f}% over last 5 days'

        return {
            'market': {'sentiment': mkt_s, 'color': mkt_color, 'desc': mkt_desc, 'vix': round(vix_now, 2)},
            'sector': {'sentiment': sec_s, 'color': sec_color, 'desc': sec_desc, 'name': sector, 'etf': etf_sym},
            'stock':  {'name': name, 'price': round(float(price), 2), 'ticker': ticker_symbol},
        }
    except Exception as e:
        return {'error': str(e)}


# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', local_ip=get_local_ip())


@app.route('/api/stock/<ticker>')
def stock_info(ticker):
    try:
        ticker  = ticker.upper().strip()
        data    = get_sentiment(ticker)
        exps    = list(yf.Ticker(ticker).options)
        data['expirations'] = exps
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/analyze', methods=['POST'])
def analyze():
    body       = request.json
    ticker     = body.get('ticker', '').upper().strip()
    expiration = body.get('expiration', '')

    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        price = info.get('currentPrice') or info.get('regularMarketPrice') or 0

        if not price:
            hist  = stock.history(period="1d")
            price = float(hist['Close'].iloc[-1]) if not hist.empty else 0

        price = float(price)
        chain = stock.option_chain(expiration)
        exp_d = datetime.strptime(expiration, '%Y-%m-%d').date()
        dte   = max((exp_d - date.today()).days, 0)
        r     = 0.05

        results = []

        def process(df, opt_type):
            for _, row in df.iterrows():
                try:
                    strike = float(row['strike'])
                    iv     = float(row['impliedVolatility']) if not pd.isna(row['impliedVolatility']) else 0.3
                    T      = max(dte / 365.0, 0.001)
                    bid    = float(row['bid'] or 0)
                    ask    = float(row['ask'] or 0)
                    mid    = (bid + ask) / 2 if ask > 0 else float(row.get('lastPrice', 0) or 0)

                    greeks              = calculate_greeks(price, strike, T, r, iv, opt_type)
                    score, grade, rsns  = score_option(row.to_dict(), greeks, dte)

                    results.append({
                        'type':         opt_type.upper(),
                        'strike':       strike,
                        'bid':          round(bid, 2),
                        'ask':          round(ask, 2),
                        'mid':          round(mid, 2),
                        'volume':       int(row['volume'])       if not pd.isna(row.get('volume', None))       else 0,
                        'openInterest': int(row['openInterest']) if not pd.isna(row.get('openInterest', None)) else 0,
                        'iv':           round(iv * 100, 1),
                        'delta':        greeks['delta'],
                        'theta':        greeks['theta'],
                        'gamma':        greeks['gamma'],
                        'vega':         greeks['vega'],
                        'dte':          dte,
                        'score':        score,
                        'grade':        grade,
                        'reasons':      rsns,
                        'inTheMoney':   bool(row.get('inTheMoney', False)),
                    })
                except Exception:
                    continue

        process(chain.calls, 'call')
        process(chain.puts,  'put')

        results.sort(key=lambda x: x['score'], reverse=True)
        for i, r in enumerate(results):
            r['rank'] = i + 1

        return jsonify({
            'results':      results,
            'ticker':       ticker,
            'currentPrice': round(price, 2),
            'expiration':   expiration,
            'dte':          dte,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    is_render = os.environ.get('RENDER') is not None

    if not is_render:
        # Running locally
        ip = get_local_ip()
        print("\n" + "="*55)
        print("  OPTIONS HUNTER is starting...")
        print("="*55)
        print(f"  Local:          http://localhost:{port}")
        print(f"  Same WiFi:      http://{ip}:{port}")
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(port, bind_tls=True).public_url
            print(f"  PUBLIC (anywhere): {public_url}")
            print(f"\n  Open that link on your phone from ANYWHERE!")
        except Exception:
            print("  (Public URL unavailable — WiFi access still works)")
        print("="*55)
        print("  Close this window to shut down the app.\n")
        threading.Thread(target=open_browser, daemon=True).start()

    app.run(host='0.0.0.0', port=port, debug=False)
