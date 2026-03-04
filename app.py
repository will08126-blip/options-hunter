"""
Options Hunter v2.0
© will0 — Personal Use Only

Three-tab trading dashboard:
  Tab 1 — Morning Briefing (market overview, crypto, movers, news)
  Tab 2 — Technical Analysis (Cipher B MTF, MACD, Volume)
  Tab 3 — Options Hunter (options chain scoring)
"""

import os
import time
import threading
import webbrowser
import socket
import requests
import concurrent.futures

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, date
from zoneinfo import ZoneInfo
from pydantic import BaseModel

import yfinance as yf
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tvDatafeed import TvDatafeed, Interval as TvInterval

# ─── RATE LIMITER ─────────────────────────────────────────────────────────────
# Protects free APIs (tvDatafeed/yfinance) from abuse and future paid APIs.
limiter = Limiter(key_func=get_remote_address, default_limits=["200/hour"])

app       = FastAPI(title="Options Hunter v2.0", docs_url="/docs")
templates = Jinja2Templates(directory="templates")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass   # empty static dir is fine

# ─── SIMPLE IN-MEMORY CACHE ───────────────────────────────────────────────────
_cache = {}
CACHE_TTL           = 600   # 10 min — market briefing, movers, news
CACHE_TTL_TECHNICAL = 300   # 5 min  — Cipher B + MACD per ticker
CACHE_TTL_STOCK     = 300   # 5 min  — sentiment + options expirations

def cache_get(key, ttl=None):
    if key in _cache:
        age = time.time() - _cache[key]['ts']
        if age < (ttl if ttl is not None else CACHE_TTL):
            return _cache[key]['data']
    return None

def cache_set(key, data):
    _cache[key] = {'data': data, 'ts': time.time()}

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

# ─── DEFAULT CRYPTO WATCHLIST ─────────────────────────────────────────────────
DEFAULT_CRYPTO = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD',
                  'SUI-USD', 'LINK-USD', 'ONDO-USD', 'UNI-USD']

CRYPTO_NAMES = {
    'BTC-USD': 'Bitcoin',   'ETH-USD': 'Ethereum',  'XRP-USD': 'XRP',
    'SOL-USD': 'Solana',    'SUI-USD': 'SUI',        'LINK-USD': 'Chainlink',
    'ONDO-USD': 'Ondo',     'UNI-USD': 'Uniswap',   'BNB-USD': 'BNB',
    'DOGE-USD': 'Dogecoin', 'ADA-USD': 'Cardano',   'AVAX-USD': 'Avalanche',
    'DOT-USD': 'Polkadot',  'MATIC-USD': 'Polygon', 'SHIB-USD': 'Shiba Inu',
}

# ─── POPULAR STOCKS FOR TOP MOVERS ────────────────────────────────────────────
POPULAR_STOCKS = [
    'AAPL','MSFT','NVDA','TSLA','META','GOOGL','AMZN','AMD','NFLX','PLTR',
    'COIN','JPM','XOM','HOOD','SOFI'
]

# ─── TRADINGVIEW SYMBOL / EXCHANGE MAP ───────────────────────────────────────
# Crypto → Binance USDT pairs (most liquid, matches TV default)
TV_SYMBOL_MAP = {
    'BTC-USD':   ('BTCUSDT',   'BINANCE'),
    'ETH-USD':   ('ETHUSDT',   'BINANCE'),
    'XRP-USD':   ('XRPUSDT',   'BINANCE'),
    'SOL-USD':   ('SOLUSDT',   'BINANCE'),
    'SUI-USD':   ('SUIUSDT',   'BINANCE'),
    'LINK-USD':  ('LINKUSDT',  'BINANCE'),
    'ONDO-USD':  ('ONDOUSDT',  'BINANCE'),
    'UNI-USD':   ('UNIUSDT',   'BINANCE'),
    'BNB-USD':   ('BNBUSDT',   'BINANCE'),
    'DOGE-USD':  ('DOGEUSDT',  'BINANCE'),
    'ADA-USD':   ('ADAUSDT',   'BINANCE'),
    'AVAX-USD':  ('AVAXUSDT',  'BINANCE'),
    'DOT-USD':   ('DOTUSDT',   'BINANCE'),
    'MATIC-USD': ('MATICUSDT', 'BINANCE'),
    # Indices
    '^GSPC': ('SPX',  'SP'),
    '^IXIC': ('COMP', 'NASDAQ'),
    '^DJI':  ('DJI',  'DJ'),
    '^RUT':  ('RUT',  'TVC'),
    '^VIX':  ('VIX',  'CBOE'),
}

# Stocks primarily listed on NYSE (everything else defaults to NASDAQ)
NYSE_STOCKS = {
    'JPM','GS','BAC','WFC','C','MS','BRK.B','XOM','CVX',
    'JNJ','PG','KO','DIS','WMT','MA','V','UNH','HD','T',
    'VZ','MRK','HOOD','SOFI',
}

def get_tv_symbol_exchange(ticker):
    """Convert a ticker (yfinance-style) to a (TV_symbol, TV_exchange) tuple."""
    t = ticker.upper().strip()
    if t in TV_SYMBOL_MAP:
        return TV_SYMBOL_MAP[t]
    # Generic -USD crypto → BINANCE USDT pair
    if t.endswith('-USD'):
        return (t.replace('-USD', '') + 'USDT', 'BINANCE')
    if t in NYSE_STOCKS:
        return (t, 'NYSE')
    return (t, 'NASDAQ')  # default for stocks

def tv_get_hist(tv, symbol, exchange, interval, n_bars=500):
    """
    Fetch OHLCV from TradingView via tvDatafeed.
    Standardises column names and converts index to EST.
    Returns None on failure.
    """
    try:
        df = tv.get_hist(symbol, exchange, interval=interval, n_bars=n_bars)
        # Exchange fallback
        if (df is None or df.empty) and exchange == 'NASDAQ':
            df = tv.get_hist(symbol, 'NYSE', interval=interval, n_bars=n_bars)
        elif (df is None or df.empty) and exchange == 'NYSE':
            df = tv.get_hist(symbol, 'NASDAQ', interval=interval, n_bars=n_bars)
        if df is None or df.empty:
            return None
        # tvDatafeed returns lowercase columns; standardise to Title case
        df = df.rename(columns={
            'open': 'Open', 'high': 'High',
            'low':  'Low',  'close': 'Close', 'volume': 'Volume',
        })
        if 'symbol' in df.columns:
            df = df.drop(columns=['symbol'])
        # Ensure EST timezone (matches TradingView US session display)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/New_York')
        return df
    except Exception as e:
        print(f"TV fetch error ({symbol}/{exchange}): {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATOR CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════

def resample_ohlcv(df, rule):
    """Resample OHLCV data to a different timeframe."""
    try:
        df = df.copy()
        # Use EST/EDT for bar alignment (matches US market sessions)
        if df.index.tz is not None:
            df.index = df.index.tz_convert('America/New_York')
        resampled = df.resample(rule).agg({
            'Open':   'first',
            'High':   'max',
            'Low':    'min',
            'Close':  'last',
            'Volume': 'sum'
        }).dropna(subset=['Close'])
        return resampled
    except Exception as e:
        print(f"Resample error ({rule}): {e}")
        return df


def calc_wavetrend(df, n1=10, n2=21, ma_len=4):
    """
    Calculate VMC Cipher B / WaveTrend Oscillator.
    Parameters match Pine Script indicator: Channel=10, Average=21, MA=4 (SMA).
    """
    if df is None or len(df) < n1 + n2 + 5:
        return None, None
    try:
        hlc3 = (df['High'] + df['Low'] + df['Close']) / 3
        esa  = hlc3.ewm(span=n1, adjust=False).mean()
        d    = (hlc3 - esa).abs().ewm(span=n1, adjust=False).mean()
        ci   = (hlc3 - esa) / (0.015 * d)
        wt1  = ci.ewm(span=n2, adjust=False).mean()
        wt2  = wt1.rolling(ma_len).mean()
        return wt1, wt2
    except Exception:
        return None, None


def get_cipher_b_signal(df, ob=53, os_level=-53, os2=-60):
    """
    Returns Cipher B signal info for a dataframe.
    Matches Pine Script crossover() + divergence logic exactly.
    Returns dict: {signal, bars_ago, wt1, wt2}
      signal: 2=gold, 1=green, -1=red, 0=none
      bars_ago: how many bars since last signal fired
      wt1/wt2: current WaveTrend values
    """
    wt1, wt2 = calc_wavetrend(df)
    if wt1 is None:
        return {'signal': 0, 'bars_ago': 99, 'wt1': 0.0, 'wt2': 0.0}

    try:
        # Pine Script: crossover(a,b) = a > b AND a[1] <= b[1]
        cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
        cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))

        green = cross_up   & (wt2 <= os_level)   # buy: cross up from oversold
        red   = cross_down & (wt2 >= ob)          # sell: cross down from overbought

        # Gold: cross up from deeply oversold WITH bullish divergence
        # Matches Pine Script: priceLL = ta.lowest(low,5) < ta.lowest(low,5)[5]
        #                      wtHL   = ta.lowest(wt2,5) > ta.lowest(wt2,5)[5]
        lb = 5
        price_ll = df['Low'].rolling(lb).min() < df['Low'].rolling(lb).min().shift(lb)
        wt_hl    = wt2.rolling(lb).min() > wt2.rolling(lb).min().shift(lb)
        gold     = cross_up & (wt2 <= os2) & price_ll & wt_hl

        signals = pd.Series(0, index=wt1.index)
        signals[red]   = -1
        signals[green] =  1
        signals[gold]  =  2   # gold overwrites green if both true

        cur_wt1 = round(float(wt1.iloc[-1]), 2)
        cur_wt2 = round(float(wt2.iloc[-1]), 2)

        # Walk backwards — find most recent non-zero signal
        sig_arr  = signals.values
        bars_ago = 99
        last_sig = 0
        for i in range(len(sig_arr) - 1, -1, -1):
            if sig_arr[i] != 0:
                bars_ago = len(sig_arr) - 1 - i
                last_sig = int(sig_arr[i])
                break

        return {
            'signal':   last_sig,
            'bars_ago': bars_ago,
            'wt1':      cur_wt1,
            'wt2':      cur_wt2,
        }
    except Exception as e:
        print(f"Cipher B signal error: {e}")
        return {'signal': 0, 'bars_ago': 99, 'wt1': 0.0, 'wt2': 0.0}


def calc_macd(df, fast=12, slow=26, signal=9, bars=60):
    """Calculate MACD — returns histogram, MACD line, and Signal line for charting."""
    if df is None or len(df) < slow + signal + 5:
        return None
    try:
        close       = df['Close'].dropna()
        ema_fast    = close.ewm(span=fast,   adjust=False).mean()
        ema_slow    = close.ewm(span=slow,   adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram   = macd_line - signal_line

        hist_t   = histogram.tail(bars)
        macd_t   = macd_line.tail(bars)
        sig_t    = signal_line.tail(bars)
        dates    = [str(d)[:10] for d in hist_t.index]

        return {
            'macd':          round(float(macd_line.iloc[-1]),   4),
            'signal':        round(float(signal_line.iloc[-1]), 4),
            'histogram':     round(float(histogram.iloc[-1]),   4),
            'bullish':       bool(histogram.iloc[-1] > 0),
            'crossing_up':   bool(histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0),
            'crossing_down': bool(histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0),
            'hist_values':   [round(float(v), 4) for v in hist_t],
            'macd_values':   [round(float(v), 4) for v in macd_t],
            'signal_values': [round(float(v), 4) for v in sig_t],
            'hist_dates':    dates,
        }
    except Exception:
        return None


def calc_volume_analysis(df):
    """Analyze current volume vs 20-day average with plain-English interpretation."""
    if df is None or len(df) < 21:
        return None
    try:
        vol     = df['Volume'].dropna()
        current = float(vol.iloc[-1])
        avg20   = float(vol.rolling(20).mean().iloc[-1])
        ratio   = current / avg20 if avg20 > 0 else 1.0

        if ratio >= 2.0:
            color = 'green'
            label = 'Very High Volume'
            interp = f'Volume is {ratio:.1f}x the 20-day average — unusually high activity. Strong conviction behind today\'s move. Pay attention.'
        elif ratio >= 1.5:
            color = 'green'
            label = 'Above Average'
            interp = f'Volume is {ratio:.1f}x the 20-day average — solid participation. The current price move has backing.'
        elif ratio >= 0.8:
            color = 'yellow'
            label = 'Average Volume'
            interp = f'Volume is near average ({ratio:.1f}x). Normal trading conditions — no strong conviction signal either way.'
        else:
            color = 'red'
            label = 'Low Volume'
            interp = f'Volume is only {ratio:.1f}x the average — weak participation. Price moves on low volume are less reliable. Be cautious.'

        return {
            'current':       int(current),
            'average_20':    int(avg20),
            'ratio':         round(ratio, 2),
            'above_average': ratio > 1.0,
            'color':         color,
            'label':         label,
            'interpretation': interp,
        }
    except Exception:
        return None


# ─── TIMEFRAME WEIGHTS ────────────────────────────────────────────────────────
# Tuned for swing trading (hours to weeks).
# Monthly & Weekly anchor the macro direction; Daily is the primary setup frame;
# 4H confirms entries; 1H/30m/15m provide precise timing with minimal influence.
TF_WEIGHTS = {
    '1M':  15.0,   # macro anchor — must align
    '1W':  12.0,   # intermediate trend
    '5D':   9.0,   # multi-day swing context
    '3D':   8.0,   # sub-weekly
    '2D':   7.0,
    '1D':  10.0,   # primary swing frame
    '12H':  5.0,   # intraday bridge
    '8H':   4.0,
    '6H':   3.5,
    '4H':   6.0,   # entry confirmation
    '2H':   3.0,
    '1H':   2.0,
    '30m':  1.5,
    '15m':  1.0,
    '5m':   0.5,
}
_MAX_WEIGHTED_SCORE = sum(TF_WEIGHTS.values())   # ~92.0


def calc_cipher_b_score(cipher_b):
    """
    Returns a normalized score in [-100, +100].
    Gold dots count as 1.5× bullish weight (divergence = higher confidence).
    Neutral (no signal) contributes 0.
    Also returns zone label and breakdowns for display.
    """
    bull_w = bear_w = 0.0
    tf_detail = {}
    for tf, data in cipher_b.items():
        w   = TF_WEIGHTS.get(tf, 1.0)
        sig = data.get('signal', 0)
        if sig == 2:      # gold — strong bull
            contribution = w * 1.5
            bull_w += contribution
        elif sig == 1:    # green
            contribution = w
            bull_w += contribution
        elif sig == -1:   # red
            contribution = -w
            bear_w += w
        else:
            contribution = 0
        tf_detail[tf] = {'weight': w, 'signal': sig, 'contribution': round(contribution, 2)}

    raw   = bull_w - bear_w
    score = round((raw / (_MAX_WEIGHTED_SCORE * 1.5)) * 100, 1)
    score = max(-100, min(100, score))

    if   score >= 65:  zone = 'Strong Bull'
    elif score >= 35:  zone = 'Bullish'
    elif score >= 12:  zone = 'Slight Bull'
    elif score >= -12: zone = 'Neutral'
    elif score >= -35: zone = 'Slight Bear'
    elif score >= -65: zone = 'Bearish'
    else:              zone = 'Strong Bear'

    return {
        'score':     score,
        'zone':      zone,
        'bull_w':    round(bull_w, 1),
        'bear_w':    round(bear_w, 1),
        'detail':    tf_detail,
    }


def get_wavetrend_chart_data(ticker, tf, ob=53, os_level=-53, os2=-60):
    """
    Fetches OHLCV via yfinance (reliable on cloud hosting) and calculates WT1/WT2.
    Returns full series + signal markers for the oscillator chart.
    yfinance intervals: 5m/15m capped at 60d; 1h capped at 730d.
    """
    YF_MAP = {
        '5m':  ('5m',  '60d',  None),
        '15m': ('15m', '60d',  None),
        '1H':  ('1h',  '2y',   None),
        '4H':  ('1h',  '2y',   '4h'),   # resample 1H → 4H
        '1D':  ('1d',  '2y',   None),
        '1W':  ('1wk', '5y',   None),
        '1M':  ('1mo', 'max',  None),
    }
    if tf not in YF_MAP:
        return None
    yf_interval, period, resample_rule = YF_MAP[tf]
    try:
        t   = yf.Ticker(ticker)
        df  = t.history(period=period, interval=yf_interval, auto_adjust=True)
        if df is None or df.empty:
            return None
        # Standardise timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/New_York')
        if resample_rule:
            df = resample_ohlcv(df, resample_rule)

        wt1, wt2 = calc_wavetrend(df)
        if wt1 is None:
            return None

        # Merge wt1/wt2 + price into one frame and drop NaN rows.
        # EWM produces NaN at the start of the series; json.dumps rejects float('nan').
        work = pd.DataFrame({
            'wt1':  wt1,
            'wt2':  wt2,
            'high': df['High'],
            'low':  df['Low'],
        }).dropna()
        if len(work) < 10:
            return None

        # Trim to a readable number of bars per timeframe so the chart isn't cramped.
        # Trim BEFORE computing signals so indices are correct.
        MAX_BARS = {'5m': 200, '15m': 150, '1H': 120, '4H': 90, '1D': 180, '1W': 104, '1M': None}
        max_bars = MAX_BARS.get(tf)
        if max_bars and len(work) > max_bars:
            work = work.iloc[-max_bars:].copy()

        lb         = 5
        cross_up   = (work['wt1'] > work['wt2']) & (work['wt1'].shift(1) <= work['wt2'].shift(1))
        cross_down = (work['wt1'] < work['wt2']) & (work['wt1'].shift(1) >= work['wt2'].shift(1))
        price_ll   = work['low'].rolling(lb).min() < work['low'].rolling(lb).min().shift(lb)
        wt_hl      = work['wt2'].rolling(lb).min() > work['wt2'].rolling(lb).min().shift(lb)

        green_mask = cross_up   & (work['wt2'] <= os_level)
        red_mask   = cross_down & (work['wt2'] >= ob)
        gold_mask  = cross_up   & (work['wt2'] <= os2) & price_ll & wt_hl

        signals = []
        for i in range(len(work)):
            wt2_val = round(float(work['wt2'].iloc[i]), 2)
            if gold_mask.iloc[i]:
                signals.append({'i': i, 'y': wt2_val, 't': 'gold'})
            elif green_mask.iloc[i]:
                signals.append({'i': i, 'y': wt2_val, 't': 'green'})
            elif red_mask.iloc[i]:
                signals.append({'i': i, 'y': wt2_val, 't': 'red'})

        # Use time-aware labels for intraday so the x-axis shows HH:MM not duplicate dates
        if tf in ('5m', '15m', '1H', '4H'):
            dates = [d.strftime('%m/%d %-I:%M %p') for d in work.index]
        else:
            dates = [d.strftime('%Y-%m-%d') for d in work.index]

        return {
            'dates':       dates,
            'wt1':         [round(float(v), 2) for v in work['wt1'].values],
            'wt2':         [round(float(v), 2) for v in work['wt2'].values],
            'signals':     signals,
            'current_wt1': round(float(work['wt1'].iloc[-1]), 2),
            'current_wt2': round(float(work['wt2'].iloc[-1]), 2),
        }
    except Exception as e:
        print(f"Wavetrend chart error ({ticker}/{tf}): {e}")
        return None


def get_all_cipher_b(ticker):
    """
    Fetch OHLCV from TradingView (tvDatafeed) and compute Cipher B for all 15 TFs.
    Uses native TV intervals where available; resamples 6H/8H/12H/2D/3D/5D from source.
    Returns (signal_dict, df_daily, df_weekly) — daily/weekly reused for MACD & Volume.
    """
    results  = {}
    df_daily = None
    df_week  = None

    try:
        symbol, exchange = get_tv_symbol_exchange(ticker)

        # Each thread gets its own TvDatafeed connection (not thread-safe to share one)
        def fetch(interval, n_bars):
            try:
                tv = TvDatafeed()
                return tv_get_hist(tv, symbol, exchange, interval, n_bars)
            except Exception as e:
                print(f"TV thread fetch error: {e}")
                return None

        # Parallel fetches — 3 workers keeps us from hammering TV simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            f5m  = ex.submit(fetch, TvInterval.in_5_minute,  500)
            f15m = ex.submit(fetch, TvInterval.in_15_minute, 500)
            f30m = ex.submit(fetch, TvInterval.in_30_minute, 300)
            f1h  = ex.submit(fetch, TvInterval.in_1_hour,    500)
            f2h  = ex.submit(fetch, TvInterval.in_2_hour,    300)  # native TV 2H
            f4h  = ex.submit(fetch, TvInterval.in_4_hour,    300)  # native TV 4H
            f1d  = ex.submit(fetch, TvInterval.in_daily,     500)
            f1wk = ex.submit(fetch, TvInterval.in_weekly,    200)
            f1mo = ex.submit(fetch, TvInterval.in_monthly,   100)

        df5m  = f5m.result()
        df15m = f15m.result()
        df30m = f30m.result()
        df1h  = f1h.result()
        df2h  = f2h.result()
        df4h  = f4h.result()
        df_daily = f1d.result()
        df_week  = f1wk.result()
        df1mo = f1mo.result()

        # (label, source_df, resample_rule or None)
        # 2H and 4H are native TV intervals — no resampling needed
        # 6H / 8H / 12H resampled from 1H; 2D/3D/5D resampled from daily
        tfs = [
            ('5m',  df5m,    None),
            ('15m', df15m,   None),
            ('30m', df30m,   None),
            ('1H',  df1h,    None),
            ('2H',  df2h,    None),
            ('4H',  df4h,    None),
            ('6H',  df1h,    '6h'),
            ('8H',  df1h,    '8h'),
            ('12H', df1h,    '12h'),
            ('1D',  df_daily, None),
            ('2D',  df_daily, '2D'),
            ('3D',  df_daily, '3D'),
            ('5D',  df_daily, '5D'),
            ('1W',  df_week,  None),
            ('1M',  df1mo,   None),
        ]

        for label, base_df, rule in tfs:
            if base_df is None or base_df.empty:
                results[label] = {'signal': 0, 'bars_ago': 99, 'wt1': 0.0, 'wt2': 0.0}
                continue
            df = resample_ohlcv(base_df, rule) if rule else base_df
            results[label] = get_cipher_b_signal(df)

    except Exception as e:
        print(f"Cipher B error: {e}")

    return results, df_daily, df_week


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIONS HUNTER — GREEKS & SCORING
# ══════════════════════════════════════════════════════════════════════════════

def calculate_greeks(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {'delta': 0.0, 'theta': 0.0, 'gamma': 0.0, 'vega': 0.0}
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
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
        return {'delta': round(delta,3), 'theta': round(theta,4),
                'gamma': round(gamma,4), 'vega': round(vega,4)}
    except Exception:
        return {'delta': 0.0, 'theta': 0.0, 'gamma': 0.0, 'vega': 0.0}


def score_option(row, greeks, dte):
    reasons = []
    delta = abs(greeks['delta'])
    theta = abs(greeks['theta'])
    bid   = float(row.get('bid', 0) or 0)
    ask   = float(row.get('ask', 0) or 0)
    mid   = (bid + ask) / 2 if ask > 0 else float(row.get('lastPrice', 0) or 0)
    vol   = int(row.get('volume', 0) or 0)
    oi    = int(row.get('openInterest', 0) or 0)
    iv    = float(row.get('impliedVolatility', 0) or 0)

    # Delta (25 pts)
    if 0.40 <= delta <= 0.65:
        d_score = 25; reasons.append(('good', f'Delta {delta:.2f} — ideal range (0.40–0.65)'))
    elif 0.30 <= delta < 0.40 or 0.65 < delta <= 0.75:
        d_score = 15; reasons.append(('ok', f'Delta {delta:.2f} — acceptable'))
    elif 0.20 <= delta < 0.30 or 0.75 < delta <= 0.85:
        d_score = 8;  reasons.append(('warn', f'Delta {delta:.2f} — outside ideal range'))
    else:
        d_score = 3;  reasons.append(('bad', f'Delta {delta:.2f} — poor exposure'))

    # Theta (25 pts)
    if mid > 0:
        theta_pct = (theta / mid) * 100
        if   theta_pct < 2:  t_score = 25; reasons.append(('good', f'Theta {theta_pct:.1f}%/day — very low decay'))
        elif theta_pct < 4:  t_score = 20; reasons.append(('good', f'Theta {theta_pct:.1f}%/day — low decay'))
        elif theta_pct < 6:  t_score = 13; reasons.append(('ok',   f'Theta {theta_pct:.1f}%/day — moderate decay'))
        elif theta_pct < 10: t_score = 6;  reasons.append(('warn', f'Theta {theta_pct:.1f}%/day — high decay'))
        else:                t_score = 1;  reasons.append(('bad',  f'Theta {theta_pct:.1f}%/day — severe decay'))
    else:
        t_score = 0; reasons.append(('bad', 'No price data'))

    # Liquidity (25 pts)
    liq = 0
    if vol > 1000: liq += 10
    elif vol > 500: liq += 8
    elif vol > 100: liq += 6
    elif vol > 50:  liq += 4
    elif vol > 10:  liq += 2
    if oi > 5000: liq += 10
    elif oi > 1000: liq += 8
    elif oi > 500:  liq += 6
    elif oi > 100:  liq += 4
    elif oi > 50:   liq += 2
    if ask > 0:
        sp = ((ask - bid) / ask) * 100
        if sp < 5: liq += 5
        elif sp < 10: liq += 3
        elif sp < 20: liq += 1
    liq = min(liq, 25)
    if liq >= 20: reasons.append(('good', f'Vol {vol:,} / OI {oi:,} — great liquidity'))
    elif liq >= 12: reasons.append(('ok',  f'Vol {vol:,} / OI {oi:,} — decent liquidity'))
    else:           reasons.append(('bad', f'Vol {vol:,} / OI {oi:,} — low liquidity'))

    # DTE (15 pts)
    if 14 <= dte <= 30:   dte_s = 15; reasons.append(('good', f'{dte} DTE — ideal for 1–5 day swing'))
    elif 7 <= dte < 14 or 30 < dte <= 45: dte_s = 10; reasons.append(('ok', f'{dte} DTE — acceptable'))
    elif 5 <= dte < 7 or 45 < dte <= 60:  dte_s = 5;  reasons.append(('warn', f'{dte} DTE — borderline'))
    else:                                  dte_s = 2;  reasons.append(('bad', f'{dte} DTE — not ideal'))

    # IV (10 pts)
    iv_pct = iv * 100
    if   iv_pct < 30:  iv_s = 10; reasons.append(('good', f'IV {iv_pct:.0f}% — cheap contract'))
    elif iv_pct < 50:  iv_s = 8;  reasons.append(('ok',   f'IV {iv_pct:.0f}% — moderate'))
    elif iv_pct < 80:  iv_s = 5;  reasons.append(('ok',   f'IV {iv_pct:.0f}% — elevated'))
    elif iv_pct < 120: iv_s = 2;  reasons.append(('warn', f'IV {iv_pct:.0f}% — expensive'))
    else:              iv_s = 0;  reasons.append(('bad',  f'IV {iv_pct:.0f}% — very expensive'))

    total = d_score + t_score + liq + dte_s + iv_s
    if   total >= 85: grade = 'A+'
    elif total >= 75: grade = 'A'
    elif total >= 65: grade = 'B+'
    elif total >= 55: grade = 'B'
    elif total >= 45: grade = 'C'
    else:             grade = 'D'

    return total, grade, reasons


# ══════════════════════════════════════════════════════════════════════════════
#  MORNING BRIEFING DATA
# ══════════════════════════════════════════════════════════════════════════════

def get_market_indices():
    """Fetch major market indices."""
    symbols = {
        '^GSPC': 'S&P 500', '^IXIC': 'NASDAQ', '^DJI': 'Dow Jones',
        '^RUT':  'Russell 2000', '^VIX': 'VIX'
    }
    results = []
    try:
        for sym, name in symbols.items():
            t    = yf.Ticker(sym)
            info = t.fast_info
            try:
                price = float(info.last_price)
                prev  = float(info.previous_close)
                chg   = price - prev
                chg_p = (chg / prev) * 100 if prev else 0
                results.append({
                    'symbol': sym, 'name': name,
                    'price':  round(price, 2),
                    'change': round(chg, 2),
                    'change_pct': round(chg_p, 2),
                    'positive': chg >= 0
                })
            except Exception:
                continue
    except Exception as e:
        print(f"Indices error: {e}")
    return results


def get_crypto_prices(symbols):
    """Fetch current crypto prices."""
    results = []
    try:
        for sym in symbols:
            try:
                t    = yf.Ticker(sym)
                info = t.fast_info
                price = float(info.last_price)
                prev  = float(info.previous_close)
                chg_p = ((price - prev) / prev * 100) if prev else 0
                results.append({
                    'symbol':     sym,
                    'name':       CRYPTO_NAMES.get(sym, sym.replace('-USD', '')),
                    'price':      round(price, 4) if price < 1 else round(price, 2),
                    'change_pct': round(chg_p, 2),
                    'positive':   chg_p >= 0
                })
            except Exception:
                continue
    except Exception as e:
        print(f"Crypto error: {e}")
    return results


def get_crypto_indices():
    """Fetch TOTAL market cap, TOTAL3 via CoinGecko; BTC dominance from TradingView (CRYPTOCAP:BTC.D)."""
    cached = cache_get('crypto_indices')
    if cached:
        return cached

    # Pull BTC.D directly from TradingView so it matches what you see in TV charts
    btc_dom_tv = None
    try:
        tv_btcd = TvDatafeed()
        df_btcd = tv_btcd.get_hist('BTC.D', 'CRYPTOCAP', interval=TvInterval.in_daily, n_bars=3)
        if df_btcd is not None and not df_btcd.empty:
            btc_dom_tv = round(float(df_btcd['close'].iloc[-1]), 2)
    except Exception as e:
        print(f"tvDatafeed BTC.D error: {e}")

    try:
        r    = requests.get('https://api.coingecko.com/api/v3/global', timeout=8)
        data = r.json().get('data', {})

        total      = data.get('total_market_cap', {}).get('usd', 0)
        chg24      = data.get('market_cap_change_percentage_24h_usd', 0)
        # Prefer TradingView value; fall back to CoinGecko if TV fetch failed
        btc_dom    = btc_dom_tv if btc_dom_tv is not None else data.get('market_cap_percentage', {}).get('btc', 0)
        eth_dom    = data.get('market_cap_percentage', {}).get('eth', 0)
        btc_mcap   = total * btc_dom / 100
        eth_mcap   = total * eth_dom / 100
        total3     = total - btc_mcap - eth_mcap

        def fmt_mcap(v):
            if v >= 1e12: return f'${v/1e12:.2f}T'
            if v >= 1e9:  return f'${v/1e9:.2f}B'
            return f'${v:.0f}'

        result = {
            'total':     {'label': 'TOTAL',  'value': fmt_mcap(total),  'raw': total,  'change_24h': round(chg24, 2)},
            'total3':    {'label': 'TOTAL3', 'value': fmt_mcap(total3), 'raw': total3, 'change_24h': None},
            'btc_dom':   {'label': 'BTC.D',  'value': f'{btc_dom:.1f}%','raw': btc_dom,'change_24h': None},
        }
        cache_set('crypto_indices', result)
        return result
    except Exception as e:
        print(f"CoinGecko error: {e}")
        return {}


def get_top_movers():
    """Get top 5 gainers and losers using fast_info per ticker (memory efficient)."""
    cached = cache_get('top_movers')
    if cached:
        return cached
    movers = []
    for sym in POPULAR_STOCKS:
        try:
            info  = yf.Ticker(sym).fast_info
            price = float(info.last_price)
            prev  = float(info.previous_close)
            if prev > 0:
                chg = round((price - prev) / prev * 100, 2)
                movers.append({'ticker': sym, 'change': chg})
        except Exception:
            continue
    if movers:
        movers.sort(key=lambda x: x['change'])
        result = {
            'losers':  movers[:5],
            'gainers': list(reversed(movers[-5:]))
        }
        cache_set('top_movers', result)
        return result
    return {'gainers': [], 'losers': []}


def get_market_news():
    """Pull latest market headlines via yfinance."""
    cached = cache_get('market_news')
    if cached:
        return cached
    news = []
    try:
        for sym in ['SPY', 'QQQ']:
            t = yf.Ticker(sym)
            raw = t.news or []
            for item in raw[:8]:
                try:
                    # Handle both old and new yfinance news formats
                    if 'content' in item:
                        content = item['content']
                        title   = content.get('title', '')
                        link    = content.get('canonicalUrl', {}).get('url', '') or content.get('clickThroughUrl', {}).get('url', '')
                        pub     = content.get('provider', {}).get('displayName', '')
                        ts      = 0
                    else:
                        title = item.get('title', '')
                        link  = item.get('link', '')
                        pub   = item.get('publisher', '')
                        ts    = item.get('providerPublishTime', 0)

                    if title and not any(n['title'] == title for n in news):
                        news.append({'title': title, 'link': link, 'publisher': pub, 'time': ts})
                except Exception:
                    continue
            if len(news) >= 10:
                break
    except Exception as e:
        print(f"News error: {e}")

    news = news[:12]
    cache_set('market_news', news)
    return news


# ══════════════════════════════════════════════════════════════════════════════
#  SENTIMENT (reused from v1)
# ══════════════════════════════════════════════════════════════════════════════

def get_sentiment(ticker_symbol):
    try:
        vix_data = yf.Ticker("^VIX").history(period="5d")
        spy_data = yf.Ticker("SPY").history(period="40d")
        qqq_data = yf.Ticker("QQQ").history(period="40d")

        vix_now   = float(vix_data['Close'].iloc[-1])
        spy_now   = float(spy_data['Close'].iloc[-1])
        spy_ema20 = float(spy_data['Close'].ewm(span=20).mean().iloc[-1])
        qqq_now   = float(qqq_data['Close'].iloc[-1])
        qqq_ema20 = float(qqq_data['Close'].ewm(span=20).mean().iloc[-1])

        bulls = sum([spy_now > spy_ema20, qqq_now > qqq_ema20])

        if vix_now < 18 and bulls == 2:
            mkt_s = 'Bullish'; mkt_color = 'green'
            mkt_desc = f'VIX {vix_now:.1f} (calm) — SPY & QQQ above 20-day average'
        elif vix_now > 28 or bulls == 0:
            mkt_s = 'Bearish'; mkt_color = 'red'
            mkt_desc = f'VIX {vix_now:.1f} (elevated) — market under pressure'
        else:
            mkt_s = 'Neutral'; mkt_color = 'yellow'
            mkt_desc = f'VIX {vix_now:.1f} — mixed signals'

        stock  = yf.Ticker(ticker_symbol)
        info   = stock.info
        name   = info.get('longName', ticker_symbol)
        price  = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        sector = info.get('sector', 'Unknown')

        if not price:
            hist  = stock.history(period="1d")
            price = float(hist['Close'].iloc[-1]) if not hist.empty else 0

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

        return {
            'market': {'sentiment': mkt_s, 'color': mkt_color, 'desc': mkt_desc, 'vix': round(vix_now, 2)},
            'sector': {'sentiment': sec_s, 'color': sec_color, 'desc': f'{sector} ({etf_sym}) {etf_5d:+.1f}% 5d', 'name': sector, 'etf': etf_sym},
            'stock':  {'name': name, 'price': round(float(price), 2), 'ticker': ticker_symbol},
        }
    except Exception as e:
        return {'error': str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI ROUTES
# ══════════════════════════════════════════════════════════════════════════════

class AnalyzeBody(BaseModel):
    ticker:     str
    expiration: str


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


# ── Morning Briefing ──────────────────────────────────────────────────────────
@app.get('/api/market')
@limiter.limit("15/minute")
def market_briefing(request: Request, crypto: str = ','.join(DEFAULT_CRYPTO)):
    crypto_symbols = [s.strip() for s in crypto.split(',') if s.strip()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        fi  = ex.submit(get_market_indices)
        fc  = ex.submit(get_crypto_prices, crypto_symbols)
        fci = ex.submit(get_crypto_indices)
        fm  = ex.submit(get_top_movers)
        fn  = ex.submit(get_market_news)

    return {
        'indices':        fi.result(),
        'crypto':         fc.result(),
        'crypto_indices': fci.result(),
        'movers':         fm.result(),
        'news':           fn.result(),
        'updated':        datetime.now(ZoneInfo('America/New_York')).strftime('%-I:%M %p ET'),
    }


# ── Technical Analysis ────────────────────────────────────────────────────────
# Tight limit — each cache miss hits TradingView 9 times concurrently
@app.get('/api/technical/{ticker}')
@limiter.limit("8/minute; 40/hour")
def technical_analysis(request: Request, ticker: str):
    ticker = ticker.upper().strip()

    # 5-minute cache per ticker — signals don't change faster than that
    cache_key = f'technical_{ticker}'
    cached = cache_get(cache_key, ttl=CACHE_TTL_TECHNICAL)
    if cached:
        return cached

    try:
        cipher_b, df_1d, df_1wk = get_all_cipher_b(ticker)

        price = 0
        try:
            price = float(yf.Ticker(ticker).fast_info.last_price)
        except Exception:
            if df_1d is not None and not df_1d.empty:
                price = float(df_1d['Close'].iloc[-1])

        macd        = calc_macd(df_1d,  bars=60)
        macd_weekly = calc_macd(df_1wk, bars=52)
        volume      = calc_volume_analysis(df_1d)
        cb_score    = calc_cipher_b_score(cipher_b)

        sig_vals   = list(cipher_b.values())
        bull_count = sum(1 for v in sig_vals if v['signal'] > 0)
        bear_count = sum(1 for v in sig_vals if v['signal'] < 0)
        neut_count = sum(1 for v in sig_vals if v['signal'] == 0)

        result = {
            'ticker':      ticker,
            'price':       round(float(price), 2),
            'cipher_b':    cipher_b,
            'cb_score':    cb_score,
            'macd':        macd,
            'macd_weekly': macd_weekly,
            'volume':      volume,
            'summary': {
                'bullish_count': bull_count,
                'bearish_count': bear_count,
                'neutral_count': neut_count,
                'macd_status':   'Bullish' if macd and macd['bullish'] else 'Bearish' if macd else 'Unknown',
                'volume_status': 'Above Average' if volume and volume['above_average'] else 'Below Average',
                'overall':       cb_score['zone'],
            }
        }
        cache_set(cache_key, result)
        return result

    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)


# ── Wavetrend Oscillator Chart ────────────────────────────────────────────────
@app.get('/api/wavetrend/{ticker}/{timeframe}')
@limiter.limit("12/minute")
def wavetrend_chart(request: Request, ticker: str, timeframe: str):
    ticker = ticker.upper().strip()
    cache_key = f'wt_{ticker}_{timeframe}'
    cached = cache_get(cache_key, ttl=CACHE_TTL_TECHNICAL)
    if cached:
        return cached
    try:
        data = get_wavetrend_chart_data(ticker, timeframe)
        if not data:
            return JSONResponse({'error': 'Could not fetch wavetrend data'}, status_code=400)
        cache_set(cache_key, data)
        return JSONResponse(data)
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({'error': f'Server error: {str(e)}'}, status_code=500)


# ── MACD by timeframe ─────────────────────────────────────────────────────────
MACD_TF_MAP = {
    '5m':  (TvInterval.in_5_minute,  200),
    '15m': (TvInterval.in_15_minute, 200),
    '30m': (TvInterval.in_30_minute, 150),
    '1H':  (TvInterval.in_1_hour,    150),
    '2H':  (TvInterval.in_2_hour,    120),
    '4H':  (TvInterval.in_4_hour,    120),
    '1D':  (TvInterval.in_daily,     120),
    '1W':  (TvInterval.in_weekly,     80),
    '1M':  (TvInterval.in_monthly,    60),
}

@app.get('/api/macd/{ticker}/{timeframe}')
@limiter.limit("12/minute")
def macd_endpoint(request: Request, ticker: str, timeframe: str):
    ticker = ticker.upper().strip()
    if timeframe not in MACD_TF_MAP:
        return JSONResponse({'error': f'Unsupported timeframe: {timeframe}'}, status_code=400)
    cache_key = f'macd_{ticker}_{timeframe}'
    cached = cache_get(cache_key, ttl=CACHE_TTL_TECHNICAL)
    if cached:
        return cached
    interval, n_bars = MACD_TF_MAP[timeframe]
    try:
        symbol, exchange = get_tv_symbol_exchange(ticker)
        tv   = TvDatafeed()
        df   = tv_get_hist(tv, symbol, exchange, interval, n_bars)
        if df is None or df.empty:
            return JSONResponse({'error': 'No data available'}, status_code=400)
        bars = 40 if timeframe == '1M' else min(60, n_bars // 2)
        result = calc_macd(df, bars=bars)
        if not result:
            return JSONResponse({'error': 'Could not calculate MACD'}, status_code=400)
        result['timeframe'] = timeframe
        cache_set(cache_key, result)
        return result
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)


# ── Volume by timeframe ────────────────────────────────────────────────────────
@app.get('/api/volume/{ticker}/{timeframe}')
@limiter.limit("12/minute")
def volume_endpoint(request: Request, ticker: str, timeframe: str):
    ticker = ticker.upper().strip()
    if timeframe not in MACD_TF_MAP:
        return JSONResponse({'error': f'Unsupported timeframe: {timeframe}'}, status_code=400)
    cache_key = f'vol_{ticker}_{timeframe}'
    cached = cache_get(cache_key, ttl=CACHE_TTL_TECHNICAL)
    if cached:
        return cached
    interval, n_bars = MACD_TF_MAP[timeframe]
    try:
        symbol, exchange = get_tv_symbol_exchange(ticker)
        tv = TvDatafeed()
        df = tv_get_hist(tv, symbol, exchange, interval, n_bars)
        if df is None or df.empty:
            return JSONResponse({'error': 'No data available'}, status_code=400)
        # Drop the last (potentially incomplete) bar for accuracy
        df_complete = df.iloc[:-1] if len(df) > 21 else df
        result = calc_volume_analysis(df_complete)
        if not result:
            return JSONResponse({'error': 'Could not calculate volume'}, status_code=400)
        result['timeframe'] = timeframe
        cache_set(cache_key, result)
        return result
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)


# ── Options Hunter ────────────────────────────────────────────────────────────
@app.get('/api/stock/{ticker}')
@limiter.limit("10/minute")
def stock_info(request: Request, ticker: str):
    ticker = ticker.upper().strip()
    cache_key = f'stock_{ticker}'
    cached = cache_get(cache_key, ttl=CACHE_TTL_STOCK)
    if cached:
        return cached
    try:
        data = get_sentiment(ticker)
        data['expirations'] = list(yf.Ticker(ticker).options)
        cache_set(cache_key, data)
        return data
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)


@app.post('/api/analyze')
@limiter.limit("6/minute; 30/hour")
def analyze(request: Request, body: AnalyzeBody):
    ticker     = body.ticker.upper().strip()
    expiration = body.expiration
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
        r_f   = 0.05
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
                    greeks = calculate_greeks(price, strike, T, r_f, iv, opt_type)
                    score, grade, rsns = score_option(row.to_dict(), greeks, dte)
                    results.append({
                        'type':         opt_type.upper(),
                        'strike':       strike,
                        'bid':          round(bid, 2),
                        'ask':          round(ask, 2),
                        'mid':          round(mid, 2),
                        'cost':         round(mid * 100, 2),
                        'volume':       int(row['volume'])       if not pd.isna(row.get('volume'))       else 0,
                        'openInterest': int(row['openInterest']) if not pd.isna(row.get('openInterest')) else 0,
                        'iv':           round(iv * 100, 1),
                        'delta':        greeks['delta'],
                        'theta':        greeks['theta'],
                        'gamma':        greeks['gamma'],
                        'vega':         greeks['vega'],
                        'breakeven':    round(strike + mid, 2) if opt_type == 'call' else round(strike - mid, 2),
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

        return {'results': results, 'ticker': ticker,
                'currentPrice': round(price, 2), 'expiration': expiration, 'dte': dte}
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS & STARTUP
# ══════════════════════════════════════════════════════════════════════════════

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return "localhost"

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    port      = int(os.environ.get('PORT', 5000))
    is_render = os.environ.get('RENDER') is not None

    if not is_render:
        ip = get_local_ip()
        print("\n" + "="*55)
        print("  OPTIONS HUNTER v2.0 is starting...")
        print("  Powered by FastAPI + Uvicorn")
        print("="*55)
        print(f"  Local:     http://localhost:{port}")
        print(f"  WiFi:      http://{ip}:{port}")
        print(f"  API docs:  http://localhost:{port}/docs")
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(port, bind_tls=True).public_url
            print(f"  PUBLIC:    {public_url}")
        except Exception:
            pass
        print("="*55 + "\n")
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host='0.0.0.0', port=port, log_level='warning')
