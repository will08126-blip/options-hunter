#!/usr/bin/env python3
"""Discord Trading Signal Bot — crypto scanner + slash commands."""

import os
import asyncio
import logging
import sqlite3
import threading
from datetime import datetime, timezone, timedelta

import discord
from discord import app_commands
from discord.ext import tasks

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
DISCORD_TOKEN    = os.environ.get('DISCORD_TOKEN', '')
DISCORD_GUILD_ID = os.environ.get('DISCORD_GUILD_ID', '')
SIGNAL_CHANNEL   = int(os.environ.get('SIGNAL_CHANNEL_ID', '0') or '0')
SCAN_INTERVAL    = int(os.environ.get('SCAN_INTERVAL_MINUTES', '1'))
DATA_DIR         = os.environ.get('DATA_DIR', '/data')
DB_PATH          = os.path.join(DATA_DIR, 'trades.db')

CRYPTO_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'PEPE/USDT',
]

EXCHANGE_PRIORITY = ['binance', 'gate', 'mexc']

# ── Exchange helper ───────────────────────────────────────────────────────────
def _get_exchange():
    """Return the first available ccxt exchange, falling back down the list."""
    try:
        import ccxt
    except ImportError:
        return None, None

    for name in EXCHANGE_PRIORITY:
        try:
            ex = getattr(ccxt, name)({'enableRateLimit': True})
            ex.load_markets()
            log.info('[marketData] Using exchange: %s', name)
            return ex, name
        except Exception as e:
            short = str(e)[:80]
            log.warning('[marketData] Exchange "%s" unavailable (%s). Falling back.', name, short)

    return None, None


def _fetch_ohlcv(symbol: str, timeframe='1h', limit=100):
    """Fetch OHLCV data for a symbol, trying exchanges in priority order."""
    try:
        import ccxt
    except ImportError:
        return None

    for name in EXCHANGE_PRIORITY:
        try:
            ex = getattr(ccxt, name)({'enableRateLimit': True})
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if data:
                return data
        except Exception:
            continue
    return None


# ── Technical analysis ────────────────────────────────────────────────────────
def _analyze(ohlcv):
    """
    Return (regime, setups) given raw OHLCV data.
    regime: one of TREND_UP / TREND_DOWN / RANGE
    setups: list of setup names detected
    """
    if not ohlcv or len(ohlcv) < 20:
        return 'UNKNOWN', []

    closes  = [c[4] for c in ohlcv]
    highs   = [c[2] for c in ohlcv]
    lows    = [c[3] for c in ohlcv]
    volumes = [c[5] for c in ohlcv]

    # ADX approximation (using TR and directional movement)
    def _adx(n=14):
        trs, pdms, ndms = [], [], []
        for i in range(1, len(closes)):
            tr  = max(highs[i] - lows[i],
                      abs(highs[i] - closes[i-1]),
                      abs(lows[i]  - closes[i-1]))
            pdm = max(highs[i] - highs[i-1], 0)
            ndm = max(lows[i-1] - lows[i], 0)
            if pdm > ndm:
                ndm = 0
            else:
                pdm = 0
            trs.append(tr); pdms.append(pdm); ndms.append(ndm)

        def _smooth(vals, n):
            s = sum(vals[:n])
            out = [s]
            for v in vals[n:]:
                s = s - s / n + v
                out.append(s)
            return out

        atr  = _smooth(trs,  n)
        pdi  = _smooth(pdms, n)
        ndi  = _smooth(ndms, n)
        dxs  = []
        for a, p, nd in zip(atr, pdi, ndi):
            di_diff = abs((p / a * 100) - (nd / a * 100)) if a else 0
            di_sum  = (p / a * 100) + (nd / a * 100) if a else 0
            dxs.append((di_diff / di_sum * 100) if di_sum else 0)

        adx_val = sum(dxs[-n:]) / n if len(dxs) >= n else sum(dxs) / len(dxs)
        return round(adx_val, 1)

    # ATR multiplier (current ATR vs 20-period average)
    def _atrx(n=14):
        trs = [max(highs[i] - lows[i],
                   abs(highs[i] - closes[i-1]),
                   abs(lows[i]  - closes[i-1]))
               for i in range(1, len(closes))]
        if not trs:
            return 1.0
        cur_atr = sum(trs[-n:]) / min(n, len(trs))
        avg_atr = sum(trs) / len(trs)
        return round(cur_atr / avg_atr, 2) if avg_atr else 1.0

    adx  = _adx()
    atrx = _atrx()

    # Regime
    sma20 = sum(closes[-20:]) / 20
    sma50 = sum(closes[-50:]) / min(50, len(closes))
    if adx > 25 and closes[-1] > sma20 > sma50:
        regime = 'TREND_UP'
    elif adx > 25 and closes[-1] < sma20 < sma50:
        regime = 'TREND_DOWN'
    else:
        regime = 'RANGE'

    # Setups
    setups = []

    # Trend Pullback: trending + price pulled back to 20 SMA within 2%
    if regime in ('TREND_UP', 'TREND_DOWN'):
        dist = abs(closes[-1] - sma20) / sma20
        if dist < 0.02:
            setups.append('Trend Pullback')

    # Breakout Retest: recent high/low break with volume spike
    recent_high = max(highs[-20:-1])
    recent_low  = min(lows[-20:-1])
    avg_vol     = sum(volumes[-20:-1]) / 19
    if closes[-1] > recent_high * 1.001 and volumes[-1] > avg_vol * 1.5:
        setups.append('Breakout Retest')
    elif closes[-1] < recent_low * 0.999 and volumes[-1] > avg_vol * 1.5:
        setups.append('Breakout Retest')

    # Liquidity Sweep: sharp wick beyond recent range then reversal
    candle_range = highs[-1] - lows[-1]
    upper_wick   = highs[-1] - max(closes[-1], closes[-2] if len(closes) > 1 else closes[-1])
    lower_wick   = min(closes[-1], closes[-2] if len(closes) > 1 else closes[-1]) - lows[-1]
    if candle_range > 0:
        if upper_wick / candle_range > 0.6 and lows[-1] < recent_low:
            setups.append('Liquidity Sweep')
        elif lower_wick / candle_range > 0.6 and highs[-1] > recent_high:
            setups.append('Liquidity Sweep')

    # Volatility Expansion: ATR spike
    if atrx > 1.5:
        setups.append('Volatility Expansion')

    return regime, setups, adx, atrx


# ── Database helper ───────────────────────────────────────────────────────────
_db_lock = threading.Lock()


def _get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_positions(status_filter: str = 'active'):
    """Return list of trade dicts filtered by status."""
    try:
        with _db_lock:
            conn = _get_db()
            rows = conn.execute(
                "SELECT * FROM simulated_trades WHERE status=? ORDER BY created_at DESC LIMIT 25",
                (status_filter,)
            ).fetchall()
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ── Bot setup ─────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
bot     = discord.Client(intents=intents)
tree    = app_commands.CommandTree(bot)

# Track scan state
_scan_state = {
    'last_scan':   None,
    'signals_found': 0,
    'signals_posted': 0,
    'scan_count':  0,
    'started_at':  datetime.now(timezone.utc),
}


# ── Slash commands ────────────────────────────────────────────────────────────
# IMPORTANT: required options MUST appear before optional options in every command.

@tree.command(name='status', description='Show bot and scan status')
async def cmd_status(interaction: discord.Interaction):
    """Show the bot's current operational status."""
    await interaction.response.defer()

    up_since  = _scan_state['started_at']
    uptime    = datetime.now(timezone.utc) - up_since
    hours, rem = divmod(int(uptime.total_seconds()), 3600)
    mins       = rem // 60

    last = _scan_state['last_scan']
    last_str = (
        f"<t:{int(last.timestamp())}:R>" if last else '*(no scan yet)*'
    )

    embed = discord.Embed(
        title='📡 Bot Status',
        color=discord.Color.green(),
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name='Uptime',      value=f'{hours}h {mins}m',           inline=True)
    embed.add_field(name='Last Scan',   value=last_str,                       inline=True)
    embed.add_field(name='Scan #',      value=str(_scan_state['scan_count']), inline=True)
    embed.add_field(
        name='Signals (session)',
        value=(
            f"Found: {_scan_state['signals_found']} | "
            f"Posted: {_scan_state['signals_posted']}"
        ),
        inline=False,
    )
    embed.add_field(
        name='Scan Interval',
        value=f'Every {SCAN_INTERVAL} min',
        inline=True,
    )
    embed.add_field(
        name='Pairs Watched',
        value=', '.join(CRYPTO_PAIRS),
        inline=False,
    )
    embed.set_footer(text='Trading Signal Bot')
    await interaction.followup.send(embed=embed)


@tree.command(name='positions', description='Show paper trade positions')
@app_commands.describe(
    # required options must come first — there are none here, all are optional
    status='Filter by status: active (default) or closed',
)
async def cmd_positions(
    interaction: discord.Interaction,
    status: str = 'active',  # optional — no required options precede it
):
    """List paper trading positions, optionally filtered by status."""
    await interaction.response.defer()

    valid = {'active', 'closed', 'expired'}
    if status not in valid:
        await interaction.followup.send(
            f'❌ Invalid status `{status}`. Choose one of: {", ".join(sorted(valid))}',
            ephemeral=True,
        )
        return

    trades = _fetch_positions(status)

    if not trades:
        await interaction.followup.send(
            f'No **{status}** positions found.',
            ephemeral=True,
        )
        return

    embed = discord.Embed(
        title=f'📊 {status.capitalize()} Positions ({len(trades)})',
        color=discord.Color.blurple(),
        timestamp=datetime.now(timezone.utc),
    )

    for t in trades[:10]:  # cap at 10 to avoid hitting embed limits
        pnl_str = ''
        if t.get('pnl_dollars') is not None:
            sign   = '+' if t['pnl_dollars'] >= 0 else ''
            pnl_str = f"  P&L: {sign}${t['pnl_dollars']:.2f} ({sign}{t.get('pnl_pct', 0):.1f}%)"

        label  = f"{t['ticker']} {t.get('direction', '').upper()} {t.get('trade_type', '')}"
        detail = (
            f"Entry: ${t.get('entry_price', 0):.2f}"
            f"{pnl_str}"
        )
        if t.get('expiration'):
            detail += f"  Exp: {t['expiration']}"

        embed.add_field(name=label, value=detail, inline=False)

    if len(trades) > 10:
        embed.set_footer(text=f'Showing 10 of {len(trades)} positions.')
    else:
        embed.set_footer(text='Trading Signal Bot')

    await interaction.followup.send(embed=embed)


@tree.command(name='scan', description='Show the latest scan results for a crypto pair')
@app_commands.describe(
    # required options first, optional last
    symbol='Crypto pair to scan, e.g. BTC/USDT (optional — scans all if omitted)',
)
async def cmd_scan(
    interaction: discord.Interaction,
    symbol: str = '',  # optional
):
    """Trigger an on-demand analysis for one pair or all watched pairs."""
    await interaction.response.defer()

    pairs = [symbol.upper()] if symbol else CRYPTO_PAIRS
    lines = []

    for pair in pairs:
        ohlcv = await asyncio.get_event_loop().run_in_executor(
            None, _fetch_ohlcv, pair, '1h', 100
        )
        if not ohlcv:
            lines.append(f'**{pair}**: ⚠️ data unavailable')
            continue

        regime, setups, adx, atrx = _analyze(ohlcv)
        setup_str = ', '.join(setups) if setups else '*(no setup detected)*'
        lines.append(
            f'**{pair}**: `{regime}` (ADX={adx}, ATRx={atrx})\n  → {setup_str}'
        )

    embed = discord.Embed(
        title='🔍 Scan Results',
        description='\n'.join(lines) or '*(no data)*',
        color=discord.Color.gold(),
        timestamp=datetime.now(timezone.utc),
    )
    embed.set_footer(text='Trading Signal Bot')
    await interaction.followup.send(embed=embed)


# ── Scan loop ─────────────────────────────────────────────────────────────────
@tasks.loop(minutes=SCAN_INTERVAL)
async def scan_loop():
    _scan_state['scan_count'] += 1
    _scan_state['last_scan']   = datetime.now(timezone.utc)
    log.info('Starting scan cycle...')

    channel = bot.get_channel(SIGNAL_CHANNEL) if SIGNAL_CHANNEL else None
    raw, posted = 0, 0

    for pair in CRYPTO_PAIRS:
        try:
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None, _fetch_ohlcv, pair, '1h', 100
            )
            if not ohlcv:
                continue

            regime, setups, adx, atrx = _analyze(ohlcv)
            log.info('%s: %s (ADX=%s, ATRx=%s)', pair, regime, adx, atrx)

            if not setups:
                for s in ['Trend Pullback', 'Breakout Retest', 'Liquidity Sweep', 'Volatility Expansion']:
                    log.info('  %s: no setup detected', s)
                continue

            raw += len(setups)
            for setup in setups:
                log.info('  %s: DETECTED', setup)

            if channel:
                closes = [c[4] for c in ohlcv]
                price  = closes[-1]
                embed  = discord.Embed(
                    title=f'🚨 Signal: {pair}',
                    color=discord.Color.red() if regime == 'TREND_DOWN' else discord.Color.green(),
                    timestamp=datetime.now(timezone.utc),
                )
                embed.add_field(name='Setup',   value=', '.join(setups), inline=True)
                embed.add_field(name='Regime',  value=regime,             inline=True)
                embed.add_field(name='Price',   value=f'${price:,.4f}',   inline=True)
                embed.add_field(name='ADX',     value=str(adx),           inline=True)
                embed.add_field(name='ATRx',    value=str(atrx),          inline=True)
                await channel.send(embed=embed)
                posted += 1

        except Exception as e:
            log.error('Error scanning %s: %s', pair, e)

    _scan_state['signals_found']  += raw
    _scan_state['signals_posted'] += posted
    log.info('Scan complete: %d raw → %d ranked → %d posted', raw, raw, posted)


# ── Bot events ────────────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    log.info('Discord bot ready — logged in as %s', bot.user)

    guild = discord.Object(id=int(DISCORD_GUILD_ID)) if DISCORD_GUILD_ID else None

    if guild:
        # Guild-scoped deploy: instant propagation
        tree.copy_global_to(guild=guild)
        await tree.sync(guild=guild)
        log.info('Slash commands synced to guild %s', DISCORD_GUILD_ID)
    else:
        # Global deploy: up to 1 h propagation
        log.info('No DISCORD_GUILD_ID — deploying global commands (up to 1h propagation)')
        await tree.sync()
        log.info('Global slash commands synced')

    log.info('Starting scan scheduler: every %d min', SCAN_INTERVAL)
    scan_loop.start()
    log.info('Running initial scan...')


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if not DISCORD_TOKEN:
        raise RuntimeError('DISCORD_TOKEN env var is not set')
    log.info('Starting Discord Trading Bot...')
    bot.run(DISCORD_TOKEN)
