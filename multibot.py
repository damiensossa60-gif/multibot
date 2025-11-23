"""
trading_multi.py
Version multi-instruments du bot PRO (Option A adaptée)
Analyse multi-timeframe + envoi Telegram + TP/SL/Trailing + logs par symbole
"""

import time, os, traceback, requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from dateutil import tz
from config import TOKEN, CHAT_ID

# ---------------------------
# CONFIG
# ---------------------------
# Liste des symboles (Yahoo Finance tickers). Commence par quelques-uns seulement.
SYMBOLS = [
    "EURUSD=X",   # Forex EUR/USD
    "GBPUSD=X",   # Forex GBP/USD
    "USDJPY=X",   # Forex USD/JPY
    "XAUUSD=X",   # Gold (XAU/USD)
    "BTC-USD",    # Bitcoin (USD)
    "^GSPC"       # S&P 500 index
]

# Intervalles et périodes par timeframe (on garde 1H/4H/1D)
TF_INTERVALS = {"1H": "1h", "4H": "4h", "1D": "1d"}
PERIODS = {"1H": "90d", "4H": "180d", "1D": "365d"}

CHECK_INTERVAL = 90  # seconds entre checks (augmente si tu as beaucoup de symboles)
SIGNAL_COOLDOWN_SECONDS = 4 * 3600  # cooldown par symbole

EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

ACCOUNT_EQUITY = 20.0
RISK_PER_TRADE = 0.01

SL_PIPS = 30
TP1_PIPS = 30
TP2_PIPS = 60
TRAILING_AFTER_TP1_PIPS = 15

LOG_CSV = "trading_multi_log.csv"
SIGNAL_CSV = "trading_multi_signals.csv"
SIGNAL_XLSX = "trading_multi_signals.xlsx"

LOCAL_TZ = tz.gettz("Africa/Porto-Novo")

# ---------------------------
# Paramètres spécifiques par symbole
# pip_size: valeur en price units pour 1 "pip"
# notes: indique si on traite les cotes comme 'pip-based' ou 'point-based'
# ---------------------------
SYMBOL_SETTINGS = {
    # Forex 4-decimals
    "EURUSD=X": {"pip_size": 0.0001, "type": "pip"},
    "GBPUSD=X": {"pip_size": 0.0001, "type": "pip"},
    # JPY pairs use 2-decimals pip
    "USDJPY=X": {"pip_size": 0.01, "type": "pip"},
    # Gold - often quoted with 2 decimals; treat 1 "pip" = 0.01 (you can change)
    "XAUUSD=X": {"pip_size": 0.01, "type": "point"},
    # Crypto - treat 1 "pip" = 1 USD (approx) or use relative ATR later
    "BTC-USD": {"pip_size": 1.0, "type": "point"},
    # Index (S&P)
    "^GSPC": {"pip_size": 0.1, "type": "point"},
}

# ---------------------------
# Helpers: indicators
# ---------------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

# ---------------------------
# Telegram
# ---------------------------
def send_telegram_message(text):
    if not TOKEN or not CHAT_ID:
        print("Telegram not configured.")
        return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print("Telegram error:", e)
        return False

# ---------------------------
# Data fetch
# ---------------------------
def fetch_ohlcv(ticker, interval, period):
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return None
        return df.dropna()
    except Exception as e:
        # print("fetch error", e)
        return None

# ---------------------------
# Utils pip/price
# ---------------------------
def pips_to_price_delta(symbol, pips):
    settings = SYMBOL_SETTINGS.get(symbol, {"pip_size": 0.0001})
    return pips * settings["pip_size"]

def suggest_lot(entry_price, sl_pips):
    pip_value_per_lot = 10.0
    risk_amount = ACCOUNT_EQUITY * RISK_PER_TRADE
    if sl_pips <= 0:
        return 0.001
    lot = max(0.001, round(risk_amount / (sl_pips * pip_value_per_lot), 4))
    return lot

# ---------------------------
# Logging helpers
# ---------------------------
def append_csv(path, row: dict):
    df_row = pd.DataFrame([row])
    header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=header, index=False)

def export_excel_from_csv(csv_path, xlsx_path):
    try:
        df = pd.read_csv(csv_path)
        df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print("Excel export error:", e)

# ---------------------------
# Multi-TF analysis per symbol
# ---------------------------
def analyze_symbol(symbol):
    results = {}
    enough_data = True
    for tf_label, interval in TF_INTERVALS.items():
        period = PERIODS.get(tf_label, "90d")
        df = fetch_ohlcv(symbol, interval, period)
        if df is None or len(df) < max(50, EMA_SLOW + 10):
            results[tf_label] = {"ok": False, "reason": "no_data"}
            enough_data = False
            continue

        df["EMA20"] = ema(df["Close"], EMA_FAST)
        df["EMA50"] = ema(df["Close"], EMA_SLOW)
        df["RSI"] = rsi(df["Close"], RSI_PERIOD)
        _, _, macdh = macd(df["Close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        df["MACD_HIST"] = macdh

        last = df.iloc[-1]
        # use iloc[0] when single-element Series warnings appear
        ema_ok = float(last["EMA20"]) > float(last["EMA50"])
        rsi_ok = float(last["RSI"]) > 50
        macd_ok = float(last["MACD_HIST"]) > 0

        ema_ok_sell = float(last["EMA20"]) < float(last["EMA50"])
        rsi_ok_sell = float(last["RSI"]) < 50
        macd_ok_sell = float(last["MACD_HIST"]) < 0

        results[tf_label] = {
            "ok": True,
            "ema_ok": ema_ok,
            "rsi_ok": rsi_ok,
            "macd_ok": macd_ok,
            "ema_ok_sell": ema_ok_sell,
            "rsi_ok_sell": rsi_ok_sell,
            "macd_ok_sell": macd_ok_sell,
            "last_price": float(last["Close"]),
            "df": df
        }

    buy = all([results[t]["ok"] and results[t]["ema_ok"] and results[t]["rsi_ok"] and results[t]["macd_ok"]
               for t in results if results[t]["ok"]])
    sell = all([results[t]["ok"] and results[t]["ema_ok_sell"] and results[t]["rsi_ok_sell"] and results[t]["macd_ok_sell"]
               for t in results if results[t]["ok"]])

    # ref price from 1H if available, else 4H, else 1D
    ref_price = None
    for pref in ("1H", "4H", "1D"):
        if results.get(pref) and results[pref]["ok"]:
            ref_price = results[pref]["last_price"]
            break

    return {"results": results, "buy": buy, "sell": sell, "price": ref_price, "enough_data": enough_data}

# ---------------------------
# Active positions per symbol
# ---------------------------
class ActivePosition:
    def __init__(self, side, entry, sl, tp1, tp2, suggested_lot):
        self.side = side
        self.entry = entry
        self.sl = sl
        self.tp1 = tp1
        self.tp2 = tp2
        self.suggested_lot = suggested_lot
        self.tp1_reached = False
        self.open = True
        self.open_time = datetime.now(timezone.utc)

    def check_price(self, current_price):
        if not self.open:
            return (None, None)
        # BUY
        if self.side == "BUY":
            if (not self.tp1_reached) and current_price >= self.tp1:
                self.tp1_reached = True
                new_sl = round(self.entry + pips_to_price_delta("EURUSD=X", TRAILING_AFTER_TP1_PIPS), 5)
                old_sl = self.sl
                self.sl = max(self.sl, new_sl)
                return ("tp1", f"TP1 reached. SL moved {old_sl}->{self.sl}")
            if current_price >= self.tp2:
                self.open = False
                return ("tp2", f"TP2 reached at {current_price:.5f}")
            if current_price <= self.sl:
                self.open = False
                return ("sl", f"SL hit at {current_price:.5f}")
        # SELL
        else:
            if (not self.tp1_reached) and current_price <= self.tp1:
                self.tp1_reached = True
                new_sl = round(self.entry - pips_to_price_delta("EURUSD=X", TRAILING_AFTER_TP1_PIPS), 5)
                old_sl = self.sl
                self.sl = min(self.sl, new_sl)
                return ("tp1", f"TP1 reached (short). SL moved {old_sl}->{self.sl}")
            if current_price <= self.tp2:
                self.open = False
                return ("tp2", f"TP2 reached at {current_price:.5f}")
            if current_price >= self.sl:
                self.open = False
                return ("sl", f"SL hit at {current_price:.5f}")
        return (None, None)

# ---------------------------
# Build & send signal per symbol
# ---------------------------
def build_and_send_signal(symbol, side, price):
    pip_size = SYMBOL_SETTINGS.get(symbol, {}).get("pip_size", 0.0001)
    sl_price = round(price - pips_to_price_delta(symbol, SL_PIPS), 5) if side == "BUY" else round(price + pips_to_price_delta(symbol, SL_PIPS), 5)
    tp1_price = round(price + pips_to_price_delta(symbol, TP1_PIPS), 5) if side == "BUY" else round(price - pips_to_price_delta(symbol, TP1_PIPS), 5)
    tp2_price = round(price + pips_to_price_delta(symbol, TP2_PIPS), 5) if side == "BUY" else round(price - pips_to_price_delta(symbol, TP2_PIPS), 5)

    suggested_lot = suggest_lot(price, SL_PIPS)

    msg = (
        f"*{symbol} SIGNAL*\n"
        f"Type: *{side}*\n"
        f"Price: `{price:.5f}`\n"
        f"SL: `{sl_price:.5f}` | TP1: `{tp1_price:.5f}` | TP2: `{tp2_price:.5f}`\n"
        f"Lot suggérée: `{suggested_lot}`\n"
        f"Règle: Multi-TF EMA20/EMA50 + RSI + MACD\n"
        f"⚠️ Teste en démo avant d'ouvrir un trade.\n"
        f"🕒 {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    ok = send_telegram_message(msg)

    sig_row = {
        "time_local": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "signal": side,
        "price": price,
        "sl": sl_price,
        "tp1": tp1_price,
        "tp2": tp2_price,
        "suggested_lot": suggested_lot,
        "sent": ok
    }
    append_csv(SIGNAL_CSV, sig_row)
    export_excel_from_csv(SIGNAL_CSV, SIGNAL_XLSX)

    return ActivePosition(side, price, sl_price, tp1_price, tp2_price, suggested_lot)

# ---------------------------
# MAIN LOOP
# ---------------------------
def main_loop():
    print("EURUSD Multi-instruments Bot démarré.")
    last_signal_time_by_symbol = {s: 0 for s in SYMBOLS}
    active_pos_by_symbol = {s: None for s in SYMBOLS}

    while True:
        try:
            for symbol in SYMBOLS:
                analysis = analyze_symbol(symbol)
                now_ts = time.time()
                append_csv(LOG_CSV, {
                    "time": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "buy": analysis["buy"],
                    "sell": analysis["sell"],
                    "price": analysis["price"]
                })

                # handle no active position
                active_pos = active_pos_by_symbol.get(symbol)
                if (active_pos is None) or (not active_pos.open):
                    if analysis["enough_data"]:
                        if analysis["buy"] and (now_ts - last_signal_time_by_symbol[symbol]) > SIGNAL_COOLDOWN_SECONDS:
                            entry_price = analysis["price"]
                            active_pos_by_symbol[symbol] = build_and_send_signal(symbol, "BUY", entry_price)
                            last_signal_time_by_symbol[symbol] = now_ts
                            print(f"[{symbol}] BUY sent @ {entry_price:.5f}")
                        elif analysis["sell"] and (now_ts - last_signal_time_by_symbol[symbol]) > SIGNAL_COOLDOWN_SECONDS:
                            entry_price = analysis["price"]
                            active_pos_by_symbol[symbol] = build_and_send_signal(symbol, "SELL", entry_price)
                            last_signal_time_by_symbol[symbol] = now_ts
                            print(f"[{symbol}] SELL sent @ {entry_price:.5f}")
                        else:
                            print(f"[{symbol}] No new signal.")
                    else:
                        print(f"[{symbol}] Pas assez de données.")
                else:
                    # monitor active position using latest 1H (or suitable tf) price
                    df1h = fetch_ohlcv(symbol, TF_INTERVALS["1H"], PERIODS["1H"])
                    if df1h is not None and not df1h.empty:
                        current_price = float(df1h.iloc[-1]["Close"])
                        evt, msg = active_pos.check_price(current_price)
                        if evt:
                            update_msg = (
                                f"*{symbol} Position update ({active_pos.side})*\n"
                                f"{msg}\n"
                                f"Current price: `{current_price:.5f}`\n"
                                f"Entry: `{active_pos.entry:.5f}` | SL:`{active_pos.sl:.5f}` | TP1:`{active_pos.tp1:.5f}` | TP2:`{active_pos.tp2:.5f}`"
                            )
                            send_telegram_message(update_msg)
                            append_csv(LOG_CSV, {"time": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                                                 "utc": datetime.now(timezone.utc).isoformat(),
                                                 "symbol": symbol, "event": evt, "message": msg, "price": current_price})
                            if evt in ("tp2", "sl"):
                                final_row = {
                                    "time_local": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                                    "utc": datetime.now(timezone.utc).isoformat(),
                                    "symbol": symbol,
                                    "event": evt,
                                    "side": active_pos.side,
                                    "entry": active_pos.entry,
                                    "close_price": current_price,
                                    "sl": active_pos.sl,
                                    "tp1": active_pos.tp1,
                                    "tp2": active_pos.tp2
                                }
                                append_csv(SIGNAL_CSV, final_row)
                                export_excel_from_csv(SIGNAL_CSV, SIGNAL_XLSX)
                                active_pos_by_symbol[symbol] = None
                    else:
                        print(f"[{symbol}] Monitoring: pas de prix 1H.")

                # petit délai entre symboles pour éviter surcharger yfinance
                time.sleep(1)

            # pause principale avant next loop
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print("Main loop error:", e)
            traceback.print_exc()
            append_csv(LOG_CSV, {"time": datetime.now(LOCAL_TZ).isoformat(), "event": f"error: {str(e)}"})
            time.sleep(10)

if __name__ == "__main__":
    main_loop()
