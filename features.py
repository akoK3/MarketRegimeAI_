import pandas as pd
import numpy as np


# ── ATR (Wilder EWM — consistent with regime_labeler.py) ─────────────────────
def add_ATR(df, period=14):
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    # EWM — consistent with regime_labeler.py
    df["ATR"] = true_range.ewm(com=period-1, adjust=False).mean()
    # Normalize ATR to percentage (comparable across time)
    df["ATR_pct"] = df["ATR"] / df["Close"]
    return df


# ── RSI (Wilder EWM — more accurate than simple rolling mean) ────────────────
def add_RSI(df, period=14):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Wilder uses EWM not simple rolling mean
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


# ── MACD ─────────────────────────────────────────────────────────────────────
def add_MACD(df, short_window=12, long_window=26, signal_window=9):
    ema_12 = df["Close"].ewm(span=short_window, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=long_window, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["Signal"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()
    # MACD histogram — captures momentum shifts
    df["MACD_hist"] = df["MACD"] - df["Signal"]
    return df


# ── Moving Averages ───────────────────────────────────────────────────────────
def add_moving_averages(df):
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()
    # Percentage distance from MAs (normalized, not absolute)
    df["Close_to_SMA20"] = (df["Close"] / df["SMA20"]) - 1
    df["Close_to_SMA50"] = (df["Close"] / df["SMA50"]) - 1
    df["Close_to_SMA200"] = (df["Close"] / df["SMA200"]) - 1
    df["SMA20_minus_SMA50"] = (df["SMA20"] / df["SMA50"]) - 1
    return df


# ── Volatility (rolling std of returns) ──────────────────────────────────────
def add_volatility(df, windows=[10, 20]):
    daily_ret = df["Close"].pct_change()
    for w in windows:
        df[f"Vol_{w}d"] = daily_ret.rolling(window=w).std()
    return df


# ── Volume Features ───────────────────────────────────────────────────────────
def add_volume_features(df, window=20):
    # Volume relative to its rolling average (normalized)
    df["Volume_SMA20"] = df["Volume"].rolling(window=window).mean()
    df["Volume_ratio"] = df["Volume"] / df["Volume_SMA20"]
    # Volume trend — is volume increasing or decreasing?
    df["Volume_change"] = df["Volume"].pct_change()
    return df


# ── Momentum (Rate of Change) ─────────────────────────────────────────────────
def add_momentum(df, windows=[5, 10, 20]):
    for w in windows:
        df[f"ROC_{w}d"] = df["Close"].pct_change(w)
    return df


# ── Lag Features ──────────────────────────────────────────────────────────────
def add_lag_features(df):
    # RSI lags
    df["RSI_lag1"] = df["RSI"].shift(1)
    df["RSI_lag5"] = df["RSI"].shift(5)
    df["RSI_lag10"] = df["RSI"].shift(10)

    # MACD lags
    df["MACD_lag1"] = df["MACD"].shift(1)
    df["MACD_lag5"] = df["MACD"].shift(5)

    # SMA distance lags
    df["Close_to_SMA20_lag1"] = df["Close_to_SMA20"].shift(1)
    df["Close_to_SMA20_lag5"] = df["Close_to_SMA20"].shift(5)

    # ATR pct lags
    df["ATR_pct_lag1"] = df["ATR_pct"].shift(1)

    return df


# ── Master Function ───────────────────────────────────────────────────────────
def add_all_features(df):
    df = add_ATR(df)
    df = add_RSI(df)
    df = add_MACD(df)
    df = add_moving_averages(df)
    df = add_volatility(df)
    df = add_volume_features(df)
    df = add_momentum(df)
    df = add_lag_features(df)
    return df
