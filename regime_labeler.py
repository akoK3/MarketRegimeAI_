import os
import numpy as np
import pandas as pd
import yfinance as yf


def calculate_atr(df, window=14):
    """Calculate Average True Range (Wilder, 1978)"""
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr = true_range.ewm(com=window-1, adjust=False).mean()
    return atr


def label_market_regimes(df, return_window=20, atr_window=14, k=1.5):
    """
    Label market regimes using forward returns and ATR-adjusted thresholds.
    
    Bull:     forward_return > K * ATR(14)
    Bear:     forward_return < -K * ATR(14)
    Sideways: everything in between
    
    Parameters:
        return_window: number of days ahead to calculate forward return
        atr_window: ATR lookback period (14 = Wilder's standard)
        k: multiplier for ATR threshold (sensitivity parameter)
    """
    df = df.copy()

    df["Forward_Return"] = df["Close"].pct_change(return_window).shift(-return_window)

    df["ATR_14"] = calculate_atr(df, window=atr_window)
    df["ATR_pct"] = df["ATR_14"] / df["Close"]

    bull_threshold = k * df["ATR_pct"]
    bear_threshold = -k * df["ATR_pct"]

    df["Regime"] = "Sideways"
    df.loc[df["Forward_Return"] > bull_threshold, "Regime"] = "Bull"
    df.loc[df["Forward_Return"] < bear_threshold, "Regime"] = "Bear"

    df.loc[df["Forward_Return"].isna(), "Regime"] = np.nan

    df = df.drop(columns=["ATR_14", "ATR_pct"], errors="ignore")
    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = yf.download("SPY", start="2000-01-01", end="2025-01-01", progress=False)
    if df.empty:
        raise ValueError("No data returned from Yahoo Finance for SPY.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv("data/raw/spy.csv")

    print("Columns in data:", df.columns.tolist())

    cols_to_numeric = ["Close", "Open", "High", "Low", "Volume"]
    df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors="coerce")

    # 5-day horizon, K=1.0 — shorter horizon, lower threshold
    df_labeled = label_market_regimes(df, return_window=5, atr_window=14, k=1.0)

    os.makedirs("data/processed", exist_ok=True)
    df_labeled.to_csv("data/processed/SPY_labeled.csv", index=True)

    print(df_labeled[["Close", "Forward_Return", "Regime"]].tail(30))

    print("\nRegime Distribution:")
    print(df_labeled["Regime"].value_counts(normalize=True).round(3))