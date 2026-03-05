import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/SPY_features.csv", parse_dates=True, index_col=0)

print(df.groupby("Regime")[["RSI", "MACD_hist", "ATR_pct", "Close_to_SMA20", "Vol_20d"]].mean().round(4))