import os
import pandas as pd
import yfinance as yf
from datetime import datetime


def fetch_data_yahoo(ticker, start="2010-01-01", end=None):
    """
    Fetch historical stock data using yfinance.
    Saves data to /data/raw/<TICKER>_data.csv
    """
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    print(f"Fetching {ticker} data from Yahoo Finance...")

    data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty:
        print(f"No data found for {ticker} using Yahoo Finance.")
        return None

    save_path = f"data/raw/{ticker}_data.csv"
    os.makedirs("data/raw", exist_ok=True)
    data.to_csv(save_path)

    print(f"Data saved to {save_path}")
    return data


def load_local_data(ticker):
    """
    Load previously downloaded data if available.
    """
    file_path = f"data/raw/{ticker}_data.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print("No local data found. Fetch data first.")
        return None


if __name__ == "__main__":
    # Test function
    fetch_data_yahoo("SPY")
