import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import fetch_data_yahoo
from regime_labeler import label_market_regimes
from features import add_all_features
from evaluation import run_evaluation
from backtest import run_backtest

# ── Config ────────────────────────────────────────────────────────────────────
TICKER = "SPY"
START = "2000-01-01"
END = "2025-01-01"
RETURN_WINDOW = 5
ATR_WINDOW = 14
K = 1.0

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH = f"data/raw/{TICKER}_data.csv"
LABELED_PATH = "data/processed/SPY_labeled.csv"
FEATURES_PATH = "data/processed/SPY_features.csv"


def main():
    print("=" * 60)
    print("MarketRegimeAI Pipeline")
    print("=" * 60)

    # ── Step 1: Load Data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    df = fetch_data_yahoo(TICKER, start=START, end=END)
    if df is None:
        raise ValueError("Data loading failed.")
    print(f"Loaded {len(df)} rows for {TICKER}")

    # ── Step 2: Label Regimes ─────────────────────────────────────────────────
    print("\n[2/5] Labeling regimes...")
    df_labeled = label_market_regimes(df, return_window=RETURN_WINDOW, atr_window=ATR_WINDOW, k=K)
    os.makedirs("data/processed", exist_ok=True)
    df_labeled.to_csv(LABELED_PATH, index=True)
    print("Regime distribution:")
    print(df_labeled["Regime"].value_counts(normalize=True).round(3))

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    print("\n[3/5] Engineering features...")
    df_features = add_all_features(df_labeled)
    df_features = df_features.loc[:, ~df_features.columns.duplicated()]
    df_features.to_csv(FEATURES_PATH, index=True)
    print(f"Features saved: {df_features.shape[1]} columns, {df_features.shape[0]} rows")

    # ── Step 4: Evaluation ────────────────────────────────────────────────────
    print("\n[4/5] Running evaluation...")
    run_evaluation(FEATURES_PATH)

    # ── Step 5: Backtest ──────────────────────────────────────────────────────
    print("\n[5/5] Running backtest...")
    df_labeled = pd.read_csv(LABELED_PATH, parse_dates=True, index_col=0)
    run_backtest(df_labeled)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()