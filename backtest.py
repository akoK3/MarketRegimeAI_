import numpy as np
import pandas as pd


def run_backtest(df_labeled, strategy_col="Regime", price_col="Close"):
    """
    Simple regime-based backtest.
    
    Bull     → invested (long SPY)
    Bear     → cash
    Sideways → cash
    
    Compares against buy-and-hold benchmark.
    """
    df = df_labeled.copy()
    df = df.dropna(subset=[strategy_col, price_col])

    # Daily returns
    df["Market_Return"] = df[price_col].pct_change()

    # Strategy — only invested during Bull
    df["Position"] = np.where(df[strategy_col] == "Bull", 1, 0)

    # Shift position by 1 — can only act on yesterday's signal
    df["Position"] = df["Position"].shift(1)

    # Strategy daily return
    df["Strategy_Return"] = df["Position"] * df["Market_Return"]

    # Cumulative returns
    df["Cumulative_Market"] = (1 + df["Market_Return"]).cumprod()
    df["Cumulative_Strategy"] = (1 + df["Strategy_Return"]).cumprod()

    # ── Metrics ───────────────────────────────────────────────────────────────
    trading_days = 252

    def annualized_return(cum_returns):
        n_years = len(cum_returns) / trading_days
        return cum_returns.iloc[-1] ** (1 / n_years) - 1

    def sharpe_ratio(daily_returns):
        mean = daily_returns.mean()
        std = daily_returns.std()
        return (mean / std) * np.sqrt(trading_days) if std != 0 else 0

    def max_drawdown(cum_returns):
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        return drawdown.min()

    market_ret = df["Market_Return"].dropna()
    strategy_ret = df["Strategy_Return"].dropna()

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)

    print("\n--- Buy & Hold (SPY) ---")
    print(f"Total Return      : {(df['Cumulative_Market'].iloc[-1] - 1) * 100:.2f}%")
    print(f"Annualized Return : {annualized_return(df['Cumulative_Market']) * 100:.2f}%")
    print(f"Sharpe Ratio      : {sharpe_ratio(market_ret):.3f}")
    print(f"Max Drawdown      : {max_drawdown(df['Cumulative_Market']) * 100:.2f}%")

    print("\n--- Regime Strategy (Bull only) ---")
    print(f"Total Return      : {(df['Cumulative_Strategy'].iloc[-1] - 1) * 100:.2f}%")
    print(f"Annualized Return : {annualized_return(df['Cumulative_Strategy']) * 100:.2f}%")
    print(f"Sharpe Ratio      : {sharpe_ratio(strategy_ret):.3f}")
    print(f"Max Drawdown      : {max_drawdown(df['Cumulative_Strategy']) * 100:.2f}%")

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/SPY_labeled.csv", parse_dates=True, index_col=0)
    run_backtest(df)