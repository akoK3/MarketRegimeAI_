import matplotlib.pyplot as plt
import pandas as pd

def plot_proper_MACD(df):
    plt.figure(figsize=(14, 6))

    # Plot MACD Line and Signal Line
    plt.plot(df.index, df['MACD'], label='MACD Line', color='blue', linewidth=1.5)
    plt.plot(df.index, df['Signal'], label='Signal Line', color='orange', linewidth=1.5)

    # Histogram (MACD - Signal)
    plt.bar(df.index, df['MACD'] - df['Signal'], color='gray', alpha=0.4, label='Histogram')

    # Zero Line
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    plt.title("Proper MACD Visualization (Zoomed for Accuracy)")
    plt.xlabel("Date")
    plt.ylabel("MACD Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data/processed/SPY_features.csv", parse_dates=True, index_col=0)
    
    plot_proper_MACD(df)
