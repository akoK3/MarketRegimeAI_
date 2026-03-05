import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_by_regime(df, feature_name):
    plt.figure(figsize=(14, 6))
    plt.plot(df[feature_name], label=feature_name, color='blue')

    regimes = {'Bull': 'green', 'Bear': 'red', 'Sideways': 'yellow'}

    for regime, color in regimes.items():
        plt.scatter(
            df.index[df['Regime'] == regime],
            df[feature_name][df['Regime'] == regime],
            color=color,
            label=regime,
            s=8
        )

    plt.title(f"{feature_name} by Market Regime")
    plt.xlabel("Date")
    plt.ylabel(feature_name)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/processed/SPY_features.csv", parse_dates=True, index_col=0)

    for feature in ['ATR', 'RSI', 'MACD']:
        plot_feature_by_regime(df, feature)
