import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_regimes(csv_path):
    # Load CSV where Date is stored as the first column (index)
    df = pd.read_csv(       # reads csv file into a dataframe
        csv_path,           # file location
        index_col=0,        # first column is Date
        parse_dates=True,   # parse it as datetime
    )
    # creating plot dimentions by 'figure'
    plt.figure(figsize=(14, 6))

    # drawing 'close' price line
    plt.plot(df['Close'], label='Close Price', color='black', linewidth=1)

    # using a dictionary to map regimes to colors
    regimes = {'Bull': 'green', 'Bear': 'red', 'Sideways': 'yellow'}

    for regime, color in regimes.items():
        plt.scatter(
            df.index[df['Regime'] == regime],       #x values (dates)   
            df['Close'][df['Regime'] == regime],    #y values
            color=color,         #dot colors
            label=regime,        #Legend label
            s=1                 #dot size
        )

    plt.title("Market Regime Detection: Bull, Bear, and Sideways")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()

    # Save the chart
    os.makedirs("reports/figures", exist_ok=True)
    output_path = "reports/figures/SPY_regime_chart.png"
    plt.savefig(output_path)
    plt.show()

    print(f"Chart saved to: {output_path}")


if __name__ == "__main__":
    csv_path = "data/processed/SPY_labeled.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Run regime_labeler.py first to generate labeled CSV.")
    plot_regimes(csv_path)
