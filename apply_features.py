import pandas as pd
from features import add_all_features

csv_path = "data/processed/SPY_labeled.csv"

df = pd.read_csv(csv_path, parse_dates=True, index_col=0)

df = add_all_features(df)

print(df.tail())

df = df.loc[:, ~df.columns.duplicated()]

df.to_csv("data/processed/SPY_features.csv")
print("Feature-enhanced dataset saved as: data/processed/SPY_features.csv")