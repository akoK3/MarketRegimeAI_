import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load feature-enhanced dataset
df = pd.read_csv("data/processed/SPY_features.csv", parse_dates=True, index_col=0)

# Select features for ML
feature_cols = [
    'ATR', 'RSI', 'MACD', 'Signal',
    'Close_to_SMA20', 'Close_to_SMA50', 'SMA20_minus_SMA50',
    'RSI_lag1', 'RSI_lag5', 'MACD_lag1', 'Close_to_SMA20_lag1'
]

X = df[feature_cols].copy()
y = df['Regime'].copy()

# Drop missing rows
X = X.dropna()
y = y.loc[X.index]
y = y.dropna()
X = X.loc[y.index]

print("Feature Data Shape (X):", X.shape)
print("Target Value Counts (y):")
print(y.value_counts())

train_size = int(len(X) * 0.8)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Standardize features
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# !!!  SCALING  !!!
scaler = StandardScaler()
# Fit on training data and transform both training and test data
# transform does what it means
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# !!!  SCALING  !!!

# Simple Logistic Regression Model
model = LogisticRegression(max_iter=1000)

# model trains on measuring calculators, which have correct labels
model.fit(X_train_scaled, y_train)

# guesses labels for test data
y_pred = model.predict(X_test_scaled)


# take first 10 predictions and take first 10 true labels, where we make series of 'values'
# and despite format of index use iloc to get first 10 values.
print("Predictions [10:20]:", y_pred[100:110])
print("True labels [10:20]:", y_test.iloc[100:110].values)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

rf_pred = rf_model.predict(X_test_scaled)

print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("\nClassification Report:\n", classification_report(y_test, rf_pred))
